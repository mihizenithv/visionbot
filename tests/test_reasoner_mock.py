"""Exercise the reasoner + router against a fake OpenAI-compatible server (no credits, no network)."""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from visionbot.config import ReasonerCfg
from visionbot.reasoner import Reasoner, Router, Trigger, parse_json
from visionbot.scene import Event

seen = {}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        seen["auth"] = self.headers.get("Authorization")
        self._send(200, {"data": [{"id": "meta/llama-3.1-8b-instruct"}, {"id": "nvidia/cosmos-reason2-8b"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen["req"] = req
        if req["model"] == "broken":
            return self._send(500, {"error": "boom"})
        answer = {"scene": "Two people talk near a table.", "people": [{"id": "#3", "activity": "waving at the robot",
                  "attention_to_robot": True, "needs_help": False}], "trigger_explanation": "Person #3 raised a hand.",
                  "hazards": [], "robot_should": "Turn towards #3 and greet them."}
        content = f"<think>\nPerson 3 has a raised hand...\n</think>\n\n<answer>\n{json.dumps(answer)}\n</answer>"
        self._send(200, {"choices": [{"message": {"content": content}}]})


srv = HTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=srv.serve_forever, daemon=True).start()
cfg = ReasonerCfg(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key_env="VB_FAKE_KEY")
import os
os.environ["VB_FAKE_KEY"] = "nvapi-test"

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)

# 1. model discovery picks the first *preferred* model the account actually has
rs = Reasoner(cfg).start()
results = []
rs.on_result = results.append
snap = {"people": [{"id": 3, "label": "person", "box": [0.2, 0.2, 0.2, 0.5], "dist_m": 1.8, "cues": {"hand_raised": ["right"]}}],
        "objects": [], "remembered": [], "events": [{"text": "Person #3 raised a hand"}]}
rs.submit(Trigger("hand_raised", 2, "Person #3 raised a hand", time.perf_counter()), np.zeros((480, 640, 3), np.uint8), snap)
t = time.time()
while not results and time.time() - t < 10:
    time.sleep(0.05)
check("result delivered", results)
res = results[0] if results else {}
check("picked preferred available model", rs.model == "nvidia/cosmos-reason2-8b")
check("bearer auth", seen.get("auth") == "Bearer nvapi-test")
msg = seen["req"]["messages"][-1]["content"]
check("image sent as base64 data URL", msg[0]["type"] == "image_url" and msg[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
check("scene state sent", '"hand_raised"' in msg[1]["text"])
check("<think>/<answer> parsed", res.get("robot_should", "").startswith("Turn towards"))

# 2. merging back into scene memory accepts '#3'-style ids
from visionbot.scene import SceneMemory, Entity
sm = SceneMemory(70)
sm.ents[3] = Entity(3, {"person": 1.0}, [0, 0, 1, 1], 0, 0)
sm.apply_reasoning(res)
check("activity merged into entity #3", sm.ents[3].activity == "waving at the robot")

# 3. ask() gets priority and returns
ans = rs.ask("What is person 3 doing?", np.zeros((480, 640, 3), np.uint8), snap, timeout=10)
check("ask() answered", "error" not in ans)

# 4. errors are contained
rs.model = "broken"
err = rs._call("scene", Trigger("t", 1, "x", 0), np.zeros((10, 10, 3), np.uint8), snap)
check("HTTP 500 returns an error dict, no exception", "error" in err)
check("parse_json tolerates junk", parse_json("no json here") is None and parse_json('```json\n{"a":1}\n```') == {"a": 1})

# 5. router: debounce merges an episode into one call; urgent events bypass pacing
r = Router(ReasonerCfg(quiet_s=0.2, max_hold_s=1.0, min_gap_s=0.0, burst=1, daily_budget=300))
evs = [Event(0, "person_entered", 2, f"Person #{i} came into view") for i in range(4)]
r.offer(evs, 0.0, 3.0, True)
check("no call while episode still active", r.ready() is None)
time.sleep(0.25)
r.offer([], 0.0, 3.0, True)
trig = r.ready()
check("one merged call after quiet period", trig is not None and trig.text.count("came into view") == 4)
r.offer([Event(0, "person_entered", 2, "Person #9 came into view")], 0.0, 3.0, True)
time.sleep(0.25)
r.offer([], 0.0, 3.0, True)
check("pacing blocks non-urgent call when bucket empty", r.ready() is None)
r.offer([Event(0, "possible_fall", 3, "Person #9 went from standing to lying")], 0.0, 3.0, True)
trig = r.ready()
check("urgent fall event overrides pacing", trig is not None and trig.priority == 3)

rs.stop()
srv.shutdown()
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
