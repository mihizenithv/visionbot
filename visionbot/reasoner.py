"""Slow lane: a vision-language model explains *what is happening*, called only when it matters.

The router decides when a call is worth it (surprise spike, important event, or a slow heartbeat),
enforces the free tier's rate limit and a daily credit budget, and always sends the *latest* frame
(never a stale backlog). The VLM receives the frame with track IDs drawn on it plus the structured
scene state, so its answer refers to the same IDs the fast lane uses and can be merged back.

Works with any OpenAI-compatible endpoint; configured for NVIDIA's free API (build.nvidia.com).
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import requests

from .config import ReasonerCfg

SYSTEM = (
    "You are the eyes of a home-assistant robot. The picture is its camera view; people and things are boxed "
    "and tagged like '#3 person' only so you can refer to them in the people list. Below are the robot's own "
    "measurements and the reason it asked you to look. Be brief and concrete. In \"scene\", describe people by "
    "where they are (for example 'the person on the left'), never by tag number. Describe faces only by what is "
    "visible (smiling, frowning, eyes closed); never guess identity, name, age or gender. Reply with ONLY a JSON "
    'object: {"scene": one sentence, "people": [{"id": tag number, "activity": short phrase, '
    '"expression": short phrase or ""}], "hazards": [short strings, usually none], '
    '"robot_should": short suggestion or "nothing"}'
)

TRIGGER_PRIORITY = {"possible_fall": 3, "hand_raised": 2, "person_entered": 2, "person_close": 2,
                    "object_taken": 2, "surprise": 2, "question": 3}


@dataclass
class Trigger:
    kind: str
    priority: int
    text: str
    t: float


class Router:
    """Decides whether now is a good moment to spend a VLM call."""

    def __init__(self, cfg: ReasonerCfg):
        self.cfg = cfg
        self.calls: deque[float] = deque()
        self.day_calls = 0
        self.day = time.strftime("%Y-%m-%d")
        self.last_call = -1e9
        self.pending: Trigger | None = None
        self.stats = {"events_seen": 0, "surprise_spikes": 0, "calls": 0, "suppressed": 0}
        self._surprise_armed = True
        self.cluster: dict | None = None
        # credit pacing: a token bucket refilled at daily_budget / active_hours, so a busy hour can't
        # spend the whole day's credits; urgent triggers may overdraw a little
        self.tokens = float(cfg.burst)
        self.refill = cfg.daily_budget / (cfg.active_hours * 3600.0)
        self._t_tok = time.perf_counter()

    def budget_left(self) -> int:
        if time.strftime("%Y-%m-%d") != self.day:
            self.day, self.day_calls = time.strftime("%Y-%m-%d"), 0
        return self.cfg.daily_budget - self.day_calls

    def offer(self, events, surprise_score: float, spike: float, anything_visible: bool) -> None:
        now = time.perf_counter()
        for ev in events:
            self.stats["events_seen"] += 1
            if ev.priority >= 2:
                self._add(ev.kind, ev.priority, ev.text, now)
        # hysteresis: one trigger per surprise episode, re-armed once the scene calms down
        if surprise_score >= spike and self._surprise_armed:
            self._surprise_armed = False
            self.stats["surprise_spikes"] += 1
            self._add("surprise", 2, f"Something changed unexpectedly", now)
        elif surprise_score < spike * 0.5:
            self._surprise_armed = True
        # promote a finished episode to the pending trigger
        c = self.cluster
        if c and (c["prio"] >= 3 or now - c["last"] >= self.cfg.quiet_s or now - c["first"] >= self.cfg.max_hold_s):
            kinds = sorted(set(c["kinds"]), key=c["kinds"].index)
            text = "; ".join(dict.fromkeys(c["texts"]))[:600]
            self._candidate(Trigger(kinds[0] if len(kinds) == 1 else "+".join(kinds[:3]), c["prio"], text, now))
            self.cluster = None
        if anything_visible and now - self.last_call > self.cfg.heartbeat_s and self.pending is None and self.cluster is None:
            self._candidate(Trigger("heartbeat", 1, "Periodic check of the scene", now))

    def _add(self, kind: str, prio: int, text: str, now: float) -> None:
        """Add an event to the current episode (debounce window)."""
        if self.cluster is None:
            self.cluster = {"first": now, "last": now, "prio": prio, "kinds": [], "texts": []}
        c = self.cluster
        c["last"], c["prio"] = now, max(c["prio"], prio)
        c["kinds"].append(kind)
        c["texts"].append(text)

    def _candidate(self, trig: Trigger) -> None:
        if self.pending is None or trig.priority >= self.pending.priority:
            if self.pending is not None:
                self.stats["suppressed"] += 1
            self.pending = trig
        else:
            self.stats["suppressed"] += 1

    def ready(self) -> Trigger | None:
        """Pop the pending trigger if limits allow a call right now."""
        if self.pending is None:
            return None
        now = time.perf_counter()
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        urgent = self.pending.priority >= 3
        self.tokens = min(float(self.cfg.burst), self.tokens + (now - self._t_tok) * self.refill)
        self._t_tok = now
        if len(self.calls) >= self.cfg.max_rpm or self.budget_left() <= 0:
            return None
        if self.tokens < 1.0 and not (urgent and self.tokens > -3.0):
            if now - self.pending.t > 10 and not urgent:
                self.pending = None
                self.stats["suppressed"] += 1
            return None
        if not urgent and now - self.last_call < self.cfg.min_gap_s:
            return None
        if now - self.pending.t > 10 and not urgent:     # too old to matter; drop it
            self.pending = None
            self.stats["suppressed"] += 1
            return None
        trig, self.pending = self.pending, None
        self.tokens -= 1.0
        self.calls.append(now)
        self.last_call = now
        self.day_calls += 1
        self.stats["calls"] += 1
        return trig


def annotate_for_vlm(bgr: np.ndarray, snapshot: dict, long_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = long_side / max(h, w)
    img = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    for e in snapshot["people"] + snapshot["objects"]:
        x, y, bw, bh = e["box"]
        p0, p1 = (int(x * W), int(y * H)), (int((x + bw) * W), int((y + bh) * H))
        col = (255, 200, 0) if e["label"] == "person" else (80, 220, 80)
        cv2.rectangle(img, p0, p1, col, 2)
        tag = f"#{e['id']} {e['label']}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p0[0], max(0, p0[1] - th - 6)), (p0[0] + tw + 6, p0[1]), col, -1)
        cv2.putText(img, tag, (p0[0] + 3, p0[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _compact(snapshot: dict) -> dict:
    keep = ("id", "label", "dist_m", "cues", "novelty", "state", "activity")
    return {
        "people": [{k: e[k] for k in keep if k in e} for e in snapshot["people"]],
        "objects": [{k: e[k] for k in keep if k in e} for e in snapshot["objects"]][:20],
        "recently_left_view": [{"id": e["id"], "label": e["label"], "seen_s_ago": e["seen_s_ago"]} for e in snapshot["remembered"][:6]],
        "recent_events": [e["text"] for e in snapshot["events"][-8:]],
    }


def parse_json(text: str) -> dict | None:
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if m:
        text = m.group(1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    try:            # trailing prose containing braces: decode just the first complete object
        obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def humanize(text: str, snap: dict, sentence: bool = True) -> str:
    """Replace tag numbers ('#3', 'person 3', '#9 cup') with where that person or thing is."""
    if not isinstance(text, str) or not text:
        return text
    from .scene import side_of
    ents = {e["id"]: e for e in snap.get("people", []) + snap.get("objects", []) + snap.get("remembered", [])}

    def name(m):
        e = ents.get(int(m.group(2)))
        return "someone" if e is None else f"the {e['label']} {side_of(e['box'])}"

    text = re.sub(r"(?i)(\bthe\s+)?\bperson\s*#?\s*(\d+)\b", name, text)
    for e in ents.values():                                  # '#9 cup': the label after the tag is redundant
        text = re.sub(rf"(?i)(\bthe\s+)?#\s*({e['id']})\s+{re.escape(e['label'])}\b", name, text)
    text = re.sub(r"(?i)(\bthe\s+)?#\s*(\d+)\b", name, text)
    return text[:1].upper() + text[1:] if (text and sentence) else text


class Reasoner:
    def __init__(self, cfg: ReasonerCfg):
        self.cfg = cfg
        self.key = cfg.api_key
        self.model: str | None = None
        self.bad_models: set[str] = set()                  # listed but not served for this account
        self.strikes: dict[str, int] = {}                  # consecutive timeouts / 5xx per model
        self.status = "no API key (set NVIDIA_API_KEY)" if not self.key else "idle"
        self.last: dict | None = None
        self.last_ms = 0.0
        self.history: deque[dict] = deque(maxlen=30)
        self._scene_job: tuple | None = None
        self._ask_job: tuple | None = None
        self.snapshots: dict[int, bytes] = {}             # small JPEG of the frame behind each entry
        self._next_entry = int(time.time()) * 100    # unique across restarts, so no browser shows a stale snapshot
        self._cv = threading.Condition()
        self._stop = False
        self.on_result = None
        self.session = requests.Session()
        if self.key:
            self.session.headers.update({"Authorization": f"Bearer {self.key}", "Accept": "application/json"})
        self._thread = threading.Thread(target=self._run, name="reasoner", daemon=True)

    @property
    def available(self) -> bool:
        return bool(self.key) and self.cfg.enabled

    def start(self) -> "Reasoner":
        if self.available:
            self._thread.start()
        return self

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    # ---------------------------------------------------------------- model discovery
    def _pick_model(self) -> str:
        if self.model:
            return self.model
        try:
            r = self.session.get(f"{self.cfg.base_url}/models", timeout=15)
            r.raise_for_status()
            have = {m["id"] for m in r.json().get("data", [])} - self.bad_models
            for m in self.cfg.models:
                if m in have:
                    self.model = m
                    return m
            vis = sorted(m for m in have if any(t in m for t in ("vision", "-vl", "cosmos-reason", "reasoner", "multimodal")))
            if vis:
                self.model = vis[0]
                return self.model
        except requests.RequestException as e:
            self.status = f"model list failed: {e.__class__.__name__}"
        self.model = next((m for m in self.cfg.models if m not in self.bad_models), self.cfg.models[-1])
        return self.model

    # ---------------------------------------------------------------- calls
    def submit(self, trig: Trigger, bgr: np.ndarray, snapshot: dict) -> dict:
        """Write a journal entry *now* from the robot's own perception, and queue the language model to
        add its interpretation. A newer job replaces an unstarted older one (latest wins)."""
        entry = {"entry": self._next_entry, "at": time.strftime("%H:%M:%S"), "trigger": trig.kind,
                 "note": trig.text, "pending": True, "model": self.model}
        h, w = bgr.shape[:2]
        sc = 360 / max(h, w)
        ok, jpg = cv2.imencode(".jpg", cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA),
                               [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            self.snapshots[self._next_entry] = jpg.tobytes()
            for k in [k for k in self.snapshots if k < self._next_entry - 40]:
                del self.snapshots[k]
        self._next_entry += 1
        self.history.append(entry)
        with self._cv:
            if self._scene_job is not None:                 # superseded before the model saw it
                self._scene_job[4]["pending"] = False
            self._scene_job = ("scene", trig, bgr, snapshot, entry)
            self._cv.notify_all()
        return entry

    def ask(self, question: str, bgr: np.ndarray, snapshot: dict, timeout: float = 110) -> dict:
        """Synchronous natural-language question about the current scene and memory."""
        if not self.available:
            return {"error": self.status}
        done = threading.Event()
        box: dict = {}
        with self._cv:
            self._ask_job = ("ask", Trigger("question", 3, question, time.perf_counter()), bgr, snapshot, (done, box))
            self._cv.notify_all()
        done.wait(timeout)
        return box or {"error": "timed out"}

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._ask_job is None and self._scene_job is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                if self._ask_job is not None:              # user questions jump the queue
                    (kind, trig, bgr, snap, reply), self._ask_job = self._ask_job, None
                else:
                    (kind, trig, bgr, snap, reply), self._scene_job = self._scene_job, None
            try:
                res = self._call(kind, trig, bgr, snap)
            except Exception as e:     # network errors must never take down perception
                res = {"error": f"{e.__class__.__name__}: {e}"[:300]}
            for k in ("scene", "robot_should", "answer"):
                if k in res:
                    res[k] = humanize(res[k], snap)
            if isinstance(res.get("hazards"), list):
                res["hazards"] = [humanize(str(x), snap) for x in res["hazards"] if str(x).strip()]
            for p in res.get("people") or []:
                if isinstance(p, dict):
                    for k in ("activity", "expression"):
                        if isinstance(p.get(k), str):
                            p[k] = humanize(p[k], snap, sentence=False)
            res.update({"model": self.model, "latency_ms": round(self.last_ms)})
            if kind == "ask":
                reply[1].update(res)
                reply[0].set()
                continue
            entry = reply                                   # the journal entry written at submit time
            entry.update(res)
            entry["pending"] = False
            if "error" not in res:
                self.last = entry
            if self.on_result:
                self.on_result(entry)

    def _call(self, kind: str, trig: Trigger, bgr: np.ndarray, snap: dict, _retry: bool = True) -> dict:
        model = self._pick_model()
        img = annotate_for_vlm(bgr, snap, self.cfg.image_long_side)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(jpg.tobytes()).decode()
        state = json.dumps(_compact(snap), separators=(",", ":"))
        if kind == "ask":
            text = (f"Scene state and memory: {state}\n\nQuestion from the user: {trig.text}\n"
                    'Answer using the image and the memory. Reply with ONLY JSON: {"answer": str, "ids": [int]}')
            system = SYSTEM.split(" Reply with ONLY")[0]
        else:
            text = f"Trigger: {trig.text}\nScene state: {state}"
            system = SYSTEM
        # Instructions go in the user turn, not a system message: Llama 3.2 Vision largely ignores system
        # prompts when an image is attached and falls back to prose. Ending on "JSON:" anchors the format.
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": f"{system}\n\n{text}\n\nJSON:"},
                ]},
            ],
            "max_tokens": 240, "temperature": 0.2, "stream": False,
        }
        self.status = f"calling {model}"
        t0 = time.perf_counter()
        try:
            r = self.session.post(f"{self.cfg.base_url}/chat/completions", json=body, timeout=self.cfg.timeout_s)
        except requests.Timeout:
            r = None
        self.last_ms = (time.perf_counter() - t0) * 1000
        if r is not None and r.status_code == 404:
            # the catalogue lists some models the free tier doesn't actually serve; drop it for good
            self.bad_models.add(model)
            self.model = None
            if _retry:
                return self._call(kind, trig, bgr, snap, _retry=False)
            self.status = "HTTP 404"
            return {"error": f"{model}: not served for this account"}
        if r is None or r.status_code >= 500:
            # a slow or hiccuping queue is usually temporary: only give up on the model after 3 in a row
            self.strikes[model] = self.strikes.get(model, 0) + 1
            if self.strikes[model] >= 3:
                self.bad_models.add(model)
                self.model = None
            self.status = "timeout" if r is None else f"HTTP {r.status_code}"
            if r is not None and _retry:
                time.sleep(2.0)
                return self._call(kind, trig, bgr, snap, _retry=False)
            return {"error": f"{model}: {self.status} (free-tier server busy)"}
        self.strikes[model] = 0
        if r.status_code >= 400:
            self.status = f"HTTP {r.status_code}"
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        content = r.json()["choices"][0]["message"].get("content") or ""
        parsed = parse_json(content)
        self.status = "idle"
        if parsed is None:
            # prose reply: keep its first paragraph, minus a leading "Scene:" label
            first = re.sub(r"^\s*(scene|answer)\s*:\s*", "", content.strip().split("\n\n")[0], flags=re.I).strip()
            return {"scene": first[:320], "people": [], "hazards": [], "robot_should": "nothing", "raw": True}
        if "scene" not in parsed:
            # small models sometimes return just one element of the schema; keep what is usable
            if "activity" in parsed and "id" in parsed:
                parsed = {"people": [parsed]}
            parsed.setdefault("scene", "")
        return parsed
