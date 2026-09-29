"""Check your NVIDIA API key: lists the vision models your account can use and makes ONE test call
(1 credit) on a sample image that is not from your camera.

    .venv/Scripts/python.exe tests/check_nvidia.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import requests

from visionbot.config import Config
from visionbot.reasoner import Reasoner, Trigger

cfg = Config().reasoner
if not cfg.api_key:
    sys.exit("No key found. Create one at build.nvidia.com, then add a line  NVIDIA_API_KEY=nvapi-...  to "
             f"{Path(__file__).resolve().parent.parent / '.env'}")
r = requests.get(f"{cfg.base_url}/models", headers={"Authorization": f"Bearer {cfg.api_key}"}, timeout=20)
print("models endpoint:", r.status_code)
ids = sorted(m["id"] for m in r.json().get("data", []))
vision = [m for m in ids if any(t in m for t in ("vision", "-vl", "cosmos", "reason", "multimodal"))]
print(f"{len(ids)} models on the account; vision-capable candidates:\n  " + "\n  ".join(vision[:30]))
print("preferred order:", cfg.models)

rs = Reasoner(cfg)
img = cv2.VideoCapture(str(Path(__file__).resolve().parent.parent / "samples" / "vtest.avi")).read()[1]
snap = {"people": [], "objects": [], "remembered": [], "events": []}
t = time.perf_counter()
res = rs._call("scene", Trigger("test", 1, "Connectivity test: describe the scene", t), img, snap)
print(f"\nmodel used: {rs.model}  ({(time.perf_counter() - t) * 1000:.0f} ms)")
print(res)
