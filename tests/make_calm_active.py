"""Build a 'home-like' test clip from vtest.avi: long calm stretches (frozen frame + sensor noise +
slow lighting drift) broken by short bursts of real motion. A home robot's camera spends most of
its time looking at a scene where nothing happens; this is what adaptive gating is designed for."""
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parent.parent / "samples"
cap = cv2.VideoCapture(str(root / "vtest.avi"))
fps = cap.get(cv2.CAP_PROP_FPS) or 10
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
h, w = frames[0].shape[:2]
out = cv2.VideoWriter(str(root / "calm_active.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
rng = np.random.default_rng(0)
# (kind, seconds, start frame in vtest)
plan = [("calm", 25, 100), ("active", 6, 100), ("calm", 25, 160), ("active", 6, 400), ("calm", 20, 460), ("active", 6, 650)]
t = 0
schedule = []
for kind, secs, start in plan:
    n = int(secs * fps)
    schedule.append((round(t / fps, 1), kind, secs))
    for i in range(n):
        if kind == "calm":
            base = frames[start].astype(np.float32)
            drift = 1.0 + 0.03 * np.sin(2 * np.pi * t / (fps * 8))          # slow lighting change
            f = np.clip(base * drift + rng.normal(0, 2.0, base.shape), 0, 255).astype(np.uint8)
        else:
            f = frames[min(start + i, len(frames) - 1)]
        out.write(f)
        t += 1
out.release()
print("wrote", root / "calm_active.mp4", f"{t / fps:.0f}s", schedule)
