"""Load each model once, check it runs on the GPU, and time it. Usage: python tests/probe_components.py [which...]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from visionbot.config import WEIGHTS, Config

WEIGHTS.mkdir(parents=True, exist_ok=True)
cfg = Config()
which = set(sys.argv[1:]) or {"torch", "jepa", "fast", "depth"}
video = Path(__file__).resolve().parent.parent / "samples" / "vtest.avi"
cap = cv2.VideoCapture(str(video))
frames = []
while len(frames) < 60:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
print(f"loaded {len(frames)} frames of {frames[0].shape}")


def timeit(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


if "torch" in which:
    print("torch", torch.__version__, torch.cuda.get_device_name(0), "bf16" if torch.cuda.is_bf16_supported() else "")

if "jepa" in which:
    from visionbot.surprise import JEPASurprise
    t = time.perf_counter()
    js = JEPASurprise(cfg.surprise)
    print(f"[jepa] loaded in {time.perf_counter()-t:.1f}s, latent dim {js.dim}, grid {cfg.surprise.grid_h}x{cfg.surprise.grid_w}")
    js.reset()
    i = [0]

    def step():
        a, b = frames[i[0] % len(frames)], frames[(i[0] + 1) % len(frames)]
        i[0] += 1
        return js.update(a, b)
    ms = timeit(step, 40)
    st = js.state
    print(f"[jepa] full step (encode+train) {ms:.1f} ms | encode {st.encode_ms:.1f} ms train {st.train_ms:.1f} ms | loss {st.loss:.4f} step {st.step}")

if "fast" in which:
    from visionbot.fast_lane import FastLane
    t = time.perf_counter()
    fl = FastLane(cfg.fast)
    print(f"[fast] loaded in {time.perf_counter()-t:.1f}s")
    i = [0]

    def step():
        d = fl.step(frames[i[0] % len(frames)])
        i[0] += 1
        return d
    ms = timeit(step, 40)
    dets = step()
    print(f"[fast] step {ms:.1f} ms (det {fl.timing['det_ms']:.1f} ms, pose {fl.timing['pose_ms']:.1f} ms) → {len(dets)} tracks:",
          sorted({d.label for d in dets}), [d.cues for d in dets if d.cues][:3])

if "depth" in which:
    from visionbot.depth import MetricDepth
    t = time.perf_counter()
    md = MetricDepth(cfg.depth)
    print(f"[depth] loaded in {time.perf_counter()-t:.1f}s")
    ms = timeit(lambda: md.infer(frames[0]), 20)
    d = md.depth
    print(f"[depth] {ms:.1f} ms, map {d.shape}, range {d.min():.2f}–{d.max():.2f} m, centre {md.sample([0.4,0.4,0.2,0.2]):.2f} m")
