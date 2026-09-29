"""Where do the fast lane's milliseconds go? Compares predict vs track, input sizes and model sizes."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import torch
from ultralytics import YOLO, YOLOE

from visionbot.config import HOME_VOCAB, WEIGHTS

cap = cv2.VideoCapture(str(Path(__file__).resolve().parent.parent / "samples" / "vtest.avi"))
frames = [cap.read()[1] for _ in range(40)]


def bench(name, fn, n=40):
    for f in frames[:5]:
        fn(f)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for f in frames[:n]:
        fn(f)
    torch.cuda.synchronize()
    print(f"{name:45s} {(time.perf_counter() - t) / n * 1000:6.1f} ms")


det = YOLOE(str(WEIGHTS / "yoloe-26s-seg.pt"))
det.set_classes(HOME_VOCAB)
for imgsz in (640, 480):
    bench(f"yoloe-26s-seg predict imgsz={imgsz} fp16", lambda f: det.predict(f, imgsz=imgsz, quantize=16, verbose=False))
bench("yoloe-26s-seg predict 640 fp32", lambda f: det.predict(f, imgsz=640, verbose=False))
bench("yoloe-26s-seg track 640 fp16", lambda f: det.track(f, imgsz=640, quantize=16, persist=True, verbose=False, tracker="bytetrack.yaml"))
nano = YOLOE(str(WEIGHTS / "yoloe-26n-seg.pt"))
nano.set_classes(HOME_VOCAB)
bench("yoloe-26n-seg predict 640 fp16", lambda f: nano.predict(f, imgsz=640, quantize=16, verbose=False))
pose = YOLO(str(WEIGHTS / "yolo26n-pose.pt"))
bench("yolo26n-pose predict 640 fp16", lambda f: pose.predict(f, imgsz=640, quantize=16, verbose=False))
bench("yolo26n-pose predict 480 fp16", lambda f: pose.predict(f, imgsz=480, quantize=16, verbose=False))
# raw model forward, no Ultralytics pre/post-processing, to see the framework overhead
m = det.model.cuda().half().eval()
x = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.half)
with torch.inference_mode():
    bench("yoloe-26s raw forward 640x640 fp16", lambda f: m(x))
