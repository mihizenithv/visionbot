"""Is the GPU slow, or is it kernel-launch overhead? Measures raw TFLOPS, per-launch cost, and CUDA-graph speedup."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

d = "cuda"
a = torch.randn(4096, 4096, device=d, dtype=torch.half)
for _ in range(3):
    a @ a
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(20):
    a @ a
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / 20
print(f"fp16 matmul: {2 * 4096**3 / dt / 1e12:.1f} TFLOPS")

x = torch.zeros(1, device=d)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(2000):
    x.add_(1)
torch.cuda.synchronize()
print(f"per-kernel launch cost: {(time.perf_counter() - t) / 2000 * 1e6:.1f} us")

from ultralytics import YOLOE
from visionbot.config import HOME_VOCAB, WEIGHTS
m = YOLOE(str(WEIGHTS / "yoloe-26s-seg.pt"))
m.set_classes(HOME_VOCAB)
net = m.model.to(d).half().eval()
inp = torch.zeros(1, 3, 640, 640, device=d, dtype=torch.half)
with torch.inference_mode():
    for _ in range(5):
        net(inp)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        net(inp)
        torch.cuda.synchronize()
    n_k = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    gpu_us = sum(e.device_time for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    t = time.perf_counter()
    for _ in range(30):
        net(inp)
    torch.cuda.synchronize()
    eager = (time.perf_counter() - t) / 30 * 1000
print(f"yoloe-26s eager forward: {eager:.1f} ms wall, {n_k} kernels, {gpu_us/1000:.1f} ms of actual GPU work")

# CUDA graph: record once, replay with a single launch
static_in = inp.clone()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.inference_mode(), torch.cuda.stream(s):
    for _ in range(3):
        net(static_in)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.inference_mode(), torch.cuda.graph(g):
    static_out = net(static_in)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(100):
    g.replay()
torch.cuda.synchronize()
print(f"yoloe-26s CUDA-graph forward: {(time.perf_counter() - t) / 100 * 1000:.2f} ms")
print("output type:", type(static_out), [getattr(o, 'shape', type(o)) for o in (static_out if isinstance(static_out, (list, tuple)) else [static_out])][:4])
