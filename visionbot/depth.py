"""Metric depth from a single webcam, so every tracked object gets a distance and a 3D position.

Depth Anything V2 Metric-Indoor-Small is trained on indoor scenes and outputs metres directly, which
fits a home robot. It runs on its own thread at ~5 Hz: object depth changes slowly compared with
the 30 Hz fast lane, so running it every frame would waste GPU time for no gain.
"""
from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np
import torch

from .config import DepthCfg

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class MetricDepth:
    def __init__(self, cfg: DepthCfg, device: str = "cuda"):
        from transformers import AutoModelForDepthEstimation
        self.cfg = cfg
        self.device = torch.device(device)
        try:        # cached copy first: the Hub round-trips add ~50 s to start-up on a slow link
            model = AutoModelForDepthEstimation.from_pretrained(cfg.model, local_files_only=True)
        except OSError:
            model = AutoModelForDepthEstimation.from_pretrained(cfg.model)
        self.model = model.to(self.device).eval().half()
        self.depth: np.ndarray | None = None     # metres, reduced resolution
        self.t = 0.0
        self.ms = 0.0
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._lock = threading.Lock()

    @torch.no_grad()
    def infer(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self.cfg.input_short / min(h, w)
        nh, nw = int(round(h * s / 14)) * 14, int(round(w * s / 14)) * 14
        t0 = time.perf_counter()
        ctx = torch.cuda.stream(self.stream) if self.stream else torch.no_grad()
        with ctx:
            rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float().div_(255)
            x = ((x - _MEAN.to(self.device)) / _STD.to(self.device)).half()
            d = self.model(pixel_values=x).predicted_depth[0].float()
            out = d.cpu().numpy()
        if self.stream:
            self.stream.synchronize()
        with self._lock:
            self.depth, self.t, self.ms = out, time.perf_counter(), (time.perf_counter() - t0) * 1000
        return out

    def sample(self, box: list[float]) -> float | None:
        """Robust distance (m) to an object: median of the central 40% of its box."""
        with self._lock:
            d = self.depth
        if d is None:
            return None
        h, w = d.shape
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        x0, x1 = int((cx - 0.2 * box[2]) * w), int(math.ceil((cx + 0.2 * box[2]) * w))
        y0, y1 = int((cy - 0.2 * box[3]) * h), int(math.ceil((cy + 0.2 * box[3]) * h))
        patch = d[max(0, y0):min(h, max(y1, y0 + 1)), max(0, x0):min(w, max(x1, x0 + 1))]
        return float(np.median(patch)) if patch.size else None


def backproject(box: list[float], z: float, hfov_deg: float, aspect: float) -> list[float]:
    """Pixel centre + depth → camera-frame XYZ in metres (x right, y down, z forward)."""
    fx = 0.5 / math.tan(math.radians(hfov_deg) / 2)            # focal length in normalised width units
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    x = (cx - 0.5) / fx * z
    y = (cy - 0.5) * aspect / fx * z                             # aspect = H / W
    return [round(x, 2), round(y, 2), round(z, 2)]
