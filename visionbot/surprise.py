"""Habituating JEPA surprise: the component that decides where and when the robot should think.

Idea
----
A frozen V-JEPA 2.1 encoder turns each short video snippet into a grid of latent vectors (one per
16x16 patch). A small predictor, trained *online while the robot runs*, learns to predict the next
latent grid from the previous K grids. This is a JEPA objective: prediction happens in
representation space, never in pixels, so lighting flicker and sensor noise that don't change
meaning barely register, while a cup falling or a person entering does.

Per-patch prediction error is then normalised by that patch's own running error statistics
("habituation"). A TV that always flickers stops being surprising after a few minutes; a new event
in the same spot still is. The output is:

* `score`   - one scalar per step: how surprising the scene is right now (robust top-k z-score)
* `zmap`    - HxW map of per-patch z-scores (for routing compute to regions and for display)
* `regions` - boxes around connected surprising patches

Unlike a fixed-rate pipeline, everything downstream (detector stride, VLM calls) is gated on this.
The predictor and statistics are checkpointed to disk, so the robot keeps what it has learned about
its home across restarts.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WEIGHTS, SurpriseCfg

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


class LatentPredictor(nn.Module):
    """Predicts the next latent grid from the last K grids. ~1.8M params, trains in ~2 ms/step."""

    def __init__(self, dim: int, k: int, squeeze: int = 128, hidden: int = 256):
        super().__init__()
        self.k = k
        self.squeeze = nn.Conv2d(dim, squeeze, 1)
        self.body = nn.Sequential(
            nn.Conv2d(squeeze * k, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        nn.init.zeros_(self.body[-1].weight)   # start as "nothing changes" (predict = last frame)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx: [B, K, D, H, W]  ->  prediction of the next grid [B, D, H, W]
        b, k, d, h, w = ctx.shape
        s = self.squeeze(ctx.reshape(b * k, d, h, w)).reshape(b, -1, h, w)
        return ctx[:, -1] + self.body(s)       # residual on the most recent grid


@dataclass
class SurpriseState:
    t: float = 0.0
    step: int = 0
    learning: bool = True
    score: float = 0.0          # scene surprise relative to what is typical here lately (used for gating)
    raw: float = 0.0            # patch-level surprise before scene-level habituation
    loss: float = 0.0
    zmap: np.ndarray | None = None               # [H, W] float32, clipped z-scores
    regions: list[dict] = field(default_factory=list)
    encode_ms: float = 0.0
    train_ms: float = 0.0


class JEPASurprise:
    def __init__(self, cfg: SurpriseCfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.encoder = self._load_encoder()
        self.dim = self._probe_dim()
        self.pred = LatentPredictor(self.dim, cfg.context).to(self.device)
        self.opt = torch.optim.AdamW(self.pred.parameters(), lr=cfg.lr, weight_decay=1e-4)
        h, w = cfg.grid_h, cfg.grid_w
        self.mu = torch.zeros(h, w, device=self.device)
        self.var = torch.ones(h, w, device=self.device)
        self.hist: deque[torch.Tensor] = deque(maxlen=cfg.context)
        # second, slower habituation level on the scene score itself: "surprising for *this* place, lately"
        self.s_mu, self.s_var = 0.0, 1.0
        self.state = SurpriseState()
        self.step = 0
        self.ckpt = WEIGHTS / "surprise_state.pt"
        self._last_save = time.perf_counter()
        self._load_state()
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- model loading
    def _load_encoder(self) -> nn.Module:
        import importlib
        import sys
        from pathlib import Path
        torch.hub.set_dir(str(WEIGHTS / "hub"))
        repo = Path(torch.hub.get_dir()) / "facebookresearch_vjepa2_main"
        if not repo.exists():   # fetch the code only; weights are loaded below
            torch.hub.load("facebookresearch/vjepa2", self.cfg.hub_entry, pretrained=False, trust_repo=True)
        # The upstream hub file ships with a developer URL (http://localhost:8300) active and the public
        # one commented out, so point it at Meta's public checkpoint server before loading weights.
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        backbones = importlib.import_module("src.hub.backbones")
        backbones.VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
        out = getattr(backbones, self.cfg.hub_entry)(pretrained=True)
        encoder = out[0] if isinstance(out, (tuple, list)) else out
        # fp32 weights + bf16 autocast: V-JEPA's RoPE attention upcasts q/k itself, so a hard .half()
        # leaves q/k/v in mixed dtypes; autocast keeps them consistent and is just as fast.
        encoder = encoder.to(self.device).eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        return encoder

    @torch.no_grad()
    def _encode_tensor(self, clip: torch.Tensor) -> torch.Tensor:
        """clip [1,3,2,H,W] normalised float → latent grid [D, gh, gw] (layer-normed, float32)."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            tokens = self.encoder(clip)
        if isinstance(tokens, (list, tuple)):      # some checkpoints return per-level outputs
            tokens = tokens[-1]
        gh, gw = self.cfg.grid_h, self.cfg.grid_w
        tokens = tokens[0, -gh * gw:].float()       # last temporal slice = most recent tubelet
        tokens = F.layer_norm(tokens, tokens.shape[-1:])
        return tokens.T.reshape(-1, gh, gw)

    def _probe_dim(self) -> int:
        clip = torch.zeros(1, 3, 2, self.cfg.grid_h * 16, self.cfg.grid_w * 16, device=self.device)
        dim = self._encode_tensor(clip).shape[0]
        self._graph = None
        if self.device.type == "cuda":
            # Same trick as the fast lane: the ViT is launch-bound on Windows, so replay it from a CUDA graph.
            try:
                self._static_in = clip
                s = torch.cuda.Stream(self.device)
                s.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._encode_tensor(self._static_in)
                torch.cuda.current_stream(self.device).wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._static_out = self._encode_tensor(self._static_in)
                self._graph = g
            except Exception as e:
                print(f"[surprise] CUDA graph capture failed ({e.__class__.__name__}); using eager encoder")
        return dim

    @torch.no_grad()
    def _encode(self, clip: torch.Tensor) -> torch.Tensor:
        if self._graph is None:
            return self._encode_tensor(clip)
        self._static_in.copy_(clip)
        self._graph.replay()
        return self._static_out.clone()

    # ---------------------------------------------------------------- persistence
    def _load_state(self) -> None:
        if not self.ckpt.exists():
            return
        try:
            s = torch.load(self.ckpt, map_location=self.device, weights_only=True)
            if s.get("dim") == self.dim and s["mu"].shape == self.mu.shape:
                self.pred.load_state_dict(s["pred"])
                self.opt.load_state_dict(s["opt"])
                self.mu, self.var, self.step = s["mu"], s["var"], int(s["step"])
                self.s_mu, self.s_var = float(s.get("s_mu", 0.0)), float(s.get("s_var", 1.0))
        except Exception as e:  # corrupt or incompatible checkpoint: start fresh rather than crash
            print(f"[surprise] ignoring checkpoint: {e}")

    def save_state(self) -> None:
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ckpt.with_suffix(".tmp")
        torch.save({"dim": self.dim, "pred": self.pred.state_dict(), "opt": self.opt.state_dict(),
                    "mu": self.mu, "var": self.var, "step": self.step, "s_mu": self.s_mu, "s_var": self.s_var}, tmp)
        tmp.replace(self.ckpt)

    def relearn(self, steps: int = 40) -> None:
        """New scene (another camera or an uploaded file): keep the learned predictor, but rebuild the
        per-patch statistics and briefly stay quiet while they settle (~4 s)."""
        with self._lock:
            self.hist.clear()
            self.mu.zero_(); self.var.fill_(1.0)
            self.s_mu, self.s_var = 0.0, 1.0
            self.relearn_until = self.step + steps

    def reset(self) -> None:
        """Forget everything learned about this environment."""
        with self._lock:
            self.pred = LatentPredictor(self.dim, self.cfg.context).to(self.device)
            self.opt = torch.optim.AdamW(self.pred.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
            self.mu.zero_(); self.var.fill_(1.0); self.hist.clear(); self.step = 0
            self.s_mu, self.s_var = 0.0, 1.0
            if self.ckpt.exists():
                self.ckpt.unlink()

    # ---------------------------------------------------------------- main step
    def _prep(self, prev_bgr: np.ndarray, cur_bgr: np.ndarray) -> torch.Tensor:
        size = (self.cfg.grid_w * 16, self.cfg.grid_h * 16)
        frames = [cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                  for f in (prev_bgr, cur_bgr)]
        x = torch.from_numpy(np.stack(frames)).to(self.device, non_blocking=True)   # [2,H,W,3] uint8
        x = x.permute(3, 0, 1, 2).unsqueeze(0).float().div_(255)                     # [1,3,2,H,W]
        return (x - _MEAN.to(self.device)) / _STD.to(self.device)

    def update(self, prev_bgr: np.ndarray, cur_bgr: np.ndarray) -> SurpriseState:
        cfg = self.cfg
        with self._lock, torch.cuda.stream(self.stream) if self.stream else _nullctx():
            t0 = time.perf_counter()
            z = self._encode(self._prep(prev_bgr, cur_bgr))
            if self.stream:
                self.stream.synchronize()
            t1 = time.perf_counter()

            score, zmap, loss_v = 0.0, None, 0.0
            if len(self.hist) == cfg.context:
                ctx = torch.stack(list(self.hist)).unsqueeze(0)
                self.pred.train()
                pred = self.pred(ctx)[0]
                err = (pred - z).abs().mean(0)                               # [H, W]
                loss = err.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pred.parameters(), 1.0)
                self.opt.step()
                loss_v = float(loss.detach())
                # Remove the scene-wide error component (lighting, auto-exposure, camera gain hit every
                # patch at once); real events are local, so surprise is measured relative to the median.
                e = err.detach()
                e = e - e.median()
                # z-score against this patch's own history, THEN update the history (habituation)
                zt = (e - self.mu) / (self.var.sqrt() + 1e-4)
                a = cfg.habituation if self.step > 30 else 0.2               # fast initial fit
                d = e - self.mu
                self.mu += a * d
                self.var = (1 - a) * (self.var + a * d * d)
                self.step += 1
                zc = zt.clamp(-2, 8)
                flat = zc.flatten()
                k = max(3, int(0.05 * flat.numel()))
                score = float(flat.topk(k).values.mean())
                zmap = zc.cpu().numpy().astype(np.float32)
                if self.stream:
                    self.stream.synchronize()
            self.hist.append(z.detach())
            t2 = time.perf_counter()

        learning = self.step < max(cfg.warmup_steps, getattr(self, "relearn_until", 0))
        rel = 0.0
        if zmap is not None:
            # z-score of the scene score against its own slow running statistics (std floored so a
            # perfectly still room doesn't become hypersensitive)
            rel = (score - self.s_mu) / max(0.5, self.s_var ** 0.5)
            b = 0.05 if self.step < cfg.warmup_steps else cfg.scene_habituation
            d = score - self.s_mu
            self.s_mu += b * d
            self.s_var = (1 - b) * (self.s_var + b * d * d)
        st = SurpriseState(
            t=time.perf_counter(), step=self.step, learning=learning,
            score=0.0 if learning else max(0.0, rel), raw=max(0.0, score), loss=loss_v,
            zmap=zmap, regions=[] if (learning or zmap is None) else self._regions(zmap),
            encode_ms=(t1 - t0) * 1000, train_ms=(t2 - t1) * 1000,
        )
        self.state = st
        if time.perf_counter() - self._last_save > 60:
            self.save_state()
            self._last_save = time.perf_counter()
        return st

    def _regions(self, zmap: np.ndarray) -> list[dict]:
        mask = (zmap > self.cfg.z_threshold).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        gh, gw = zmap.shape
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            peak = float(zmap[lab == i].max())
            out.append({"box": [x / gw, y / gh, w / gw, h / gh], "peak": round(peak, 2), "cells": int(area)})
        out.sort(key=lambda r: -r["peak"])
        return out[:6]

    def novelty_in(self, box: list[float]) -> float:
        """Mean surprise z inside a normalised [x,y,w,h] box (0 if no map yet)."""
        zm = self.state.zmap
        if zm is None or self.state.learning:
            return 0.0
        gh, gw = zm.shape
        x0, y0 = int(box[0] * gw), int(box[1] * gh)
        x1, y1 = max(x0 + 1, int(np.ceil((box[0] + box[2]) * gw))), max(y0 + 1, int(np.ceil((box[1] + box[3]) * gh)))
        patch = zm[max(0, y0):min(gh, y1), max(0, x0):min(gw, x1)]
        return float(patch.mean()) if patch.size else 0.0


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
