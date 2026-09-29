"""Fast lane: open-vocabulary detection + tracking + human pose, every frame, in a few milliseconds.

* YOLOE-26 (Ultralytics, 2026) detects whatever the vocabulary names — no retraining to add
  "medicine bottle" or "keys". Text prompts are fused into the head once at start-up.
* ByteTrack keeps IDs stable across frames.
* YOLO26-pose gives 17 body keypoints for people; simple geometry turns them into interaction cues
  (hand raised, facing the robot, posture).

Why this file doesn't just call `model.track()`: on Windows each CUDA kernel launch costs ~15 µs
and a detector launches hundreds of them, so the stock Python path is launch-bound (~18 ms for a
model whose actual GPU work is a few ms, and the nano model is no faster than the small one). We
record each network once as a **CUDA Graph** and replay it with a single launch per frame, and we do
letterboxing, NMS and mask decoding on the GPU. If graph capture fails for any reason, it falls back
to the stock Ultralytics path automatically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import WEIGHTS, FastLaneCfg

# COCO keypoint indices
NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKNEE, RKNEE, LANK, RANK = range(17)


@dataclass
class Det:
    tid: int
    label: str
    conf: float
    box: list[float]                     # normalised [x, y, w, h]
    poly: np.ndarray | None = None       # mask outline in pixels (for drawing)
    kpts: np.ndarray | None = None       # [17, 3] pixels + confidence (people only)
    cues: dict = field(default_factory=dict)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def body_cues(k: np.ndarray, box_px: tuple[float, float, float, float]) -> dict:
    """Interaction cues from COCO keypoints. k: [17,3] (x, y, conf) in pixels."""
    ok = lambda *ids: all(k[i, 2] > 0.5 for i in ids)
    cues: dict = {}
    sh_y = np.mean([k[i, 1] for i in (LSH, RSH) if k[i, 2] > 0.5]) if (k[LSH, 2] > 0.5 or k[RSH, 2] > 0.5) else None
    if sh_y is not None:
        torso = None
        if ok(LSH, LHIP) or ok(RSH, RHIP):
            s, h = (LSH, LHIP) if ok(LSH, LHIP) else (RSH, RHIP)
            torso = abs(k[h, 1] - k[s, 1])
        margin = 0.15 * torso if torso else 0.05 * box_px[3]
        raised = [side for side, w in (("left", LWR), ("right", RWR)) if k[w, 2] > 0.5 and k[w, 1] < sh_y - margin]
        if raised:
            cues["hand_raised"] = raised
    if ok(NOSE, LEYE, REYE):
        eye_mid = (k[LEYE, 0] + k[REYE, 0]) / 2
        eye_span = abs(k[LEYE, 0] - k[REYE, 0]) + 1e-3
        cues["facing_camera"] = bool(abs(k[NOSE, 0] - eye_mid) / eye_span < 0.35)
    if ok(LSH, LHIP) or ok(RSH, RHIP):
        sh = np.mean([k[i, :2] for i in (LSH, RSH) if k[i, 2] > 0.5], axis=0)
        hp = np.mean([k[i, :2] for i in (LHIP, RHIP) if k[i, 2] > 0.5], axis=0)
        v = sh - hp
        tilt = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))    # 0 = upright
        if tilt > 60:
            cues["posture"] = "lying"
        else:
            knees = [i for i in (LKNEE, RKNEE) if k[i, 2] > 0.5]
            if knees:
                kn_y = np.mean([k[i, 1] for i in knees])
                thigh = abs(kn_y - hp[1]) / (np.linalg.norm(v) + 1e-6)
                cues["posture"] = "sitting" if thigh < 0.45 else "standing"
    return cues


def face_crop(bgr: np.ndarray, k: np.ndarray, min_eye_px: float = 10.0) -> np.ndarray | None:
    """64x64 greyscale face, cut out using the pose keypoints (no separate face detector needed).
    Only for faces turned towards the camera and big enough to read."""
    if not (k[LEYE, 2] > 0.5 and k[REYE, 2] > 0.5 and k[NOSE, 2] > 0.5):
        return None
    le, re_, nose = k[LEYE, :2], k[REYE, :2], k[NOSE, :2]
    d = float(np.linalg.norm(le - re_))
    if d < min_eye_px:
        return None
    cx = (le[0] + re_[0]) / 2
    cy = (le[1] + re_[1]) / 2 + 0.45 * d                 # the face centre sits a little below the eyes
    half = 1.25 * d
    h, w = bgr.shape[:2]
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(cy - half * 1.1), int(cy + half * 1.1)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    face = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA)


FERPLUS_URL = ("https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
               "emotion_ferplus/model/emotion-ferplus-8.onnx")


class Expressions:
    """FER+ (Barsoum et al., 2016): 8 facial-expression classes from a 64x64 face, ~1 ms on the CPU."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, face64: np.ndarray) -> np.ndarray:
        x = face64.astype(np.float32).reshape(1, 1, 64, 64)
        z = self.sess.run(None, {self.inp: x})[0][0]
        z = np.exp(z - z.max())
        return z / z.sum()


# ----------------------------------------------------------------------------- GPU plumbing
class Letterbox:
    """Upload a BGR frame once and letterbox it on the GPU to a fixed network input shape."""

    def __init__(self, frame_hw: tuple[int, int], imgsz: int, device: torch.device, stride: int = 32):
        h, w = frame_hw
        r = min(imgsz / h, imgsz / w)
        nh, nw = round(h * r), round(w * r)
        self.in_h, self.in_w = int(np.ceil(nh / stride) * stride), int(np.ceil(nw / stride) * stride)
        self.r, self.nh, self.nw = r, nh, nw
        self.top, self.left = (self.in_h - nh) // 2, (self.in_w - nw) // 2
        self.device = device
        self.pinned = torch.empty((h, w, 3), dtype=torch.uint8).pin_memory()
        self.out = torch.full((1, 3, self.in_h, self.in_w), 114 / 255, device=device, dtype=torch.half)

    def __call__(self, bgr: np.ndarray) -> torch.Tensor:
        self.pinned.numpy()[...] = bgr
        x = self.pinned.to(self.device, non_blocking=True).permute(2, 0, 1).flip(0).unsqueeze(0).half().div_(255)
        x = F.interpolate(x, size=(self.nh, self.nw), mode="bilinear", align_corners=False)
        self.out[:, :, self.top:self.top + self.nh, self.left:self.left + self.nw] = x
        return self.out

    def unmap(self, xy: torch.Tensor) -> torch.Tensor:
        """Network-input pixel coords → original frame pixel coords (works on [...,2k] xyxy or xy pairs)."""
        xy = xy.clone()
        xy[..., 0::2] = (xy[..., 0::2] - self.left) / self.r
        xy[..., 1::2] = (xy[..., 1::2] - self.top) / self.r
        return xy


class Graphed:
    """Replays a network's forward pass from a CUDA Graph at one fixed input shape."""

    def __init__(self, net: torch.nn.Module, shape: tuple[int, int], device: torch.device):
        self.inp = torch.zeros(1, 3, *shape, device=device, dtype=torch.half)
        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.inference_mode(), torch.cuda.stream(s):
            for _ in range(3):                       # warm-up also caches anchors for this shape
                net(self.inp)
        torch.cuda.current_stream(device).wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.g):
            out = net(self.inp)
        self.out = out[0] if isinstance(out, (list, tuple)) and isinstance(out[0], (list, tuple)) else out

    def __call__(self, x: torch.Tensor):
        self.inp.copy_(x)
        self.g.replay()
        return self.out


class _TrackInput:
    """The minimal Results-like view ByteTrack needs."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    @property
    def xywh(self) -> np.ndarray:
        c = np.empty_like(self.xyxy)
        c[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
        c[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
        c[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        c[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return c

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, m):
        return _TrackInput(self.xyxy[m], self.conf[m], self.cls[m])


def _new_tracker():
    # track_buffer: a track survives ~3 s unseen (it was ~1 s, which made people behind a chair "new");
    # new_track_thresh: a new identity needs a reasonably confident box, which cuts spurious IDs.
    from ultralytics.trackers.byte_tracker import BYTETracker
    return BYTETracker(SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.3, track_low_thresh=0.1,
                                       new_track_thresh=0.45, track_buffer=90, match_thresh=0.8, fuse_score=True))


# ----------------------------------------------------------------------------- fast lane
class FastLane:
    CAND_CONF = 0.1          # ByteTrack's second stage wants low-score boxes too

    def __init__(self, cfg: FastLaneCfg, device: str = "cuda"):
        from ultralytics import YOLO, YOLOE
        self.cfg = cfg
        self.device = torch.device(device)
        self.dev_str = device
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.det_model = YOLOE(str(WEIGHTS / cfg.detector))      # Ultralytics downloads known assets here
        self.det_model.set_classes(cfg.vocab)
        self.pose_model = YOLO(str(WEIGHTS / cfg.pose))
        self.names = list(cfg.vocab)
        self.frame_no = 0
        self.last_pose: list[tuple[list[float], np.ndarray]] = []
        self.timing = {"det_ms": 0.0, "pose_ms": 0.0}
        self.tracker = _new_tracker()
        self.graphed = self.device.type == "cuda"
        ferp = WEIGHTS / "emotion-ferplus-8.onnx"
        try:
            if not ferp.exists():                                # 34 MB, MIT-licensed, from the ONNX model zoo
                import urllib.request
                print("[fast] downloading the FER+ expression model…")
                urllib.request.urlretrieve(FERPLUS_URL, ferp)
            self.expr = Expressions(ferp)
        except Exception as e:
            print(f"[fast] expression model unavailable: {e}")
            self.expr = None
        self.want_masks = False
        self._lb = self._det_g = self._pose_g = None
        self._frame_hw = None
        self._det_net = self._fused(self.det_model)
        self._pose_net = self._fused(self.pose_model)

    def _fused(self, model) -> torch.nn.Module:
        """Run one stock prediction so Ultralytics fuses conv+BN and bakes text embeddings into the head."""
        model.predict(np.zeros((64, 64, 3), np.uint8), imgsz=64, quantize=16, device=self.dev_str, verbose=False)
        net = model.predictor.model.model
        # YOLOE keeps its text-prompt embeddings on the CPU and copies them to the GPU every forward;
        # move them once (this also makes the forward pass CUDA-graph capturable).
        if isinstance(getattr(net, "pe", None), torch.Tensor):
            net.pe = net.pe.to(self.device, torch.half)
        return net

    def _setup(self, hw: tuple[int, int]) -> None:
        self._frame_hw = hw
        self._lb = Letterbox(hw, self.cfg.imgsz, self.device)
        shape = (self._lb.in_h, self._lb.in_w)
        if self.graphed:
            try:
                self._det_g = Graphed(self._det_net, shape, self.device)
                self._pose_g = Graphed(self._pose_net, shape, self.device)
            except Exception as e:                               # e.g. a future Ultralytics change
                print(f"[fast] CUDA graph capture failed ({e.__class__.__name__}: {e}); using eager mode")
                self.graphed = False

    def set_vocab(self, words: list[str]) -> None:
        self.cfg.vocab = list(words)
        self.names = list(words)
        self.det_model.set_classes(self.cfg.vocab)
        self._det_net = self._fused(self.det_model)
        self._frame_hw = None                                    # force graph re-capture
        self.tracker = _new_tracker()

    # ------------------------------------------------------------------ raw nets
    def _run(self, graph, net, x):
        if self.graphed:
            return graph(x)
        with torch.inference_mode():
            out = net(x)
        return out[0] if isinstance(out, (list, tuple)) and isinstance(out[0], (list, tuple)) else out

    @torch.inference_mode()
    def _decode(self, pred: torch.Tensor, n_cls: int, conf: float, max_det: int = 100):
        """pred [1, 4+n_cls+extra, N] → (xyxy frame px, conf, cls, extra) after NMS."""
        from torchvision.ops import batched_nms
        p = pred[0].T                                            # [N, C]
        scores, cls = p[:, 4:4 + n_cls].max(1)
        keep = scores > conf
        p, scores, cls = p[keep], scores[keep], cls[keep]
        if p.shape[0] > 1000:
            top = scores.topk(1000).indices
            p, scores, cls = p[top], scores[top], cls[top]
        c = p[:, :4].float()
        xyxy = torch.cat([c[:, :2] - c[:, 2:] / 2, c[:, :2] + c[:, 2:] / 2], 1)
        k = batched_nms(xyxy, scores.float(), cls, 0.7)[:max_det]
        return self._lb.unmap(xyxy[k]), scores[k].float(), cls[k], p[k, 4 + n_cls:]

    # ------------------------------------------------------------------ public
    @torch.inference_mode()
    def step(self, bgr: np.ndarray) -> list[Det]:
        h, w = bgr.shape[:2]
        if self._frame_hw != (h, w):
            self._setup((h, w))
        t0 = time.perf_counter()
        x = self._lb(bgr)
        out = self._run(self._det_g, self._det_net, x)
        pred = out[0] if isinstance(out, (list, tuple)) else out
        proto = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else None
        xyxy, conf, cls, coef = self._decode(pred, len(self.names), self.CAND_CONF)
        xyxy[:, 0::2].clamp_(0, w)
        xyxy[:, 1::2].clamp_(0, h)
        xyxy_np, conf_np, cls_np = xyxy.cpu().numpy(), conf.cpu().numpy(), cls.cpu().numpy()
        tracks = self.tracker.update(_TrackInput(xyxy_np, conf_np, cls_np))
        polys = self._masks(proto, coef, xyxy, tracks, (h, w)) if (self.want_masks and proto is not None and len(tracks)) else {}
        dets: list[Det] = []
        for x0, y0, x1, y1, tid, score, c, idx in tracks:
            dets.append(Det(int(tid), self.names[int(c)], float(score),
                            [float(x0 / w), float(y0 / h), float((x1 - x0) / w), float((y1 - y0) / h)],
                            poly=polys.get(int(idx))))
        t1 = time.perf_counter()
        self.timing["det_ms"] = (t1 - t0) * 1000

        people = [d for d in dets if d.label == "person"]
        if people and self.frame_no % self.cfg.pose_every == 0:
            pout = self._run(self._pose_g, self._pose_net, x)
            ppred = pout[0] if isinstance(pout, (list, tuple)) else pout
            pxy, pconf, _, kp = self._decode(ppred, 1, 0.4, max_det=20)
            self.last_pose = []
            if len(pconf):
                kp = kp.float().view(-1, 17, 3)
                kp[..., :2] = self._lb.unmap(kp[..., :2].reshape(-1, 34)).view(-1, 17, 2)
                for b, k in zip(pxy.cpu().numpy(), kp.cpu().numpy()):
                    self.last_pose.append(([float(b[0] / w), float(b[1] / h), float((b[2] - b[0]) / w), float((b[3] - b[1]) / h)], k))
            self.timing["pose_ms"] = (time.perf_counter() - t1) * 1000
        elif not people:
            self.timing["pose_ms"] = 0.0
        fresh_pose = people and (self.frame_no % self.cfg.pose_every == 0)
        for d in people:                                         # attach the best-matching skeleton
            best = max(self.last_pose, key=lambda pk: _iou(pk[0], d.box), default=None)
            if best is not None and _iou(best[0], d.box) > 0.4:
                d.kpts = best[1]
                d.cues = body_cues(best[1], (d.box[0] * w, d.box[1] * h, d.box[2] * w, d.box[3] * h))
                if self.expr is not None and fresh_pose:         # expressions on fresh skeletons only
                    face = face_crop(bgr, best[1])
                    if face is not None:
                        d.cues["expr"] = self.expr(face).tolist()
        self.frame_no += 1
        return dets

    @torch.inference_mode()
    def _masks(self, proto, coef, xyxy, tracks, hw) -> dict[int, np.ndarray]:
        """Instance masks → outlines, only for tracked boxes and only when someone is watching."""
        idx = torch.as_tensor(tracks[:, 7].astype(np.int64), device=coef.device)
        m = (coef[idx].float() @ proto[0].float().view(proto.shape[1], -1)).sigmoid()
        ph, pw = proto.shape[2:]
        m = m.view(-1, ph, pw).cpu().numpy()
        lb, out = self._lb, {}
        sx, sy = pw / lb.in_w, ph / lb.in_h
        for j, i in enumerate(idx.tolist()):
            x0, y0, x1, y1 = xyxy[i].tolist()
            # crop in proto space (frame px → net px → proto px), threshold, trace outline
            a0, b0 = int((x0 * lb.r + lb.left) * sx), int((y0 * lb.r + lb.top) * sy)
            a1, b1 = int(np.ceil((x1 * lb.r + lb.left) * sx)), int(np.ceil((y1 * lb.r + lb.top) * sy))
            crop = np.zeros((ph, pw), np.uint8)
            crop[b0:b1, a0:a1] = (m[j, b0:b1, a0:a1] > 0.5)
            cs, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                c[:, 0] = (c[:, 0] / sx - lb.left) / lb.r
                c[:, 1] = (c[:, 1] / sy - lb.top) / lb.r
                out[i] = c.astype(np.int32)
        return out
