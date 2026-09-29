"""Orchestrator: runs the three lanes concurrently and measures latency end to end.

Threads
  main      camera frame → fast lane (detect/track/pose) → scene memory → router      (every frame)
  aux       JEPA surprise at ~10 Hz and metric depth at ~5 Hz on their own CUDA streams
  reasoner  NVIDIA VLM calls, only when the router fires

The main loop never waits on aux or reasoner; it always uses their most recent results.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from collections import deque

import cv2
import numpy as np

from .capture import LatestFrameSource
from .config import Config
from .fast_lane import Det, FastLane, _new_tracker
from .reasoner import Reasoner, Router
from .scene import SceneMemory

# Overlay pigments (BGR), matching the dashboard: cinabro for people, verderame for animals, ocra for things.
CAT_COLORS = {"person": (34, 58, 178), "animal": (100, 117, 63), "object": (41, 113, 154)}
GESSO = (216, 231, 239)          # cartellino paper and chalk
UMBER = (25, 33, 42)             # ink
CINABRO = np.array([34, 58, 178], np.float32)
LABEL_FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL   # the one serif face OpenCV has
ANIMALS = {"cat", "dog"}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


class Rolling:
    def __init__(self, n: int = 300):
        self.v: deque[float] = deque(maxlen=n)

    def add(self, x: float) -> None:
        self.v.append(x)

    def pct(self, q: float) -> float:
        if not self.v:
            return 0.0
        return float(np.percentile(np.fromiter(self.v, float), q))


class Engine:
    def __init__(self, cfg: Config, log=print):
        self.cfg, self.log = cfg, log
        log("[engine] opening camera…")
        self.src = LatestFrameSource(cfg.camera)          # opened now, started once models are loaded
        log("[engine] loading fast lane (YOLOE-26 + pose)…")
        self.fast = FastLane(cfg.fast, cfg.device)
        self.depth = None
        if cfg.depth.enabled:
            log("[engine] loading metric depth…")
            from .depth import MetricDepth
            self.depth = MetricDepth(cfg.depth, cfg.device)
        self.surprise = None
        if cfg.surprise.enabled:
            log("[engine] loading V-JEPA 2.1 encoder for surprise…")
            from .surprise import JEPASurprise
            self.surprise = JEPASurprise(cfg.surprise, cfg.device)
        self.scene = SceneMemory(cfg.camera.hfov_deg)
        self.lock = threading.RLock()
        self.router = Router(cfg.reasoner)
        self.reasoner = Reasoner(cfg.reasoner)
        self.reasoner.on_result = self._on_reasoning
        self.reasoner.start()

        self.latest = None
        self.last_dets: list[Det] = []
        self.last_det_t = 0.0
        self.want_overlay = False
        self.show_heatmap = True
        self.overlay_jpeg: bytes | None = None
        self.overlay_id = -1
        self.running = False
        self.frames = 0
        self.det_frames = 0
        self.m = {k: Rolling() for k in ("e2e_ms", "e2e_full_ms", "det_ms", "pose_ms", "scene_ms", "loop_ms", "surprise_ms", "depth_ms")}
        self.fps = 0.0
        self._t_last = None
        self._threads: list[threading.Thread] = []
        self.calm_why: dict[str, int] = {}
        self.exit_on_end = True             # benchmarks stop at the end of a file; the server keeps running
        self.brightness = None
        self._render_cv = threading.Condition()
        self._render_req = None
        self._switch_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "Engine":
        self.src.start()
        self.running = True
        for fn, name in ((self._main_loop, "main"), (self._aux_loop, "aux"), (self._render_loop, "render")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self.running = False
        with self._render_cv:
            self._render_cv.notify_all()
        for t in self._threads:
            t.join(timeout=3)
        self.src.stop()
        self.reasoner.stop()
        if self.surprise:
            self.surprise.save_state()

    @property
    def source_info(self) -> dict:
        w, h = self.src.size
        return {"kind": self.src.kind, "name": self.src.name, "width": w, "height": h}

    def switch_source(self, source: str | int) -> dict:
        """Replace the camera/video/image being watched, and start the world state afresh for it."""
        with self._switch_lock:
            is_file = not isinstance(source, int)
            cam = dataclasses.replace(self.cfg.camera, source=source, loop=is_file or self.cfg.camera.loop)
            new = LatestFrameSource(cam)                  # raises before anything is torn down if it can't open
            old, self.src = self.src, new
            new.start()
            old.stop()
            with self.lock:
                self.scene = SceneMemory(self.cfg.camera.hfov_deg)
                self.fast.tracker = _new_tracker()
                self.last_dets, self.latest = [], None
                self.router.cluster, self.router.pending = None, None
                self.frames = self.det_frames = 0
                self.fps, self._t_last = 0.0, None
                self.calm_why = {}
                for r in self.m.values():
                    r.v.clear()
            if self.surprise:
                self.surprise.relearn()
            self.log(f"[engine] now watching {new.kind}: {new.name}")
            return self.source_info

    @property
    def finished(self) -> bool:
        return self.src.ended and not self._threads[0].is_alive()

    # ------------------------------------------------------------------ loops
    def _calm(self) -> bool:
        """True when nothing is changing, so the detector may skip frames."""
        if not self.cfg.fast.adaptive or self.surprise is None:
            return False
        st = self.surprise.state
        # thresholds are deliberately loose on noise: a lone odd patch or ~1 sigma wobble isn't activity
        strong = [r for r in st.regions if r["peak"] >= 5.0 or r["cells"] >= 3]
        why = ("learning" if st.learning else "score" if st.score > 2.0 else "regions" if strong else None)
        if why is None:
            with self.lock:
                for e in self.scene.visible():
                    if e.state != "visible":
                        why = "occluded_track"
                        break
                    if abs(e.vel[0]) + abs(e.vel[1]) > 0.03:
                        why = "moving_track"
                        break
        self.calm_why[why or "calm"] = self.calm_why.get(why or "calm", 0) + 1
        return why is None

    def _main_loop(self) -> None:
        while self.running:
            src = self.src
            f = src.read(timeout=0.5)
            if f is None:
                if src.ended and src is self.src:
                    if self.exit_on_end:
                        break
                    time.sleep(0.05)
                continue
            if src is not self.src:          # a frame from a source that was just replaced
                continue
            self.latest = f
            t0 = time.perf_counter()
            if self.frames % 15 == 0:                 # is the camera actually showing anything?
                self.brightness = float(f.image[::16, ::16].mean())
            stride = self.cfg.fast.calm_stride if self._calm() else 1
            ran = (self.frames % stride == 0)
            if ran:
                self.fast.want_masks = self.want_overlay           # outlines only matter to a viewer
                dets = self.fast.step(f.image)
                self.last_dets, self.last_det_t = dets, t0
                self.det_frames += 1
                self.m["det_ms"].add(self.fast.timing["det_ms"])
                if self.fast.timing["pose_ms"]:
                    self.m["pose_ms"].add(self.fast.timing["pose_ms"])
            t1 = time.perf_counter()
            with self.lock:
                if ran:
                    self.scene.update(dets, t1,
                                      depth_fn=self.depth.sample if self.depth else None,
                                      novelty_fn=self.surprise.novelty_in if self.surprise else None,
                                      frame_hw=f.image.shape[:2], frame=f.image)
                events = self.scene.pop_events()
                st = self.surprise.state if self.surprise else None
                self.router.offer(events, st.score if st else 0.0, self.cfg.surprise.spike_score,
                                  bool(self.scene.visible()))
                trig = self.router.ready() if self.reasoner.available else None
                snap = self.scene.snapshot(t1) if trig else None
            for ev in events:
                if ev.priority >= 1:
                    self.log(f"[event] {ev.text}")
            if trig:
                self.reasoner.submit(trig, f.image.copy(), snap)
            t2 = time.perf_counter()
            self.m["scene_ms"].add((t2 - t1) * 1000)
            self.m["e2e_ms"].add((t2 - f.t_capture) * 1000)
            if ran:                                  # frames that were fully analysed (the honest latency)
                self.m["e2e_full_ms"].add((t2 - f.t_capture) * 1000)
            self.m["loop_ms"].add((t2 - t0) * 1000)
            if self._t_last is not None:
                dt = t2 - self._t_last
                self.fps = 0.9 * self.fps + 0.1 * (1 / dt) if self.fps else 1 / dt
            self._t_last = t2
            self.frames += 1
            if self.want_overlay:                    # drawing + JPEG happen on their own thread, off the hot path
                with self._render_cv:
                    self._render_req = f
                    self._render_cv.notify()

    def _aux_loop(self) -> None:
        prev_img = None
        next_s = next_d = 0.0
        last_idx = -1
        while self.running:
            f = self.latest
            now = time.perf_counter()
            if f is None or f.idx == last_idx:
                time.sleep(0.003)
                continue
            did = False
            if self.surprise and now >= next_s:
                if prev_img is not None:
                    st = self.surprise.update(prev_img, f.image)
                    self.m["surprise_ms"].add(st.encode_ms + st.train_ms)
                prev_img = f.image
                next_s = now + self.cfg.surprise.period_s
                did = True
            if self.depth and now >= next_d:
                self.depth.infer(f.image)
                self.m["depth_ms"].add(self.depth.ms)
                next_d = now + self.cfg.depth.period_s
                did = True
            if did:
                last_idx = f.idx
            else:
                time.sleep(0.003)

    def _render_loop(self) -> None:
        while self.running:
            with self._render_cv:
                while self._render_req is None and self.running:
                    self._render_cv.wait(0.5)
                f, self._render_req = self._render_req, None
            if f is not None:
                try:
                    self._render(f)
                except Exception as e:           # a drawing glitch must never stop the stream
                    self.log(f"[render] {e.__class__.__name__}: {e}")

    def _on_reasoning(self, res: dict) -> None:
        with self.lock:
            if "error" not in res:
                self.scene.apply_reasoning(res)
        self.log(f"[reasoner] {res.get('trigger')}: {res.get('scene') or res.get('error')} ({res.get('latency_ms')} ms)")

    # ------------------------------------------------------------------ outputs
    def ask(self, question: str) -> dict:
        f = self.latest
        if f is None:
            return {"error": "no frame yet"}
        with self.lock:
            snap = self.scene.snapshot(time.perf_counter(), recent_events=30)
        return self.reasoner.ask(question, f.image.copy(), snap)

    def state(self) -> dict:
        now = time.perf_counter()
        with self.lock:
            snap = self.scene.snapshot(now)
        st = self.surprise.state if self.surprise else None
        return {
            **snap,
            "perf": {
                "fps": round(self.fps, 1),
                "e2e_p50": round(self.m["e2e_full_ms"].pct(50), 1), "e2e_p95": round(self.m["e2e_full_ms"].pct(95), 1),
                "e2e_all_p50": round(self.m["e2e_ms"].pct(50), 1),
                "det_ms": round(self.m["det_ms"].pct(50), 1), "pose_ms": round(self.m["pose_ms"].pct(50), 1),
                "surprise_ms": round(self.m["surprise_ms"].pct(50), 1), "depth_ms": round(self.m["depth_ms"].pct(50), 1),
                "det_fraction": round(self.det_frames / max(1, self.frames), 3),
                "dropped": self.src.dropped, "frames": self.frames, "calm_why": dict(self.calm_why),
                "brightness": None if self.brightness is None else round(self.brightness, 1),
            },
            "surprise": None if st is None else {
                "score": round(st.score, 2), "raw": round(st.raw, 2), "learning": st.learning, "step": st.step,
                "warmup": self.cfg.surprise.warmup_steps, "regions": st.regions, "loss": round(st.loss, 4),
            },
            "source": self.source_info,
            "router": {**self.router.stats, "budget_left": self.router.budget_left(), "pending": getattr(self.router.pending, "kind", None)},
            "reasoner": {"available": self.reasoner.available, "status": self.reasoner.status,
                         "model": self.reasoner.model, "last": self.reasoner.last,
                         "history": list(self.reasoner.history)[-8:]},
        }

    def _render(self, f) -> None:
        img = f.image.copy()
        h, w = img.shape[:2]
        st = self.surprise.state if self.surprise else None
        if self.show_heatmap and st is not None and st.zmap is not None and not st.learning:
            # a vermilion wash where the robot was surprised, like a glaze over the underdrawing
            z = np.clip((st.zmap - 1.0) / 5.0, 0, 1)
            # clip after resizing: cubic interpolation overshoots below 0, which would tint away from red
            a = np.clip(cv2.resize(z.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)[..., None] * 0.45
            img = (img * (1 - a) + CINABRO * a).astype(np.uint8)
        with self.lock:
            ents = {e.tid: e for e in self.scene.visible()}
        age = time.perf_counter() - self.last_det_t
        for d in self.last_dets:
            e = ents.get(d.tid)
            if e is None or e.state == "tentative":
                continue
            box = d.box
            if age > 0.01:                                # detector skipped this frame: coast on velocity
                box = [box[0] + e.vel[0] * age, box[1] + e.vel[1] * age, box[2], box[3]]
            cat = "person" if e.is_person else "animal" if e.label in ANIMALS else "object"
            col = CAT_COLORS[cat]
            x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), int((box[0] + box[2]) * w), int((box[1] + box[3]) * h)
            if d.poly is not None and age <= 0.01:
                cv2.polylines(img, [d.poly], True, GESSO, 1, cv2.LINE_AA)
            # a hairline frame with heavier corner marks, like a squared-up drawing
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 1, cv2.LINE_AA)
            L = max(6, min(16, (x1 - x0) // 5, (y1 - y0) // 5))
            for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                cv2.line(img, (cx, cy), (cx + sx * L, cy), col, 2, cv2.LINE_AA)
                cv2.line(img, (cx, cy), (cx, cy + sy * L), col, 2, cv2.LINE_AA)
            if d.kpts is not None and age <= 0.01:       # the figure in chalk, as on toned paper
                k = d.kpts
                for a_, b_ in SKELETON:
                    if k[a_, 2] > 0.5 and k[b_, 2] > 0.5:
                        cv2.line(img, (int(k[a_, 0]), int(k[a_, 1])), (int(k[b_, 0]), int(k[b_, 1])), GESSO, 1, cv2.LINE_AA)
            # the cartellino: a small paper label pinned to the corner
            parts = [e.label]                          # no tracker numbers: they mean nothing to a person
            if e.depth_m:
                parts.append(f"{e.depth_m:.1f} m")
            if e.posture_since:
                parts.append(e.posture_since[0])
            if e.cues.get("hand_raised"):
                parts.append("hand raised")
            if e.expression and e.expression != "a neutral expression":
                parts.append(e.expression)
            tag = ", ".join(parts)
            (tw, th), base = cv2.getTextSize(tag, LABEL_FONT, 0.75, 1)
            ly1 = y0 - 3 if y0 - th - 12 >= 0 else min(h - 1, y1 + th + 12)
            ly0 = ly1 - th - 9
            lx = max(0, min(x0, w - tw - 13))              # keep the label on the picture
            cv2.rectangle(img, (lx, ly0), (lx + tw + 12, ly1), GESSO, -1)
            cv2.rectangle(img, (lx, ly0), (lx + tw + 12, ly1), col, 1)
            cv2.putText(img, tag, (lx + 6, ly1 - 5), LABEL_FONT, 0.75, UMBER, 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.server.jpeg_quality])
        if ok:
            self.overlay_jpeg, self.overlay_id = jpg.tobytes(), f.idx
