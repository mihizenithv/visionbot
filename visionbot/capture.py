"""Camera capture that always hands out the *newest* frame.

The single biggest latency bug in hobby vision pipelines is a queue of stale frames: if inference
is slower than the camera, OpenCV's internal buffer fills and you process frames that are
hundreds of milliseconds old. Here a background thread drains the camera continuously and keeps
only the latest frame, so consumers never wait on history.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraCfg


@dataclass
class Frame:
    idx: int
    t_capture: float          # time.perf_counter() when the frame left the camera driver
    image: np.ndarray         # BGR, HxWx3


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


class LatestFrameSource:
    def __init__(self, cfg: CameraCfg):
        self.cfg = cfg
        self.is_file = not isinstance(cfg.source, int) and not str(cfg.source).isdigit()
        src = cfg.source if self.is_file else int(cfg.source)
        self.still = None
        self.kind = "camera"
        self.name = f"camera {src}" if not self.is_file else str(cfg.source).replace("\\", "/").split("/")[-1]
        if self.is_file and ("." + self.name.rsplit(".", 1)[-1].lower()) in IMAGE_EXTS:
            # a still picture is served as a steady 10 fps stream of the same frame
            self.still = cv2.imdecode(np.fromfile(str(cfg.source), np.uint8), cv2.IMREAD_COLOR)
            if self.still is None:
                raise RuntimeError(f"Could not read image {cfg.source!r}")
            self.kind, self.cap, self.native_fps = "image", None, 10.0
            self._init_state()
            return
        if self.is_file:
            self.kind = "video"
        backend = cv2.CAP_DSHOW if (sys.platform == "win32" and not self.is_file) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {cfg.source!r}")
        if not self.is_file:
            # MJPG lets USB webcams deliver 720p30; the default YUY2 often caps at 720p10.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or cfg.fps
        self._init_state()

    def _init_state(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: Frame | None = None
        self._idx = 0
        self._stop = False
        self.ended = False
        self.dropped = 0
        self._last_taken = -1
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)

    def start(self) -> "LatestFrameSource":
        if not self._thread.is_alive():
            self._thread.start()
        return self

    def _run(self) -> None:
        period = 1.0 / self.native_fps if (self.is_file and self.cfg.realtime_file) else 0.0
        next_t = time.perf_counter()
        while not self._stop:
            ok, img = (True, self.still) if self.still is not None else self.cap.read()
            if not ok:
                if self.is_file and self.cfg.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self.is_file:
                    self.ended = True
                    with self._cond:
                        self._cond.notify_all()
                    break
                time.sleep(0.01)
                continue
            t = time.perf_counter()
            with self._cond:
                if self._frame is not None and self._frame.idx != self._last_taken:
                    self.dropped += 1          # consumer was busy; the stale frame is discarded
                self._frame = Frame(self._idx, t, img)
                self._idx += 1
                self._cond.notify_all()
            if period:
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()

    def read(self, timeout: float = 1.0) -> Frame | None:
        """Block until a frame newer than the last one returned exists, then return it."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while (self._frame is None or self._frame.idx == self._last_taken) and not self.ended:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._frame is None or self._frame.idx == self._last_taken:
                return None
            self._last_taken = self._frame.idx
            return self._frame

    @property
    def size(self) -> tuple[int, int]:
        if self.still is not None:
            return self.still.shape[1], self.still.shape[0]
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def stop(self) -> None:
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
