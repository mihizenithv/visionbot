"""HTTP/WebSocket interface: live annotated video, JSON world state, and natural-language questions.

A robot controller can consume the same `/ws` JSON stream the dashboard uses; nothing here is
dashboard-specific.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import re

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .capture import IMAGE_EXTS, VIDEO_EXTS
from .config import ROOT
from .engine import Engine

UPLOADS = ROOT / "uploads"
MAX_UPLOAD = 1024 * 1024 * 1024        # 1 GB

STATIC = Path(__file__).parent / "static"


class Question(BaseModel):
    question: str


class Toggle(BaseModel):
    heatmap: bool | None = None
    adaptive: bool | None = None
    vocab: list[str] | None = None
    reset_habituation: bool | None = None


def make_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="VisionBot")
    viewers = {"n": 0}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/stream.mjpg")
    async def stream():
        async def gen():
            viewers["n"] += 1
            engine.want_overlay = True
            last = -1
            try:
                while True:
                    if engine.overlay_jpeg is not None and engine.overlay_id != last:
                        last = engine.overlay_id
                        jpg = engine.overlay_jpeg
                        yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                    await asyncio.sleep(1 / engine.cfg.server.stream_fps)
            finally:
                viewers["n"] -= 1
                engine.want_overlay = viewers["n"] > 0
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/state")
    def state():
        return Response(json.dumps(engine.state(), default=float), media_type="application/json")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                await sock.send_text(json.dumps(engine.state(), default=float))
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.post("/api/ask")
    async def ask(q: Question):
        text = q.question.strip()[:500]
        if not text:
            return JSONResponse({"error": "empty question"}, status_code=400)
        t0 = time.perf_counter()
        res = await asyncio.to_thread(engine.ask, text)
        res.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000))
        return JSONResponse(res)

    @app.post("/api/source")
    async def source(request: Request):
        """Switch what the robot watches: JSON {"camera": 0} for a webcam, or a raw file body
        (image or video) with an X-Filename header."""
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            target: str | int = int(body.get("camera", 0))
        else:
            name = request.headers.get("x-filename", "upload")
            name = re.sub(r"[^A-Za-z0-9._ -]", "_", name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])[:120] or "upload"
            ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
            if ext not in IMAGE_EXTS | VIDEO_EXTS:
                return JSONResponse({"error": f"Unsupported file type '{ext or name}'. Use a picture or a video."}, status_code=400)
            UPLOADS.mkdir(exist_ok=True)
            dest = UPLOADS / f"{int(time.time())}_{name}"
            size = 0
            with open(dest, "wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        f.close()
                        dest.unlink(missing_ok=True)
                        return JSONResponse({"error": "That file is larger than 1 GB."}, status_code=413)
                    f.write(chunk)
            target = str(dest)
        try:
            info = await asyncio.to_thread(engine.switch_source, target)
        except Exception as e:
            return JSONResponse({"error": f"Could not open it: {e}"}, status_code=400)
        return JSONResponse(info)

    @app.get("/api/entry/{entry}.jpg")
    def entry(entry: int):
        jpg = engine.reasoner.snapshots.get(entry)
        if jpg is None:
            return Response(status_code=404)
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.post("/api/toggle")
    def toggle(t: Toggle):
        if t.heatmap is not None:
            engine.show_heatmap = t.heatmap
        if t.adaptive is not None:
            engine.cfg.fast.adaptive = t.adaptive
        if t.vocab:
            engine.fast.set_vocab([w.strip() for w in t.vocab if w.strip()][:80])
        if t.reset_habituation and engine.surprise:
            engine.surprise.reset()
        return {"heatmap": engine.show_heatmap, "adaptive": engine.cfg.fast.adaptive, "vocab": engine.cfg.fast.vocab}

    return app
