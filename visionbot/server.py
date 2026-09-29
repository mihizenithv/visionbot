"""HTTP/WebSocket interface: live annotated video, JSON world state, and natural-language questions.

A robot controller can consume the same `/ws` JSON stream the page uses; nothing here is
page-specific.

Access. The page can be opened three ways:
  * from this machine at http://127.0.0.1:8770 — trusted, no key needed;
  * from the Vercel-hosted copy of the page (another origin) — needs the access key;
  * through a tunnel (another host name) — needs the access key.
The key lives in `.env` as VISIONBOT_KEY and is generated on first start. Browsers can't attach
headers to <img> and WebSocket requests, so the page trades the key for a short-lived ticket
(`POST /api/session`) and uses that in their URLs; the long-term key never appears in a URL.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .capture import IMAGE_EXTS, VIDEO_EXTS
from .config import ROOT
from .engine import Engine

UPLOADS = ROOT / "uploads"
MAX_UPLOAD = 1024 * 1024 * 1024        # 1 GB
WEB = ROOT / "web"                     # the same folder Vercel publishes
TICKET_TTL = 12 * 3600
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
LOCAL_CLIENTS = {"127.0.0.1", "::1"}


def load_access_key() -> str:
    """VISIONBOT_KEY from the environment or .env; created (and saved to .env) if missing."""
    key = os.environ.get("VISIONBOT_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k.strip() == "VISIONBOT_KEY" and v.strip():
                key = v.strip().strip('"').strip("'")
    if not key:
        key = secrets.token_urlsafe(18)
        with open(env, "a", encoding="utf-8") as f:
            f.write(("\n" if env.exists() and env.read_text(encoding="utf-8").strip() else "") + f"VISIONBOT_KEY={key}\n")
        print("[visionbot] created an access key for remote pages and saved it to .env (VISIONBOT_KEY)")
    return key


class Question(BaseModel):
    question: str


class Toggle(BaseModel):
    heatmap: bool | None = None
    adaptive: bool | None = None
    vocab: list[str] | None = None
    reset_habituation: bool | None = None


def make_app(engine: Engine, allowed_origins: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="VisionBot")
    viewers = {"n": 0}
    access_key = load_access_key()
    tickets: dict[str, float] = {}

    # ------------------------------------------------------------------ access control
    def is_local(headers, client_host: str | None) -> bool:
        """A page served by this very server, opened on this very machine."""
        h = headers.get("host", "").lower()
        host = h.split("]")[0] + "]" if h.startswith("[") else h.split(":")[0]
        site = headers.get("sec-fetch-site")
        return host in LOCAL_HOSTS and client_host in LOCAL_CLIENTS and site in (None, "same-origin", "none")

    def key_ok(headers, query) -> bool:
        k = headers.get("x-visionbot-key")
        if k and hmac.compare_digest(k.encode(), access_key.encode()):
            return True
        t = query.get("ticket")
        if t and tickets.get(t, 0) > time.time():
            return True
        return False

    def allowed(headers, query, client_host) -> bool:
        return is_local(headers, client_host) or key_ok(headers, query)

    extra = [o.strip() for o in (allowed_origins or os.environ.get("VISIONBOT_ORIGINS", "").split(",")) if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=extra,
        allow_origin_regex=r"https://[a-z0-9-]+\.vercel\.app",     # the Vercel-hosted page
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-visionbot-key", "x-filename"],
        max_age=600,
        allow_private_network=True,                                  # Chrome's local-network check for 127.0.0.1
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS":
            resp = await call_next(request)
            # Chrome's private/local-network access check for public pages talking to 127.0.0.1
            if request.headers.get("access-control-request-private-network"):
                resp.headers["Access-Control-Allow-Private-Network"] = "true"
            return resp
        protected = (path.startswith("/api/") and path not in ("/api/health",)) or path == "/stream.mjpg"
        if protected and not allowed(request.headers, request.query_params, request.client.host if request.client else None):
            return JSONResponse({"error": "access key required"}, status_code=401)
        return await call_next(request)

    # ------------------------------------------------------------------ page + health
    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/health")
    def health(request: Request):
        return {"visionbot": True,
                "needs_key": not is_local(request.headers, request.client.host if request.client else None)}

    @app.post("/api/session")
    def session():
        """Swap the key (checked by the guard) for a ticket usable in <img>/WebSocket URLs."""
        now = time.time()
        for t in [t for t, exp in tickets.items() if exp < now]:
            del tickets[t]
        t = secrets.token_urlsafe(24)
        tickets[t] = now + TICKET_TTL
        return {"ticket": t, "expires_in": TICKET_TTL}

    # ------------------------------------------------------------------ live data
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
        origin = sock.headers.get("origin", "")
        cross = origin and not re.match(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$", origin)
        client = sock.client.host if sock.client else None
        if (cross or not is_local(sock.headers, client)) and not key_ok(sock.headers, sock.query_params):
            await sock.close(code=1008)
            return
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
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

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
