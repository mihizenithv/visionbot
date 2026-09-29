"""Run VisionBot:  python -m visionbot [--source 0|video.mp4] [--no-depth] [--no-surprise] [--no-reasoner]"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from .config import Config


def main() -> None:
    ap = argparse.ArgumentParser(prog="visionbot")
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--detector", default=None, help="e.g. yoloe-26n-seg.pt for more speed, yoloe-26m-seg.pt for accuracy")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-surprise", action="store_true")
    ap.add_argument("--no-reasoner", action="store_true")
    ap.add_argument("--no-adaptive", action="store_true", help="run the detector on every frame")
    ap.add_argument("--headless", action="store_true", help="no web server; print state to the console")
    ap.add_argument("--loop", action="store_true", help="loop a video file")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    cfg = Config()
    cfg.camera.source = int(a.source) if a.source.isdigit() else a.source
    cfg.camera.width, cfg.camera.height = a.width, a.height
    cfg.camera.loop = a.loop
    cfg.server.port = a.port
    cfg.depth.enabled = not a.no_depth
    cfg.surprise.enabled = not a.no_surprise
    cfg.reasoner.enabled = not a.no_reasoner
    cfg.fast.adaptive = not a.no_adaptive
    if a.detector:
        cfg.fast.detector = a.detector

    from .engine import Engine
    engine = Engine(cfg)
    engine.exit_on_end = a.headless           # the server keeps running (and accepts new sources)
    engine.start()
    print(f"[visionbot] reasoner: {engine.reasoner.status if not engine.reasoner.available else 'NVIDIA API ready'}")

    if a.headless:
        signal.signal(signal.SIGINT, lambda *_: (engine.stop(), sys.exit(0)))
        while not engine.finished:
            s = engine.state()
            p, su = s["perf"], s["surprise"] or {}
            print(f"fps {p['fps']:5.1f} | e2e p50 {p['e2e_p50']:5.1f} ms | det {p['det_ms']:4.1f} ms "
                  f"({p['det_fraction']*100:.0f}% frames) | people {len(s['people'])} objects {len(s['objects'])} | "
                  f"surprise {su.get('score', 0):.1f}{' (learning)' if su.get('learning') else ''}")
            time.sleep(1)
        engine.stop()
        return

    import uvicorn
    from .server import make_app
    print(f"[visionbot] dashboard → http://{cfg.server.host}:{cfg.server.port}")
    try:
        uvicorn.run(make_app(engine), host=cfg.server.host, port=cfg.server.port, log_level="warning")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
