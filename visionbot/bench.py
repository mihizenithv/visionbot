"""Benchmark: does JEPA-surprise gating actually help?

Runs the full pipeline on a video (paced like a real camera) and reports:

1. Latency per stage and end to end (camera frame → updated world state), p50 / p95.
2. Compute saved by the adaptive fast lane (fraction of frames where the detector ran).
3. Slow-lane efficiency: VLM calls made by the surprise/event router versus fixed-rate
   baselines, and how long an important event waits before the VLM looks at it.

The reasoner runs in dry-run mode (no API calls, no credits), so this is free to repeat.

    python -m visionbot.bench path/to/video.mp4 [--seconds 60] [--compare-fixed]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .config import Config


def run(video: str, seconds: float, adaptive: bool, surprise: bool, warm_state: bool) -> dict:
    from .engine import Engine
    cfg = Config()
    cfg.camera.source = video
    cfg.camera.realtime_file = True
    cfg.fast.adaptive = adaptive
    cfg.surprise.enabled = surprise
    cfg.reasoner.enabled = False                      # dry run: router decides, nothing is sent
    eng = Engine(cfg, log=lambda *_: None)
    if surprise and not warm_state:
        eng.surprise.reset()
    eng.router_dry_calls: list[tuple[float, str]] = []
    orig_ready = eng.router.ready

    def spy():
        trig = orig_ready()
        if trig:
            eng.router_dry_calls.append((time.perf_counter(), trig.kind))
        return trig

    eng.router.ready = spy
    t_start = time.perf_counter()
    eng.start()
    # the engine only consults the router when a reasoner is available; poll it ourselves instead
    while not eng.finished and time.perf_counter() - t_start < seconds:
        with eng.lock:
            eng.router.ready()
        time.sleep(0.05)
    dur = time.perf_counter() - t_start
    st = eng.state()
    events = [e for e in eng.scene.events]
    eng.stop()

    calls = [t - t_start for t, _ in eng.router_dry_calls]
    important = [e.t + (eng.scene.t0 - t_start) for e in events if e.priority >= 2]
    cov, waits = coverage(important, calls)
    return {
        "adaptive": adaptive, "surprise": surprise, "seconds": round(dur, 1), "frames": st["perf"]["frames"],
        "fps": st["perf"]["fps"], "e2e_p50_ms": st["perf"]["e2e_p50"], "e2e_p95_ms": st["perf"]["e2e_p95"],
        "det_ms": st["perf"]["det_ms"], "pose_ms": st["perf"]["pose_ms"], "surprise_ms": st["perf"]["surprise_ms"],
        "depth_ms": st["perf"]["depth_ms"], "detector_frame_fraction": st["perf"]["det_fraction"],
        "dropped_frames": st["perf"]["dropped"], "calm_why": st["perf"]["calm_why"],
        "events_total": len(events), "events_important": len(important),
        "surprise_spikes": eng.router.stats["surprise_spikes"],
        "vlm_calls": len(calls), "vlm_calls_per_min": round(60 * len(calls) / dur, 2),
        "call_kinds": {k: sum(1 for _, kk in eng.router_dry_calls if kk == k) for k in {kk for _, kk in eng.router_dry_calls}},
        "coverage_10s": cov, "covered_wait_s_mean": waits,
        "events_per_call": round(len(important) / max(1, len(calls)), 2),
        "important_times": important,
    }


def coverage(important: list[float], calls: list[float], window: float = 10.0):
    """Share of important events that a VLM call looked at within `window` s, and the mean wait for those."""
    if not important:
        return None, None
    waits = [min((c - t for c in calls if 0 <= c - t <= window), default=None) for t in important]
    got = [w for w in waits if w is not None]
    return round(len(got) / len(important), 3), (round(float(np.mean(got)), 2) if got else None)


def fixed_rate(important: list[float], dur: float, period: float) -> dict:
    ticks = list(np.arange(period, dur + 1e-9, period))
    cov, waits = coverage(important, ticks)
    return {"policy": f"fixed every {period:g}s", "vlm_calls": len(ticks), "vlm_calls_per_min": round(60 / period, 2),
            "coverage_10s": cov, "covered_wait_s_mean": waits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--compare-fixed", action="store_true", help="also run with adaptive compute off")
    ap.add_argument("--warm", action="store_true", help="keep previously learned habituation")
    ap.add_argument("--out", default="bench_report.json")
    a = ap.parse_args()

    results = {"gated": run(a.video, a.seconds, adaptive=True, surprise=True, warm_state=a.warm)}
    if a.compare_fixed:
        results["always_on"] = run(a.video, a.seconds, adaptive=False, surprise=True, warm_state=a.warm)
    g = results["gated"]
    results["fixed_rate_baselines"] = [fixed_rate(g["important_times"], g["seconds"], p) for p in (2, 5, 10)]
    for r in results.values():
        if isinstance(r, dict):
            r.pop("important_times", None)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
