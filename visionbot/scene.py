"""Scene memory: turns per-frame detections into a persistent world state with object permanence.

The fast lane only knows about this frame. The robot needs to know "the keys were on the table two
minutes ago", "the person on the left has been sitting for a while", "someone just picked up the
cup". This module keeps entities alive after they leave view, smooths their labels by voting, fuses
depth into 3D positions, re-recognises people who come back, and emits discrete *events*, which
(together with JEPA surprise) are what wake the slow reasoning lane.

Tracker IDs are internal bookkeeping. Nothing a person reads (event text, journal, labels) contains
them; things are described by where they are instead.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .depth import backproject

EXPRESSIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
# how each FER+ class is described: what is visible, not a claim about inner feelings
EXPRESSION_WORDS = {"neutral": "a neutral expression", "happiness": "smiling", "surprise": "looking surprised",
                    "sadness": "looking sad", "anger": "frowning", "disgust": "grimacing",
                    "fear": "looking startled", "contempt": "smirking"}
# rarer, less reliable classes need more evidence before they are reported
EXPRESSION_MIN = {"neutral": 0.45, "happiness": 0.45, "surprise": 0.5, "sadness": 0.5, "anger": 0.55,
                  "disgust": 0.65, "fear": 0.65, "contempt": 0.7}


def where(box: list[float], dist: float | None = None) -> str:
    """'on the left, about 2.3 m away' — how a person would point something out."""
    cx = box[0] + box[2] / 2
    side = "on the left" if cx < 0.36 else "on the right" if cx > 0.64 else "in the middle"
    if dist is None:
        return side
    d = f"{dist:.1f}" if dist < 10 else f"{dist:.0f}"
    return f"{side}, about {d} m away"


def side_of(box: list[float]) -> str:
    return where(box).split(",")[0]


def appearance(bgr: np.ndarray, box: list[float]) -> np.ndarray | None:
    """Colour signature of a person's clothes: hue/saturation histograms of torso and legs.
    Cheap, and good enough to tell the people in one home apart after a short absence."""
    h, w = bgr.shape[:2]
    x0, y0 = int(box[0] * w), int(box[1] * h)
    x1, y1 = int((box[0] + box[2]) * w), int((box[1] + box[3]) * h)
    if x1 - x0 < 12 or y1 - y0 < 24:
        return None
    crop = bgr[max(0, y0):min(h, y1), max(0, x0 + (x1 - x0) // 6):min(w, x1 - (x1 - x0) // 6)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    parts = []
    for a, b in ((0.2, 0.55), (0.55, 1.0)):                     # torso, legs (skip the head: hair/skin)
        seg = hsv[int(a * 64):int(b * 64)]
        hist = cv2.calcHist([seg], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        parts.append(hist / (hist.sum() + 1e-6))
    return np.concatenate(parts).astype(np.float32)


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.minimum(a, b).sum() / 2.0)                 # histogram intersection, 0..1


@dataclass
class Entity:
    tid: int
    votes: dict[str, float]
    box: list[float]
    first_seen: float
    last_seen: float
    hits: int = 1
    conf: float = 0.0
    vel: list[float] = field(default_factory=lambda: [0.0, 0.0])      # normalised units / s
    depth_m: float | None = None
    xyz: list[float] | None = None
    cues: dict = field(default_factory=dict)
    novelty: float = 0.0
    activity: str | None = None                                        # filled by the reasoner
    state: str = "tentative"                                           # tentative|visible|occluded|left
    home_box: list[float] | None = None                                # where a static object normally sits
    posture_since: tuple[str, float] | None = None     # debounced (stable) posture and since when
    posture_cand: tuple[str, float] | None = None      # latest raw posture, waiting to be confirmed
    hand_frames: int = 0
    announced: set = field(default_factory=set)
    app: np.ndarray | None = None                      # clothing colour signature (people)
    expr: np.ndarray | None = None                     # smoothed FER+ probabilities
    expression: str | None = None

    @property
    def label(self) -> str:
        return max(self.votes, key=self.votes.get)

    @property
    def is_person(self) -> bool:
        return self.label == "person"

    def where(self) -> str:
        return where(self.box, self.depth_m)

    def to_dict(self, now: float) -> dict:
        d = {"id": self.tid, "label": self.label, "state": self.state, "conf": round(self.conf, 2),
             "box": [round(v, 3) for v in self.box], "seen_s_ago": round(now - self.last_seen, 1),
             "age_s": round(now - self.first_seen, 1), "novelty": round(self.novelty, 2),
             "side": side_of(self.box)}
        if self.depth_m is not None:
            d["dist_m"] = round(self.depth_m, 2)
            d["xyz_m"] = self.xyz
        cues = {k: v for k, v in self.cues.items() if k != "expr"}
        if self.posture_since:
            cues["posture"] = self.posture_since[0]                   # the debounced posture, not the raw one
        if self.expression:
            cues["expression"] = self.expression
        if cues:
            d["cues"] = cues
        if self.activity:
            d["activity"] = self.activity
        return d


@dataclass
class Event:
    t: float
    kind: str
    priority: int           # 0 = log only, 1 = worth a look, 2 = important, 3 = urgent
    text: str
    ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"t": round(self.t, 2), "kind": self.kind, "priority": self.priority, "text": self.text, "ids": self.ids}


class SceneMemory:
    CONFIRM_HITS = 3
    OCCLUDED_AFTER = 0.5
    LEFT_AFTER = 2.0
    FORGET_AFTER = 30 * 60.0
    REID_WINDOW = {"person": 180.0}           # seconds a departed person can still be recognised on return
    REID_WINDOW_OBJECT = 900.0
    REID_MIN_SIM = 0.62

    def __init__(self, hfov_deg: float):
        self.hfov = hfov_deg
        self.aspect = 9 / 16
        self.ents: dict[int, Entity] = {}
        self.alias: dict[int, int] = {}      # new tracker id -> the entity it turned out to be
        self.events: deque[Event] = deque(maxlen=300)
        self._pending: list[Event] = []
        self.narrative: dict | None = None
        self.t0 = time.perf_counter()
        self.reids = 0

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, prio: int, text: str, ids: list[int] | None = None) -> None:
        ev = Event(time.perf_counter() - self.t0, kind, prio, text, ids or [])
        self.events.append(ev)
        self._pending.append(ev)

    def pop_events(self) -> list[Event]:
        out, self._pending = self._pending, []
        return out

    # ------------------------------------------------------------------ update
    def update(self, dets, now: float, depth_fn=None, novelty_fn=None, frame_hw=(720, 1280), frame=None) -> None:
        self.aspect = frame_hw[0] / frame_hw[1]
        seen = set()
        for d in dets:
            tid = self.alias.get(d.tid, d.tid)
            e = self.ents.get(tid)
            if e is None:
                e = self.ents[tid] = Entity(tid, {d.label: d.conf}, d.box, now, now, conf=d.conf)
            else:
                dt = max(1e-3, now - e.last_seen)
                cx0, cy0 = e.box[0] + e.box[2] / 2, e.box[1] + e.box[3] / 2
                cx1, cy1 = d.box[0] + d.box[2] / 2, d.box[1] + d.box[3] / 2
                e.vel = [0.7 * e.vel[0] + 0.3 * (cx1 - cx0) / dt, 0.7 * e.vel[1] + 0.3 * (cy1 - cy0) / dt]
                for k in e.votes:
                    e.votes[k] *= 0.95
                e.votes[d.label] = e.votes.get(d.label, 0.0) + d.conf
                e.box, e.conf, e.last_seen = d.box, 0.8 * e.conf + 0.2 * d.conf, now
                e.hits += 1
            e.cues = d.cues or ({} if e.is_person else e.cues)
            if frame is not None and e.is_person and d.conf > 0.4 and (e.hits < 10 or e.hits % 5 == 0):
                a = appearance(frame, d.box)
                if a is not None:
                    e.app = a if e.app is None else 0.8 * e.app + 0.2 * a
            if e.is_person and d.cues.get("expr") is not None:
                p = np.asarray(d.cues["expr"], np.float32)
                e.expr = p if e.expr is None else 0.75 * e.expr + 0.25 * p
                k = int(e.expr.argmax())
                name = EXPRESSIONS[k]
                e.expression = EXPRESSION_WORDS[name] if e.expr[k] >= EXPRESSION_MIN[name] else None
            if depth_fn:
                z = depth_fn(e.box)
                if z is not None and 0.1 < z < 20:
                    e.depth_m = z if e.depth_m is None else 0.7 * e.depth_m + 0.3 * z
                    e.xyz = backproject(e.box, e.depth_m, self.hfov, self.aspect)
            if novelty_fn:
                e.novelty = novelty_fn(e.box)
            seen.add(e.tid)
            self._on_seen(e, now)

        for tid, e in list(self.ents.items()):
            if tid in seen:
                continue
            gone = now - e.last_seen
            if e.state == "tentative" and gone > self.OCCLUDED_AFTER:
                del self.ents[tid]                                  # never confirmed: flicker, forget
            elif e.state == "visible" and gone > self.OCCLUDED_AFTER:
                e.state = "occluded"
            elif e.state == "occluded" and gone > self.LEFT_AFTER:
                e.state = "left"
                self._on_left(e, now)
            elif e.state == "left" and gone > self.FORGET_AFTER:
                del self.ents[tid]
        if len(self.alias) > 2000:                                  # aliases of long-forgotten tracks
            live = set(self.ents)
            self.alias = {k: v for k, v in self.alias.items() if v in live}

    # ------------------------------------------------------------------ re-identification
    def _reidentify(self, e: Entity, now: float) -> Entity | None:
        """Is this 'new' track actually someone/something we already know, back after an absence?"""
        best, best_s = None, 0.0
        for o in self.ents.values():
            if o is e or o.label != e.label or o.state not in ("occluded", "left"):
                continue
            window = self.REID_WINDOW.get(e.label, self.REID_WINDOW_OBJECT)
            if now - o.last_seen > window:
                continue
            if e.is_person:
                s = similarity(e.app, o.app)
                # someone who vanished a moment ago and reappears nearby is very likely the same person
                if now - o.last_seen < 3.0:
                    gap = abs((e.box[0] + e.box[2] / 2) - (o.box[0] + o.box[2] / 2))
                    s += max(0.0, 0.25 - gap)
            else:
                # things don't walk: same kind of object in (nearly) the same place is the same object
                gap = abs((e.box[0] + e.box[2] / 2) - (o.box[0] + o.box[2] / 2)) + abs((e.box[1] + e.box[3] / 2) - (o.box[1] + o.box[3] / 2))
                s = 1.0 - gap * 4
            if s > best_s:
                best, best_s = o, s
        return best if best is not None and best_s >= self.REID_MIN_SIM else None

    def _merge_into(self, old: Entity, new: Entity) -> None:
        old.box, old.vel, old.last_seen, old.conf = new.box, new.vel, new.last_seen, new.conf
        old.hits += new.hits
        old.cues = new.cues or old.cues
        if new.app is not None:
            old.app = new.app if old.app is None else 0.7 * old.app + 0.3 * new.app
        if new.depth_m is not None:
            old.depth_m, old.xyz = new.depth_m, new.xyz
        self.alias[new.tid] = old.tid
        for k, v in list(self.alias.items()):
            if v == new.tid:
                self.alias[k] = old.tid
        self.ents.pop(new.tid, None)
        self.reids += 1

    def _on_seen(self, e: Entity, now: float) -> None:
        if e.state == "tentative" and e.hits >= self.CONFIRM_HITS:
            known = self._reidentify(e, now)
            if known is not None:
                was_left = known.state == "left"
                self._merge_into(known, e)
                known.state = "visible"
                if was_left:
                    what = "The person" if known.is_person else f"The {known.label}"
                    self._emit("returned", 1, f"{what} {side_of(known.box)} is back", [known.tid])
                e = known
            else:
                e.state = "visible"
                if e.is_person:
                    self._emit("person_entered", 2, f"Someone came into view {e.where()}", [e.tid])
                else:
                    self._emit("object_appeared", 1 if now - self.t0 > 5 else 0,
                               f"{'An' if e.label[0] in 'aeiou' else 'A'} {e.label} appeared {side_of(e.box)}", [e.tid])
                    e.home_box = list(e.box)
        elif e.state in ("occluded", "left"):
            was_left = e.state == "left"
            e.state = "visible"
            if was_left:
                what = "The person" if e.is_person else f"The {e.label}"
                self._emit("returned", 1, f"{what} {side_of(e.box)} is back", [e.tid])
        if e.state != "visible":
            return
        if e.is_person:
            self._person_cues(e, now)
        elif e.home_box is not None:
            moved = abs((e.box[0] + e.box[2] / 2) - (e.home_box[0] + e.home_box[2] / 2)) + \
                abs((e.box[1] + e.box[3] / 2) - (e.home_box[1] + e.home_box[3] / 2))
            if moved > 0.12 and abs(e.vel[0]) + abs(e.vel[1]) < 0.05:     # moved and has come to rest
                self._emit("object_moved", 1, f"The {e.label} was moved; it is now {side_of(e.box)}", [e.tid])
                e.home_box = list(e.box)

    def _person_cues(self, e: Entity, now: float) -> None:
        # Pose estimates wobble frame to frame; cues must persist before they become events, so a single
        # bad skeleton can't raise a (priority-3) fall alarm.
        c = e.cues
        e.hand_frames = e.hand_frames + 1 if c.get("hand_raised") else 0
        if e.hand_frames >= 3 and "hand" not in e.announced:
            e.announced.add("hand")
            self._emit("hand_raised", 2, f"The person {side_of(e.box)} raised a hand", [e.tid])
        elif e.hand_frames == 0:
            e.announced.discard("hand")
        posture = c.get("posture")
        if posture:
            if not e.posture_cand or e.posture_cand[0] != posture:
                e.posture_cand = (posture, now)
            elif now - e.posture_cand[1] >= 0.8 and (not e.posture_since or e.posture_since[0] != posture):
                prev = e.posture_since[0] if e.posture_since else None
                e.posture_since = (posture, now)
                if posture == "lying" and prev in ("standing", "sitting"):
                    self._emit("possible_fall", 3, f"The person {side_of(e.box)} went from {prev} to lying down", [e.tid])
        if e.depth_m is not None and e.depth_m < 1.0 and "close" not in e.announced:
            e.announced.add("close")
            self._emit("person_close", 2, f"The person {side_of(e.box)} came within a metre of the robot", [e.tid])
        elif e.depth_m is not None and e.depth_m > 1.4:
            e.announced.discard("close")

    def _on_left(self, e: Entity, now: float) -> None:
        if e.is_person:
            self._emit("person_left", 1, f"The person {side_of(e.box)} went out of view", [e.tid])
            return
        # object vanished while a person was right next to it → probably picked up
        cx, cy = e.box[0] + e.box[2] / 2, e.box[1] + e.box[3] / 2
        for p in self.ents.values():
            if p.is_person and p.state == "visible":
                px0, py0, pw, ph = p.box
                if px0 - 0.05 < cx < px0 + pw + 0.05 and py0 - 0.05 < cy < py0 + ph + 0.05:
                    self._emit("object_taken", 2, f"The {e.label} {side_of(e.box)} disappeared next to someone; probably picked up", [e.tid, p.tid])
                    return
        self._emit("object_gone", 1, f"The {e.label} {side_of(e.box)} is no longer visible", [e.tid])

    # ------------------------------------------------------------------ views
    def visible(self) -> list[Entity]:
        return [e for e in self.ents.values() if e.state in ("visible", "occluded")]

    def remembered(self) -> list[Entity]:
        return [e for e in self.ents.values() if e.state == "left"]

    def snapshot(self, now: float, recent_events: int = 12) -> dict:
        vis = sorted(self.visible(), key=lambda e: (not e.is_person, e.tid))
        return {
            "t": round(now - self.t0, 2),
            "people": [e.to_dict(now) for e in vis if e.is_person],
            "objects": [e.to_dict(now) for e in vis if not e.is_person],
            "remembered": [e.to_dict(now) for e in sorted(self.remembered(), key=lambda e: -e.last_seen)[:15]],
            "events": [ev.to_dict() for ev in list(self.events)[-recent_events:]],
            "narrative": self.narrative,
            "reidentified": self.reids,
        }

    def apply_reasoning(self, result: dict) -> None:
        """Merge the slow lane's interpretation back into the entities."""
        self.narrative = result
        for p in result.get("people", []) or []:
            try:
                tid = int(str(p.get("id")).lstrip("#"))
            except (TypeError, ValueError):
                continue
            e = self.ents.get(self.alias.get(tid, tid))
            if e is not None and p.get("activity"):
                e.activity = str(p["activity"])[:120]
