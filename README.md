# VisionBot: JEPA-gated perception for a home robot

A real-time vision system that runs every frame in a few milliseconds and **only thinks hard when
something surprising happens**. A small world model learns what "normal" looks like in your home
while it runs. Its prediction error decides where the GPU looks and when a vision-language model
(NVIDIA's free API) is asked what's going on.

```
camera ──► fast lane (every frame, ~8 ms) ─────────────► scene memory ──► JSON world state / dashboard
            YOLOE-26 open-vocab detection                  object permanence      ▲
            ByteTrack IDs · YOLO26 pose → cues              events (entered, hand  │
            (CUDA-graph replay, GPU letterbox + NMS)        raised, fall, taken…)  │
                    ▲ detector stride                             │               │
                    │                                             ▼               │
       JEPA surprise lane (10 Hz, ~14 ms) ───────────────► router ──► NVIDIA vision-language model
         frozen V-JEPA 2.1 encoder                          debounce · credit pacing   "what's happening"
         + online-trained latent predictor                  surprise / event / heartbeat
         + two-level habituation
       metric depth (5 Hz, ~11 ms): Depth Anything V2 indoor → distances + 3D positions
```

## What you see

A local web page (http://127.0.0.1:8770) laid out like an observer's notebook:

- **The live view**, framed at the picture's own proportions. Open a photo or video from disk (or
  drag it onto the view) and the whole system switches to it; "Use the camera" switches back.
- **Right now**: the scene in plain sentences. "Two people are in view. The person on the left is
  about 2.4 m away, facing the robot and smiling." People are described by where they are, never by
  a tracker number.
- **Journal**: a note is written the moment something happens (someone arrives, raises a hand, may
  have fallen, something unexpected), with a small snapshot. The language model's reading of the
  scene is added to the same note when it arrives, typically 4 to 30 s later on NVIDIA's free tier.
- **Where its attention went**: the surprise grid, plus the last minute of surprise.
- **Ask it something**: questions answered from memory and the current view.

Faces turned towards the camera are read for **visible expression** (smiling, frowning, surprised)
with the FER+ model. It reports what a face shows, not what someone feels, and it often calls a
subtle closed-mouth smile neutral. People who leave and come back within a few minutes are
**recognised again** by the colours of their clothing instead of being counted as new. On the
pedestrian test clip this halved false "someone came into view" events (31–33 down to 17).

## What's new here, and what isn't

Existing parts: YOLOE-26 open-vocabulary detection, ByteTrack, YOLO26-pose, Depth Anything V2, and
V-JEPA 2.1 (Meta, Mar 2026). Using V-JEPA surprise to decide *which memories to store* was published
in June 2026 ("Worth Remembering"); event-triggered inference for robot action models also exists.

The contribution of this system is how these fit together:

1. **Habituating online JEPA predictor.** V-JEPA 2.1's own predictor can't be used with the small
   ViT-B (it predicts into the distilled teacher's space). So a ~1.8M-parameter predictor is trained
   *online, on the robot*, in V-JEPA's latent space. That's the JEPA objective: predict
   representations, not pixels. It learns the specific home, and it saves itself to disk so the
   learning survives restarts.
2. **Two-level habituation.** Each patch's error is z-scored against that patch's own history
   (so a flickering TV stops counting), and the scene score is z-scored against the recent scene
   history (so a busy hallway stops counting). The scene-wide error component is subtracted first,
   because lighting and auto-exposure hit every patch at once while real events are local. Only
   surprise relative to *this place, lately* passes.
3. **One surprise signal gates both lanes.** It sets the detector stride in the fast lane (the
   tracker coasts through calm periods) and triggers vision-language calls in the slow lane.
4. **Credit-aware slow lane.** Events are debounced into episodes, calls are paced with a token
   bucket (free tier: 40 RPM, ~1000 credits), and urgent events (possible fall) jump the queue.
5. **Launch-bound fix.** On Windows the stock Ultralytics path is dominated by ~15 µs per-kernel
   launch overhead. Replaying the detector, pose model and V-JEPA encoder from **CUDA Graphs** made
   the fast lane 3.3× faster and the JEPA encoder 4.5× faster.

These are engineering and systems claims backed by the benchmarks below. They are not yet a
peer-reviewed result; see "Next steps" for what that would need.

## Measured (RTX 5060 Laptop, 8 GB)

| Stage | Before | Now |
|---|---|---|
| Detector + tracker (YOLOE-26s, 640) | 21.8 ms | 5.8 ms |
| Pose (YOLO26n-pose) | 11.6 ms | 2.2 ms |
| V-JEPA 2.1 ViT-B encode + online train | 44 ms | 13.7 ms |
| Metric depth (336 px) | — | 10.7 ms (5 Hz) |
| **Frame → world state, p50 / p95** | — | **7.6 / 14.1 ms** |

Tests: `tests/test_reasoner_mock.py` (router + NVIDIA client against a fake server, 14 checks),
`tests/probe_components.py` (each model on the GPU), `tests/check_nvidia.py` (your key, 1 credit).

**Gating benchmark** on `samples/calm_active.mp4` (88 s; long calm stretches with sensor noise and
lighting drift, then bursts of real motion; 25 important events). VLM calls are dry-run:

| Policy | VLM calls | Important events covered within 10 s | Mean wait |
|---|---|---|---|
| **Surprise + event router (this system)** | **6** | **96–100 %** (two runs) | ~2.4 s |
| Fixed-rate every 2 s | 44 | 100 % | 1.3 s |
| Fixed-rate every 5 s | 17 | 96 % | 2.7 s |
| Fixed-rate every 10 s | 8 | 68 % | 5.5 s |

Adaptive compute: the detector ran on 74% of frames, about 66% once the 15 s warm-up is excluded,
and median latency dropped from 9.5 ms to 7.6 ms. On a crowded scene (`vtest.avi`) the
gate correctly never engages. Honest caveats: this is one synthetic clip with 25 events. A real
evaluation needs labelled home footage.

The benchmark harness replays a video like a live camera and compares the gated system with an
always-on detector and fixed-rate VLM polling (dry run, no credits used):

```
python -m visionbot.bench samples/calm_active.mp4 --seconds 90 --compare-fixed
```

## Run it

```bash
# one-time setup (Python 3.11; RTX 50-series needs CUDA 12.8+ wheels)
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# NVIDIA API key (free) from build.nvidia.com: copy the template, then paste your key after the "="
cp .env.example .env        # .env is git-ignored; never commit it or paste the key anywhere public

.venv/Scripts/python.exe -m visionbot                 # webcam 0, dashboard at http://127.0.0.1:8770
.venv/Scripts/python.exe -m visionbot --source clip.mp4 --no-reasoner
.venv/Scripts/python.exe -m visionbot --headless      # console only
.venv/Scripts/python.exe tests/check_nvidia.py        # verify your NVIDIA key (uses 1 credit, sample image)
```

The first run downloads about 2 GB of weights into `weights/`. The surprise model needs about
15 seconds of ordinary scene to warm up before it reports anything.

Which language model answers depends on your NVIDIA account: the client tries Cosmos Reason first and
falls back automatically. On a free account in September 2026 only `meta/llama-3.2-11b-vision-instruct`
was actually served (Cosmos Reason was listed but returned 404; the 90B model timed out).

The benchmark clips aren't in the repository. `tests/make_calm_active.py` builds `calm_active.mp4`
from OpenCV's `vtest.avi` sample (download it into `samples/` first).

## Robot integration

`ws://127.0.0.1:8770/ws` streams the world state at 10 Hz: people (distance, 3D position, posture,
hand raised, facing robot, activity), objects, remembered objects that left view, events, surprise
and the latest interpretation. `POST /api/ask {"question": ...}` answers questions using memory and
the current view. The dashboard uses exactly these endpoints.

## Privacy

Frames only leave the machine when the router makes a VLM call, and then it's one downscaled JPEG
sent to NVIDIA's API. With `--no-reasoner`, or without an API key, everything stays local. The system
never tries to identify people: no face recognition, no names, and the language model is told not to
guess identity, age or gender. Expressions are read locally and described only as what is visible.

## Next steps

- **Moving robot:** condition the predictor on ego-motion (odometry or optical flow) so the robot's
  own movement isn't "surprising".
- **TensorRT FP16/INT8 engines** for the detector and encoder (a further ~2× expected), then a
  Jetson port.
- **Evaluation:** a labelled home dataset to report event recall versus calls per hour against
  fixed-rate and optical-flow gating baselines. That's what turns the systems claim into a research result.
- **Local fallback VLM** (e.g. Cosmos-Reason2-2B, 4-bit) for offline operation.

## Built on

Ultralytics YOLOE-26 and YOLO26-pose (AGPL-3.0), Meta's V-JEPA 2.1 (loaded from its official
repository at run time), Depth Anything V2 Metric Indoor Small, the FER+ expression model from the
ONNX model zoo, ByteTrack as shipped in Ultralytics, and NVIDIA's hosted models via build.nvidia.com.
Check each project's licence before reusing this code; the Ultralytics AGPL-3.0 licence in particular
applies to anything that includes it. No licence has been chosen yet for this repository's own code.
