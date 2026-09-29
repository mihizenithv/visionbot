"""Central configuration. Every tunable lives here so experiments are one-line changes."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
os.environ.setdefault("HF_HOME", str(WEIGHTS / "hf"))          # keep every model file inside the project

# Open vocabulary for a home / people-interaction robot. YOLOE turns these words into
# class embeddings once at start-up, so changing the list costs nothing per frame.
HOME_VOCAB = [
    "person", "cat", "dog",
    "cup", "mug", "bottle", "glass", "bowl", "plate", "knife",
    "cell phone", "laptop", "keyboard", "computer mouse", "remote control", "television", "monitor",
    "book", "bag", "backpack", "keys", "wallet", "eyeglasses", "headphones", "charger cable",
    "chair", "sofa", "table", "desk", "bed", "door", "window", "lamp", "potted plant",
    "shoe", "pillow", "box", "toy", "medicine bottle", "scissors", "clock", "fan",
]


@dataclass
class CameraCfg:
    source: str | int = 0            # webcam index or a video file path
    width: int = 1280
    height: int = 720
    fps: int = 30
    hfov_deg: float = 70.0           # typical laptop webcam; used to back-project depth into 3D
    realtime_file: bool = True       # pace video files at their native fps (simulates a camera)
    loop: bool = False               # restart video files at the end (demos)


@dataclass
class FastLaneCfg:
    detector: str = "yoloe-26s-seg.pt"
    pose: str = "yolo26n-pose.pt"
    imgsz: int = 640
    conf: float = 0.25
    half: bool = True
    vocab: list[str] = field(default_factory=lambda: list(HOME_VOCAB))
    pose_every: int = 2              # pose model runs every Nth frame when people are present
    # Surprise-gated stride: when the scene is calm the detector runs every `calm_stride` frames and
    # the tracker's Kalman filter coasts in between; any surprise snaps it back to every frame.
    calm_stride: int = 3
    adaptive: bool = True


@dataclass
class DepthCfg:
    enabled: bool = True
    model: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    input_short: int = 336           # multiple of 14 (ViT patch); lower = faster
    period_s: float = 0.2            # 5 Hz is plenty: depth of a tracked object changes slowly


@dataclass
class SurpriseCfg:
    enabled: bool = True
    hub_entry: str = "vjepa2_1_vit_base_384"
    grid_w: int = 20                 # input = grid*16 px → 320x240 for 20x15 patches
    grid_h: int = 15
    period_s: float = 0.1            # 10 Hz latent stream
    context: int = 3                 # predictor sees the last K latent maps
    lr: float = 1e-3
    warmup_steps: int = 150          # predictor must learn "normal" before surprise is reported
    habituation: float = 0.01        # EMA rate of per-patch error statistics (~10 s memory at 10 Hz)
    scene_habituation: float = 0.002 # EMA rate of the scene-score statistics (~50 s memory at 10 Hz)
    z_threshold: float = 3.0         # patch is surprising when error z-score exceeds this
    spike_score: float = 3.0         # scene-level (relative) score that counts as a surprise event


@dataclass
class ReasonerCfg:
    enabled: bool = True
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    # First model the account can access wins. Cosmos Reason is NVIDIA's physical-AI VLM.
    models: list[str] = field(default_factory=lambda: [
        "nvidia/cosmos3-nano-reasoner",
        "nvidia/cosmos-reason2-8b",
        "nvidia/nemotron-nano-12b-v2-vl",
        "meta/llama-3.2-11b-vision-instruct",     # verified on a free account (~19 s per call)
        "meta/llama-3.2-90b-vision-instruct",
        "microsoft/phi-4-multimodal-instruct",
    ])
    max_rpm: int = 20                # stay well under the free tier's 40 RPM
    daily_budget: int = 300          # free tier ships ~1000 credits; don't burn them in a day
    active_hours: float = 16.0       # the budget is paced evenly over this many waking hours
    burst: int = 6                   # calls that may be spent back-to-back before pacing kicks in
    heartbeat_s: float = 90.0        # refresh the narrative at most this often when nothing happens
    min_gap_s: float = 4.0           # never call more often than this, whatever the trigger
    # Debounce: gather an episode of events into ONE call, fired when things go quiet for `quiet_s`
    # or after `max_hold_s` at most. Urgent (priority 3) events skip the wait.
    quiet_s: float = 0.8
    max_hold_s: float = 2.5
    timeout_s: float = 45.0         # free-tier queues are slow; 11B vision answered in ~19 s
    image_long_side: int = 512          # smaller picture: faster upload and prefill, still readable

    @property
    def api_key(self) -> str | None:
        key = os.environ.get(self.api_key_env)
        if not key:
            env = ROOT / ".env"
            if env.exists():
                for line in env.read_text(encoding="utf-8-sig").splitlines():
                    line = line.strip()
                    k, sep, v = line.partition("=")
                    if sep and k.strip() == self.api_key_env:
                        key = v.strip().strip('"').strip("'")
                    elif not sep and line.startswith("nvapi-"):     # a bare key pasted on its own line
                        key = line
        return key or None


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"
    port: int = 8770
    jpeg_quality: int = 80
    stream_fps: int = 30


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    fast: FastLaneCfg = field(default_factory=FastLaneCfg)
    depth: DepthCfg = field(default_factory=DepthCfg)
    surprise: SurpriseCfg = field(default_factory=SurpriseCfg)
    reasoner: ReasonerCfg = field(default_factory=ReasonerCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    device: str = "cuda"
