"""Launcher that works from any working directory:  python path/to/run.py [visionbot args]"""
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
os.chdir(here)

from visionbot.__main__ import main  # noqa: E402

main()
