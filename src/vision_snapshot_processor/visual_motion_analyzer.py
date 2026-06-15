"""Deprecated compatibility shim for the Self Mirror visual analyzer.

The Visual Motion Analyzer authority now lives under:
runtime/visual-motion-analyzer/src/self_mirror_visual_analyzer.

This module remains only to keep older import paths working during the
migration window. Do not add analyzer implementation logic here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_RUNTIME_SRC = _REPO_ROOT / "runtime" / "visual-motion-analyzer" / "src"
if str(_RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_SRC))

from self_mirror_visual_analyzer.visual_motion_analyzer import (  # noqa: F401
    analyze_config,
    analyze_frames,
    main,
    write_outputs,
)

__all__ = ["analyze_config", "analyze_frames", "main", "write_outputs"]
