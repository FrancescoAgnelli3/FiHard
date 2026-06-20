"""Runtime compatibility helpers for newer PyTorch versions."""

from __future__ import annotations

import collections.abc as container_abcs
import math
import sys
import types


def ensure_torch_six_compat() -> None:
    """Provide a minimal torch._six shim for legacy deps (e.g., old timm)."""
    if "torch._six" in sys.modules:
        return

    shim = types.ModuleType("torch._six")
    shim.container_abcs = container_abcs
    shim.string_classes = (str, bytes)
    shim.int_classes = (int,)
    shim.inf = math.inf
    sys.modules["torch._six"] = shim

