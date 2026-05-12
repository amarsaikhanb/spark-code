"""IO helpers: JSON-safe serialization for our typical artifact mix.

Handles dataclasses, numpy scalars/arrays, Path, and torch dtypes — everything
we might want to dump in a metrics or rollout file.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch

    _TORCH_DTYPE = torch.dtype
except Exception:
    torch = None  # type: ignore[assignment]
    _TORCH_DTYPE = None


def json_safe(obj: Any) -> Any:
    """Recursively convert an object into JSON-serializable Python primitives."""
    if dataclasses.is_dataclass(obj):
        return json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if _TORCH_DTYPE is not None and isinstance(obj, _TORCH_DTYPE):
        return str(obj)
    return obj


def save_json(path: Path, obj: Any) -> None:
    """Atomic-ish save: ensure parent dir, then write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


def mkdir(path: Path) -> Path:
    """``mkdir -p`` and return the path for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path
