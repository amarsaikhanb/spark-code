"""Logging: tee stdout/stderr to a file, plus wandb init and metric pushes.

The :class:`Tee` writer is installed by ``scripts/run_experiment.py`` BEFORE
torch is imported, so import-time messages are captured too. We expose Tee
here so other entrypoints can install it the same way if they want.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

from spark_code.config import Config
from spark_code.utils.io import json_safe

try:
    import wandb

    WANDB_OK = True
except Exception:
    wandb = None
    WANDB_OK = False


class Tee:
    """File-like wrapper that mirrors writes to a stream and a file object."""

    def __init__(self, stream, file_obj):
        self.stream = stream
        self.file_obj = file_obj

    def write(self, data):
        self.stream.write(data)
        self.file_obj.write(data)
        self.flush()

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.file_obj.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def log_metrics(metrics: Dict[str, Any], step: int) -> None:
    """Print a one-line summary of metrics and forward to wandb if enabled."""
    parts = [
        f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
    ]
    print(f"  [step {step}] " + " | ".join(parts))
    if WANDB_OK and wandb.run is not None:
        wandb.log(metrics, step=step)


def init_wandb(cfg: Config, condition: str) -> Optional[Any]:
    """Initialize a wandb run for a given condition. Returns None if disabled."""
    if not cfg.wandb_enabled or not WANDB_OK:
        return None
    return wandb.init(
        project=cfg.wandb_project,
        name=f"cond-{condition}_s{cfg.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=json_safe(asdict(cfg)),
        dir=cfg.output_dir,
        reinit=True,
    )
