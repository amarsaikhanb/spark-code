"""Reproducibility-oriented seeding.

We seed Python random, NumPy, PyTorch (CPU + all CUDA devices), HuggingFace
``set_seed``, and PYTHONHASHSEED. The eval-time seed is offset deterministically
by iteration so different iterations get different sample noise, but every run
of iteration N uses the exact same seed.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from transformers import set_seed


def set_all_seeds(seed: int) -> None:
    """Seed everything reachable. Call at startup and at the start of each condition."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_eval_seed(cfg, iteration: int) -> None:
    """Deterministic per-iteration eval seed (offset by 10000 + 100*iter)."""
    set_all_seeds(cfg.seed + 10000 + iteration * 100)
