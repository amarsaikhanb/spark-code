"""Cross-cutting utilities: seeding, IO, logging, reporting.

The torch-free helpers (IO, reporting, Tee logging) are re-exported here.
``set_all_seeds`` and ``set_eval_seed`` live in ``spark_code.utils.seeds``
and depend on torch — import them directly from that submodule.
"""

from spark_code.utils.io import json_safe, mkdir, save_json
from spark_code.utils.logging import Tee, init_wandb, log_metrics
from spark_code.utils.reporting import (
    export_csv,
    print_comparison,
    save_rollout_summary,
    summarize_rollouts,
)

__all__ = [
    "json_safe",
    "save_json",
    "mkdir",
    "Tee",
    "log_metrics",
    "init_wandb",
    "summarize_rollouts",
    "save_rollout_summary",
    "export_csv",
    "print_comparison",
]
