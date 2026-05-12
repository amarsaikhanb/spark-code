"""SPARK-Code: Co-Evolving Policy & Reward for Code Generation.

CSC 675 Capstone — Amarsaikhan Batjargal, DePaul University, Spring 2026.

This package adapts the SPARK framework (Liu et al., 2025) from mathematical
reasoning to code generation, where execution provides deterministic
verification AND rich diagnostic feedback (stderr, error type, failing tests).

Public API surface (most users only need these):

    from spark_code import Config, CodeProblem, Rollout, ExecResult
    from spark_code.data import load_mbpp, load_humaneval, split_mbpp_train_heldout
    from spark_code.sandbox import execute_code, execute_tests_individually
    from spark_code.training import run_condition

For end-to-end runs use the CLI: ``python scripts/run_experiment.py``.
"""

from spark_code.config import Config
from spark_code.data.structures import AuxExample, CodeProblem, ExecResult, Rollout

__version__ = "1.0.0"

__all__ = [
    "Config",
    "CodeProblem",
    "ExecResult",
    "Rollout",
    "AuxExample",
    "__version__",
]
