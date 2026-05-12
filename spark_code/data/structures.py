"""Plain data structures shared across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CodeProblem:
    """A single code generation problem (MBPP or HumanEval)."""

    task_id: str
    prompt_text: str
    test_code: str
    entry_point: str
    canonical_solution: str
    source: str
    test_list: List[str] = field(default_factory=list)


@dataclass
class ExecResult:
    """Result of executing generated code against a test suite."""

    passed: bool
    stdout: str
    stderr: str
    error_type: str
    runtime_ms: float
    tests_passed: int = 0
    tests_total: int = 0
    reward: float = 0.0


@dataclass
class Rollout:
    """One generation rollout, with everything needed for GRPO + KL.

    ``ref_logprobs`` is computed once at rollout time (in eval mode) using the
    PEFT ``disable_adapter`` context, so the GRPO step does not need to flip
    adapters in and out during training.
    """

    task_id: str
    prompt: str
    completion: str
    extracted_code: str
    exec_result: ExecResult
    reward: float
    old_logprobs: List[float]
    ref_logprobs: List[float]
    prompt_len: int
    full_ids: List[int]


@dataclass
class AuxExample:
    """A single SPARK-style auxiliary training example.

    ``type`` is one of {"pointwise", "pairwise", "reflection", "sft"}.
    """

    type: str
    user_msg: str
    asst_msg: str
