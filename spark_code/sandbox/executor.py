"""Subprocess sandbox for executing generated code.

Each candidate solution runs in a fresh subprocess via ``sys.executable``,
with a wall-clock timeout, a CPU-time rlimit, and an address-space rlimit on
POSIX systems. We classify stderr into one of {syntax, runtime, wrong_answer,
timeout, none} so the partial-reward path can apply error-type-specific
penalties.

Two execution modes are supported:

* :func:`execute_code` runs all assertions in a single subprocess and returns
  binary pass/fail. Used for evaluation and the ``binary_rewards`` training
  path.
* :func:`execute_tests_individually` runs each assertion in its own subprocess
  and returns a partial reward proportional to the number of tests passed.
  Used for the default training path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

from spark_code.config import Config
from spark_code.data.structures import CodeProblem, ExecResult

try:
    import resource

    RESOURCE_OK = True
except Exception:
    resource = None
    RESOURCE_OK = False


def _classify(stderr: str) -> str:
    """Classify a stderr string into one of our canonical error types."""
    s = (stderr or "").lower()
    if "syntaxerror" in s or "indentationerror" in s:
        return "syntax"
    if "assertionerror" in s:
        return "wrong_answer"
    if "timeout" in s:
        return "timeout"
    for err in [
        "nameerror",
        "typeerror",
        "indexerror",
        "keyerror",
        "valueerror",
        "attributeerror",
        "zerodivisionerror",
        "recursionerror",
        "importerror",
        "modulenotfounderror",
    ]:
        if err in s:
            return "runtime"
    return "runtime" if s.strip() else "none"


def _safe_preexec(memory_mb: int):
    """Build a POSIX ``preexec_fn`` that imposes RLIMIT_AS and RLIMIT_CPU.

    Returns None on non-POSIX systems or if resource is unavailable.
    """
    if not RESOURCE_OK or memory_mb <= 0:
        return None

    def _limit():
        try:
            mem = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        except Exception:
            pass

    return _limit


def run_python_program(program: str, timeout: int, memory_mb: int) -> ExecResult:
    """Execute one self-contained Python program in a sandboxed subprocess.

    Returns an ExecResult with ``passed=True`` only if the subprocess exits 0.
    The default ``reward`` is binary pass/fail (1.0 / 0.0); callers wanting
    partial credit should use :func:`execute_tests_individually` instead.
    """
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PATH": os.environ.get("PATH", ""),
    }
    with tempfile.TemporaryDirectory(prefix="spark_exec_") as tmpdir:
        path = Path(tmpdir) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        t0 = time.time()
        try:
            r = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                preexec_fn=_safe_preexec(memory_mb) if os.name == "posix" else None,
            )
            ms = (time.time() - t0) * 1000
            if r.returncode == 0:
                return ExecResult(True, r.stdout[:1000], "", "none", ms, reward=1.0)
            stderr = (r.stderr or "")[:2000]
            return ExecResult(
                False,
                (r.stdout or "")[:1000],
                stderr,
                _classify(stderr),
                ms,
                reward=0.0,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                False, "", f"Timeout: {timeout}s", "timeout", timeout * 1000, reward=0.0
            )
        except Exception as e:
            stderr = f"{type(e).__name__}: {e}"
            return ExecResult(
                False, "", stderr[:1000], "runtime", (time.time() - t0) * 1000, reward=0.0
            )


def execute_code(code: str, test_code: str, cfg: Config) -> ExecResult:
    """Run ``code`` followed by ``test_code`` in one subprocess. Binary reward."""
    program = code.strip() + "\n\n" + test_code.strip() + "\n"
    er = run_python_program(program, cfg.exec_timeout, cfg.exec_memory_mb)
    er.tests_total = 1
    er.tests_passed = 1 if er.passed else 0
    er.reward = 1.0 if er.passed else 0.0
    return er


def execute_tests_individually(code: str, tests: List[str], cfg: Config) -> ExecResult:
    """Run each assertion in its own subprocess. Returns a partial reward.

    The partial reward is the fraction of tests passed, modified by error-type
    penalties (syntax / runtime / timeout / wrong-answer floor) from cfg.
    """
    passed = 0
    total = len(tests)
    first_stdout = first_stderr = ""
    first_error = "none"
    total_runtime = 0.0
    for test in tests:
        program = code.strip() + "\n\n" + test.strip() + "\n"
        er = run_python_program(program, cfg.exec_timeout, cfg.exec_memory_mb)
        total_runtime += er.runtime_ms
        if er.passed:
            passed += 1
        elif not first_stderr:
            first_stdout, first_stderr, first_error = er.stdout, er.stderr, er.error_type

    base = passed / max(total, 1)
    error_type = "none" if passed == total else first_error
    reward = base
    if error_type == "syntax":
        reward = cfg.syntax_penalty
    elif error_type == "timeout":
        reward = max(base + cfg.timeout_penalty, cfg.timeout_penalty)
    elif error_type == "runtime":
        reward = max(base + cfg.runtime_penalty, cfg.runtime_penalty)
    elif error_type == "wrong_answer":
        reward = max(base, cfg.wrong_answer_floor)

    return ExecResult(
        passed=(passed == total),
        stdout=first_stdout[:1000],
        stderr=first_stderr[:2000],
        error_type=error_type,
        runtime_ms=total_runtime,
        tests_passed=passed,
        tests_total=total,
        reward=float(reward),
    )


def evaluate_generated_code(
    code: str, prob: CodeProblem, cfg: Config, training: bool
) -> ExecResult:
    """Top-level dispatcher.

    During training with ``partial_rewards=True``, MBPP problems use per-test
    partial credit. Everything else uses binary pass/fail.
    """
    if training and cfg.partial_rewards and prob.test_list:
        return execute_tests_individually(code, prob.test_list, cfg)
    return execute_code(code, prob.test_code, cfg)
