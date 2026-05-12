"""Tests for the subprocess sandbox.

The sandbox itself has no heavy dependencies (only stdlib subprocess), so
these tests run unconditionally in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the package importable when running pytest from the repo root,
# without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_code.config import Config
from spark_code.sandbox.executor import (
    _classify,
    execute_code,
    execute_tests_individually,
)


@pytest.fixture
def cfg():
    return Config(exec_timeout=5, exec_memory_mb=1024)



# Stderr classification — pure logic


def test_classify_empty():
    assert _classify("") == "none"
    assert _classify("   \n\n  ") == "none"


def test_classify_syntax():
    assert _classify("SyntaxError: invalid syntax") == "syntax"
    assert _classify("IndentationError: unexpected indent") == "syntax"
    assert _classify("  File foo, line 1\n    def bad(\n         ^\nSyntaxError: x") == "syntax"


def test_classify_wrong_answer():
    assert _classify("AssertionError") == "wrong_answer"
    assert _classify("Traceback...\nAssertionError: 5 != 6") == "wrong_answer"


def test_classify_timeout():
    assert _classify("Timeout: 10s") == "timeout"


def test_classify_runtime():
    for err in [
        "ZeroDivisionError: division by zero",
        "KeyError: 'foo'",
        "TypeError: unsupported operand",
        "ValueError: bad input",
        "NameError: name 'x' is not defined",
        "IndexError: list out of range",
        "AttributeError: 'NoneType' has no attribute 'x'",
        "RecursionError: maximum recursion depth exceeded",
        "ImportError: no module named foo",
    ]:
        assert _classify(err) == "runtime", f"failed to classify: {err!r}"


def test_classify_unknown_with_content_is_runtime():
    """Anything non-empty that doesn't match known patterns falls through to runtime."""
    assert _classify("some unknown garbage") == "runtime"



# execute_code (binary)


def test_execute_code_passing(cfg):
    code = "def add(a, b):\n    return a + b\n"
    tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
    er = execute_code(code, tests, cfg)
    assert er.passed is True
    assert er.error_type == "none"
    assert er.reward == 1.0
    assert er.tests_passed == 1
    assert er.tests_total == 1


def test_execute_code_assertion_failure(cfg):
    code = "def add(a, b):\n    return a - b\n"  # wrong
    tests = "assert add(2, 3) == 5\n"
    er = execute_code(code, tests, cfg)
    assert er.passed is False
    assert er.error_type == "wrong_answer"
    assert er.reward == 0.0


def test_execute_code_syntax_error(cfg):
    code = "def broken(\n    return 1\n"
    tests = "assert broken() == 1\n"
    er = execute_code(code, tests, cfg)
    assert er.passed is False
    assert er.error_type == "syntax"


def test_execute_code_runtime_error(cfg):
    code = "def divide(a, b):\n    return a / b\n"
    tests = "assert divide(1, 0) == 0\n"
    er = execute_code(code, tests, cfg)
    assert er.passed is False
    assert er.error_type == "runtime"


def test_execute_code_timeout(cfg):
    cfg.exec_timeout = 2
    code = "def loop():\n    while True:\n        pass\n"
    tests = "loop()\n"
    er = execute_code(code, tests, cfg)
    assert er.passed is False
    assert er.error_type == "timeout"



# execute_tests_individually (partial reward)


def test_partial_all_pass(cfg):
    code = "def mul(a, b):\n    return a * b\n"
    tests = ["assert mul(2, 3) == 6", "assert mul(-1, 1) == -1", "assert mul(0, 5) == 0"]
    er = execute_tests_individually(code, tests, cfg)
    assert er.passed is True
    assert er.tests_passed == 3
    assert er.tests_total == 3
    assert er.reward == 1.0


def test_partial_some_fail(cfg):
    """Partial credit: 2/3 tests pass, error_type from first failure."""
    code = "def mul(a, b):\n    return a * b if a > 0 else 0\n"  # broken on negatives
    tests = ["assert mul(2, 3) == 6", "assert mul(-1, 1) == -1", "assert mul(0, 5) == 0"]
    er = execute_tests_individually(code, tests, cfg)
    assert er.passed is False
    assert er.tests_passed == 2
    assert er.tests_total == 3
    assert er.error_type == "wrong_answer"
    # wrong_answer floor = 0.0 by default; base reward is 2/3 which is above the floor
    assert er.reward == pytest.approx(2 / 3)


def test_partial_syntax_error_uses_penalty(cfg):
    """Syntax errors override partial credit with the configured penalty."""
    code = "def mul(a, b\n    return a * b\n"  # syntax error
    tests = ["assert mul(2, 3) == 6", "assert mul(-1, 1) == -1"]
    er = execute_tests_individually(code, tests, cfg)
    assert er.passed is False
    assert er.tests_passed == 0
    assert er.error_type == "syntax"
    assert er.reward == cfg.syntax_penalty  # -0.2 default


def test_partial_runtime_floor(cfg):
    """All tests fail with runtime errors → reward floors at runtime_penalty."""
    code = "def f(a, b):\n    raise ValueError('boom')\n"
    tests = ["assert f(1, 2) == 3", "assert f(0, 0) == 0"]
    er = execute_tests_individually(code, tests, cfg)
    assert er.passed is False
    assert er.tests_passed == 0
    assert er.error_type == "runtime"
    # 0/2 + runtime_penalty (-0.1), floored at runtime_penalty
    assert er.reward == cfg.runtime_penalty
