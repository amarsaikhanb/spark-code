"""Tests for code extraction and stderr cleaning."""

from __future__ import annotations

import re


# Inline copies of the public functions from spark_code.model.prompts so tests
# can run with no torch dependency installed.

def extract_code(response: str) -> str:
    s = (response or "").strip()
    for pat in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        matches = re.findall(pat, s, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
    s = re.sub(r"^Here(?:'s| is).*?:\s*", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    candidates = [
        i for i in [s.find("def "), s.find("from "), s.find("import "), s.find("class ")] if i >= 0
    ]
    return s[min(candidates):].strip() if candidates else s


def clean_stderr(stderr: str, max_lines: int = 6) -> str:
    lines = [l for l in (stderr or "").strip().split("\n") if l.strip()]
    return "\n".join(lines[-max_lines:]) if lines else "No error message"


# --- extract_code -----------------------------------------------------------

def test_extract_python_fence():
    s = "Here's the answer:\n```python\ndef foo():\n    return 1\n```\nThat's it."
    assert extract_code(s) == "def foo():\n    return 1"


def test_extract_generic_fence():
    s = "```\ndef foo():\n    return 1\n```"
    assert extract_code(s) == "def foo():\n    return 1"


def test_extract_fence_case_insensitive():
    s = "```PYTHON\ndef foo():\n    return 1\n```"
    assert extract_code(s) == "def foo():\n    return 1"


def test_extract_no_fence_with_preamble():
    s = "Here is the function:\n\ndef foo():\n    return 1"
    assert extract_code(s) == "def foo():\n    return 1"


def test_extract_no_fence_with_imports():
    s = "Sure!\nimport math\n\ndef foo():\n    return math.pi"
    out = extract_code(s)
    assert out.startswith("import math")
    assert "def foo" in out


def test_extract_class_definition():
    s = "class Foo:\n    def bar(self):\n        return 1"
    assert extract_code(s) == s


def test_extract_picks_earliest_anchor():
    """Should pick the earliest of def / from / import / class as the start."""
    s = "import os\ndef foo():\n    return 1\nfrom typing import List"
    out = extract_code(s)
    assert out.startswith("import os")


def test_extract_empty():
    assert extract_code("") == ""


def test_extract_only_prose():
    s = "I cannot solve this problem."
    # No def/import/class anchor → returns stripped prose unchanged
    assert extract_code(s) == "I cannot solve this problem."


def test_extract_first_fence_wins():
    s = "```python\ndef a():\n    return 1\n```\n\nor:\n\n```python\ndef b():\n    return 2\n```"
    out = extract_code(s)
    assert "def a" in out
    assert "def b" not in out


# --- clean_stderr -----------------------------------------------------------

def test_clean_stderr_keeps_last_lines():
    err = "\n".join(f"line {i}" for i in range(10))
    out = clean_stderr(err, max_lines=3)
    assert out == "line 7\nline 8\nline 9"


def test_clean_stderr_drops_blank_lines():
    err = "line a\n\n\nline b\n\nline c"
    out = clean_stderr(err)
    assert out == "line a\nline b\nline c"


def test_clean_stderr_empty_returns_placeholder():
    assert clean_stderr("") == "No error message"
    assert clean_stderr("   \n\n   ") == "No error message"


def test_clean_stderr_short_unchanged():
    err = "Traceback (most recent call last):\n  File foo.py\nValueError: bad"
    out = clean_stderr(err, max_lines=6)
    assert out == err


def test_clean_stderr_default_six_lines():
    err = "\n".join(f"line {i}" for i in range(20))
    out = clean_stderr(err)
    assert out.count("\n") == 5  # 6 lines → 5 newlines
    assert out.endswith("line 19")
