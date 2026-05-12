"""Chat templating, system prompts, and post-generation text cleaning.

The two system prompts split the model's two roles: SYS_CODER for generation
(where it should emit only Python code) and SYS_REVIEWER for the auxiliary
judgment / reflection objectives.
"""

from __future__ import annotations

import re

SYS_CODER = "You are an expert Python programmer. Return only correct Python code."
SYS_REVIEWER = "You are an expert Python programmer and code reviewer."


def extract_code(response: str) -> str:
    """Strip markdown, prose preambles, and noise from a model response.

    Tries fenced ```python``` blocks first, then any fenced block, then falls
    back to the first def/import/class line.
    """
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
    """Trim stderr to the last few non-empty lines (the actual error trace)."""
    lines = [l for l in (stderr or "").strip().split("\n") if l.strip()]
    return "\n".join(lines[-max_lines:]) if lines else "No error message"


def chat_prompt(tok, user_msg: str) -> str:
    """Build a generation-ready coder prompt with the SYS_CODER system role."""
    return tok.apply_chat_template(
        [
            {"role": "system", "content": SYS_CODER},
            {"role": "user", "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def aux_prefix(tok, user_msg: str) -> str:
    """Reviewer-mode prompt prefix (system + user only). Used for SFT training."""
    return tok.apply_chat_template(
        [
            {"role": "system", "content": SYS_REVIEWER},
            {"role": "user", "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def aux_full(tok, user_msg: str, asst_msg: str) -> str:
    """Reviewer-mode full sequence (prefix + assistant response + EOS).

    The label-mask in :func:`spark_code.training.auxiliary.sft_step` zeros out
    the prefix tokens so loss is only computed on the assistant span.
    """
    eos = tok.eos_token or ""
    return aux_prefix(tok, user_msg) + asst_msg + eos
