"""MBPP-sanitized loading and train/held-out splitting."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import concatenate_datasets, load_dataset

from spark_code.config import Config
from spark_code.data.structures import CodeProblem

_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*:", re.MULTILINE)


def extract_signature(code: str) -> Optional[str]:
    """Extract the first ``def foo(...):`` signature line from a code blob."""
    m = _DEF_RE.search(code or "")
    return m.group(0).strip() if m else None


def infer_entry_point_from_assert(assertion: str) -> Optional[str]:
    """Pull the function name out of an ``assert foo(...) == ...`` line."""
    m = re.match(r"\s*assert\s+([A-Za-z_]\w*)\s*\(", assertion or "")
    return m.group(1) if m else None


def make_mbpp_prompt(
    item: Dict[str, Any], text_field: str, entry_point: str, cfg: Config
) -> str:
    """Build the user-facing prompt for one MBPP problem.

    With ``prompt_style="complete_function"`` and a recoverable signature, the
    prompt looks like a HumanEval-style function stub. Otherwise falls back to
    a natural-language description.
    """
    task = item[text_field]
    sig = extract_signature(item.get("code", ""))
    if cfg.prompt_style == "complete_function" and sig:
        return (
            "Write the complete Python function satisfying this signature and docstring. "
            "Return only Python code, no markdown, no explanation.\n\n"
            f"{sig}\n"
            f'    """{task}"""\n'
        )
    return (
        f"Write a Python function `{entry_point}` that solves the following task.\n\n"
        f"Task: {task}\n\nReturn only Python code, no markdown, no explanation."
    )


def load_mbpp(cfg: Config, max_n: Optional[int] = None) -> List[CodeProblem]:
    """Load MBPP sanitized problems across all available splits.

    Concatenating splits is intentional — we then build a fresh deterministic
    train/heldout split in :func:`split_mbpp_train_heldout`.
    """
    raw = load_dataset("google-research-datasets/mbpp", "sanitized")
    if hasattr(raw, "keys"):
        print(f"[data] MBPP splits: {list(raw.keys())}")
        ds = concatenate_datasets([raw[k] for k in raw.keys()])
    else:
        ds = raw
    print(f"[data] MBPP fields: {list(ds.features.keys())}")
    text_field = "text" if "text" in ds.features else "prompt"

    problems: List[CodeProblem] = []
    seen_task_ids = set()
    cap = max_n
    for item in ds:
        task_id_raw = item.get("task_id")
        task_id = f"mbpp/{task_id_raw}"
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)

        tests = list(item.get("test_list") or [])
        if cfg.use_mbpp_challenge_tests and item.get("challenge_test_list"):
            tests += list(item.get("challenge_test_list") or [])
        if not tests:
            continue
        ep = infer_entry_point_from_assert(tests[0])
        if not ep:
            continue
        asserts = "\n".join(tests)
        test_code = f"{asserts}\n"
        prompt = make_mbpp_prompt(item, text_field, ep, cfg)
        problems.append(
            CodeProblem(
                task_id=task_id,
                prompt_text=prompt,
                test_code=test_code,
                entry_point=ep,
                canonical_solution=item.get("code", ""),
                source="mbpp",
                test_list=tests,
            )
        )
        if cap is not None and len(problems) >= cap:
            break
    print(f"[data] Loaded {len(problems)} valid MBPP problems")
    return problems


def split_mbpp_train_heldout(
    problems: List[CodeProblem], cfg: Config
) -> Tuple[List[CodeProblem], List[CodeProblem]]:
    """Deterministically split MBPP into a training pool and a held-out eval set.

    The held-out slice should never be used by frontier filtering, GRPO,
    auxiliary SFT, or SFT warmup. The split is reproducible from cfg.seed.
    """
    rng = random.Random(cfg.seed + 2026)
    shuffled = list(problems)
    rng.shuffle(shuffled)

    if not cfg.eval_mbpp_heldout:
        return shuffled, []

    max_possible_holdout = max(0, len(shuffled) - cfg.max_train_problems)
    holdout_n = min(cfg.mbpp_eval_size, max_possible_holdout)
    if holdout_n <= 0:
        print(
            "[data] WARNING: Not enough MBPP problems for a held-out MBPP split; "
            "disabling MBPP held-out eval."
        )
        return shuffled, []

    mbpp_heldout = shuffled[-holdout_n:]
    train_pool = shuffled[:-holdout_n]
    train_ids = {p.task_id for p in train_pool}
    held_ids = {p.task_id for p in mbpp_heldout}
    assert train_ids.isdisjoint(held_ids), "MBPP train/held-out leakage detected"
    print(
        f"[data] MBPP train pool: {len(train_pool)} | "
        f"MBPP held-out eval: {len(mbpp_heldout)}"
    )
    return train_pool, mbpp_heldout
