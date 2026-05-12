"""Reflection evaluation.

Greedy two-step protocol on HumanEval (or any held-out problem set):

1. Generate one greedy solution.
2. If it fails, give the model the failing code and the cleaned stderr trace,
   and ask for a corrected solution. Test the repair.

The fix rate is fixed / tested, where ``tested`` counts only problems that
failed the initial pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from spark_code.config import Config
from spark_code.data.structures import CodeProblem
from spark_code.model.prompts import chat_prompt, clean_stderr, extract_code
from spark_code.sandbox.executor import execute_code
from spark_code.utils.io import save_json


def eval_reflection(
    model, tok, problems: List[CodeProblem], cfg: Config, out_path: Optional[Path] = None
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Greedy reflection eval: model generates → if fails, given trace, fix → retest."""
    model.eval()
    n_eval = min(cfg.reflection_eval_problems, len(problems))
    sample = problems[:n_eval]
    tested = 0
    fixed = 0
    rows: List[Dict[str, Any]] = []
    print(f"  [refl] Greedy reflection eval on {n_eval} problems")
    for p in sample:
        prompt_text = chat_prompt(tok, p.prompt_text)
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(cfg.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        code = extract_code(tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True))
        er = execute_code(code, p.test_code, cfg)
        if er.passed:
            continue
        err = clean_stderr(er.stderr)
        fix_user = (
            "Fix this failing Python code.\n\n"
            f"Problem:\n{p.prompt_text}\n\n"
            f"Failing code:\n```python\n{code}\n```\n\n"
            f"Execution diagnostic:\n```\n{err}\n```\n\n"
            "Return only the corrected Python code."
        )
        fp = chat_prompt(tok, fix_user)
        fe = tok(fp, return_tensors="pt", add_special_tokens=False).to(cfg.device)
        with torch.no_grad():
            fo = model.generate(
                input_ids=fe.input_ids,
                attention_mask=fe.attention_mask,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        fcode = extract_code(tok.decode(fo[0][fe.input_ids.shape[1]:], skip_special_tokens=True))
        fer = execute_code(fcode, p.test_code, cfg)
        tested += 1
        if fer.passed:
            fixed += 1
        rows.append(
            {
                "task_id": p.task_id,
                "initial_error": er.error_type,
                "fixed": fer.passed,
                "fixed_error": fer.error_type,
                "initial_code_preview": code[:500],
                "fixed_code_preview": fcode[:500],
            }
        )

    metrics: Dict[str, float] = {
        "refl/fix_rate": fixed / max(tested, 1),
        "refl/tested": tested,
        "refl/fixed": fixed,
    }
    if out_path is not None:
        save_json(out_path, rows)
    print(f"  [refl] fixed={fixed}/{tested} rate={metrics['refl/fix_rate']:.4f}")
    return metrics, rows
