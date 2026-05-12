"""pass@k evaluation on code datasets (HumanEval, held-out MBPP).

A single generic evaluator drives both benchmarks; the only differences are
the metric prefix (``eval/`` vs ``eval_mbpp/``) and the dataset name in logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from spark_code.config import Config
from spark_code.data.structures import CodeProblem
from spark_code.eval.metrics import pass_at_k
from spark_code.model.prompts import chat_prompt, extract_code
from spark_code.sandbox.executor import execute_code
from spark_code.utils.io import save_json


def eval_code_dataset(
    model,
    tok,
    problems: List[CodeProblem],
    cfg: Config,
    dataset_name: str,
    metric_prefix: str,
    out_path: Optional[Path] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Generic pass@k evaluator.

    For each problem we draw ``cfg.eval_num_samples`` solutions at the eval
    sampling temperature, run each in the sandbox, and compute pass@1/5/10
    using the unbiased estimator.
    """
    model.eval()
    rows: List[Dict[str, Any]] = []
    print(
        f"  [eval] {dataset_name}: {len(problems)} problems, "
        f"{cfg.eval_num_samples} samples/problem"
    )
    for i, p in enumerate(problems):
        prompt_text = chat_prompt(tok, p.prompt_text)
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(cfg.device)
        correct = 0
        samples: List[Dict[str, Any]] = []
        for sidx in range(cfg.eval_num_samples):
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.eval_temperature,
                    do_sample=True,
                    top_p=cfg.eval_top_p,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            gen_ids = out[0][enc.input_ids.shape[1]:]
            completion = tok.decode(gen_ids, skip_special_tokens=True)
            code = extract_code(completion)
            er = execute_code(code, p.test_code, cfg)
            if er.passed:
                correct += 1
            samples.append(
                {
                    "sample": sidx,
                    "passed": er.passed,
                    "error_type": er.error_type,
                    "runtime_ms": er.runtime_ms,
                    "code_preview": code[:500],
                }
            )
        n = cfg.eval_num_samples
        row: Dict[str, Any] = {
            "task_id": p.task_id,
            "n": n,
            "c": correct,
            "pass@1": pass_at_k(n, correct, 1),
        }
        if n >= 5:
            row["pass@5"] = pass_at_k(n, correct, 5)
        if n >= 10:
            row["pass@10"] = pass_at_k(n, correct, 10)
        row["samples"] = samples
        rows.append(row)
        if (i + 1) % 30 == 0 or i == len(problems) - 1:
            p1 = float(np.mean([r["pass@1"] for r in rows]))
            msg = f"    [{i+1}/{len(problems)}] {metric_prefix}/pass@1={p1:.4f}"
            if n >= 5:
                msg += f" {metric_prefix}/pass@5={np.mean([r['pass@5'] for r in rows]):.4f}"
            if n >= 10:
                msg += f" {metric_prefix}/pass@10={np.mean([r['pass@10'] for r in rows]):.4f}"
            print(msg)

    metrics: Dict[str, float] = {
        f"{metric_prefix}/pass@1": float(np.mean([r["pass@1"] for r in rows])),
    }
    if cfg.eval_num_samples >= 5:
        metrics[f"{metric_prefix}/pass@5"] = float(np.mean([r["pass@5"] for r in rows]))
    if cfg.eval_num_samples >= 10:
        metrics[f"{metric_prefix}/pass@10"] = float(np.mean([r["pass@10"] for r in rows]))
    if out_path is not None:
        save_json(out_path, rows)
    print("  [eval] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics, rows


def eval_humaneval(
    model, tok, problems: List[CodeProblem], cfg: Config, out_path: Optional[Path] = None
):
    return eval_code_dataset(
        model,
        tok,
        problems,
        cfg,
        dataset_name="HumanEval",
        metric_prefix="eval",
        out_path=out_path,
    )


def eval_mbpp_heldout(
    model, tok, problems: List[CodeProblem], cfg: Config, out_path: Optional[Path] = None
):
    return eval_code_dataset(
        model,
        tok,
        problems,
        cfg,
        dataset_name="MBPP held-out",
        metric_prefix="eval_mbpp",
        out_path=out_path,
    )
