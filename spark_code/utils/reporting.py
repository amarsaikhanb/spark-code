"""Result aggregation: rollout summaries, CSV export, comparison tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from spark_code.data.structures import Rollout
from spark_code.utils.io import json_safe, save_json


def summarize_rollouts(groups: List[List[Rollout]]) -> Dict[str, Any]:
    """Compute aggregate training-time rollout statistics for one iteration."""
    rewards = [r.reward for g in groups for r in g]
    pass_flags = [r.exec_result.passed for g in groups for r in g]
    group_stds = [float(np.std([r.reward for r in g])) for g in groups]
    sizes = [len(g) for g in groups]
    err_counts: Dict[str, int] = {}
    test_fracs: List[float] = []
    for g in groups:
        for r in g:
            err_counts[r.exec_result.error_type] = err_counts.get(r.exec_result.error_type, 0) + 1
            if r.exec_result.tests_total:
                test_fracs.append(r.exec_result.tests_passed / r.exec_result.tests_total)
    return {
        "train/pass_rate": float(np.mean(pass_flags)) if pass_flags else 0.0,
        "train/mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "train/reward_std": float(np.std(rewards)) if rewards else 0.0,
        "train/informative_groups": int(sum(1 for s in group_stds if s > 1e-8)),
        "train/num_groups": len(groups),
        "train/num_rollouts": int(sum(sizes)),
        "train/mean_group_size": float(np.mean(sizes)) if sizes else 0.0,
        "train/error_counts": err_counts,
        "train/mean_test_pass_frac": float(np.mean(test_fracs)) if test_fracs else 0.0,
    }


def save_rollout_summary(path: Path, groups: List[List[Rollout]]) -> None:
    """Persist a per-rollout summary table for offline inspection / debugging."""
    rows = []
    for gi, group in enumerate(groups):
        for ri, r in enumerate(group):
            rows.append(
                {
                    "group_index": gi,
                    "rollout_index": ri,
                    "task_id": r.task_id,
                    "reward": r.reward,
                    "passed": r.exec_result.passed,
                    "tests_passed": r.exec_result.tests_passed,
                    "tests_total": r.exec_result.tests_total,
                    "error_type": r.exec_result.error_type,
                    "runtime_ms": r.exec_result.runtime_ms,
                    "stderr_preview": r.exec_result.stderr[:300],
                    "code_preview": r.extracted_code[:500],
                }
            )
    save_json(path, rows)


def export_csv(all_results: List[Dict[str, Any]], path: Path) -> None:
    """Flatten cross-condition metrics history into a single wide CSV."""
    keys = set()
    flat_rows = []
    for res in all_results:
        for m in res["metrics"]:
            flat = {}
            for k, v in m.items():
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(json_safe(v), ensure_ascii=False)
                else:
                    flat[k] = v
                keys.add(k)
            flat_rows.append(flat)
    keys = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in flat_rows:
            w.writerow({k: row.get(k, "") for k in keys})
    print(f"[export] CSV: {path}")


def print_comparison(all_results: List[Dict[str, Any]]) -> None:
    """Pretty-print the headline cross-condition comparison table."""
    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    print(
        f"{'Cond':<6} {'Iter':<5} {'HE@1':<10} {'HE@5':<10} "
        f"{'MBPP@1':<10} {'MBPP@5':<10} {'train_pass':<12} {'KL':<10} {'fix_rate':<10}"
    )
    print("-" * 100)
    for res in all_results:
        cond = res["condition"]
        for m in res["metrics"]:

            def fmt(x):
                return f"{x:.4f}" if isinstance(x, float) else str(x)

            print(
                f"{cond:<6} {m.get('iteration', ''):<5} "
                f"{fmt(m.get('eval/pass@1', '')):<10} "
                f"{fmt(m.get('eval/pass@5', '')):<10} "
                f"{fmt(m.get('eval_mbpp/pass@1', '')):<10} "
                f"{fmt(m.get('eval_mbpp/pass@5', '')):<10} "
                f"{fmt(m.get('train/pass_rate', '')):<12} "
                f"{fmt(m.get('grpo/kl', '')):<10} "
                f"{fmt(m.get('refl/fix_rate', '')):<10}"
            )
    print("=" * 100 + "\n")
