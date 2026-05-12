"""Per-condition orchestration loop.

This module owns the outer training loop. It is intentionally not split
further: orchestration code reads better as one linear sequence than as a web
of helpers.

Flow:

1. Initialize model, tokenizer, optimizer, wandb.
2. Optional SFT warmup on canonical MBPP.
3. Frontier filter (or load saved task IDs).
4. Build deterministic per-iteration training batches.
5. For each iteration:
   a. Generate K rollouts per problem.
   b. GRPO update.
   c. Condition C only: build aux data, run aux SFT step.
   d. Evaluate HumanEval, held-out MBPP, and (Condition C) reflection.
   e. Save checkpoint and metrics.
6. Save final model and metrics; optional HF Hub push.
"""

from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import AdamW

from spark_code.config import Config
from spark_code.data.structures import CodeProblem
from spark_code.eval.code import eval_humaneval, eval_mbpp_heldout
from spark_code.eval.reflection import eval_reflection
from spark_code.model.loading import load_model
from spark_code.training.auxiliary import build_aux_data, build_sft_warmup_data, sft_step
from spark_code.training.frontier import frontier_filter, load_frontier_from_file
from spark_code.training.grpo import compute_advantages, grpo_step
from spark_code.training.rollouts import generate_rollouts
from spark_code.utils.io import json_safe, mkdir, save_json
from spark_code.utils.logging import init_wandb, log_metrics
from spark_code.utils.reporting import save_rollout_summary, summarize_rollouts
from spark_code.utils.seeds import set_eval_seed

try:
    import wandb

    WANDB_OK = True
except Exception:
    wandb = None
    WANDB_OK = False


def run_condition(
    condition: str,
    cfg: Config,
    train_pool: List[CodeProblem],
    eval_problems: List[CodeProblem],
    mbpp_eval_problems: List[CodeProblem],
    output_root: Path,
) -> Dict[str, Any]:
    """Run one full A or C condition end-to-end. Returns metrics history."""
    cond_dir = mkdir(output_root / f"condition_{condition}")
    ckpt_dir = mkdir(cond_dir / "checkpoints")
    eval_dir = mkdir(cond_dir / "eval")
    rollout_dir = mkdir(cond_dir / "rollouts")

    print("\n" + "=" * 70)
    label = "Exec-only GRPO" if condition == "A" else "GRPO + SPARK auxiliary recycling"
    print(f"CONDITION {condition}: {label}")
    print(f"Output: {cond_dir}")
    print(
        f"Train pool candidates: {len(train_pool)} | "
        f"HumanEval eval: {len(eval_problems)} | "
        f"MBPP held-out eval: {len(mbpp_eval_problems)}"
    )
    print("=" * 70)

    run = init_wandb(cfg, condition)
    model, tok = load_model(cfg)

    # SINGLE optimizer for both GRPO and aux phases
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=cfg.grpo_lr, weight_decay=0.01)

    metrics_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Optional SFT warmup
    # ------------------------------------------------------------------
    if cfg.sft_warmup:
        print("[warmup] SFT warmup on canonical MBPP solutions")
        warm = build_sft_warmup_data(train_pool, cfg)
        warm_metrics = sft_step(
            model,
            tok,
            warm,
            opt,
            cfg,
            cfg.sft_epochs,
            cfg.aux_max_len,
            "warmup",
            loss_scale=cfg.sft_loss_scale,
        )
        log_metrics({"iteration": -1, "condition": condition, **warm_metrics}, -1)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Frontier filtering (or load from file)
    # ------------------------------------------------------------------
    if cfg.load_frontier_path:
        train_problems = load_frontier_from_file(cfg.load_frontier_path, train_pool)
        if len(train_problems) < 20:
            print(
                f"[frontier] Loaded too few problems; falling back to first "
                f"{cfg.max_train_problems} of pool."
            )
            train_problems = train_pool[: cfg.max_train_problems]
    elif cfg.use_frontier_filter:
        scan_pool = train_pool[: cfg.frontier_pool_size]
        filtered = frontier_filter(model, tok, scan_pool, cfg, target_n=cfg.max_train_problems)
        if len(filtered) < 20:
            print(
                f"[frontier] Too few frontier problems ({len(filtered)}); "
                f"falling back to first {cfg.max_train_problems} of pool."
            )
            train_problems = train_pool[: cfg.max_train_problems]
        else:
            train_problems = filtered
    else:
        train_problems = train_pool[: cfg.max_train_problems]

    # Always save the chosen task IDs so the OTHER condition can re-use them
    save_json(cond_dir / "frontier_task_ids.json", [p.task_id for p in train_problems])
    print(f"[train] Using {len(train_problems)} training problems")

    # Build deterministic per-iter batches
    rng = random.Random(cfg.seed + 999)
    train_batches = [
        rng.sample(train_problems, min(cfg.max_train_problems, len(train_problems)))
        for _ in range(cfg.num_iterations)
    ]
    save_json(
        cond_dir / "train_batches_task_ids.json",
        [[p.task_id for p in b] for b in train_batches],
    )

    # ------------------------------------------------------------------
    # Iteration 0 baseline
    # ------------------------------------------------------------------
    print("[iter 0] Baseline evaluation")
    set_eval_seed(cfg, 0)
    eval_metrics, _ = eval_humaneval(
        model, tok, eval_problems, cfg, eval_dir / "iter0_humaneval.json"
    )
    mbpp_eval_metrics: Dict[str, Any] = {}
    if mbpp_eval_problems:
        mbpp_eval_metrics, _ = eval_mbpp_heldout(
            model, tok, mbpp_eval_problems, cfg, eval_dir / "iter0_mbpp_heldout.json"
        )
    base = {"iteration": 0, "condition": condition, **eval_metrics, **mbpp_eval_metrics}
    if condition == "C":
        rm0, _ = eval_reflection(
            model, tok, eval_problems, cfg, eval_dir / "iter0_reflection.json"
        )
        base.update(rm0)
    metrics_history.append(base)
    log_metrics(base, 0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for it in range(1, cfg.num_iterations + 1):
        t0 = time.time()
        print("\n" + "─" * 70)
        print(f"[iter {it}/{cfg.num_iterations}] Condition {condition}")
        batch = train_batches[it - 1]

        print(f"[iter {it}] Generating rollouts (K={cfg.num_rollouts})")
        groups = generate_rollouts(model, tok, batch, cfg)
        roll_metrics = summarize_rollouts(groups)
        save_rollout_summary(rollout_dir / f"iter{it}_summary.json", groups)
        print(
            f"[iter {it}] Rollout summary: "
            f"{json.dumps(json_safe(roll_metrics), indent=2)}"
        )

        print(f"[iter {it}] GRPO step")
        advs = compute_advantages(groups)
        grpo_metrics = grpo_step(model, groups, advs, opt, cfg)
        print(f"[iter {it}] GRPO: {json.dumps(json_safe(grpo_metrics), indent=2)}")

        aux_metrics: Dict[str, Any] = {}
        sf_meta: Dict[str, Any] = {}
        if condition == "C":
            print(f"[iter {it}] Building auxiliary data with self-fix reflection")
            aux_data, sf_meta = build_aux_data(groups, batch, model, tok, cfg)
            save_json(
                cond_dir / f"aux_counts_iter{it}.json",
                {
                    "n": len(aux_data),
                    "counts": {
                        t: sum(1 for x in aux_data if x.type == t)
                        for t in ["pointwise", "pairwise", "reflection"]
                    },
                    **sf_meta,
                },
            )
            if aux_data:
                print(
                    f"[iter {it}] Auxiliary SFT (n={len(aux_data)}, "
                    f"loss_scale={cfg.aux_loss_scale})"
                )
                aux_metrics = sft_step(
                    model,
                    tok,
                    aux_data,
                    opt,
                    cfg,
                    cfg.aux_epochs,
                    cfg.aux_max_len,
                    "aux",
                    loss_scale=cfg.aux_loss_scale,
                )
                print(f"[iter {it}] Aux: {json.dumps(json_safe(aux_metrics), indent=2)}")

        print(f"[iter {it}] HumanEval evaluation")
        set_eval_seed(cfg, it)
        eval_metrics, _ = eval_humaneval(
            model, tok, eval_problems, cfg, eval_dir / f"iter{it}_humaneval.json"
        )
        mbpp_eval_metrics = {}
        if mbpp_eval_problems:
            print(f"[iter {it}] MBPP held-out evaluation")
            mbpp_eval_metrics, _ = eval_mbpp_heldout(
                model, tok, mbpp_eval_problems, cfg, eval_dir / f"iter{it}_mbpp_heldout.json"
            )

        refl_metrics: Dict[str, Any] = {}
        if condition == "C":
            print(f"[iter {it}] Reflection evaluation (greedy)")
            refl_metrics, _ = eval_reflection(
                model, tok, eval_problems, cfg, eval_dir / f"iter{it}_reflection.json"
            )

        elapsed = (time.time() - t0) / 60.0
        combined = {
            "iteration": it,
            "condition": condition,
            "time_min": elapsed,
            **roll_metrics,
            **grpo_metrics,
            **aux_metrics,
            **sf_meta,
            **eval_metrics,
            **mbpp_eval_metrics,
            **refl_metrics,
        }
        metrics_history.append(combined)
        save_json(cond_dir / "metrics_so_far.json", metrics_history)
        log_metrics(combined, it)

        ckpt = ckpt_dir / f"iter{it}"
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        print(f"[iter {it}] Saved checkpoint: {ckpt}")
        mbpp_msg = ""
        if mbpp_eval_metrics:
            mbpp_msg = (
                f" | mbpp_pass@1={mbpp_eval_metrics.get('eval_mbpp/pass@1', float('nan')):.4f}"
            )
        print(
            f"[iter {it}] Done in {elapsed:.2f}m  "
            f"humaneval_pass@1={eval_metrics.get('eval/pass@1', float('nan')):.4f}"
            f"{mbpp_msg}"
        )

        del groups, advs
        if condition == "C":
            try:
                del aux_data
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_dir = mkdir(cond_dir / "final")
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    save_json(final_dir / "metrics.json", metrics_history)
    print(f"[final] Saved final model: {final_dir}")

    if cfg.hf_repo_id and cfg.hf_token:
        try:
            model.push_to_hub(cfg.hf_repo_id + f"-{condition}", token=cfg.hf_token, private=True)
        except Exception as e:
            print(f"[hf] Push failed: {e}")

    if run is not None and WANDB_OK:
        wandb.finish()
    del model, tok, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"condition": condition, "metrics": metrics_history}
