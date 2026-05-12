#!/usr/bin/env python3
"""SPARK-Code experiment runner.

This is the main entrypoint. It does GPU-visibility selection and Tee logging
BEFORE importing torch, then dispatches to per-condition orchestration in
:mod:`spark_code.training.runner`.

Example launches (remote server, two GPUs in parallel):

    nohup python scripts/run_experiment.py --gpu 2 --condition A \\
        --output ./output_A_3B \\
        --base-model Qwen/Qwen2.5-Coder-3B-Instruct &

    nohup python scripts/run_experiment.py --gpu 3 --condition C \\
        --output ./output_C_3B \\
        --base-model Qwen/Qwen2.5-Coder-3B-Instruct \\
        --load-frontier ./output_A_3B/condition_A/frontier_task_ids.json &

Smoke test:

    python scripts/run_experiment.py --gpu 0 --condition C --output ./smoke \\
        --iterations 1 --rollouts 4 --train-problems 20 --eval-samples 2 \\
        --skip-frontier --no-wandb --reflection-eval-problems 10
"""


# 0. Early CLI parsing, GPU selection, tee logging — BEFORE torch import


from __future__ import annotations

import argparse
import atexit
import os
import sys
from datetime import datetime
from pathlib import Path

_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--gpu", type=str, default=None)
_PRE.add_argument("--condition", type=str, default="both")
_PRE.add_argument("--output", type=str, default="./spark_code_output")
_PRE.add_argument("--config", type=str, default=None)
_PRE_ARGS, _ = _PRE.parse_known_args()

if _PRE_ARGS.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.gpu

_OUTPUT_ROOT_EARLY = Path(_PRE_ARGS.output).expanduser().resolve()
_OUTPUT_ROOT_EARLY.mkdir(parents=True, exist_ok=True)
_TS_EARLY = datetime.now().strftime("%Y%m%d_%H%M%S")
_GPU_TAG = f"gpu{_PRE_ARGS.gpu}" if _PRE_ARGS.gpu is not None else "gpu_auto"
_LOG_PATH_EARLY = (
    _OUTPUT_ROOT_EARLY / f"run_{_PRE_ARGS.condition}_{_GPU_TAG}_{_TS_EARLY}.log"
)


class _EarlyTee:
    """Local copy so we can install before any spark_code imports happen."""

    def __init__(self, stream, file_obj):
        self.stream = stream
        self.file_obj = file_obj

    def write(self, data):
        self.stream.write(data)
        self.file_obj.write(data)
        self.flush()

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.file_obj.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


_LOG_FILE_EARLY = open(_LOG_PATH_EARLY, "a", buffering=1, encoding="utf-8")
sys.stdout = _EarlyTee(sys.stdout, _LOG_FILE_EARLY)
sys.stderr = _EarlyTee(sys.stderr, _LOG_FILE_EARLY)
atexit.register(lambda: _LOG_FILE_EARLY.close())

print(f"[startup] Output root: {_OUTPUT_ROOT_EARLY}")
print(f"[startup] Mirroring stdout/stderr to: {_LOG_PATH_EARLY}")
if _PRE_ARGS.gpu is not None:
    print(f"[startup] CUDA_VISIBLE_DEVICES={_PRE_ARGS.gpu}")



# 1. Heavy imports (now safe — env vars + tee already installed)


import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import asdict  # noqa: E402

import torch  # noqa: E402

from spark_code.config import Config  # noqa: E402
from spark_code.data.humaneval import load_humaneval  # noqa: E402
from spark_code.data.mbpp import load_mbpp, split_mbpp_train_heldout  # noqa: E402
from spark_code.sandbox.executor import (  # noqa: E402
    execute_code,
    execute_tests_individually,
)
from spark_code.training.runner import run_condition  # noqa: E402
from spark_code.utils.io import json_safe, mkdir, save_json  # noqa: E402
from spark_code.utils.reporting import export_csv, print_comparison  # noqa: E402
from spark_code.utils.seeds import set_all_seeds  # noqa: E402



# 2. CLI


def parse_args():
    p = argparse.ArgumentParser(description="SPARK-Code experiment runner")
    # GPU / experiment selection
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument("--condition", type=str, default="both", choices=["A", "C", "both"])
    p.add_argument("--config", type=str, default=None,
                   help="Path to JSON config file. CLI flags override JSON values.")
    p.add_argument("--base-model", "--model", dest="model", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    # Output
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    # Training
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--rollouts", type=int, default=None)
    p.add_argument("--train-problems", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    # Adaptive
    p.add_argument("--disable-adaptive-rollouts", action="store_true")
    p.add_argument("--max-adaptive-rollouts", type=int, default=None)
    p.add_argument("--adaptive-extra-step", type=int, default=None)
    # Reward
    p.add_argument("--binary-rewards", action="store_true",
                   help="Use binary pass/fail instead of partial per-test rewards")
    # Frontier
    p.add_argument("--skip-frontier", action="store_true")
    p.add_argument("--frontier-pool", type=int, default=None)
    p.add_argument("--frontier-k", type=int, default=None)
    p.add_argument("--load-frontier", type=str, default=None,
                   help="Path to frontier_task_ids.json from a previous run.")
    # Optimizer
    p.add_argument("--grpo-lr", type=float, default=None)
    p.add_argument("--kl-coeff", type=float, default=None)
    p.add_argument("--aux-loss-scale", type=float, default=None,
                   help="Multiplier on aux SFT loss; effective aux LR = grpo_lr * aux_loss_scale")
    p.add_argument("--aux-epochs", type=int, default=None)
    p.add_argument("--aux-micro-batch", type=int, default=None)
    # Aux caps/weights
    p.add_argument("--aux-max-pointwise", type=int, default=None)
    p.add_argument("--aux-max-pairwise", type=int, default=None)
    p.add_argument("--aux-max-reflection", type=int, default=None)
    p.add_argument("--aux-weight-pointwise", type=float, default=None)
    p.add_argument("--aux-weight-pairwise", type=float, default=None)
    p.add_argument("--aux-weight-reflection", type=float, default=None)
    p.add_argument("--reflection-target-mode", type=str, default=None,
                   choices=["correct_or_canonical", "self_fix"])
    # SFT warmup
    p.add_argument("--sft-warmup", action="store_true")
    p.add_argument("--sft-loss-scale", type=float, default=None)
    p.add_argument("--sft-epochs", type=int, default=None)
    p.add_argument("--sft-max-examples", type=int, default=None)
    # Eval
    p.add_argument("--eval-samples", type=int, default=None)
    p.add_argument("--eval-temperature", type=float, default=None)
    p.add_argument("--eval-top-p", type=float, default=None)
    p.add_argument("--reflection-eval-problems", type=int, default=None)
    p.add_argument("--mbpp-eval-size", type=int, default=None)
    p.add_argument("--disable-mbpp-eval", action="store_true")
    # Sandbox
    p.add_argument("--exec-timeout", type=int, default=None)
    p.add_argument("--exec-memory-mb", type=int, default=None)
    # Hub
    p.add_argument("--hf-repo-id", type=str, default=None)
    p.add_argument("--hf-token", type=str, default=None)
    return p.parse_args()


def apply_args_to_config(args, cfg: Config) -> Config:
    """Apply CLI args on top of a (possibly JSON-loaded) Config.

    CLI takes precedence over JSON, which takes precedence over Config defaults.
    """
    if args.model:
        cfg.model_name = args.model
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output:
        cfg.output_dir = args.output
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.no_wandb:
        cfg.wandb_enabled = False

    if args.iterations is not None:
        cfg.num_iterations = args.iterations
    if args.rollouts is not None:
        cfg.num_rollouts = args.rollouts
    if args.train_problems is not None:
        cfg.max_train_problems = args.train_problems
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_p is not None:
        cfg.top_p = args.top_p

    if args.disable_adaptive_rollouts:
        cfg.adaptive_rollouts = False
    if args.max_adaptive_rollouts is not None:
        cfg.max_adaptive_rollouts = args.max_adaptive_rollouts
    if args.adaptive_extra_step is not None:
        cfg.adaptive_extra_step = args.adaptive_extra_step
    if args.binary_rewards:
        cfg.partial_rewards = False

    if args.skip_frontier:
        cfg.use_frontier_filter = False
    if args.frontier_pool is not None:
        cfg.frontier_pool_size = args.frontier_pool
    if args.frontier_k is not None:
        cfg.frontier_k = args.frontier_k
    if args.load_frontier:
        cfg.load_frontier_path = args.load_frontier

    if args.grpo_lr is not None:
        cfg.grpo_lr = args.grpo_lr
    if args.kl_coeff is not None:
        cfg.kl_coeff = args.kl_coeff
    if args.aux_loss_scale is not None:
        cfg.aux_loss_scale = args.aux_loss_scale
    if args.aux_epochs is not None:
        cfg.aux_epochs = args.aux_epochs
    if args.aux_micro_batch is not None:
        cfg.aux_micro_batch = args.aux_micro_batch

    if args.aux_max_pointwise is not None:
        cfg.aux_max_pointwise = args.aux_max_pointwise
    if args.aux_max_pairwise is not None:
        cfg.aux_max_pairwise = args.aux_max_pairwise
    if args.aux_max_reflection is not None:
        cfg.aux_max_reflection = args.aux_max_reflection
    if args.aux_weight_pointwise is not None:
        cfg.aux_weight_pointwise = args.aux_weight_pointwise
    if args.aux_weight_pairwise is not None:
        cfg.aux_weight_pairwise = args.aux_weight_pairwise
    if args.aux_weight_reflection is not None:
        cfg.aux_weight_reflection = args.aux_weight_reflection
    if args.reflection_target_mode is not None:
        cfg.reflection_target_mode = args.reflection_target_mode

    if args.sft_warmup:
        cfg.sft_warmup = True
    if args.sft_loss_scale is not None:
        cfg.sft_loss_scale = args.sft_loss_scale
    if args.sft_epochs is not None:
        cfg.sft_epochs = args.sft_epochs
    if args.sft_max_examples is not None:
        cfg.sft_max_examples = args.sft_max_examples

    if args.eval_samples is not None:
        cfg.eval_num_samples = args.eval_samples
    if args.eval_temperature is not None:
        cfg.eval_temperature = args.eval_temperature
    if args.eval_top_p is not None:
        cfg.eval_top_p = args.eval_top_p
    if args.reflection_eval_problems is not None:
        cfg.reflection_eval_problems = args.reflection_eval_problems
    if args.mbpp_eval_size is not None:
        cfg.mbpp_eval_size = args.mbpp_eval_size
    if args.disable_mbpp_eval:
        cfg.eval_mbpp_heldout = False

    if args.exec_timeout is not None:
        cfg.exec_timeout = args.exec_timeout
    if args.exec_memory_mb is not None:
        cfg.exec_memory_mb = args.exec_memory_mb

    if args.hf_repo_id is not None:
        cfg.hf_repo_id = args.hf_repo_id
    if args.hf_token is not None:
        cfg.hf_token = args.hf_token

    cfg.run_log_path = str(_LOG_PATH_EARLY)
    return cfg


def load_config_from_json(path: str) -> Config:
    """Load a Config from a JSON file. Unknown keys are ignored with a warning."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = Config()
    valid_fields = {f for f in cfg.__dataclass_fields__}
    for k, v in data.items():
        if k in valid_fields:
            setattr(cfg, k, v)
        else:
            print(f"[config] WARNING: ignoring unknown key '{k}' from {path}")
    print(f"[config] Loaded base config from {path}")
    return cfg



# 3. Main


def main():
    args = parse_args()
    cfg = load_config_from_json(args.config) if args.config else Config()
    cfg = apply_args_to_config(args, cfg)

    output_root = Path(cfg.output_dir).expanduser().resolve()
    mkdir(output_root)
    mkdir(output_root / "artifacts")

    print("\n" + "=" * 70)
    print("SPARK-Code experiment")
    print("=" * 70)
    print(f"Output root: {output_root}")
    print(f"Run log: {cfg.run_log_path}")
    print(f"Condition: {args.condition}")
    print(f"Model: {cfg.model_name}")
    print(f"Seed: {cfg.seed}")
    if cfg.load_frontier_path:
        print(f"Load frontier: {cfg.load_frontier_path}")
    print("\n[config]")
    print(json.dumps(json_safe(asdict(cfg)), indent=2))
    save_json(output_root / "config.json", asdict(cfg))
    save_json(
        output_root / "command.json",
        {
            "argv": sys.argv,
            "log_path": cfg.run_log_path,
            "started_at": datetime.now().isoformat(),
        },
    )

    set_all_seeds(cfg.seed)
    print("\n[system]")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count visible: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / 1e9:.2f} GB")

    # ----- Smoke test the sandbox before any expensive work -----
    smoke_bin = execute_code(
        "def add(a,b):\n    return a+b",
        "assert add(2,3)==5\nassert add(-1,1)==0",
        cfg,
    )
    print(
        f"\n[sandbox] binary smoke passed={smoke_bin.passed} "
        f"type={smoke_bin.error_type} reward={smoke_bin.reward}"
    )
    if not smoke_bin.passed:
        raise RuntimeError("Binary execution sandbox smoke test failed")

    smoke_part = execute_tests_individually(
        "def mul(a,b):\n    return a*b",
        ["assert mul(2,3)==6", "assert mul(-1,1)==-1", "assert mul(0,5)==0"],
        cfg,
    )
    print(
        f"[sandbox] partial smoke passed={smoke_part.passed} "
        f"reward={smoke_part.reward} tests={smoke_part.tests_passed}/{smoke_part.tests_total}"
    )
    if not smoke_part.passed:
        raise RuntimeError("Partial-reward execution sandbox smoke test failed")

    # ----- Load datasets -----
    print("\n[data] Loading datasets")
    all_mbpp_problems = load_mbpp(cfg, max_n=None)
    train_pool, mbpp_eval_problems = split_mbpp_train_heldout(all_mbpp_problems, cfg)
    eval_problems = load_humaneval()
    save_json(output_root / "all_mbpp_task_ids.json", [p.task_id for p in all_mbpp_problems])
    save_json(output_root / "train_pool_task_ids.json", [p.task_id for p in train_pool])
    save_json(
        output_root / "mbpp_heldout_task_ids.json",
        [p.task_id for p in mbpp_eval_problems],
    )
    save_json(output_root / "eval_task_ids.json", [p.task_id for p in eval_problems])

    # ----- Run conditions -----
    conditions = ["A", "C"] if args.condition == "both" else [args.condition]
    all_results = []
    start_all = time.time()

    original_load_frontier_path = cfg.load_frontier_path
    try:
        for cond in conditions:
            # When running both, force C to reuse A's frontier task IDs
            if (
                args.condition == "both"
                and cond == "C"
                and cfg.use_frontier_filter
                and not original_load_frontier_path
            ):
                a_frontier = output_root / "condition_A" / "frontier_task_ids.json"
                if a_frontier.exists():
                    cfg.load_frontier_path = str(a_frontier)
                    print(f"[main] Reusing A frontier for C: {cfg.load_frontier_path}")
            else:
                cfg.load_frontier_path = original_load_frontier_path

            set_all_seeds(cfg.seed)
            result = run_condition(
                cond, cfg, train_pool, eval_problems, mbpp_eval_problems, output_root
            )
            all_results.append(result)
            save_json(output_root / "all_results_so_far.json", all_results)
    except Exception:
        print("\n[error] Experiment failed:")
        traceback.print_exc()
        save_json(
            output_root / "error.json",
            {"traceback": traceback.format_exc(), "argv": sys.argv},
        )
        raise
    finally:
        elapsed = (time.time() - start_all) / 60.0
        print(f"\n[done] Elapsed minutes: {elapsed:.2f}")

    save_json(output_root / "all_results.json", all_results)
    export_csv(all_results, output_root / "results.csv")
    print_comparison(all_results)

    print(f"\n[complete] Results: {output_root}")
    print(f"[complete] Log: {cfg.run_log_path}")
    print("[complete] Top-level files:")
    for p in sorted(output_root.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
