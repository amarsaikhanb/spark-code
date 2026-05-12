"""Configuration dataclass for SPARK-Code experiments.

A single flat dataclass that matches the CLI 1:1. Sub-configs are deliberately
avoided — flat is easier to scan, easier to serialize, and easier to override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Config:
    
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    
    # Data
    max_train_problems: int = 200
    # Held-out MBPP evaluation separates in-distribution generalization from
    # cross-benchmark HumanEval transfer. The effective holdout size is capped
    # so at least max_train_problems remain available for training.
    eval_mbpp_heldout: bool = True
    mbpp_eval_size: int = 100
    use_mbpp_challenge_tests: bool = True
    prompt_style: str = "complete_function"  # complete_function or original

    
    # Frontier filtering (one-time, before training)
    use_frontier_filter: bool = True
    frontier_pool_size: int = 300
    frontier_k: int = 4
    load_frontier_path: str = ""  # if set, load task IDs instead of scanning

    
    # Generation / rollout collection
    num_iterations: int = 3
    num_rollouts: int = 4
    adaptive_rollouts: bool = True
    max_adaptive_rollouts: int = 8
    adaptive_extra_step: int = 2
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.98

    
    # Reward
    partial_rewards: bool = True
    syntax_penalty: float = -0.2
    runtime_penalty: float = -0.1
    timeout_penalty: float = -0.3
    wrong_answer_floor: float = 0.0

    
    # GRPO
    grpo_lr: float = 5e-6
    grpo_grad_accum: int = 4
    clip_ratio: float = 0.2
    kl_coeff: float = 0.0  # OFF by default; set >0 to enable frozen-reference k3 KL
    max_grad_norm: float = 1.0

    
    # Auxiliary SPARK-style SFT
    # NOTE: single optimizer with grpo_lr; aux phase uses aux_loss_scale to
    # effectively train at a different learning rate without a second optimizer.
    aux_loss_scale: float = 0.4
    aux_epochs: int = 1
    aux_micro_batch: int = 2
    aux_max_len: int = 1024
    aux_max_pointwise: int = 200
    aux_max_pairwise: int = 200
    aux_max_reflection: int = 400
    aux_weight_pointwise: float = 0.1
    aux_weight_pairwise: float = 0.3
    aux_weight_reflection: float = 1.0
    refl_self_fix_temperature: float = 0.7
    # Default uses an already-correct rollout, or MBPP canonical solution as
    # fallback. Use "self_fix" to train reflection only on the model's own
    # successful repairs.
    reflection_target_mode: str = "correct_or_canonical"  # or "self_fix"

    
    # Optional supervised warmup on MBPP canonical solutions
    sft_warmup: bool = False
    sft_loss_scale: float = 0.4
    sft_epochs: int = 1
    sft_max_examples: int = 200

    
    # Evaluation
    eval_num_samples: int = 5
    eval_temperature: float = 0.2
    eval_top_p: float = 0.95
    reflection_eval_problems: int = 164  # full HumanEval, greedy

    
    # Execution sandbox
    exec_timeout: int = 10
    exec_memory_mb: int = 2048

    
    # Infrastructure
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./spark_code_output"
    wandb_project: str = "spark-code"
    wandb_enabled: bool = True
    hf_repo_id: str = ""
    hf_token: str = ""
    run_log_path: str = ""

    
    # Derived properties
    @property
    def torch_dtype(self) -> Any:
        # Lazy import so this module can be loaded without torch installed
        # (e.g. by lightweight CI tests that only exercise data/sandbox code).
        import torch

        return torch.bfloat16

    @property
    def aux_weights(self) -> Dict[str, float]:
        return {
            "pointwise": self.aux_weight_pointwise,
            "pairwise": self.aux_weight_pairwise,
            "reflection": self.aux_weight_reflection,
            "sft": 1.0,
        }
