"""GRPO update.

Group-relative advantage normalization (DeepSeekMath-style), clipped surrogate
loss with importance ratios, and an optional k3 KL penalty against the frozen
base model. Reference log-probs are read from the stored Rollout — they were
computed once at rollout time in eval mode, so this step never has to flip
PEFT adapters.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from spark_code.config import Config
from spark_code.data.structures import Rollout
from spark_code.training.logprobs import get_new_logprobs


def compute_advantages(groups: List[List[Rollout]]) -> List[List[float]]:
    """Z-score rewards within each group. Constant-reward groups → all zeros."""
    advs: List[List[float]] = []
    for g in groups:
        r = np.array([x.reward for x in g], dtype=np.float32)
        mu, sd = float(r.mean()), float(r.std())
        if sd < 1e-8:
            advs.append([0.0] * len(g))
        else:
            advs.append(((r - mu) / (sd + 1e-8)).tolist())
    return advs


def grpo_step(
    model,
    groups: List[List[Rollout]],
    advantages: List[List[float]],
    opt,
    cfg: Config,
) -> Dict[str, Any]:
    """One GRPO update across all groups, with gradient accumulation."""
    model.train()
    opt.zero_grad(set_to_none=True)
    total_loss = total_policy = total_kl = total_abs_adv = 0.0
    n_seq = n_tok = accum = 0

    for group, group_advs in zip(groups, advantages):
        for ro, adv in zip(group, group_advs):
            if abs(adv) < 1e-8 or len(ro.old_logprobs) < 1:
                continue
            new = get_new_logprobs(model, ro, cfg.device)
            gl = min(len(ro.old_logprobs), new.shape[0])
            if gl <= 0:
                continue

            old = torch.tensor(ro.old_logprobs[:gl], dtype=torch.float32, device=cfg.device)
            new = new[:gl]

            ratio = torch.exp(new - old)
            adv_t = torch.tensor(float(adv), dtype=torch.float32, device=cfg.device)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_t
            policy_loss = -torch.min(s1, s2).mean()

            # Read ref log-probs from rollout (computed once at rollout time)
            if cfg.kl_coeff > 0 and ro.ref_logprobs:
                rgl = min(gl, len(ro.ref_logprobs))
                if rgl > 0:
                    ref = torch.tensor(
                        ro.ref_logprobs[:rgl], dtype=torch.float32, device=cfg.device
                    )
                    cur = new[:rgl]
                    log_ratio = ref - cur
                    kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
                else:
                    kl = torch.zeros((), dtype=torch.float32, device=cfg.device)
            else:
                kl = torch.zeros((), dtype=torch.float32, device=cfg.device)

            loss = policy_loss + cfg.kl_coeff * kl
            (loss / cfg.grpo_grad_accum).backward()

            total_loss += float(loss.detach().cpu())
            total_policy += float(policy_loss.detach().cpu())
            total_kl += float(kl.detach().cpu())
            total_abs_adv += abs(float(adv))
            n_seq += 1
            n_tok += gl
            accum += 1

            if accum >= cfg.grpo_grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum = 0

    if accum > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)

    denom = max(n_seq, 1)
    return {
        "grpo/loss": total_loss / denom,
        "grpo/policy_loss": total_policy / denom,
        "grpo/kl": total_kl / denom,
        "grpo/n_seq": n_seq,
        "grpo/n_tokens": n_tok,
        "grpo/mean_abs_adv": total_abs_adv / denom,
    }
