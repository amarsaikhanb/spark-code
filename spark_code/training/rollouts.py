"""Rollout generation.

For each problem we sample K candidate completions, evaluate each in the
sandbox, and store everything GRPO needs (old log-probs, optionally reference
log-probs, full token IDs, prompt length).

If a group is "uninformative" — every rollout has the same reward — we top it
up with extra samples so GRPO has a non-zero advantage signal.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from spark_code.config import Config
from spark_code.data.structures import CodeProblem, Rollout
from spark_code.model.prompts import chat_prompt, extract_code
from spark_code.sandbox.executor import evaluate_generated_code
from spark_code.training.logprobs import get_reference_logprobs_eval, get_token_logprobs


def generate_one_rollout(
    model, tok, prob: CodeProblem, cfg: Config, prompt_text: str, enc, prompt_len: int
) -> Rollout:
    """Generate one rollout. Caller MUST ensure ``model.eval()`` is active."""
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=True,
            top_p=cfg.top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    full_ids = out[0]
    gen_ids = full_ids[prompt_len:]
    completion_len = int(gen_ids.shape[0])
    completion = tok.decode(gen_ids, skip_special_tokens=True)
    code = extract_code(completion)
    er = evaluate_generated_code(code, prob, cfg, training=True)

    fid_gpu = full_ids.unsqueeze(0).to(cfg.device)
    with torch.no_grad():
        # Old log-probs: current policy at sampling time
        old_lps = (
            get_token_logprobs(model, fid_gpu, prompt_len, completion_len)
            .detach()
            .cpu()
            .float()
            .tolist()
        )
        # Reference log-probs: frozen base, computed once here in eval mode
        ref_lps: List[float] = []
        if cfg.kl_coeff > 0:
            ref_lps = get_reference_logprobs_eval(model, fid_gpu, prompt_len, completion_len)

    return Rollout(
        task_id=prob.task_id,
        prompt=prompt_text,
        completion=completion,
        extracted_code=code,
        exec_result=er,
        reward=float(er.reward),
        old_logprobs=old_lps,
        ref_logprobs=ref_lps,
        prompt_len=prompt_len,
        full_ids=full_ids.detach().cpu().tolist(),
    )


def group_is_informative(group: List[Rollout]) -> bool:
    """A group is informative iff its reward std is non-trivial.

    GRPO advantages are zero-mean within a group, so a constant-reward group
    contributes zero gradient and is wasted compute.
    """
    return float(np.std([r.reward for r in group])) > 1e-8


def generate_rollouts(
    model, tok, problems: List[CodeProblem], cfg: Config
) -> List[List[Rollout]]:
    """Generate K (or up to max_adaptive_rollouts) rollouts per problem.

    Adaptive top-up: if after K samples the group is still uninformative, we
    keep adding samples until either (a) the group becomes informative or
    (b) we hit ``max_adaptive_rollouts``.
    """
    model.eval()
    groups: List[List[Rollout]] = []
    n_total = 0
    for i, prob in enumerate(problems):
        prompt_text = chat_prompt(tok, prob.prompt_text)
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(cfg.device)
        prompt_len = enc.input_ids.shape[1]
        group: List[Rollout] = []
        for _ in range(cfg.num_rollouts):
            group.append(generate_one_rollout(model, tok, prob, cfg, prompt_text, enc, prompt_len))
        # Adaptive: top up uninformative groups
        while (
            cfg.adaptive_rollouts
            and len(group) < cfg.max_adaptive_rollouts
            and not group_is_informative(group)
        ):
            for _ in range(cfg.adaptive_extra_step):
                if len(group) >= cfg.max_adaptive_rollouts:
                    break
                group.append(
                    generate_one_rollout(model, tok, prob, cfg, prompt_text, enc, prompt_len)
                )
        groups.append(group)
        n_total += len(group)
        if (i + 1) % 25 == 0 or i == 0 or i == len(problems) - 1:
            rewards = [r.reward for r in group]
            passes = sum(1 for r in group if r.exec_result.passed)
            print(
                f"  [gen] {i+1}/{len(problems)} n={len(group)} "
                f"pass={passes}/{len(group)} mean_r={np.mean(rewards):.3f} "
                f"std={np.std(rewards):.3f}"
            )
    print(f"  [gen] Total rollouts: {n_total}")
    return groups
