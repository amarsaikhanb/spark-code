"""Per-token log-probability extraction for GRPO.

Three flavors:

* :func:`get_token_logprobs` — log p(completion | prefix) under whatever model
  is passed. Used both for "old" log-probs at rollout time (model in eval mode)
  and "new" log-probs during the GRPO step (model in train mode).
* :func:`get_reference_logprobs_eval` — frozen-base log-probs via PEFT
  ``disable_adapter``. Computed ONCE at rollout time so the GRPO loop never
  has to flip adapters in and out mid-training.
* :func:`get_new_logprobs` — convenience wrapper that builds the right tensor
  shape from a stored Rollout.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from spark_code.data.structures import Rollout


def get_token_logprobs(
    model, input_ids: torch.Tensor, prompt_len: int, completion_len: int
) -> torch.Tensor:
    """Return log p(completion tokens | prefix) for one sequence.

    ``input_ids`` should be shape (1, T). The returned tensor is shape
    ``(completion_len,)`` and lives on the model's device.
    """
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    logps = F.log_softmax(logits, dim=-1)
    token_logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = max(prompt_len - 1, 0)
    end = start + completion_len
    return token_logps[0, start:end]


def get_reference_logprobs_eval(
    model, input_ids: torch.Tensor, prompt_len: int, completion_len: int
) -> List[float]:
    """Frozen-base log-probs via PEFT ``disable_adapter``.

    Call ONLY in eval mode. We materialize the result back to a Python list of
    floats so it can be stored on the Rollout and consumed later by the GRPO
    step without keeping a graph or extra GPU memory alive.
    """
    if not hasattr(model, "disable_adapter"):
        with torch.no_grad():
            return (
                get_token_logprobs(model, input_ids, prompt_len, completion_len)
                .detach()
                .cpu()
                .float()
                .tolist()
            )
    with model.disable_adapter():
        with torch.no_grad():
            return (
                get_token_logprobs(model, input_ids, prompt_len, completion_len)
                .detach()
                .cpu()
                .float()
                .tolist()
            )


def get_new_logprobs(model, rollout: Rollout, device: str) -> torch.Tensor:
    """Recompute current-policy log-probs for a stored rollout (training mode)."""
    input_ids = torch.tensor(rollout.full_ids, dtype=torch.long, device=device).unsqueeze(0)
    completion_len = len(rollout.old_logprobs)
    if input_ids.shape[1] < rollout.prompt_len + completion_len:
        completion_len = input_ids.shape[1] - rollout.prompt_len
    if completion_len <= 0:
        return torch.zeros(0, device=device)
    return get_token_logprobs(model, input_ids, rollout.prompt_len, completion_len)
