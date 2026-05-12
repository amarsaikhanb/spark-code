"""Auxiliary SPARK-style SFT.

After each GRPO update under Condition C, the same rollouts are recycled into
three auxiliary objectives, all trained on the SAME shared LoRA weights:

* **Pointwise**: minimal binary "Correct"/"Incorrect" judgment.
* **Pairwise**: minimal "A"/"B" preference between one correct and one
  incorrect rollout for the same problem, with randomized order.
* **Reflection**: execution-grounded repair. Failing code + actual stderr →
  corrected solution. The repair target is either an already-correct rollout
  from the same group, the canonical MBPP solution as fallback, OR — when
  ``reflection_target_mode="self_fix"`` — the model's own successful repair.

The single optimizer is shared with GRPO; auxiliary loss is multiplied by
``aux_loss_scale`` to effectively train at a smaller learning rate.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from spark_code.config import Config
from spark_code.data.structures import AuxExample, CodeProblem, Rollout
from spark_code.model.prompts import aux_full, aux_prefix, chat_prompt, clean_stderr, extract_code
from spark_code.sandbox.executor import execute_code


def build_aux_data(
    groups: List[List[Rollout]],
    problems: List[CodeProblem],
    model,
    tok,
    cfg: Config,
) -> Tuple[List[AuxExample], Dict[str, Any]]:
    """Build SPARK-style pointwise / pairwise / reflection auxiliary examples."""
    by_type: Dict[str, List[AuxExample]] = {
        "pointwise": [],
        "pairwise": [],
        "reflection": [],
    }
    n_self_fix_attempts = 0
    n_self_fix_passed = 0
    model.eval()

    for group, prob in zip(groups, problems):
        correct = [r for r in group if r.exec_result.passed]
        incorrect = [r for r in group if not r.exec_result.passed]

        #  Pointwise: binary judgment, no stderr leakage 
        for r in group:
            user = (
                f"Problem:\n{prob.prompt_text}\n\n"
                f"Candidate code:\n```python\n{r.extracted_code}\n```\n\n"
                "Does this code correctly solve the problem? "
                "Answer 'Correct' or 'Incorrect'."
            )
            asst = "Correct." if r.exec_result.passed else "Incorrect."
            by_type["pointwise"].append(AuxExample("pointwise", user, asst))

        #  Pairwise: A/B preference with randomized order 
        if correct and incorrect:
            for ic in incorrect[:2]:
                co = random.choice(correct)
                if random.random() < 0.5:
                    ca, cb, b = co.extracted_code, ic.extracted_code, "A"
                else:
                    ca, cb, b = ic.extracted_code, co.extracted_code, "B"
                user = (
                    f"Problem:\n{prob.prompt_text}\n\n"
                    f"A:\n```python\n{ca}\n```\n\n"
                    f"B:\n```python\n{cb}\n```\n\n"
                    "Which solution is correct? Answer 'A' or 'B'."
                )
                by_type["pairwise"].append(AuxExample("pairwise", user, f"{b}."))

        #  Reflection: execution-grounded repair 
        if incorrect:
            if cfg.reflection_target_mode == "self_fix":
                # Strict SPARK self-fix: train only on the model's own successful repairs.
                for ic in incorrect[:2]:
                    err = clean_stderr(ic.exec_result.stderr)
                    fix_msg = (
                        "The following code fails. Fix it.\n\n"
                        f"Problem:\n{prob.prompt_text}\n\n"
                        f"Failing code:\n```python\n{ic.extracted_code}\n```\n\n"
                        f"Error:\n```\n{err}\n```\n\n"
                        "Provide the corrected Python function."
                    )
                    fp = chat_prompt(tok, fix_msg)
                    fe = tok(fp, return_tensors="pt", add_special_tokens=False).to(cfg.device)
                    with torch.no_grad():
                        fo = model.generate(
                            input_ids=fe.input_ids,
                            attention_mask=fe.attention_mask,
                            max_new_tokens=cfg.max_new_tokens,
                            temperature=cfg.refl_self_fix_temperature,
                            do_sample=True,
                            top_p=0.95,
                            pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id,
                        )
                    fix_code = extract_code(
                        tok.decode(fo[0][fe.input_ids.shape[1]:], skip_special_tokens=True)
                    )
                    n_self_fix_attempts += 1
                    if execute_code(fix_code, prob.test_code, cfg).passed:
                        asst = f"```python\n{fix_code}\n```"
                        by_type["reflection"].append(AuxExample("reflection", fix_msg, asst))
                        n_self_fix_passed += 1
            else:
                # Default: use a correct rollout, or canonical MBPP solution as fallback.
                target_code = None
                if correct:
                    target_code = random.choice(correct).extracted_code
                elif prob.canonical_solution:
                    target_code = prob.canonical_solution

                if target_code:
                    for ic in incorrect[:2]:
                        err = clean_stderr(ic.exec_result.stderr)
                        fix_msg = (
                            "The following code fails. Fix it.\n\n"
                            f"Problem:\n{prob.prompt_text}\n\n"
                            f"Failing code:\n```python\n{ic.extracted_code}\n```\n\n"
                            f"Error:\n```\n{err}\n```\n\n"
                            "Provide the corrected Python function."
                        )
                        asst = f"```python\n{target_code.strip()}\n```"
                        by_type["reflection"].append(AuxExample("reflection", fix_msg, asst))

    # Cap each type and shuffle
    caps = {
        "pointwise": cfg.aux_max_pointwise,
        "pairwise": cfg.aux_max_pairwise,
        "reflection": cfg.aux_max_reflection,
    }
    final: List[AuxExample] = []
    counts: Dict[str, int] = {}
    for typ, examples in by_type.items():
        random.shuffle(examples)
        kept = examples[: caps[typ]]
        counts[typ] = len(kept)
        final.extend(kept)
    random.shuffle(final)

    sf_rate = n_self_fix_passed / max(n_self_fix_attempts, 1)
    print(f"  [aux] {len(final)} examples after caps: {counts}")
    print(f"  [aux] reflection_target_mode={cfg.reflection_target_mode}")
    print(f"  [aux] self-fix rate: {n_self_fix_passed}/{n_self_fix_attempts}={sf_rate:.3f}")
    meta = {
        "aux/self_fix_rate": float(sf_rate),
        "aux/n_self_fix_passed": int(n_self_fix_passed),
        "aux/n_self_fix_attempts": int(n_self_fix_attempts),
    }
    return final, meta


def build_sft_warmup_data(problems: List[CodeProblem], cfg: Config) -> List[AuxExample]:
    """Build supervised warmup examples from canonical MBPP solutions."""
    examples: List[AuxExample] = []
    for p in problems[: cfg.sft_max_examples]:
        if not p.canonical_solution:
            continue
        examples.append(
            AuxExample(
                type="sft",
                user_msg=p.prompt_text,
                asst_msg=f"```python\n{p.canonical_solution.strip()}\n```",
            )
        )
    return examples


def sft_step(
    model,
    tok,
    data: List[AuxExample],
    opt,
    cfg: Config,
    epochs: int,
    max_len: int,
    tag: str,
    loss_scale: float = 1.0,
) -> Dict[str, Any]:
    """One SFT pass over an auxiliary dataset.

    The single optimizer is shared with GRPO. ``loss_scale`` lets us
    effectively train the aux phase at a fraction of the GRPO learning rate
    without instantiating a second optimizer.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_used = 0
    for ep in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), cfg.aux_micro_batch):
            batch = data[i : i + cfg.aux_micro_batch]
            opt.zero_grad(set_to_none=True)
            batch_loss_value = 0.0
            used_in_batch = 0
            for ex in batch:
                prefix = aux_prefix(tok, ex.user_msg)
                full = aux_full(tok, ex.user_msg, ex.asst_msg)
                prefix_ids = tok(prefix, add_special_tokens=False).input_ids
                enc = tok(
                    full,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                ).to(cfg.device)
                ids = enc.input_ids
                prompt_len = min(len(prefix_ids), ids.shape[1])
                if prompt_len >= ids.shape[1] - 1:
                    continue
                out = model(input_ids=ids, use_cache=False)
                labels = ids.clone()
                labels[:, :prompt_len] = -100
                logits = out.logits[:, :-1, :].contiguous()
                shifted = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    shifted.view(-1),
                    ignore_index=-100,
                )
                weight = cfg.aux_weights.get(ex.type, 1.0)
                weighted = loss * float(weight) * float(loss_scale)
                (weighted / max(len(batch), 1)).backward()
                batch_loss_value += float(weighted.detach().cpu())
                used_in_batch += 1
            if used_in_batch == 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            total_loss += batch_loss_value / used_in_batch
            n_batches += 1
            n_used += used_in_batch

    return {
        f"{tag}/loss": total_loss / max(n_batches, 1),
        f"{tag}/n": n_used,
        f"{tag}/loss_scale": float(loss_scale),
    }
