"""Frontier filtering."""

from __future__ import annotations

import json
import time
from typing import List

import numpy as np
import torch

from spark_code.config import Config
from spark_code.data.structures import CodeProblem
from spark_code.model.prompts import chat_prompt, extract_code
from spark_code.sandbox.executor import evaluate_generated_code


def frontier_filter(
    model, tok, problems: List[CodeProblem], cfg: Config, target_n: int
) -> List[CodeProblem]:
    """Keep problems whose K rollouts produce non-trivial reward variance."""
    model.eval()
    print(
        f"[frontier] Scanning {len(problems)} problems with k={cfg.frontier_k} "
        f"using {'partial' if cfg.partial_rewards else 'binary'} rewards..."
    )
    k = cfg.frontier_k
    keep: List[CodeProblem] = []
    t0 = time.time()
    for i, p in enumerate(problems):
        prompt_text = chat_prompt(tok, p.prompt_text)
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(cfg.device)
        rewards = []
        for _ in range(k):
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
            code = extract_code(
                tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
            )
            er = evaluate_generated_code(code, p, cfg, training=True)
            rewards.append(float(er.reward))
        # Keep if rewards are non-trivially varied (informative group)
        if float(np.std(rewards)) > 1e-3:
            keep.append(p)
        if (i + 1) % 50 == 0:
            mins = (time.time() - t0) / 60
            print(f"  [frontier] {i+1}/{len(problems)} scanned, {len(keep)} kept ({mins:.1f}m)")
        if len(keep) >= target_n:
            print(f"  [frontier] Target n={target_n} reached, stopping early.")
            break
    mins = (time.time() - t0) / 60
    print(f"[frontier] Kept {len(keep)}/{len(problems)} ({mins:.1f}m)")
    return keep


def load_frontier_from_file(
    path: str, train_pool: List[CodeProblem]
) -> List[CodeProblem]:
    """Load a frontier task-ID list from a previous run for fair A/C comparison."""
    print(f"[frontier] Loading task IDs from {path}")
    with open(path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    by_id = {p.task_id: p for p in train_pool}
    loaded = [by_id[i] for i in ids if i in by_id]
    missing = [i for i in ids if i not in by_id]
    if missing:
        print(f"[frontier] WARNING: {len(missing)} task IDs from {path} not found in pool")
    print(f"[frontier] Loaded {len(loaded)} problems")
    return loaded
