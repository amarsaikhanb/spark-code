"""HumanEval loading. Used as a fully held-out evaluation benchmark."""

from __future__ import annotations

from typing import List

from datasets import load_dataset

from spark_code.data.structures import CodeProblem


def load_humaneval() -> List[CodeProblem]:
    """Load all 164 HumanEval problems with executable test code.

    The prompt is the HumanEval function stub prefixed with a short instruction
    so the model returns just the body, in code-block form. The test code wraps
    HumanEval's ``check`` function to make it directly executable.
    """
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems: List[CodeProblem] = []
    for item in ds:
        prompt = (
            "Write the complete Python function satisfying this signature and docstring. "
            "Return only Python code, no markdown, no explanation.\n\n"
            + item["prompt"]
        )
        test_code = item["test"] + f"\ncheck({item['entry_point']})\n"
        canonical = item["prompt"] + item["canonical_solution"]
        problems.append(
            CodeProblem(
                task_id=item["task_id"],
                prompt_text=prompt,
                test_code=test_code,
                entry_point=item["entry_point"],
                canonical_solution=canonical,
                source="humaneval",
                test_list=[],
            )
        )
    print(f"[data] Loaded {len(problems)} HumanEval problems")
    return problems
