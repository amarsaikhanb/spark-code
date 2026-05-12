"""Standard pass@k estimator from the HumanEval.

Given ``n`` total samples and ``c`` correct ones, returns the unbiased estimate
of the probability that at least one of ``k`` randomly drawn samples passes.
"""

from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator. Raises if k > n."""
    if k <= 0:
        return 0.0
    if n < k:
        raise ValueError(f"Cannot compute pass@{k} with n={n}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
