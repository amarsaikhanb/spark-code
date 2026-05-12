"""Tests for the pass@k unbiased estimator."""

from __future__ import annotations

import math

import pytest



# Inline copy of spark_code.eval.metrics.pass_at_k — keep in sync.


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



# Tests


def test_zero_correct_is_zero():
    """No correct samples → pass@k = 0 for any k."""
    for k in [1, 5, 10]:
        assert pass_at_k(10, 0, k) == 0.0


def test_all_correct_is_one():
    """All samples correct → pass@k = 1.0 for any k ≤ n."""
    for k in [1, 5, 10]:
        assert pass_at_k(10, 10, k) == 1.0


def test_n_minus_c_less_than_k_is_one():
    """When n - c < k, picking k samples is guaranteed to include a correct one."""
    assert pass_at_k(5, 4, 2) == 1.0
    assert pass_at_k(10, 9, 5) == 1.0


def test_pass_at_1_equals_c_over_n():
    """pass@1 = c/n exactly (modulo float precision)."""
    assert math.isclose(pass_at_k(10, 5, 1), 0.5)
    assert math.isclose(pass_at_k(10, 1, 1), 0.1)
    assert math.isclose(pass_at_k(100, 73, 1), 0.73)


def test_known_value_pass_at_5():
    """Spot-check pass@5 with a known computation.

    n=10, c=2, k=5: P(at least one correct in 5 draws from 10 with 2 correct)
    = 1 - C(8,5)/C(10,5) = 1 - 56/252 ≈ 0.7778
    """
    val = pass_at_k(10, 2, 5)
    expected = 1 - (math.comb(8, 5) / math.comb(10, 5))
    assert math.isclose(val, expected, rel_tol=1e-9)
    assert math.isclose(val, 1 - 56 / 252, rel_tol=1e-9)


def test_k_zero_returns_zero():
    """Defensive: k=0 should not blow up."""
    assert pass_at_k(10, 5, 0) == 0.0


def test_negative_k_returns_zero():
    """Defensive: negative k should not blow up."""
    assert pass_at_k(10, 5, -1) == 0.0


def test_n_less_than_k_raises():
    """Cannot estimate pass@k when n < k."""
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def test_monotonic_in_c():
    """For fixed n, k: pass@k is non-decreasing in c."""
    n, k = 20, 5
    prev = -1.0
    for c in range(n + 1):
        v = pass_at_k(n, c, k)
        assert v >= prev - 1e-12, f"non-monotonic at c={c}: {prev} -> {v}"
        prev = v


def test_monotonic_in_k():
    """For fixed n, c: pass@k is non-decreasing in k."""
    n, c = 20, 5
    prev = -1.0
    for k in range(1, n + 1):
        v = pass_at_k(n, c, k)
        assert v >= prev - 1e-12, f"non-monotonic at k={k}: {prev} -> {v}"
        prev = v


def test_in_unit_interval():
    """pass@k is always a probability."""
    for n in [1, 5, 10, 50, 200]:
        for c in range(n + 1):
            for k in range(1, n + 1):
                v = pass_at_k(n, c, k)
                assert 0.0 <= v <= 1.0, f"out of range: pass_at_k({n},{c},{k})={v}"


def test_inline_matches_canonical():
    """If the canonical module IS importable in this environment (torch present),
    sanity-check that the inline copy agrees with it on a grid."""
    try:
        from spark_code.eval.metrics import pass_at_k as canonical
    except Exception:
        pytest.skip("spark_code not importable in this environment")
    for n in [1, 5, 20]:
        for c in range(n + 1):
            for k in range(1, n + 1):
                assert math.isclose(
                    pass_at_k(n, c, k), canonical(n, c, k), rel_tol=1e-12
                )
