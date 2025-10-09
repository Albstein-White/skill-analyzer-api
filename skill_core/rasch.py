"""Rasch 1PL utilities used by the adaptive engine.

This module provides a minimal set of helpers for computing logistic
probabilities, Fisher information, and a damped MAP (Newton) update for the
person ability parameter.  Functions are intentionally lightweight so that
future policy tweaks can reuse them from both the API runtime and developer
CLI tools.
"""
from __future__ import annotations

import math
from typing import List, Tuple

__all__ = [
    "sigma",
    "rasch_p",
    "item_info",
    "map_update",
    "se_from_info",
]

_EPS = 1e-6


def sigma(x: float) -> float:
    """Return the logistic function ``σ(x) = 1 / (1 + e^{−x})``.

    The implementation guards against overflow for large negative inputs by
    handling the positive and negative halves of the real line separately.
    """

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def rasch_p(theta: float, b: float) -> float:
    """Compute the Rasch 1PL probability of a correct answer.

    Parameters
    ----------
    theta: float
        Current estimate of the learner ability.
    b: float
        Item difficulty parameter.

    Returns
    -------
    float
        ``σ(theta − b)``
    """

    return sigma(theta - b)


def item_info(theta: float, b: float, a: float = 1.0) -> float:
    """Fisher information contributed by a single 1PL item.

    The Rasch formulation treats discrimination ``a`` as 1.0, but the
    parameter is left explicit for future flexibility.
    """

    p = rasch_p(theta, b)
    info = (a * a) * p * (1.0 - p)
    return max(info, 0.0)


def map_update(
    theta: float,
    b: float,
    correct: bool,
    a: float = 1.0,
    prior_var: float = 1.0,
    eta: float = 1.0,
) -> float:
    """Perform a damped Newton/MAP update for the Rasch person ability.

    Parameters mirror the pseudocode from the task description.  ``prior_var``
    provides Gaussian shrinkage towards zero while ``eta`` can be used to
    throttle the step size.
    """

    p = rasch_p(theta, b)
    grad = ((1.0 if correct else 0.0) - p) * a
    info = (a * a) * p * (1.0 - p) + (1.0 / max(prior_var, _EPS))
    step = eta * grad / max(info, _EPS)
    return float(theta + step)


def se_from_info(info_total: float) -> float:
    """Convert accumulated Fisher information into a standard error."""

    return 1.0 / math.sqrt(max(info_total, _EPS))


def _demo_sequence() -> Tuple[List[float], List[float]]:
    """Run a developer demo with synthetic data.

    Returns the history of ``theta`` and ``se`` for downstream assertions.
    """

    theta_hist: List[float] = []
    se_hist: List[float] = []
    theta = 0.0
    info_total = 1.0
    seq_b = [-1, -1, 0, 0, +1, +1, +1, +2]
    seq_correct = [1, 1, 1, 0, 1, 1, 0, 1]

    print("step | b | correct | theta    | info_tot |   SE")
    for idx, (b, correct) in enumerate(zip(seq_b, seq_correct), start=1):
        theta_new = map_update(theta, b, bool(correct))
        info_i = item_info(theta_new, b)
        info_total = max(info_total + info_i, _EPS)
        se = se_from_info(info_total)
        theta = theta_new

        theta_hist.append(theta)
        se_hist.append(se)
        print(f" {idx:2d}  | {b:+d} |   {correct:d}     | {theta:7.4f} | {info_total:8.4f} | {se:5.4f}")

    if theta_hist:
        assert theta_hist[-1] > theta_hist[0] - 1e-6, "θ should increase overall"
    for prev, cur in zip(se_hist, se_hist[1:]):
        assert cur <= prev + 1e-6, "SE should be non-increasing"
    return theta_hist, se_hist


if __name__ == "__main__":  # pragma: no cover - developer utility
    # Inline sanity checks for quick confidence during refactors.
    assert abs(sigma(0.0) - 0.5) < 1e-9
    assert rasch_p(1.0, 0.0) > 0.5
    assert abs(item_info(0.0, 0.0) - 0.25) < 1e-6
    theta_hist, se_hist = _demo_sequence()
    print(f"Final θ: {theta_hist[-1]:.4f}, Final SE: {se_hist[-1]:.4f}")
    print(
        "Rasch/SE enabled: domain state now includes {theta, se, info_total, level, b_peak, b_stable} and updates on OBJ answers."
    )
