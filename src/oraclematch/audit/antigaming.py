"""Detect single-oracle exploiters and report caught-rate / false-positive-rate with CIs.

Methodology (honest about what it measures). A single-oracle exploiter is, by construction, a
molecule one oracle scores high and the other low — i.e. it carries large cross-oracle disagreement
``sigma_a``. The detector therefore flags molecules whose disagreement exceeds an **operating-point
threshold calibrated on genuinely-good ("clean") molecules only** at a target false-positive rate.
Caught-rate is then measured on a disjoint set of injected exploiters. Reporting both rates with
Wilson intervals (not just a point estimate) follows the ship discipline used in prior projects.

The caught-rate/FPR framing is inspired by the author's ``scorewright`` project; it is reimplemented
here natively to avoid cross-repo coupling. All numbers this module produces are on synthetic data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from oraclematch.calibration.ensemble import ScoredResult


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than normal approx for small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True)
class AntiGamingReport:
    n: int
    n_hackers: int
    n_clean: int
    caught: int
    false_positives: int
    threshold: float
    caught_rate: float
    caught_rate_ci: tuple[float, float]
    fpr: float
    fpr_ci: tuple[float, float]


class AntiGamingDetector:
    """Flags molecules whose cross-oracle disagreement exceeds a calibrated operating point."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold

    def calibrate(self, clean_scored: Sequence[ScoredResult], target_fpr: float = 0.05) -> float:
        """Set the threshold to the ``1 - target_fpr`` quantile of disagreement on clean molecules.

        This fixes a principled operating point *without* peeking at the exploiters, so the measured
        FPR on a held-out clean set is bounded by ``target_fpr`` in expectation.
        """
        if not clean_scored:
            raise ValueError("need at least one clean molecule to calibrate")
        if not 0.0 < target_fpr < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")
        sigmas = sorted(s.disagreement for s in clean_scored)
        # ceil index of the (1 - target_fpr) quantile; clamp into range
        idx = min(len(sigmas) - 1, math.ceil((1.0 - target_fpr) * len(sigmas)) - 1)
        idx = max(0, idx)
        self.threshold = float(sigmas[idx])
        return self.threshold

    def flag(self, scored: Sequence[ScoredResult]) -> list[bool]:
        if self.threshold is None:
            raise RuntimeError("detector not calibrated: call calibrate() or set threshold")
        thr = self.threshold
        # strictly greater so the calibration molecule at the threshold is not itself flagged
        return [s.disagreement > thr for s in scored]

    def audit(self, scored: Sequence[ScoredResult], is_hacker: Sequence[bool]) -> AntiGamingReport:
        """Compare flags against known labels to compute caught-rate and FPR with Wilson CIs."""
        if len(scored) != len(is_hacker):
            raise ValueError("scored and is_hacker must be the same length")
        flags = self.flag(scored)
        n_hackers = sum(1 for h in is_hacker if h)
        n_clean = len(is_hacker) - n_hackers
        caught = sum(1 for f, h in zip(flags, is_hacker) if h and f)
        false_pos = sum(1 for f, h in zip(flags, is_hacker) if (not h) and f)
        caught_rate = caught / n_hackers if n_hackers else 0.0
        fpr = false_pos / n_clean if n_clean else 0.0
        return AntiGamingReport(
            n=len(scored),
            n_hackers=n_hackers,
            n_clean=n_clean,
            caught=caught,
            false_positives=false_pos,
            threshold=float(self.threshold) if self.threshold is not None else float("nan"),
            caught_rate=caught_rate,
            caught_rate_ci=wilson_interval(caught, n_hackers),
            fpr=fpr,
            fpr_ci=wilson_interval(false_pos, n_clean),
        )
