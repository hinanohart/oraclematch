"""The calibrated ensemble fitness and its cross-oracle disagreement penalty.

This is the intellectual core of oraclematch. Given, for one population of molecules, the
predictions of K heterogeneous oracles, we compute for each molecule x::

    F(x) = a_bar(x)  -  z * sigma_a(x) / sqrt(K)  -  mu * max(0, delta - c_bar(x))

where

    a_bar(x)   = mean, across oracles, of the *per-oracle rank-normalized* affinity
    sigma_a(x) = std,  across oracles, of the *per-oracle rank-normalized* affinity   <-- penalty core
    c_bar(x)   = mean, across oracles, of the confidence in [0, 1]
    delta      = confidence floor below which the multiplicative-style gate bites
    z, mu      = penalty weights (config; calibrated by the KC-2 pilot in a GPU env)

The single defensible novelty is that ``sigma_a`` — the *cross-paradigm* disagreement
between a DL co-folder and a classical docker — is used as an explicit, lower-confidence-bound
style fitness penalty inside quality-diversity (MAP-Elites) search, to structurally suppress
single-oracle reward hacking. ``sigma_a`` is computed ONLY after per-oracle rank-normalization
(see :mod:`oraclematch.core.normalize`); computing it on raw scores is a units artifact and is
treated as a correctness bug, not a tuning choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from oraclematch.core.normalize import normalize
from oraclematch.core.protocol import Molecule, PredictionResult


@dataclass(frozen=True)
class ScoredResult:
    """A molecule's calibrated fitness with an uncertainty interval.

    ``fitness_ci_low/high`` is a bootstrap interval obtained by resampling the K per-oracle
    normalized affinities. It is honest about small K: with K=2 it is wide and crude (there are
    only a handful of distinct resamples) and is a *diagnostic*, not the headline statistic. The
    statistic used for improvement claims is the population-level bootstrap in
    :mod:`oraclematch.experiments.controls`.
    """

    molecule: Molecule
    fitness: float
    fitness_ci_low: float
    fitness_ci_high: float
    mean_affinity: float
    disagreement: float
    mean_confidence: float
    per_oracle: dict[str, float] = field(default_factory=dict)


class EnsembleCalibrator:
    """Turns per-oracle predictions over a population into calibrated :class:`ScoredResult`s."""

    def __init__(
        self,
        oracle_ids: list[str],
        *,
        z: float = 1.0,
        mu: float = 1.0,
        delta: float = 0.5,
        normalize_method: str = "rank",
        bootstrap_n: int = 1000,
        seed: int = 0,
    ) -> None:
        if len(oracle_ids) < 1:
            raise ValueError("at least one oracle is required")
        if len(set(oracle_ids)) != len(oracle_ids):
            raise ValueError(f"duplicate oracle ids: {oracle_ids}")
        self.oracle_ids = list(oracle_ids)
        self.k = len(self.oracle_ids)
        self.z = float(z)
        self.mu = float(mu)
        self.delta = float(delta)
        self.normalize_method = normalize_method
        self.bootstrap_n = int(bootstrap_n)
        self.seed = int(seed)

    def score_population(
        self, results_by_oracle: dict[str, list[PredictionResult]]
    ) -> list[ScoredResult]:
        """Score a whole population at once (normalization is population-relative).

        ``results_by_oracle`` maps each oracle id to a list of :class:`PredictionResult`, one
        per molecule, all lists aligned to the same molecule order. Raises on empty input or
        ragged/missing oracle columns.
        """
        if not results_by_oracle:
            raise ValueError("results_by_oracle is empty")
        missing = set(self.oracle_ids) - set(results_by_oracle)
        if missing:
            raise ValueError(f"missing predictions for oracles: {sorted(missing)}")

        columns = [results_by_oracle[oid] for oid in self.oracle_ids]
        n = len(columns[0])
        if n == 0:
            raise ValueError("population is empty")
        if any(len(c) != n for c in columns):
            raise ValueError("ragged input: every oracle must score every molecule")

        # (K, N) raw affinity / confidence matrices.
        raw_aff = np.array([[r.affinity for r in col] for col in columns], dtype=float)
        conf = np.array([[r.confidence for r in col] for col in columns], dtype=float)

        # CRITICAL: rank-normalize *within each oracle* (per row) before any cross-oracle stat.
        norm_aff = np.vstack([normalize(raw_aff[i], self.normalize_method) for i in range(self.k)])

        a_bar = norm_aff.mean(axis=0)  # (N,)
        sigma_a = norm_aff.std(axis=0)  # (N,) — disagreement penalty core
        c_bar = conf.mean(axis=0)  # (N,)

        gate = self.mu * np.maximum(0.0, self.delta - c_bar)
        fitness = a_bar - self.z * sigma_a / np.sqrt(self.k) - gate

        ci_low, ci_high = self._bootstrap_affinity_ci(norm_aff)

        # recover the molecule objects from the first oracle's results' raw payload if present;
        # otherwise rebuild a placeholder. Backends attach the molecule under raw["molecule"].
        mols = self._extract_molecules(columns[0])

        out: list[ScoredResult] = []
        for j in range(n):
            out.append(
                ScoredResult(
                    molecule=mols[j],
                    fitness=float(fitness[j]),
                    fitness_ci_low=float(ci_low[j]),
                    fitness_ci_high=float(ci_high[j]),
                    mean_affinity=float(a_bar[j]),
                    disagreement=float(sigma_a[j]),
                    mean_confidence=float(c_bar[j]),
                    per_oracle={
                        oid: float(norm_aff[i, j]) for i, oid in enumerate(self.oracle_ids)
                    },
                )
            )
        return out

    def _bootstrap_affinity_ci(
        self, norm_aff: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Percentile bootstrap of the across-oracle mean, resampling oracles with replacement.

        Returns ``(low, high)`` arrays of shape (N,). With K=2 this is intentionally crude; see
        :class:`ScoredResult`.
        """
        k = norm_aff.shape[0]
        if k == 1:
            v = norm_aff[0]
            return v.copy(), v.copy()
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, k, size=(self.bootstrap_n, k))
        # (bootstrap_n, N): mean over resampled oracles for every molecule
        means = norm_aff[idx, :].mean(axis=1)
        low = np.quantile(means, alpha / 2, axis=0)
        high = np.quantile(means, 1 - alpha / 2, axis=0)
        return low, high

    @staticmethod
    def _extract_molecules(col: list[PredictionResult]) -> list[Molecule]:
        mols: list[Molecule] = []
        for r in col:
            m = r.raw.get("molecule") if isinstance(r.raw, dict) else None
            mols.append(m if isinstance(m, Molecule) else Molecule(smiles="", genome=()))
        return mols
