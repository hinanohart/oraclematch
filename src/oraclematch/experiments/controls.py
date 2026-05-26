"""Controlled synthetic experiments.

These run entirely on the deterministic mock landscape, where each molecule has a *known* ground
truth affinity. That is the whole point: it lets us measure, with no GPU and no confounds, whether
the disagreement penalty actually helps. **Results here are about the mechanism on synthetic data
and say nothing about real binding affinity.**

Control A — search efficiency
    Compare four selection strategies under a matched evaluation budget, scoring each strategy's
    *self-selected* top-k by the hidden ground truth, across many seeds, with a bootstrap CI over
    seeds. Improvement is claimed ONLY when confidence intervals are disjoint.

Control B — anti-gaming
    Inject known single-oracle exploiters, calibrate the detector on clean molecules only, and
    measure caught-rate / FPR with Wilson intervals on a disjoint test set.
"""

from __future__ import annotations

import numpy as np

from oraclematch.audit.antigaming import AntiGamingDetector, AntiGamingReport
from oraclematch.backends.mock import (
    GENOME_DIM,
    ground_truth_affinity,
    make_oracle_pair,
    sample_clean_molecule,
    sample_hacker_molecule,
    sample_random_molecule,
)
from oraclematch.calibration.ensemble import EnsembleCalibrator, ScoredResult
from oraclematch.core.normalize import rank_normalize
from oraclematch.core.protocol import Molecule
from oraclematch.evolution.mapelites import MapElites
from oraclematch.evolution.mutate import random_genome


def _topk_ground_truth(elites: list[ScoredResult], w_true: np.ndarray, k: int) -> float:
    """Mean ground-truth affinity of the k elites the *method itself* ranks best (by fitness)."""
    ranked = sorted(elites, key=lambda s: s.fitness, reverse=True)[:k]
    if not ranked:
        return float("nan")
    return float(np.mean([ground_truth_affinity(s.molecule, w_true) for s in ranked]))


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def _paired_verdict(diffs: np.ndarray, name_a: str, name_b: str, n_boot: int = 1000) -> dict:
    """Paired bootstrap of per-seed differences ``a - b``; honest when the CI spans zero.

    Returns the mean difference, its 95% CI, and a verdict string that only declares a winner when
    the paired CI excludes 0.
    """
    lo, hi = _bootstrap_ci(diffs, n_boot=n_boot, seed=43)
    mean_diff = float(np.mean(diffs)) if diffs.size else float("nan")
    if lo > 0:
        verdict = f"{name_a} > {name_b} (paired 95% CI excludes 0)"
    elif hi < 0:
        verdict = f"{name_b} > {name_a} (paired 95% CI excludes 0)"
    else:
        verdict = f"no measurable difference between {name_a} and {name_b} (paired CI spans 0)"
    return {
        "mean_diff": round(mean_diff, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "verdict": verdict,
    }


def control_a(
    *,
    n_seeds: int = 12,
    generations: int = 15,
    population: int = 24,
    top_k: int = 5,
    shared_fraction: float = 0.5,
    z: float = 1.0,
    mu: float = 1.0,
    delta: float = 0.5,
    n_boot: int = 1000,
) -> dict:
    """Run the four-way search-efficiency comparison. Returns a JSON-serializable dict."""
    methods = ("random", "greedy_single", "ensemble_mean", "oracle_matched")
    per_seed: dict[str, list[float]] = {m: [] for m in methods}

    for seed in range(n_seeds):
        dl, dock, w_true = make_oracle_pair(seed=seed, shared_fraction=shared_fraction)
        budget = generations * population

        # random: sample `budget` molecules, rank by the single DL oracle (naive selection)
        rng = np.random.default_rng(1000 + seed)
        rand_mols = [Molecule(genome=random_genome(rng, GENOME_DIM)) for _ in range(budget)]
        rand_scored = EnsembleCalibrator([dl.oracle_id]).score_population(
            {dl.oracle_id: dl.predict_batch(rand_mols)}
        )
        per_seed["random"].append(_topk_ground_truth(rand_scored, w_true, top_k))

        # greedy_single: optimize the single DL oracle only (no penalty)
        gs = MapElites([dl], EnsembleCalibrator([dl.oracle_id], z=0.0, mu=0.0), seed=seed)
        gs.run(generations=generations, population=population)
        per_seed["greedy_single"].append(_topk_ground_truth(gs.all_elites(), w_true, top_k))

        # ensemble_mean: two oracles, NO disagreement penalty (isolates penalty from averaging)
        em = MapElites(
            [dl, dock],
            EnsembleCalibrator([dl.oracle_id, dock.oracle_id], z=0.0, mu=mu, delta=delta),
            seed=seed,
        )
        em.run(generations=generations, population=population)
        per_seed["ensemble_mean"].append(_topk_ground_truth(em.all_elites(), w_true, top_k))

        # oracle_matched: two oracles WITH the disagreement penalty
        om = MapElites(
            [dl, dock],
            EnsembleCalibrator([dl.oracle_id, dock.oracle_id], z=z, mu=mu, delta=delta),
            seed=seed,
        )
        om.run(generations=generations, population=population)
        per_seed["oracle_matched"].append(_topk_ground_truth(om.all_elites(), w_true, top_k))

    summary: dict[str, dict] = {}
    for m in methods:
        vals = np.array(per_seed[m])
        lo, hi = _bootstrap_ci(vals, n_boot=n_boot, seed=42)
        summary[m] = {
            "seed_means": [round(v, 4) for v in vals.tolist()],
            "grand_mean": round(float(vals.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }
    # Paired analysis is the correct test: every method ran on the SAME per-seed landscape, so
    # pairing cancels landscape-to-landscape variance that swamps the unpaired CIs above.
    om_vals = np.array(per_seed["oracle_matched"])
    verdicts = {
        "penalty_vs_greedy": _paired_verdict(
            om_vals - np.array(per_seed["greedy_single"]), "oracle_matched", "greedy_single", n_boot
        ),
        "penalty_vs_ensemble_mean": _paired_verdict(
            om_vals - np.array(per_seed["ensemble_mean"]), "oracle_matched", "ensemble_mean", n_boot
        ),
    }
    return {
        "experiment": "control_a_search_efficiency",
        "backend": "mock",
        "note": "synthetic landscape with known ground truth; not evidence about real affinity",
        "config": {
            "n_seeds": n_seeds,
            "generations": generations,
            "population": population,
            "top_k": top_k,
            "shared_fraction": shared_fraction,
            "z": z,
            "mu": mu,
            "delta": delta,
        },
        "methods": summary,
        "verdicts": verdicts,
    }


def control_b(
    *,
    n_clean_calib: int = 40,
    n_clean_test: int = 40,
    n_hackers: int = 40,
    n_random: int = 40,
    shared_fraction: float = 0.5,
    target_fpr: float = 0.05,
    seed: int = 0,
) -> dict:
    """Run the anti-gaming audit. Returns a JSON-serializable dict including a Wilson-CI report."""
    dl, dock, w_true = make_oracle_pair(seed=seed, shared_fraction=shared_fraction)
    calib = EnsembleCalibrator([dl.oracle_id, dock.oracle_id])
    rng = np.random.default_rng(seed + 7)

    # Because disagreement (sigma_a) is rank-normalized WITHIN a population, the calibration set,
    # the clean test set, the exploiters and a random background must all be scored TOGETHER so the
    # operating point transfers. The split is by *role* (which molecules set the threshold vs. which
    # are evaluated), not by separate normalizations.
    clean_calib = [sample_clean_molecule(rng, w_true) for _ in range(n_clean_calib)]
    clean_test = [sample_clean_molecule(rng, w_true) for _ in range(n_clean_test)]
    hackers = [sample_hacker_molecule(rng, dl, dock) for _ in range(n_hackers)]
    background = [sample_random_molecule(rng) for _ in range(n_random)]
    all_mols = clean_calib + clean_test + hackers + background
    scored = calib.score_population({o.oracle_id: o.predict_batch(all_mols) for o in (dl, dock)})

    i = 0
    sc_calib = scored[i : i + n_clean_calib]
    i += n_clean_calib
    sc_clean_test = scored[i : i + n_clean_test]
    i += n_clean_test
    sc_hackers = scored[i : i + n_hackers]

    test_scored = sc_clean_test + sc_hackers
    is_hacker = [False] * n_clean_test + [True] * n_hackers

    # Intrinsic inter-oracle Spearman rho, measured over a NEUTRAL random ligand population (not the
    # adversarial audit set, which is anti-correlated by construction). This is the KC-2 analog: the
    # quantity the real GPU pilot would measure to decide whether the disagreement penalty is viable.
    neutral = [sample_random_molecule(rng) for _ in range(256)]
    a_dl = rank_normalize(np.array([r.affinity for r in dl.predict_batch(neutral)]))
    a_dock = rank_normalize(np.array([r.affinity for r in dock.predict_batch(neutral)]))
    rho = float(np.corrcoef(a_dl, a_dock)[0, 1])

    detector = AntiGamingDetector()
    detector.calibrate(sc_calib, target_fpr=target_fpr)  # operating point from clean molecules only
    report: AntiGamingReport = detector.audit(test_scored, is_hacker)

    return {
        "experiment": "control_b_anti_gaming",
        "backend": "mock",
        "note": "synthetic exploiters; caught-rate/FPR are on synthetic data only",
        "config": {
            "shared_fraction": shared_fraction,
            "target_fpr": target_fpr,
            "n_clean_test": n_clean_test,
            "n_hackers": n_hackers,
        },
        "inter_oracle_spearman_rho": round(rho, 3),
        "threshold": round(report.threshold, 4),
        "caught_rate": round(report.caught_rate, 4),
        "caught_rate_ci95": [round(c, 4) for c in report.caught_rate_ci],
        "fpr": round(report.fpr, 4),
        "fpr_ci95": [round(c, 4) for c in report.fpr_ci],
        "counts": {
            "caught": report.caught,
            "n_hackers": report.n_hackers,
            "false_positives": report.false_positives,
            "n_clean": report.n_clean,
        },
    }
