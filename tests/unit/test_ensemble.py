import numpy as np
import pytest

from oraclematch.backends import (
    make_oracle_pair,
    sample_clean_molecule,
    sample_hacker_molecule,
    sample_random_molecule,
)
from oraclematch.calibration import EnsembleCalibrator
from oraclematch.core.protocol import Molecule, PredictionResult


def _col(oracle_id, affs, confs=None):
    confs = confs if confs is not None else [1.0] * len(affs)
    return [
        PredictionResult(oracle_id, a, c, raw={"molecule": Molecule(genome=(i,))})
        for i, (a, c) in enumerate(zip(affs, confs))
    ]


def test_empty_input_raises():
    with pytest.raises(ValueError):
        EnsembleCalibrator(["a"]).score_population({})


def test_missing_oracle_raises():
    calib = EnsembleCalibrator(["a", "b"])
    with pytest.raises(ValueError):
        calib.score_population({"a": _col("a", [1.0, 2.0])})


def test_ragged_input_raises():
    calib = EnsembleCalibrator(["a", "b"])
    with pytest.raises(ValueError):
        calib.score_population({"a": _col("a", [1.0, 2.0]), "b": _col("b", [1.0])})


def test_duplicate_oracle_ids_raise():
    with pytest.raises(ValueError):
        EnsembleCalibrator(["a", "a"])


def test_sigma_zero_when_oracles_agree_on_ranking_despite_scale():
    """THE core invariant: disagreement is computed AFTER per-oracle rank-normalization.

    Oracle B reports the same ordering as A but on a 1000x larger, shifted scale. On raw scores the
    cross-oracle std would be enormous; after rank-normalization the two agree perfectly, so sigma_a
    must be ~0. This guards against the units-artifact bug.
    """
    a = [0.0, 1.0, 2.0, 3.0, 4.0]
    b = [500.0, 1500.0, 2500.0, 3500.0, 4500.0]  # == 1000*a + 500, identical ranking
    calib = EnsembleCalibrator(["a", "b"], z=1.0, mu=0.0)
    scored = calib.score_population({"a": _col("a", a), "b": _col("b", b)})
    assert max(s.disagreement for s in scored) < 1e-9


def test_sigma_high_when_oracles_rank_oppositely():
    a = [0.0, 1.0, 2.0, 3.0]
    b = [3.0, 2.0, 1.0, 0.0]  # reversed ranking => maximal disagreement
    calib = EnsembleCalibrator(["a", "b"], z=1.0, mu=0.0)
    scored = calib.score_population({"a": _col("a", a), "b": _col("b", b)})
    assert min(s.disagreement for s in scored) > 0.0
    # the extreme-rank molecules disagree most (norm 0 vs 1 => std 0.5)
    assert max(s.disagreement for s in scored) == pytest.approx(0.5, abs=1e-9)


def test_fitness_equals_mean_affinity_when_no_disagreement_no_gate():
    a = [0.0, 1.0, 2.0, 3.0]
    calib = EnsembleCalibrator(["a", "b"], z=1.0, mu=1.0, delta=0.5)
    scored = calib.score_population({"a": _col("a", a), "b": _col("b", a)})
    # identical oracles, all confidence 1.0 -> sigma 0, gate 0 -> fitness == normalized affinity
    for s in scored:
        assert s.fitness == pytest.approx(s.mean_affinity, abs=1e-9)
        assert s.disagreement == pytest.approx(0.0, abs=1e-9)


def test_confidence_gate_penalizes_low_confidence():
    a = [0.0, 1.0, 2.0, 3.0]
    confs = [0.0, 0.0, 0.0, 0.0]  # well below delta
    calib = EnsembleCalibrator(["a", "b"], z=0.0, mu=2.0, delta=0.5)
    scored = calib.score_population({"a": _col("a", a, confs), "b": _col("b", a, confs)})
    # fitness = a_bar - mu*(delta - 0) = a_bar - 2*0.5 = a_bar - 1.0
    for s in scored:
        assert s.fitness == pytest.approx(s.mean_affinity - 1.0, abs=1e-9)


def test_z_weight_scales_penalty():
    a = [0.0, 1.0, 2.0, 3.0]
    b = [3.0, 2.0, 1.0, 0.0]
    low = EnsembleCalibrator(["a", "b"], z=0.0, mu=0.0).score_population(
        {"a": _col("a", a), "b": _col("b", b)}
    )
    high = EnsembleCalibrator(["a", "b"], z=4.0, mu=0.0).score_population(
        {"a": _col("a", a), "b": _col("b", b)}
    )
    # with z=0 fitness==mean; with z>0 fitness is strictly lower wherever disagreement>0
    for lo, hi in zip(low, high):
        if lo.disagreement > 0:
            assert hi.fitness < lo.fitness


def test_bootstrap_ci_brackets_or_equals_mean():
    a = [0.0, 1.0, 2.0, 3.0]
    b = [1.0, 0.0, 3.0, 2.0]
    scored = EnsembleCalibrator(["a", "b"], bootstrap_n=500).score_population(
        {"a": _col("a", a), "b": _col("b", b)}
    )
    for s in scored:
        assert s.fitness_ci_low <= s.mean_affinity + 1e-9
        assert s.fitness_ci_high >= s.mean_affinity - 1e-9


def test_single_oracle_has_zero_disagreement():
    a = [0.0, 1.0, 2.0, 3.0]
    scored = EnsembleCalibrator(["a"]).score_population({"a": _col("a", a)})
    assert all(s.disagreement == 0.0 for s in scored)


def test_score_population_recovers_molecules():
    calib = EnsembleCalibrator(["a", "b"])
    scored = calib.score_population({"a": _col("a", [1.0, 2.0]), "b": _col("b", [2.0, 1.0])})
    assert [s.molecule.genome for s in scored] == [(0,), (1,)]


def test_hacker_has_higher_disagreement_than_clean():
    dl, dock, w = make_oracle_pair(seed=0, shared_fraction=0.5)
    calib = EnsembleCalibrator([dl.oracle_id, dock.oracle_id])
    rng = np.random.default_rng(0)
    pop = (
        [sample_clean_molecule(rng, w) for _ in range(8)]
        + [sample_hacker_molecule(rng, dl, dock) for _ in range(8)]
        + [sample_random_molecule(rng) for _ in range(16)]
    )
    labels = ["clean"] * 8 + ["hacker"] * 8 + ["random"] * 16
    scored = calib.score_population({o.oracle_id: o.predict_batch(pop) for o in (dl, dock)})
    clean = np.mean([s.disagreement for s, lab in zip(scored, labels) if lab == "clean"])
    hacker = np.mean([s.disagreement for s, lab in zip(scored, labels) if lab == "hacker"])
    assert hacker > clean
