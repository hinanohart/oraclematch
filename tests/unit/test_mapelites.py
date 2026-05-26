import pytest

from oraclematch.backends import make_oracle_pair
from oraclematch.calibration import EnsembleCalibrator
from oraclematch.evolution import MapElites
from oraclematch.evolution.mapelites import default_descriptor
from oraclematch.evolution.mutate import crossover, mutate_genome, random_genome


def _runner(qd=True, n_islands=3, seed=0):
    dl, dock, _ = make_oracle_pair(seed=seed, shared_fraction=0.5)
    calib = EnsembleCalibrator([dl.oracle_id, dock.oracle_id], z=1.0, mu=1.0, delta=0.5)
    return MapElites([dl, dock], calib, n_islands=n_islands, qd=qd, seed=seed)


def test_requires_at_least_one_predictor():
    with pytest.raises(ValueError):
        MapElites([], EnsembleCalibrator(["a"]))


def test_run_fills_archive_and_counts_evaluations():
    me = _runner()
    elites = me.run(generations=8, population=16)
    assert len(elites) > 0
    assert me.coverage() > 1
    assert me.n_evaluations > 0
    assert me.best() is not None


def test_qd_covers_more_cells_than_ga():
    qd = _runner(qd=True)
    qd.run(generations=8, population=16)
    ga = _runner(qd=False, n_islands=1)
    ga.run(generations=8, population=16)
    assert qd.coverage() > ga.coverage()
    assert ga.coverage() == 1  # GA collapses to a single elitist cell


def test_run_is_deterministic_given_seed():
    a = _runner(seed=3)
    b = _runner(seed=3)
    a.run(generations=6, population=12)
    b.run(generations=6, population=12)
    assert a.best().fitness == b.best().fitness


def test_history_length_matches_generations():
    me = _runner()
    me.run(generations=7, population=12)
    assert len(me.history) == 7


def test_default_descriptor_is_bounded():
    for genome in [(0,) * 8, (9,) * 8, (5, 0, 9, 3, 1, 8, 2, 7)]:
        b0, b1 = default_descriptor(type("M", (), {"genome": genome})())
        assert 0 <= b0 < 8 and 0 <= b1 < 8


def test_mutate_genome_changes_at_least_one_gene():
    import numpy as np

    rng = np.random.default_rng(0)
    g = (5, 5, 5, 5, 5, 5, 5, 5)
    # even at rate 0.0 the contract guarantees at least one gene is resampled
    changed = mutate_genome(g, rng, rate=0.0)
    assert sum(1 for x, y in zip(g, changed) if x != y) >= 1


def test_crossover_and_random_genome_shapes():
    import numpy as np

    rng = np.random.default_rng(0)
    g1, g2 = random_genome(rng), random_genome(rng)
    child = crossover(g1, g2, rng)
    assert len(child) == len(g1)
    assert all(c in (a, b) for a, b, c in zip(g1, g2, child))
