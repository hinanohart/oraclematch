import numpy as np
import pytest

from oraclematch.backends import (
    Boltz2Predictor,
    MockPredictor,
    VinaPredictor,
    ground_truth_affinity,
    make_oracle_pair,
)
from oraclematch.backends.mock import GENOME_DIM
from oraclematch.core.protocol import Molecule


def test_mock_is_deterministic():
    dl, _, _ = make_oracle_pair(0)
    m = Molecule(genome=(1, 2, 3, 4, 5, 6, 7, 8))
    assert dl.predict(m).affinity == dl.predict(m).affinity
    assert dl.predict(m).confidence == dl.predict(m).confidence


def test_mock_always_available():
    dl, dock, _ = make_oracle_pair(0)
    assert dl.is_available() and dock.is_available()


def test_mock_confidence_in_unit_interval():
    dl, _, _ = make_oracle_pair(0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = tuple(int(x) for x in rng.integers(0, 10, size=GENOME_DIM))
        assert 0.0 <= dl.predict(Molecule(genome=g)).confidence <= 1.0


def test_make_oracle_pair_correlation_is_controllable():
    def realized_rho(sf):
        dl, dock, _ = make_oracle_pair(seed=0, shared_fraction=sf)
        rng = np.random.default_rng(0)
        mols = [
            Molecule(genome=tuple(int(x) for x in rng.integers(0, 10, GENOME_DIM)))
            for _ in range(200)
        ]
        a = np.array([r.affinity for r in dl.predict_batch(mols)])
        b = np.array([r.affinity for r in dock.predict_batch(mols)])
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return np.corrcoef(ra, rb)[0, 1]

    assert realized_rho(0.9) > realized_rho(0.5)  # more shared signal => higher correlation


def test_invalid_shared_fraction_raises():
    with pytest.raises(ValueError):
        make_oracle_pair(shared_fraction=1.5)


def test_weight_wrong_length_raises():
    with pytest.raises(ValueError):
        MockPredictor("x", np.ones(3))


def test_ground_truth_is_deterministic():
    _, _, w = make_oracle_pair(0)
    m = Molecule(genome=(9,) * GENOME_DIM)
    assert ground_truth_affinity(m, w) == ground_truth_affinity(m, w)


def test_boltz2_limit_and_unavailable_here():
    b = Boltz2Predictor("MKV")
    assert b.max_ligand_atoms == 50
    assert b.is_available() is False  # no GPU/boltz in this environment


def test_vina_unavailable_here_and_raises_on_predict():
    v = VinaPredictor("rec.pdbqt", (0.0, 0.0, 0.0))
    assert v.is_available() is False
    with pytest.raises(RuntimeError):
        v.predict(Molecule(smiles="CCO"))


def test_boltz2_predict_raises_when_unavailable():
    b = Boltz2Predictor("MKV")
    with pytest.raises(RuntimeError):
        b.predict(Molecule(smiles="CCO"))
