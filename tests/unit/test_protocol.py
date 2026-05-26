import pytest

from oraclematch.backends import MockPredictor, make_oracle_pair
from oraclematch.core.protocol import Molecule, PredictionResult, Predictor


def test_molecule_is_hashable_and_frozen():
    m = Molecule(smiles="CCO", genome=(1, 2, 3))
    assert hash(m) == hash(Molecule(smiles="CCO", genome=(1, 2, 3)))
    with pytest.raises(AttributeError):  # FrozenInstanceError subclasses AttributeError
        m.smiles = "X"  # frozen


def test_meta_excluded_from_equality():
    a = Molecule(genome=(1, 2), meta={"label": "clean"})
    b = Molecule(genome=(1, 2), meta={"label": "hacker"})
    assert a == b  # meta does not participate in equality/compare


def test_prediction_result_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        PredictionResult(oracle_id="x", affinity=1.0, confidence=1.5)
    with pytest.raises(ValueError):
        PredictionResult(oracle_id="x", affinity=1.0, confidence=-0.1)


def test_prediction_result_accepts_boundary_confidence():
    PredictionResult(oracle_id="x", affinity=1.0, confidence=0.0)
    PredictionResult(oracle_id="x", affinity=1.0, confidence=1.0)


def test_mock_predictor_satisfies_protocol_at_runtime():
    dl, _, _ = make_oracle_pair(0)
    assert isinstance(dl, Predictor)
    assert isinstance(dl, MockPredictor)
