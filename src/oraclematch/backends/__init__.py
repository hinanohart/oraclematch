"""Oracle backends. ``MockPredictor`` is GPU-free and always available; ``VinaPredictor``
(classical docking, CPU) and ``Boltz2Predictor`` (DL co-folding, GPU) are import-guarded and
report ``is_available() == False`` unless their optional dependencies are installed."""

from oraclematch.backends.boltz2 import Boltz2Predictor
from oraclematch.backends.mock import (
    MockPredictor,
    ground_truth_affinity,
    make_oracle_pair,
    sample_clean_molecule,
    sample_hacker_molecule,
    sample_random_molecule,
)
from oraclematch.backends.vina import VinaPredictor

__all__ = [
    "MockPredictor",
    "VinaPredictor",
    "Boltz2Predictor",
    "make_oracle_pair",
    "ground_truth_affinity",
    "sample_clean_molecule",
    "sample_hacker_molecule",
    "sample_random_molecule",
]
