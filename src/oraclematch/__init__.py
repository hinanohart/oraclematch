"""oraclematch — cross-paradigm oracle-disagreement as a calibrated fitness penalty
for MAP-Elites molecular search.

PRE-ALPHA (v0.1.0a1). This release ships the *framework* and a *synthetic/mock*
demonstration only. It makes NO drug-discovery or therapeutic-efficacy claim, and the
cross-oracle correlation pilot that gates the scientific novelty claim (KC-2) has NOT
been run in the release environment (no GPU). See README "Status & honest limitations".
"""

from oraclematch.calibration.ensemble import EnsembleCalibrator, ScoredResult
from oraclematch.core.normalize import rank_normalize, zscore_normalize
from oraclematch.core.protocol import Molecule, PredictionResult, Predictor

__version__ = "0.1.0a1"

__all__ = [
    "Molecule",
    "PredictionResult",
    "Predictor",
    "rank_normalize",
    "zscore_normalize",
    "EnsembleCalibrator",
    "ScoredResult",
    "__version__",
]
