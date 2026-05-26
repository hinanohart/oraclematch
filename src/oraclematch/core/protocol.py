"""The oracle-agnostic data model and the ``Predictor`` protocol.

Sign convention (IMPORTANT). Every backend MUST return ``PredictionResult.affinity`` in a
"**higher is better**" convention. Real oracles disagree on native units and sign:
Boltz-2 affinity is a (log-pIC50-like) score where higher is better, while AutoDock-Vina
returns a docking energy in kcal/mol where *lower* (more negative) is better. The Vina
backend therefore negates its score before returning it. Cross-oracle disagreement
(``sigma_a``) is only meaningful after both a sign convention (here) and a per-oracle
rank-normalization (:mod:`oraclematch.core.normalize`) are applied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Molecule:
    """A candidate molecule.

    ``smiles`` is the chemical representation consumed by real backends (Boltz-2, Vina).
    ``genome`` is an abstract integer vector used by the GPU-free :class:`MockPredictor`
    and by deterministic mutation, so the *evolutionary and calibration machinery* — which
    is where the novelty lives — is fully testable without any chemistry toolkit or GPU.
    """

    smiles: str = ""
    genome: tuple[int, ...] = ()
    meta: dict = field(default_factory=dict, compare=False)

    def __hash__(self) -> int:
        return hash((self.smiles, self.genome))


@dataclass(frozen=True)
class PredictionResult:
    """One oracle's prediction for one molecule.

    affinity:
        Binding strength in a "higher is better" convention (oracle-native scale; it is
        rank-normalized across the population before disagreement is computed).
    confidence:
        Physical/structural plausibility in ``[0, 1]`` (e.g. a PoseBusters pass fraction
        or a confidence/pLDDT-like value). Feeds the multiplicative validity gate.
    """

    oracle_id: str
    affinity: float
    confidence: float = 1.0
    raw: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r} for oracle {self.oracle_id!r}"
            )


@runtime_checkable
class Predictor(Protocol):
    """Structural protocol every oracle backend implements.

    Implementations: :class:`oraclematch.backends.mock.MockPredictor` (deterministic,
    GPU-free), :class:`oraclematch.backends.vina.VinaPredictor` (classical docking, CPU),
    :class:`oraclematch.backends.boltz2.Boltz2Predictor` (DL co-folding, GPU).
    """

    oracle_id: str

    def predict(self, mol: Molecule) -> PredictionResult:
        """Score one molecule."""
        ...

    def predict_batch(self, mols: Sequence[Molecule]) -> list[PredictionResult]:
        """Score many molecules (backends may override for batched compute)."""
        ...

    def is_available(self) -> bool:
        """True if this backend's dependencies/weights are importable and usable now."""
        ...

    @property
    def max_ligand_atoms(self) -> int | None:
        """Hard ligand-size limit, or ``None`` if unbounded (Boltz-2 is ~50)."""
        ...
