"""A deterministic, GPU-free oracle and a synthetic landscape with *known ground truth*.

This module exists so the disagreement penalty and anti-gaming machinery can be exercised and
tested without any GPU, chemistry toolkit, or network. The synthetic world is honest about what
it is: a controlled setting where the "true" affinity of a molecule is defined *by construction*,
so we can measure whether disagreement-penalized search stays closer to ground truth than greedy
single-oracle search. **None of this is evidence about real binding affinity or drug discovery.**

Construction
------------
A molecule's ``genome`` is a length-``GENOME_DIM`` integer vector. The hidden ground-truth
affinity is a linear function ``w_true . g_hat`` of the normalized genome. Each oracle is a
*biased* view of ground truth: ``w_oracle = w_true + bias_direction``. Two oracles with different
bias directions form a heterogeneous pair — a molecule aligned with one oracle's bias but not the
other scores high on that oracle alone, i.e. it is a single-oracle exploiter ("hacker"), and the
two oracles disagree (large ``sigma_a``) on exactly such molecules.
"""

from __future__ import annotations

import numpy as np

from oraclematch.core.protocol import Molecule, PredictionResult

GENOME_DIM = 8
GENE_MAX = 9  # genes are ints in [0, GENE_MAX]


def _normalized(genome: tuple[int, ...]) -> np.ndarray:
    g = np.asarray(genome, dtype=float)
    if g.size != GENOME_DIM:
        # pad/truncate so arbitrary genomes are still scorable (mutation may vary length)
        g = np.resize(g, GENOME_DIM)
    return np.clip(g, 0, GENE_MAX) / GENE_MAX


class MockPredictor:
    """A deterministic linear oracle: ``affinity = w . g_hat + bias`` (higher is better).

    Always available; never touches a GPU. ``confidence`` is a deterministic, genome-derived
    value in ``[0, 1]`` so the validity gate can be exercised.
    """

    def __init__(
        self,
        oracle_id: str,
        weight: np.ndarray,
        *,
        bias: float = 0.0,
        confidence_weight: np.ndarray | None = None,
        max_ligand_atoms: int | None = None,
    ) -> None:
        self.oracle_id = oracle_id
        self.weight = np.asarray(weight, dtype=float)
        if self.weight.size != GENOME_DIM:
            raise ValueError(f"weight must have length {GENOME_DIM}")
        self.bias = float(bias)
        self.confidence_weight = (
            np.asarray(confidence_weight, dtype=float)
            if confidence_weight is not None
            else np.ones(GENOME_DIM)
        )
        self._max_ligand_atoms = max_ligand_atoms

    def predict(self, mol: Molecule) -> PredictionResult:
        g = _normalized(mol.genome)
        affinity = float(self.weight @ g + self.bias)
        conf_latent = float(self.confidence_weight @ g) / GENOME_DIM
        confidence = float(1.0 / (1.0 + np.exp(-4.0 * (conf_latent - 0.5))))  # logistic in (0,1)
        return PredictionResult(
            oracle_id=self.oracle_id,
            affinity=affinity,
            confidence=confidence,
            raw={"molecule": mol},
        )

    def predict_batch(self, mols):
        return [self.predict(m) for m in mols]

    def is_available(self) -> bool:
        return True

    @property
    def max_ligand_atoms(self) -> int | None:
        return self._max_ligand_atoms


def _orthonormal_basis(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` orthonormal vectors in R^GENOME_DIM (rows), deterministic in ``rng``."""
    mat = rng.normal(0, 1, size=(GENOME_DIM, n))
    q, _ = np.linalg.qr(mat)
    return q.T[:n]


def make_oracle_pair(
    seed: int = 0, shared_fraction: float = 0.5
) -> tuple[MockPredictor, MockPredictor, np.ndarray]:
    """Build a heterogeneous oracle pair plus the hidden ground-truth weight vector.

    The two oracles are built on an orthonormal basis ``(u_true, u_a, u_b)`` so their correlation
    is *set explicitly*: each weight is ``w = sqrt(shared)·u_true + sqrt(1-shared)·u_private``. With
    orthonormal directions the population-level correlation between the oracles is approximately
    ``shared_fraction``. ``shared_fraction`` near 1 reproduces the KC-2 failure mode (oracles agree,
    the disagreement penalty is meaningless); the default 0.5 keeps them genuinely heterogeneous so
    single-oracle exploiters exist and are catchable. Deterministic in ``seed``.

    Returns ``(oracle_dl, oracle_dock, w_true)``.
    """
    if not 0.0 <= shared_fraction <= 1.0:
        raise ValueError("shared_fraction must be in [0, 1]")
    rng = np.random.default_rng(seed)
    basis = _orthonormal_basis(rng, 3)
    u_true, u_a, u_b = basis[0], basis[1], basis[2]
    s, p = np.sqrt(shared_fraction), np.sqrt(1.0 - shared_fraction)
    scale = float(np.sqrt(GENOME_DIM))  # keep affinities on an O(1) scale
    w_true = scale * u_true
    w_a = scale * (s * u_true + p * u_a)
    w_b = scale * (s * u_true + p * u_b)
    oracle_dl = MockPredictor("mock-dl", w_a, max_ligand_atoms=50)
    oracle_dock = MockPredictor("mock-dock", w_b)
    return oracle_dl, oracle_dock, w_true


def ground_truth_affinity(mol: Molecule, w_true: np.ndarray) -> float:
    """The by-construction true affinity used to *evaluate* search quality (never as an oracle)."""
    return float(np.asarray(w_true, dtype=float) @ _normalized(mol.genome))


def sample_clean_molecule(rng: np.random.Generator, w_true: np.ndarray) -> Molecule:
    """A genuinely good molecule: a bang-bang genome aligned with ground truth.

    It maximizes ``w_true · g``, so it scores well on the *shared* component both oracles agree on,
    giving similar ranks on both oracles and therefore low disagreement.
    """
    base = np.where(np.asarray(w_true) > 0.0, GENE_MAX, 0)
    jitter = rng.integers(-1, 2, size=GENOME_DIM)
    genome = tuple(int(np.clip(b + j, 0, GENE_MAX)) for b, j in zip(base, jitter))
    return Molecule(smiles="", genome=genome, meta={"label": "clean"})


def sample_hacker_molecule(
    rng: np.random.Generator, oracle_high: MockPredictor, oracle_low: MockPredictor
) -> Molecule:
    """A single-oracle exploiter that scores high on ``oracle_high`` and low on ``oracle_low``.

    We pick a bang-bang genome that maximizes the *raw gap* ``w_high · g − w_low · g``: gene ``i``
    is switched on (``GENE_MAX``) where ``w_high[i] > w_low[i]`` and off otherwise. Such a molecule
    ranks high on one oracle and low on the other within a varied population, producing a large
    ``sigma_a`` — the adversary the anti-gaming audit must catch. A little jitter yields distinct
    individuals without erasing the exploit.
    """
    gap = np.asarray(oracle_high.weight) - np.asarray(oracle_low.weight)
    base = np.where(gap > 0.0, GENE_MAX, 0)
    jitter = rng.integers(-1, 2, size=GENOME_DIM)
    genome = tuple(int(np.clip(b + j, 0, GENE_MAX)) for b, j in zip(base, jitter))
    return Molecule(smiles="", genome=genome, meta={"label": "hacker"})


def sample_random_molecule(rng: np.random.Generator) -> Molecule:
    """A uniformly random genome, used to give a population a meaningful rank spread."""
    genome = tuple(int(x) for x in rng.integers(0, GENE_MAX + 1, size=GENOME_DIM))
    return Molecule(smiles="", genome=genome, meta={"label": "random"})
