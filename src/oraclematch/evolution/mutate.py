"""Variation operators.

The default operators act on the abstract integer ``genome`` so search is deterministic and
chemistry-free in CI. A chemically-valid SMILES mutation path (RDKit) and a program-level operator
(openevolve) are intended optional extras; they are out of scope for the GPU-free v0.1.0a1 core and
plug in behind the same ``mutate_fn`` signature.
"""

from __future__ import annotations

import numpy as np

from oraclematch.backends.mock import GENE_MAX, GENOME_DIM
from oraclematch.core.protocol import Molecule


def random_genome(
    rng: np.random.Generator, dim: int = GENOME_DIM, gene_max: int = GENE_MAX
) -> tuple[int, ...]:
    return tuple(int(x) for x in rng.integers(0, gene_max + 1, size=dim))


def mutate_genome(
    genome: tuple[int, ...],
    rng: np.random.Generator,
    rate: float = 0.3,
    gene_max: int = GENE_MAX,
) -> tuple[int, ...]:
    """Per-gene resample with probability ``rate``; never a silent no-op.

    At least one gene is always resampled, and if the resample happens to reproduce the parent
    exactly, one gene is shifted to a guaranteed-different value (when ``gene_max > 0``), so a
    mutation never wastes an evaluation on an unchanged genome.
    """
    g = list(genome)
    n = len(g)
    if n == 0:
        return tuple(g)
    mask = rng.random(n) < rate
    if not mask.any():
        mask[rng.integers(0, n)] = True
    for i in range(n):
        if mask[i]:
            g[i] = int(rng.integers(0, gene_max + 1))
    if tuple(g) == genome and gene_max > 0:
        i = int(rng.integers(0, n))
        g[i] = (genome[i] + 1) % (gene_max + 1)
    return tuple(g)


def crossover(
    g1: tuple[int, ...], g2: tuple[int, ...], rng: np.random.Generator
) -> tuple[int, ...]:
    """Uniform crossover of two equal-length genomes."""
    n = min(len(g1), len(g2))
    pick = rng.random(n) < 0.5
    return tuple(int(g1[i] if pick[i] else g2[i]) for i in range(n))


def mutate_molecule(mol: Molecule, rng: np.random.Generator, rate: float = 0.3) -> Molecule:
    return Molecule(smiles=mol.smiles, genome=mutate_genome(mol.genome, rng, rate))
