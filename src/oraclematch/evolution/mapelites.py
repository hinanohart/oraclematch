"""MAP-Elites quality-diversity search over molecules, scored by the calibrated ensemble.

Design notes
------------
* The calibrated fitness is *population-relative* (rank-normalization happens within a batch). To
  keep fitness internally consistent, each generation re-evaluates the union of current elites and
  new offspring **together**, then refills the archive. Final search quality is reported elsewhere
  by an external ground-truth metric, so this batch-relativity never leaks into a claim.
* Setting ``qd=False`` collapses the behavior archive into a single elitist population — a plain
  (mu+lambda) GA — which is exactly the ablation prior-art review asked for (QD vs. GA, same
  disagreement penalty).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from oraclematch.backends.mock import GENE_MAX, GENOME_DIM
from oraclematch.calibration.ensemble import EnsembleCalibrator, ScoredResult
from oraclematch.core.protocol import Molecule, Predictor
from oraclematch.evolution.mutate import mutate_genome, random_genome

Descriptor = Callable[[Molecule], tuple[int, ...]]


def default_descriptor(mol: Molecule, bins: int = 8, gene_max: int = GENE_MAX) -> tuple[int, int]:
    """A 2-D behavior descriptor from the genome: (fraction of 'on' genes, mean gene level)."""
    g = np.asarray(mol.genome, dtype=float)
    if g.size == 0:
        return (0, 0)
    frac_on = float((g > gene_max / 2).mean())
    mean_level = float(g.mean()) / gene_max
    b0 = min(int(frac_on * bins), bins - 1)
    b1 = min(int(mean_level * bins), bins - 1)
    return (b0, b1)


class MapElites:
    """Illuminates the behavior space with molecules, ranked by the disagreement-penalized fitness."""

    def __init__(
        self,
        predictors: Sequence[Predictor],
        calibrator: EnsembleCalibrator,
        *,
        descriptor_fn: Descriptor | None = None,
        n_islands: int = 4,
        bins: int = 8,
        gene_max: int = GENE_MAX,
        genome_dim: int = GENOME_DIM,
        mutation_rate: float = 0.3,
        migration_interval: int = 5,
        migration_size: int = 2,
        qd: bool = True,
        seed: int = 0,
    ) -> None:
        if not predictors:
            raise ValueError("at least one predictor is required")
        self.predictors = list(predictors)
        self.calibrator = calibrator
        self.descriptor_fn = descriptor_fn or (lambda m: default_descriptor(m, bins, gene_max))
        self.n_islands = max(1, n_islands)
        self.bins = bins
        self.gene_max = gene_max
        self.genome_dim = genome_dim
        self.mutation_rate = mutation_rate
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.qd = qd
        self.rng = np.random.default_rng(seed)
        self.archives: list[dict[tuple[int, ...], ScoredResult]] = [
            {} for _ in range(self.n_islands)
        ]
        self.n_evaluations = 0
        self.history: list[float] = []

    # -- evaluation -----------------------------------------------------------------
    def _evaluate(self, mols: list[Molecule]) -> list[ScoredResult]:
        if not mols:
            return []
        results_by_oracle = {p.oracle_id: p.predict_batch(mols) for p in self.predictors}
        self.n_evaluations += len(mols) * len(self.predictors)
        return self.calibrator.score_population(results_by_oracle)

    def _place(self, island: int, scored: list[ScoredResult]) -> None:
        archive = self.archives[island]
        for s in scored:
            if self.qd:
                cell = self.descriptor_fn(s.molecule)
            else:
                cell = (-1, -1)  # single sentinel cell => elitist (mu+lambda) population
            cur = archive.get(cell)
            if cur is None or s.fitness > cur.fitness:
                archive[cell] = s

    # -- main loop ------------------------------------------------------------------
    def run(self, generations: int = 20, population: int = 32) -> list[ScoredResult]:
        for isl in range(self.n_islands):
            init = [
                Molecule(genome=random_genome(self.rng, self.genome_dim, self.gene_max))
                for _ in range(population)
            ]
            self._place(isl, self._evaluate(init))

        for gen in range(generations):
            for isl in range(self.n_islands):
                elites = list(self.archives[isl].values())
                if not elites:
                    continue
                offspring = []
                for _ in range(population):
                    parent = elites[self.rng.integers(0, len(elites))]
                    child_genome = mutate_genome(
                        parent.molecule.genome, self.rng, self.mutation_rate, self.gene_max
                    )
                    offspring.append(Molecule(genome=child_genome))
                # re-evaluate elites + offspring together so normalization is consistent
                union = [e.molecule for e in elites] + offspring
                self.archives[isl] = {}
                self._place(isl, self._evaluate(union))
            if self.qd and self.n_islands > 1 and (gen + 1) % self.migration_interval == 0:
                self._migrate()
            best = self.best()
            self.history.append(best.fitness if best else float("nan"))
        return self.all_elites()

    def _migrate(self) -> None:
        for src in range(self.n_islands):
            dst = (src + 1) % self.n_islands
            donors = sorted(self.archives[src].values(), key=lambda s: s.fitness, reverse=True)
            for s in donors[: self.migration_size]:
                self._place(dst, [s])

    # -- accessors ------------------------------------------------------------------
    def all_elites(self) -> list[ScoredResult]:
        out: list[ScoredResult] = []
        for archive in self.archives:
            out.extend(archive.values())
        return out

    def best(self) -> ScoredResult | None:
        elites = self.all_elites()
        return max(elites, key=lambda s: s.fitness) if elites else None

    def coverage(self) -> int:
        """Number of distinct behavior cells filled across all islands (QD coverage)."""
        cells: set[tuple[int, ...]] = set()
        for archive in self.archives:
            cells.update(archive.keys())
        return len(cells)
