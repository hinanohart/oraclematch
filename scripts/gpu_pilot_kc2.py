#!/usr/bin/env python
"""KC-2 pilot: measure the inter-oracle correlation that gates oraclematch's whole premise.

The disagreement penalty is only meaningful if the two oracles are NOT near-perfectly correlated.
This script docks/folds a small set of ligands against a target with BOTH Boltz-2 and AutoDock-Vina
and reports the Spearman correlation of their (sign-unified, rank-normalized) affinities.

    PROCEED if rho < 0.7   (oracles genuinely heterogeneous; penalty is meaningful)
    CAUTION if 0.7-0.9     (lower z; document honestly)
    ABORT   if rho > 0.9   (swap the second oracle for a more orthogonal engine)

This requires a GPU + the 'boltz' and 'vina' extras + a prepared receptor, so it is NOT run in CI
and was NOT run for the v0.1.0a1 release (CPU-only build env). It is the documented prerequisite for
the empirical novelty claim. Run it before wiring live inference in v0.1.1.

Usage:
    python scripts/gpu_pilot_kc2.py --target TARGET.fasta --receptor REC.pdbqt \\
        --center X Y Z --ligands ligands.smi
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from oraclematch.backends import Boltz2Predictor, VinaPredictor
from oraclematch.core.normalize import rank_normalize
from oraclematch.core.protocol import Molecule


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = rank_normalize(np.asarray(a))
    rb = rank_normalize(np.asarray(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True, help="target FASTA for Boltz-2")
    p.add_argument("--receptor", required=True, help="prepared receptor .pdbqt for Vina")
    p.add_argument("--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--ligands", required=True, help="SMILES file, one ligand per line")
    args = p.parse_args(argv)

    boltz = Boltz2Predictor(open(args.target).read().strip())
    vina = VinaPredictor(args.receptor, tuple(args.center))
    for oracle in (boltz, vina):
        if not oracle.is_available():
            print(f"[KC-2] {oracle.oracle_id} unavailable — install extras and provide a GPU.")
            return 2

    mols = [Molecule(smiles=s.strip()) for s in open(args.ligands) if s.strip()]
    a = np.array([boltz.predict(m).affinity for m in mols])
    b = np.array([vina.predict(m).affinity for m in mols])
    rho = spearman(a, b)
    verdict = "PROCEED" if rho < 0.7 else ("CAUTION" if rho <= 0.9 else "ABORT")
    print(f"[KC-2] n={len(mols)} Spearman rho(Boltz-2, Vina) = {rho:.3f} -> {verdict}")
    return 0 if verdict != "ABORT" else 1


if __name__ == "__main__":
    sys.exit(main())
