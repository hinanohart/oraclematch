"""Command-line entry point: ``oraclematch demo [--controls] [--ablation]``.

Everything here runs on the deterministic mock backend (no GPU, no network). Output is JSON with an
explicit ``"backend": "mock"`` field so results can never be mistaken for real-oracle measurements.
"""

from __future__ import annotations

import argparse
import json
import sys

from oraclematch import __version__
from oraclematch.backends import make_oracle_pair
from oraclematch.calibration import EnsembleCalibrator
from oraclematch.evolution import MapElites
from oraclematch.experiments.controls import control_a, control_b


def _demo(seed: int, generations: int, population: int) -> dict:
    dl, dock, _ = make_oracle_pair(seed=seed, shared_fraction=0.5)
    calib = EnsembleCalibrator([dl.oracle_id, dock.oracle_id], z=1.0, mu=1.0, delta=0.5)
    me = MapElites([dl, dock], calib, seed=seed)
    me.run(generations=generations, population=population)
    best = me.best()
    return {
        "experiment": "demo_map_elites",
        "backend": "mock",
        "oracles": [dl.oracle_id, dock.oracle_id],
        "generations": generations,
        "population": population,
        "n_evaluations": me.n_evaluations,
        "coverage_cells": me.coverage(),
        "n_elites": len(me.all_elites()),
        "best_fitness": round(best.fitness, 4) if best else None,
        "best_disagreement": round(best.disagreement, 4) if best else None,
        "best_mean_affinity": round(best.mean_affinity, 4) if best else None,
    }


def _ablation(seed: int, generations: int, population: int) -> dict:
    dl, dock, _ = make_oracle_pair(seed=seed, shared_fraction=0.5)
    calib = EnsembleCalibrator([dl.oracle_id, dock.oracle_id], z=1.0, mu=1.0, delta=0.5)
    qd = MapElites([dl, dock], calib, qd=True, seed=seed)
    qd.run(generations=generations, population=population)
    ga = MapElites([dl, dock], calib, qd=False, n_islands=1, seed=seed)
    ga.run(generations=generations, population=population)
    qb, gb = qd.best(), ga.best()
    assert qb is not None and gb is not None  # archives are non-empty after run()
    return {
        "experiment": "ablation_qd_vs_ga",
        "backend": "mock",
        "note": "QD (MAP-Elites) vs plain (mu+lambda) GA, same disagreement penalty",
        "qd": {"coverage_cells": qd.coverage(), "best_fitness": round(qb.fitness, 4)},
        "ga": {"coverage_cells": ga.coverage(), "best_fitness": round(gb.fitness, 4)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oraclematch", description=__doc__)
    parser.add_argument("command", nargs="?", default="demo", choices=["demo", "version"])
    parser.add_argument("--controls", action="store_true", help="run Control A and Control B")
    parser.add_argument("--ablation", action="store_true", help="run the QD-vs-GA ablation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--population", type=int, default=24)
    args = parser.parse_args(argv)

    if args.command == "version":
        print(json.dumps({"oraclematch": __version__}))
        return 0

    out: dict = {"oraclematch": __version__, "results": []}
    out["results"].append(_demo(args.seed, args.generations, args.population))
    if args.ablation:
        out["results"].append(_ablation(args.seed, args.generations, args.population))
    if args.controls:
        out["results"].append(control_a(generations=args.generations, population=args.population))
        out["results"].append(control_b(shared_fraction=0.5))
        out["results"].append(control_b(shared_fraction=0.9))
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
