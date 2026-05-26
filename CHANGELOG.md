# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org/) (pre-releases use the PEP 440 `aN` suffix).

## [Unreleased]

Post-release code-quality pass. No behaviour change: the synthetic Control A / Control B
outputs reproduce byte-for-byte, all 60 tests pass, coverage 93%.

### Changed
- Removed the unused `evolve` optional-dependency extra (`openevolve`): it advertised a
  program-level mutation operator that the v0.1.0a1 core does not wire in. The genome-level
  operators remain the only mutation path; a real chemical/program operator is a v0.2 roadmap item.
- Removed the dead, unreferenced `mutate_molecule` helper from `evolution/mutate.py`.
- Eliminated a redundant `argsort` in `core.normalize.rank_normalize` (the per-oracle
  normalization on the σ_a critical path) by threading the existing sort order into `_average_ties`.

## [0.1.0a1] — 2026-05-27

First public pre-alpha. Ships the **framework** and a **deterministic, GPU-free synthetic
demonstration** only. Makes no drug-discovery or therapeutic-efficacy claim.

### Added
- `Predictor` protocol and `Molecule` / `PredictionResult` data model with a documented
  "higher is better" affinity sign convention.
- `EnsembleCalibrator`: the calibrated fitness `F = ā − z·σ_a/√K − μ·max(0, δ − c̄)` with the
  cross-oracle disagreement `σ_a` computed **after** per-oracle rank-normalization, plus a
  bootstrap CI per molecule.
- Backends: `MockPredictor` (deterministic, GPU-free, with a synthetic landscape of known ground
  truth and tunable oracle correlation); `VinaPredictor` and `Boltz2Predictor` (import-guarded,
  provided but **not exercised** in this CPU-only release).
- `MapElites` quality-diversity search with island migration; `qd=False` collapses to a
  `(mu+lambda)` GA for the QD-vs-GA ablation.
- `AntiGamingDetector`: detects single-oracle exploiters via disagreement, reporting caught-rate /
  FPR with Wilson intervals from an operating point calibrated on clean molecules only.
- Controlled experiments (`control_a`, `control_b`) and an `oraclematch` CLI demo.
- `scripts/gpu_pilot_kc2.py`: the deferred KC-2 oracle-correlation pilot (requires a GPU).

### Known limitations
- **KC-2 not run.** The real-oracle (Boltz-2 × Vina) correlation pilot that gates the empirical
  novelty claim requires a GPU and is deferred to v0.1.1.
- All reported numbers are on the synthetic mock landscape.
- The disagreement penalty's search-efficiency edge over plain two-oracle averaging is **not**
  statistically measurable on the synthetic landscape; its demonstrated value is anti-gaming
  robustness, which degrades as oracle correlation rises.

### Roadmap
- **v0.1.1** — run KC-2 on GPU; wire live Boltz-2 + Vina inference.
- **v0.2** — VeriEvolve-Bio: reuse `scorewright`'s evolution/anti-gaming layer; K≥3 oracles
  (Chai-1 / Protenix); peptide targets.
