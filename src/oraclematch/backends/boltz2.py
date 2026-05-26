"""Boltz-2 backend — deep-learning co-folding oracle (GPU). MIT.

PROVIDED, NOT EXERCISED in v0.1.0a1: requires the ``boltz`` extra, model weights, and a CUDA GPU
(~20 s/ligand). Import-guarded so the package imports fine without it. Boltz-2 has a practical
ligand-size limit (~50 heavy atoms), surfaced via :attr:`max_ligand_atoms` so callers can route
larger ligands to a docking oracle instead.

Sign convention: Boltz-2's binding-affinity head already follows a higher-is-better scale (a
log-pIC50-like value), so it is returned as-is; the predicted-complex confidence is mapped into
``[0, 1]`` to feed the validity gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from oraclematch.core.protocol import Molecule, PredictionResult

_MAX_LIGAND_ATOMS = 50


class Boltz2Predictor:
    """Co-folds a target+ligand and returns Boltz-2's binding-affinity score and confidence."""

    def __init__(
        self,
        target_sequence: str,
        *,
        device: str = "cuda",
        cache_dir: str | None = None,
        oracle_id: str = "boltz-2",
    ) -> None:
        self.target_sequence = target_sequence
        self.device = device
        self.cache_dir = cache_dir
        self.oracle_id = oracle_id

    def is_available(self) -> bool:
        try:
            import boltz  # noqa: F401
            import torch
        except Exception:
            return False
        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())

    def predict(self, mol: Molecule) -> PredictionResult:
        if not self.is_available():
            raise RuntimeError(
                "Boltz2Predictor requires the 'boltz' extra (pip install 'oraclematch[boltz]'), "
                "model weights, and a CUDA GPU; it is not exercised in the v0.1.0a1 release. Run "
                "scripts/gpu_pilot_kc2.py in a GPU environment to validate KC-2 first."
            )
        # Real path (intentionally not executed in CI): run a Boltz-2 prediction for the
        # target+ligand complex, read the affinity head (higher is better) and the complex
        # confidence (mapped to [0, 1]). Left as a thin documented adapter for v0.1.1.
        raise NotImplementedError(
            "live Boltz-2 inference is wired in v0.1.1 after the KC-2 GPU pilot; see ROADMAP."
        )

    def predict_batch(self, mols: Sequence[Molecule]) -> list[PredictionResult]:
        return [self.predict(m) for m in mols]

    @property
    def max_ligand_atoms(self) -> int | None:
        return _MAX_LIGAND_ATOMS
