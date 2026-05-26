"""AutoDock-Vina backend — classical-physics docking oracle (CPU). Apache-2.0.

PROVIDED, NOT EXERCISED in v0.1.0a1: requires the ``vina`` extra and a prepared receptor. It is
import-guarded so the package imports fine without it; :meth:`is_available` reports the truth and
:meth:`predict` raises a clear error if called when unavailable.

Sign convention: Vina returns a docking *energy* in kcal/mol where lower (more negative) is better.
This adapter **negates** it so ``affinity`` follows the package-wide "higher is better" convention
required before cross-oracle disagreement is computed (see :mod:`oraclematch.core.protocol`).
"""

from __future__ import annotations

from collections.abc import Sequence

from oraclematch.core.protocol import Molecule, PredictionResult


class VinaPredictor:
    """Docks a SMILES ligand into a fixed receptor box and returns ``-docking_energy``."""

    def __init__(
        self,
        receptor_pdbqt: str,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
        *,
        exhaustiveness: int = 8,
        oracle_id: str = "autodock-vina",
    ) -> None:
        self.receptor_pdbqt = receptor_pdbqt
        self.center = center
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness
        self.oracle_id = oracle_id

    def is_available(self) -> bool:
        try:
            import vina  # noqa: F401
            from meeko import MoleculePreparation  # noqa: F401
        except Exception:
            return False
        return True

    def predict(self, mol: Molecule) -> PredictionResult:
        if not self.is_available():
            raise RuntimeError(
                "VinaPredictor requires the 'vina' extra (pip install 'oraclematch[vina]') and a "
                "prepared receptor; it is not exercised in the v0.1.0a1 CPU-only release."
            )
        # Real path (intentionally not executed in CI): prepare the ligand with Meeko, run Vina in
        # the configured box, take the best mode's energy, and negate it for the higher-is-better
        # convention. Confidence is left at 1.0 here; a PoseBusters gate would supply it.
        from vina import Vina  # type: ignore

        v = Vina(sf_name="vina")
        v.set_receptor(self.receptor_pdbqt)
        v.compute_vina_maps(center=list(self.center), box_size=list(self.box_size))
        ligand_pdbqt = self._prepare_ligand(mol)
        v.set_ligand_from_string(ligand_pdbqt)
        v.dock(exhaustiveness=self.exhaustiveness)
        energy = float(v.energies(n_poses=1)[0][0])  # kcal/mol, lower is better
        return PredictionResult(
            oracle_id=self.oracle_id,
            affinity=-energy,  # negate -> higher is better
            confidence=1.0,
            raw={"molecule": mol, "docking_energy_kcal_mol": energy},
        )

    def predict_batch(self, mols: Sequence[Molecule]) -> list[PredictionResult]:
        return [self.predict(m) for m in mols]

    @staticmethod
    def _prepare_ligand(mol: Molecule) -> str:
        from meeko import MoleculePreparation  # type: ignore
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore

        rdmol = Chem.MolFromSmiles(mol.smiles)
        if rdmol is None:
            raise ValueError(f"invalid SMILES: {mol.smiles!r}")
        rdmol = Chem.AddHs(rdmol)
        AllChem.EmbedMolecule(rdmol, randomSeed=0)
        prep = MoleculePreparation()
        prep.prepare(rdmol)
        return prep.write_pdbqt_string()

    @property
    def max_ligand_atoms(self) -> int | None:
        return None
