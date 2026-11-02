from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Optional, Sequence, TypeAlias
import joblib
from joblib import Parallel, delayed

import polars as pl
from rdkit import Chem
from tqdm.auto import tqdm

from vantami.chem.molecule import DeepMolecule
from vantami.io import write_pl


MolID: TypeAlias = str | int
MoleculeInput: TypeAlias = tuple[MolID, str]
ConformerStatus: TypeAlias = Literal["success", "failed", "skipped"]


def _run_conformer_search(mol_id: MolID, smiles: str, molecule: DeepMolecule, options: Mapping[str, Any]):
    """
    Execute one isolated conformer search, including in joblib workers.
    """
    try:
        conformer_id, energy = molecule.lowest_energy_conformer(**options)
        return mol_id, smiles, molecule, {
            "status": "success",
            "conformer_id": conformer_id,
            "energy": energy,
            "method": options["method"],
            "error": None,
        }
    except Exception as exc:
        return mol_id, smiles, molecule, {
            "status": "failed",
            "conformer_id": None,
            "energy": None,
            "method": options["method"],
            "error": f"{type(exc).__name__}: {exc}",
        }


class DeepMoleculeManager:
    """
    Manage a SMILES-unique collection of < DeepMolecule > instances.

    Parameters
    ----------
    conformer_defaults
        Defaults passed to < DeepMolecule.lowest_energy_conformer >.
    """

    DEFAULT_CONFORMER_OPTIONS: dict[str, Any] = {
        "n_conformers": 16,
        "method": "MMFF94s",
        "seed": 42,
        "prune_threshold": 0.25,
        "max_iterations": 1024,
        "charge": None,
        "uhf": None,
    }

    def __init__(self, conformer_defaults: Optional[Mapping[str, Any]] = None) -> None:
        self.molecules: dict[str, DeepMolecule] = {}
        self.mol_ids: dict[str, MolID] = {}
        self.failed_smiles: dict[str, str] = {}
        self.errors: dict[str, dict[str, str]] = {}
        self.search_results: dict[str, dict[str, Any]] = {}

        self.conformer_defaults = self.DEFAULT_CONFORMER_OPTIONS.copy()
        if conformer_defaults:
            self.conformer_defaults.update(conformer_defaults)
        self._validate_conformer_options(self.conformer_defaults)

    def __len__(self) -> int:
        return len(self.molecules)

    def __contains__(self, smiles: object) -> bool:
        return isinstance(smiles, str) and smiles in self.molecules

    def __iter__(self) -> Iterator[str]:
        return iter(self.molecules)

    def items(self) -> Iterator[tuple[str, DeepMolecule]]:
        """
        Yield < (smiles, DeepMolecule) > pairs for valid molecules.
        """
        return iter(self.molecules.items())

    @staticmethod
    def _validate_conformer_options(options: Mapping[str, Any]) -> None:
        required = set(DeepMoleculeManager.DEFAULT_CONFORMER_OPTIONS)
        unknown = set(options) - required
        if unknown:
            raise ValueError(f"Unknown conformer option(s): {sorted(unknown)}")

        if options["n_conformers"] < 1:
            raise ValueError("n_conformers must be at least 1.")
        if options["max_iterations"] < 1:
            raise ValueError("max_iterations must be at least 1.")

    def _record_error(self, smiles: str, stage: str, message: str) -> None:
        self.errors.setdefault(smiles, {})[stage] = message

    def clear_errors(self, smiles: Optional[str] = None) -> None:
        """
        Clear all errors, or all recorded errors for one SMILES.
        """
        if smiles is None:
            self.errors.clear()
        else:
            self.errors.pop(smiles, None)

    def add(self, mol_id: MolID, smiles: str) -> bool:
        """
        Validate and add one molecule, returning whether it was added.
        """
        if smiles in self.molecules:
            raise ValueError(f"SMILES is already registered: {smiles!r}")

        try:
            rdkit_mol = Chem.MolFromSmiles(smiles)
        except Exception as exc:
            rdkit_mol = None
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = "RDKit could not construct a molecule."

        if rdkit_mol is None:
            self.failed_smiles[smiles] = mol_id
            self._record_error(smiles, "validation", error)
            return False

        try:
            molecule = DeepMolecule(smiles)
        except Exception as exc:
            self.failed_smiles[smiles] = mol_id
            self._record_error(smiles, "initialization", f"{type(exc).__name__}: {exc}")
            return False

        self.molecules[smiles] = molecule
        self.mol_ids[smiles] = mol_id
        self.failed_smiles.pop(smiles, None)
        self.errors.pop(smiles, None)
        self.search_results.pop(smiles, None)
        return True

    def add_many(self, records: Iterable[MoleculeInput]) -> dict[str, int]:
        """
        Add < (mol_id, smiles) > records and return valid/invalid counts.
        """
        counts = {"added": 0, "invalid": 0}
        for mol_id, smiles in records:
            if self.add(mol_id, smiles):
                counts["added"] += 1
            else:
                counts["invalid"] += 1
        print(counts)
        return

    def get(self, smiles: str) -> DeepMolecule:
        """
        Retrieve a valid molecule by its original, input SMILES.
        """
        try:
            return self.molecules[smiles]
        except KeyError as exc:
            raise KeyError(f"No valid molecule registered for SMILES: {smiles!r}") from exc

    def is_valid(self, smiles: str) -> bool:
        """
        Return whether a SMILES has a successfully constructed molecule.
        """
        return smiles in self.molecules

    def select(self, valid: bool = True) -> dict[str, DeepMolecule] | dict[str, MolID]:
        """
        Return valid molecules or invalid SMILES mapped to their MolIDs.
        """
        return self.molecules.copy() if valid else self.failed_smiles.copy()

    def _search_options(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        options = self.conformer_defaults | dict(overrides)
        self._validate_conformer_options(options)
        return options

    def search_conformers(self, smiles: str, **overrides: Any) -> dict[str, Any]:
        """
        Run a conformer search for one valid molecule without raising errors.
        """
        molecule = self.get(smiles)
        options = self._search_options(overrides)
        _, _, updated_molecule, result = _run_conformer_search(self.mol_ids[smiles], smiles, molecule, options)
        self.molecules[smiles] = updated_molecule
        self.search_results[smiles] = result

        if result["status"] == "failed":
            self._record_error(smiles, "conformer_search", result["error"])
        else:
            self.errors.get(smiles, {}).pop("conformer_search", None)
            if not self.errors.get(smiles):
                self.errors.pop(smiles, None)
        return result.copy()

    def search_conformers_many(self, smiles: Optional[Sequence[str]] = None, *, n_jobs: int = 1,
                               batch_size: int | str = "auto", **overrides: Any) -> None:
        """Run independent conformer searches and update < .search_results >.
        """
        options = self._search_options(overrides)
        requested_smiles = list(self.molecules if smiles is None else smiles)

        unknown = [item for item in requested_smiles if item not in self.molecules]
        if unknown:
            raise KeyError(f"Cannot search unregistered or invalid SMILES: {unknown!r}")

        tasks = (
            delayed(_run_conformer_search)(self.mol_ids[item], item, self.molecules[item], options)
            for item in requested_smiles
        )

        completed = Parallel(n_jobs=n_jobs, batch_size=batch_size, return_as="generator")(tasks)

        with tqdm(completed, total=len(requested_smiles), desc="Conformer search", unit="molecule") as progress:
            for _, item, molecule, result in progress:
                self.molecules[item] = molecule
                self.search_results[item] = result
                if result["status"] == "failed":
                    self._record_error(item, "conformer_search", result["error"])
                else:
                    self.errors.get(item, {}).pop("conformer_search", None)
                    if not self.errors.get(item):
                        self.errors.pop(item, None)

    def summary(self) -> dict[str, int]:
        """
        Return basic collection and conformer-search status counts.
        """
        statuses = [result["status"] for result in self.search_results.values()]
        return {
            "valid": len(self.molecules),
            "invalid": len(self.failed_smiles),
            "conformer_searches": len(statuses),
            "conformer_successes": statuses.count("success"),
            "conformer_failures": statuses.count("failed"),
        }

    def save(self, path: str | Path) -> Path:
        """
        Save the full manager state, including valid and invalid inputs.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "DeepMoleculeManager":
        """
        Restore a manager saved with :meth:`save`.
        """
        manager = joblib.load(Path(path))
        if not isinstance(manager, cls):
            raise TypeError(f"Saved object is not a {cls.__name__}.")
        return manager

    def to_dataframe(self) -> pl.DataFrame:
        """
        Return one row per supplied SMILES as a Polars DataFrame.
        """
        rows: list[dict[str, Any]] = []
        for smiles in self.molecules:
            result = self.search_results.get(smiles, {})
            rows.append({
                "mol_id": self.mol_ids[smiles],
                "smiles": smiles,
                "valid": True,
                "conformer_status": result.get("status", "skipped"),
                "conformer_id": result.get("conformer_id"),
                "energy": result.get("energy"),
                "method": result.get("method"),
                "error": result.get("error"),
            })

        for smiles, mol_id in self.failed_smiles.items():
            rows.append({
                "mol_id": mol_id,
                "smiles": smiles,
                "valid": False,
                "conformer_status": "skipped",
                "conformer_id": None,
                "energy": None,
                "method": None,
                "error": self.errors.get(smiles, {}).get("validation")
                or self.errors.get(smiles, {}).get("initialization"),
            })

        return pl.DataFrame(rows)

    def export_results(self, path: str | Path) -> Path:
        """
        Export :meth:`to_dataframe`.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_pl(self.to_dataframe(), output_path)
        return output_path
