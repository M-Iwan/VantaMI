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


def _run_conformer_search(mol_id: MolID, smiles: str, molecule: DeepMolecule, options: Mapping[str, Any]):
    """
    Run a conformer search and capture any exception as a result record.

    Parameters
    ----------
    mol_id: MolID
        Identifier associated with the molecule.
    smiles: str
        SMILES representation of the molecule.
    molecule: DeepMolecule
        Molecule instance for which to generate and optimize conformers.
    options: Mapping[str, Any]
        Keyword arguments passed to `DeepMolecule.lowest_energy_conformer`.

    Returns
    -------
    tuple[MolID, str, DeepMolecule, dict[str, Any]]
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
        """
        Initialize an empty collection of SMILES-unique `DeepMolecule` instances.

        Parameters
        ----------
        conformer_defaults: Optional[Mapping[str, Any]]
            Default options passed to conformer-search calculations. Values override
            the built-in conformer-search defaults.
        """
        self.molecules: dict[str, DeepMolecule] = {}
        self.mol_ids: dict[str, MolID] = {}
        self.failed_smiles: dict[str, str] = {}
        self.errors: dict[str, dict[str, str]] = {}
        self.conformer_results: dict[str, dict[str, Any]] = {}

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
        """
        Store an error message for a molecule-processing stage.
        """
        self.errors.setdefault(smiles, {})[stage] = message

    def clear_errors(self, smiles: Optional[str] = None) -> None:
        """
        Clear all recorded errors or errors associated with one SMILES string.

        Parameters
        ----------
        smiles: Optional[str]
            SMILES representation whose errors will be removed. If `None`, all errors are removed.
        """
        if smiles is None:
            self.errors.clear()
        else:
            self.errors.pop(smiles, None)

    def add(self, mol_id: MolID, smiles: str) -> bool:
        """
        Validate and register one molecule from its identifier and SMILES string.

        Parameters
        ----------
        mol_id: MolID
            Identifier associated with the molecule.
        smiles: str
            SMILES representation to validate and register.

        Returns
        -------
        bool
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
        self.conformer_results.pop(smiles, None)
        return True

    def add_many(self, records: Iterable[MoleculeInput]):
        """
        Validate and register multiple molecule records.

        Parameters
        ----------
        records: Iterable[MoleculeInput]
            Iterable of `(mol_id, smiles)` records to add.
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
        Retrieve a valid registered molecule by its SMILES string.

        Parameters
        ----------
        smiles: str
            Original SMILES representation of the registered molecule.

        Returns
        -------
        DeepMolecule
        """
        try:
            return self.molecules[smiles]
        except KeyError as exc:
            raise KeyError(f"No valid molecule registered for SMILES: {smiles!r}") from exc

    def get_valid_conformers(self):
        """
        Return molecules whose conformer searches completed successfully.

        Returns
        -------
        list[DeepMolecule]
        """
        if not self.conformer_results:
            raise ValueError(f"No conformer search has been performed yet")

        return [self.get(smi) for smi, res in self.conformer_results.items() is res["status"] == "success"]

    def select(self, valid: bool = True) -> dict[str, DeepMolecule] | dict[str, MolID]:
        """
        Return valid molecules or invalid SMILES records.

        Parameters
        ----------
        valid: bool
            Whether to return valid molecules. If `False`, return invalid SMILES strings mapped to their molecule identifiers.

        Returns
        -------
        dict[str, DeepMolecule] | dict[str, MolID]
        """

        return self.molecules.copy() if valid else self.failed_smiles.copy()

    def select_conformers(self, valid: bool = True) -> dict[str, DeepMolecules]:
        """
        Return valid conformers or invalid SMILES records.

        Parameters
        ----------
        valid: bool
            Whether to return valid molecules. If `False`, return invalid SMILES strings mapped to their molecule identifiers.

        Returns
        -------
        dict[str, DeepMolecule] | dict[str, MolID]
        """
        if not self.conformer_results:
            raise ValueError(f"No conformer search has been performed yet")

        if valid:
            return [self.get(smi) for smi, res in self.conformer_results.items() is res["status"] == "success"]
        else:
            return [self.get(smi) for smi, res in self.conformer_results.items() is res["status"] != "success"]

    def _search_options(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        """
        Merge conformer-search defaults with per-call option overrides.
        """
        options = self.conformer_defaults | dict(overrides)
        self._validate_conformer_options(options)
        return options

    def search_conformers(self, smiles: str, **overrides: Any) -> dict[str, Any]:
        """
        Run a conformer search for one valid molecule and record its result.

        Parameters
        ----------
        smiles: str
            SMILES representation of the valid registered molecule.
        **overrides: Any
            Conformer-search options that override the configured defaults.

        Returns
        -------
        dict[str, Any]
        """
        molecule = self.get(smiles)
        options = self._search_options(overrides)
        _, _, updated_molecule, result = _run_conformer_search(self.mol_ids[smiles], smiles, molecule, options)
        self.molecules[smiles] = updated_molecule
        self.conformer_results[smiles] = result

        if result["status"] == "failed":
            self._record_error(smiles, "conformer_search", result["error"])
        else:
            self.errors.get(smiles, {}).pop("conformer_search", None)
            if not self.errors.get(smiles):
                self.errors.pop(smiles, None)
        return result.copy()

    def search_conformers_many(self, smiles: Optional[Sequence[str]] = None, *, n_jobs: int = 1,
                               batch_size: Union[int, str] = "auto", **overrides: Any) -> None:
        """
        Run conformer searches for multiple valid molecules.

        Parameters
        ----------
        smiles: Optional[Sequence[str]]
            SMILES strings for molecules to process. If `None`, all valid registered molecules are processed.
        n_jobs: int
            Number of parallel joblib workers. Use `1` for sequential processing.
        batch_size: int | str
            Number of tasks dispatched to each joblib worker batch, or `"auto"` to let joblib determine an appropriate batch size.
        **overrides: Any
            Conformer-search options that override the configured defaults.
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
                self.conformer_results[item] = result
                if result["status"] == "failed":
                    self._record_error(item, "conformer_search", result["error"])
                else:
                    self.errors.get(item, {}).pop("conformer_search", None)
                    if not self.errors.get(item):
                        self.errors.pop(item, None)

    def retain_lowest_energy_conformers(self, smiles: Optional[Sequence[str]] = None, *,
                                        keep_global_charge_sets: bool = True) -> None:
        """
        Prune successful conformer-search results to one conformer per molecule.
        """
        requested_smiles = list(self.molecules if smiles is None else smiles)

        unknown = [smi for smi in requested_smiles if smi not in self.molecules]
        if unknown:
            raise KeyError(f"Cannot prune unregistered molecules: {unknown!r}")

        for smi in requested_smiles:
            result = self.conformer_results.get(smi)

            if result is None or result["status"] != "success":
                continue

            molecule = self.molecules[smi]
            conf_id, energy = molecule.retain_lowest_energy_conformer(
                method=result["method"],
                keep_global_charge_sets=keep_global_charge_sets,
            )

            result["conformer_id"] = conf_id
            result["energy"] = energy

    def summary(self) -> dict[str, int]:
        """
        Return counts for molecule validation and conformer-search results.

        Returns
        -------
        dict[str, int]
        """
        statuses = [result["status"] for result in self.conformer_results.values()]
        return {
            "valid": len(self.molecules),
            "invalid": len(self.failed_smiles),
            "conformer_searches": len(statuses),
            "conformer_successes": statuses.count("success"),
            "conformer_failures": statuses.count("failed"),
        }

    def save(self, path: Union[str, Path]) -> Path:
        """
        Serialize the complete manager state to disk.

        Parameters
        ----------
        path: Union[str, Path]
            Destination path for the serialized manager.

        Returns
        -------
        Path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeepMoleculeManager":
        """
        Restore a serialized manager instance from disk.

        Parameters
        ----------
        path: Union[str, Path]
            Path to the serialized manager.

        Returns
        -------
        DeepMoleculeManager
        """
        manager = joblib.load(Path(path))
        if not isinstance(manager, cls):
            raise TypeError(f"Saved object is not a {cls.__name__}.")
        return manager

    def to_dataframe(self) -> pl.DataFrame:
        """
        Convert validation and conformer-search results to a Polars DataFrame.

        Returns
        -------
        pl.DataFrame
        """
        rows: list[dict[str, Any]] = []
        for smiles in self.molecules:
            result = self.conformer_results.get(smiles, {})
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

    def export_results(self, path: Union[str, Path]) -> Path:
        """
        Export validation and conformer-search results to a file.

        Parameters
        ----------
        path: Union[str, Path]
            Destination path for the exported results.

        Returns
        -------
        Path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_pl(self.to_dataframe(), output_path)
        return output_path
