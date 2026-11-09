"""
Small-molecule preparation, docking, and interaction-fingerprint utilities.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import numpy.typing as npt

from rdkit import Chem
from rdkit.Chem import Draw, rdPartialCharges
from rdkit.Chem.AllChem import (EmbedMultipleConfs, MMFFGetMoleculeForceField,
                                MMFFGetMoleculeProperties, MMFFOptimizeMolecule,
                                UFFGetMoleculeForceField, UFFHasAllMoleculeParams,
                                UFFOptimizeMolecule)

try:
    from rdkit.Chem.AllChem import ETKDGv4
except ImportError:
    from rdkit.Chem.AllChem import ETKDGv3 as ETKDGv4

    warnings.warn("ETKDGv4 is unavailable in this RDKit installation; falling back to ETKDGv3.",
                  UserWarning, stacklevel=2)


MMFFVariant = Literal["MMFF94", "MMFF94s"]
EnergyMethod = Literal["MMFF94", "MMFF94s", "UFF", "GFN2-xTB"]
ChargeMethod = Literal["Gasteiger", "MMFF94", "MMFF94s", "GFN2-xTB"]


class Molecule:
    """
    Shared RDKit molecule preparation functionality.
    """

    def __init__(self, smiles: str, id: int | str):
        """
        Initialize a molecule from SMILES.

        Parameters
        ----------
        smiles: str
            SMILES representation used to construct the RDKit molecule.
        id: str or None
            Optional molecule identifier.
        """
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        if not isinstance(id, (int, str)):
            raise TypeError(f"ID must be an int or str: {id}")

        self.smiles = smiles
        self.id = id
        self.mol = mol
        self.energy: float | None = None
        self.energy_method: EnergyMethod | None = None
        self.charge: int | None = None
        self.uhf: int | None = None
        self.n_optimized: int = 0
        self.lowest_energy_conf_id: int | None = None
        self.lowest_energy_method: EnergyMethod | None = None
        self._energies: dict[tuple[int, EnergyMethod], float] = {}
        self._charge_sets: dict[tuple[int | None, ChargeMethod], npt.NDArray[np.float32]] = {}
        self._descriptors: dict[tuple[int, EnergyMethod], dict[str, Any]] = {}
        self._active_charge_method: ChargeMethod | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__} object of < {self.smiles} >"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(smiles={self.smiles!r}, id={self.id!r}, "
            f"n_atoms={self.mol.GetNumAtoms()}, n_conformers={self.mol.GetNumConformers()})"
        )

    @staticmethod
    def _with_hs(mol: Chem.Mol, keep_coords: bool = False) -> Chem.Mol:
        """
        Copy a molecule with explicit hydrogens.

        Parameters
        ----------
        mol: rdkit.Chem.Mol
            Molecule to copy.
        keep_coords: bool
            Whether to preserve existing conformer coordinates.

        Returns
        -------
        rdkit.Chem.Mol
            Copied molecule with explicit hydrogen atoms.
        """
        copied = Chem.Mol(mol)

        if not keep_coords:
            copied.RemoveAllConformers()

        return Chem.AddHs(copied, addCoords=keep_coords)

    @staticmethod
    def _embedding_parameters(seed: int, prune_threshold: float) -> Any:
        """
        Create ETKDG embedding parameters.

        Parameters
        ----------
        seed: int
            Random seed for conformer embedding.
        prune_threshold: float
            RMSD pruning threshold.

        Returns
        -------
        rdkit.Chem.rdDistGeom.EmbedParameters
            Configured embedding parameters.
        """
        parameters = ETKDGv4()
        parameters.randomSeed = seed
        parameters.useSmallRingTorsions = True
        parameters.pruneRmsThresh = prune_threshold
        return parameters

    def _resolve_conf_id(self, conf_id: int = -1) -> int:
        """
        Resolve an RDKit conformer identifier.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.

        Returns
        -------
        int
            Resolved conformer identifier.
        """
        if self.mol.GetNumConformers() == 0:
            raise ValueError("The molecule does not contain a conformer.")

        return self.mol.GetConformer(conf_id).GetId()

    def _clear_calculation_results(self) -> None:
        """
        Clear cached molecular calculation results.
        """
        self.energy = None
        self.energy_method = None
        self.charge = None
        self.uhf = None
        self.n_optimized = 0
        self.lowest_energy_conf_id = None
        self.lowest_energy_method = None
        self._energies.clear()
        self._charge_sets.clear()
        self._descriptors.clear()
        self._active_charge_method = None

        for atom in self.mol.GetAtoms():
            for property_name in ("partial_charge", "partial_charge_method"):
                if atom.HasProp(property_name):
                    atom.ClearProp(property_name)

        if self.mol.HasProp("partial_charge_method"):
            self.mol.ClearProp("partial_charge_method")

    def _clear_conformer_results(self, conf_id: int) -> None:
        """
        Clear cached results for a conformer.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        """
        conf_id = self._resolve_conf_id(conf_id)

        for key in list(self._energies):
            if key[0] == conf_id:
                del self._energies[key]

        for key in list(self._descriptors):
            if key[0] == conf_id:
                del self._descriptors[key]

        for key in list(self._charge_sets):
            if key[0] == conf_id:
                del self._charge_sets[key]

        if self.lowest_energy_conf_id == conf_id:
            self.lowest_energy_conf_id = None
            self.lowest_energy_method = None

    def _store_energy(self, energy: float, method: EnergyMethod, conf_id: int, **descriptors: Any) -> float:
        """
        Store a conformer energy and descriptors.

        Parameters
        ----------
        energy: float
            Calculated conformer energy.
        method: EnergyMethod
            Energy calculation method.
        conf_id: int
            RDKit conformer identifier.
        **descriptors: Any
            Additional calculation descriptors.

        Returns
        -------
        float
            Stored energy.
        """
        energy = float(energy)
        conf_id = self._resolve_conf_id(conf_id)
        self._energies[(conf_id, method)] = energy
        self._descriptors[(conf_id, method)] = {
            "energy": energy,
            "method": method,
            "conf_id": conf_id,
            **descriptors,
        }
        self.energy = energy
        self.energy_method = method
        return energy

    def _store_charges(self, charges: npt.ArrayLike, method: ChargeMethod, conf_id: int | None = None) -> npt.NDArray[np.float32]:
        """
        Store atomic partial charges.

        Parameters
        ----------
        charges: numpy.typing.ArrayLike
            Partial charges in RDKit atom-index order.
        method: ChargeMethod
            Charge calculation method.
        conf_id: int or None
            Optional RDKit conformer identifier.

        Returns
        -------
        numpy.ndarray
            Stored partial charges.
        """
        values = np.asarray(charges, dtype=np.float32).reshape(self.mol.GetNumAtoms())

        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{method} produced non-finite partial charges.")

        if conf_id is not None:
            conf_id = self._resolve_conf_id(conf_id)

        self._charge_sets[(conf_id, method)] = values.copy()
        self._active_charge_method = method

        for atom, partial_charge in zip(self.mol.GetAtoms(), values):
            atom.SetDoubleProp("partial_charge", float(partial_charge))
            atom.SetProp("partial_charge_method", method)

        self.mol.SetProp("partial_charge_method", method)
        return values.copy()

    def _get_cached_charges(self, method: ChargeMethod, conf_id: int | None) -> npt.NDArray[np.float32] | None:
        """
        Return cached charges for a method.

        Parameters
        ----------
        method: ChargeMethod
            Charge calculation method.
        conf_id: int | None
            Optional RDKit conformer identifier.

        Returns
        -------
        numpy.ndarray | None
            Cached partial charges when available.
        """
        if conf_id is not None:
            conf_id = self._resolve_conf_id(conf_id)

        charges = self._charge_sets.get((conf_id, method))

        if charges is None and method in ("Gasteiger", "MMFF94", "MMFF94s"):
            charges = self._charge_sets.get((None, method))

        return None if charges is None else charges.copy()

    def _prepare_for_charge_calculation(self) -> None:
        """
        Add explicit hydrogens for charge calculations.
        """
        h_mol = self._with_hs(self.mol, keep_coords=self.mol.GetNumConformers() > 0)

        if h_mol.GetNumAtoms() != self.mol.GetNumAtoms():
            self.mol = h_mol
            self._clear_calculation_results()

    def _get_xtb_atoms(self, conf_id: int, charge: int | None, uhf: int | None, method: str) -> tuple[Any, int, int]:
        """
        Build an ASE Atoms object for xTB.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        charge: int or None
            Molecular charge.
        uhf: int or None
            Number of unpaired electrons.
        method: str
            xTB method name.

        Returns
        -------
        tuple
            ASE atoms object, charge, and unpaired-electron count.
        """
        try:
            from ase import Atoms
            from xtb.ase.calculator import XTB
        except ImportError as exc:
            raise ImportError("xTB calculations require ASE and xtb-python.") from exc

        conf_id = self._resolve_conf_id(conf_id)
        conformer = self.mol.GetConformer(conf_id)

        if charge is None:
            charge = sum(atom.GetFormalCharge() for atom in self.mol.GetAtoms())

        if uhf is None:
            uhf = sum(atom.GetNumRadicalElectrons() for atom in self.mol.GetAtoms())

        atoms = Atoms(
            numbers=[atom.GetAtomicNum() for atom in self.mol.GetAtoms()],
            positions=np.asarray(conformer.GetPositions(), dtype=float),
            pbc=False,
        )
        atoms.calc = XTB(method=method, charge=charge, uhf=uhf)
        return atoms, charge, uhf

    def _update_conformer_positions(self, conf_id: int, positions: npt.ArrayLike) -> None:
        """
        Update RDKit conformer coordinates.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        positions: numpy.typing.ArrayLike
            Cartesian coordinates.
        """
        conformer = self.mol.GetConformer(self._resolve_conf_id(conf_id))

        for atom_index, position in enumerate(np.asarray(positions, dtype=float)):
            conformer.SetAtomPosition(atom_index, position)

    def validate(self) -> None:
        """
        Sanitize the current molecule.
        """
        Chem.SanitizeMol(self.mol)

    def show(self) -> Any:
        """
        Render the current molecule.

        Returns
        -------
        PIL.Image.Image
            RDKit molecular depiction.
        """
        return Draw.MolToImage(self.mol, size=(512, 512))

    def get_atom_indices(self, heavy_only: bool = False) -> npt.NDArray[np.int64]:
        """
        Return atom indices.

        Parameters
        ----------
        heavy_only: bool
            Whether to return only non-hydrogen atoms.

        Returns
        -------
        numpy.ndarray
            RDKit atom indices.
        """
        return np.asarray(
            [atom.GetIdx() for atom in self.mol.GetAtoms() if not heavy_only or atom.GetAtomicNum() > 1],
            dtype=np.int64,
        )

    def get_atoms(self, heavy_only: bool = False) -> tuple[Chem.rdchem.Atom, ...]:
        """
        Return RDKit atoms.

        Parameters
        ----------
        heavy_only: bool
            Whether to return only non-hydrogen atoms.

        Returns
        -------
        tuple
            RDKit atoms.
        """
        return tuple(atom for atom in self.mol.GetAtoms() if not heavy_only or atom.GetAtomicNum() > 1)

    def get_bonds(self, heavy_only: bool = False) -> tuple[Chem.rdchem.Bond, ...]:
        """
        Return RDKit bonds.

        Parameters
        ----------
        heavy_only: bool
            Whether to return only bonds between non-hydrogen atoms.

        Returns
        -------
        tuple
            RDKit bonds.
        """
        return tuple(
            bond for bond in self.mol.GetBonds()
            if not heavy_only or (bond.GetBeginAtom().GetAtomicNum() > 1 and bond.GetEndAtom().GetAtomicNum() > 1)
        )

    def get_conformer_positions(self, conf_id: int = -1, heavy_only: bool = False) -> npt.NDArray[np.float32]:
        """
        Return conformer coordinates.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        heavy_only: bool
            Whether to return only non-hydrogen coordinates.

        Returns
        -------
        numpy.ndarray
            Cartesian coordinates.
        """
        conformer = self.mol.GetConformer(self._resolve_conf_id(conf_id))
        positions = np.asarray(conformer.GetPositions(), dtype=np.float32)

        if heavy_only:
            positions = positions[self.get_atom_indices(heavy_only=True)]

        return positions

    @property
    def available_charge_methods(self) -> tuple[str, ...]:
        """
        Return calculated charge methods.

        Returns
        -------
        tuple
            Available charge methods.
        """
        return tuple(sorted({method for _, method in self._charge_sets}))

    @property
    def charge_method(self) -> ChargeMethod | None:
        """
        Return the active charge method.

        Returns
        -------
        ChargeMethod | None
            Active charge method.
        """
        return self._active_charge_method

    @property
    def charges(self) -> npt.NDArray[np.float32] | None:
        """
        Return active partial charges.

        Returns
        -------
        numpy.ndarray | None
            Active partial charges.
        """
        if self._active_charge_method is None:
            return None

        if self.lowest_energy_conf_id is not None:
            charges = self._get_cached_charges(self._active_charge_method, self.lowest_energy_conf_id)

            if charges is not None:
                return charges

        return self._get_cached_charges(self._active_charge_method, None)

    def conformers(self, n_conformers: int = 1, seed: int = 42, prune_threshold: float = 0.25) -> tuple[int, ...]:
        """
        Generate ETKDG conformers.

        Parameters
        ----------
        n_conformers: int
            Number of conformers to generate.
        seed: int
            Random seed for conformer embedding.
        prune_threshold: float
            RMSD pruning threshold.

        Returns
        -------
        tuple
            Generated RDKit conformer identifiers.
        """
        if n_conformers < 1:
            raise ValueError("n_conformers must be at least 1.")

        embedded_mol = self._with_hs(self.mol)
        embedded_mol.RemoveAllConformers()
        parameters = self._embedding_parameters(seed=seed, prune_threshold=prune_threshold)
        conformer_ids = tuple(
            int(conf_id)
            for conf_id in EmbedMultipleConfs(embedded_mol, numConfs=n_conformers, params=parameters)
        )

        if not conformer_ids:
            raise RuntimeError(f"RDKit could not generate conformers for < {self.smiles!r} >.")

        self.mol = embedded_mol
        self._clear_calculation_results()
        return conformer_ids

    def lowest_energy_conformer(self, n_conformers: int = 16, method: EnergyMethod = "MMFF94", seed: int = 42,
                                prune_threshold: float = 0.25, max_iterations: int = 1024, charge: int | None = None,
                                uhf: int | None = None) -> tuple[int, float]:
        """
        Generate and optimize conformers.

        Parameters
        ----------
        n_conformers: int
            Number of conformers to generate.
        method: EnergyMethod
            Optimization method.
        seed: int
            Random seed for conformer embedding.
        prune_threshold: float
            RMSD pruning threshold.
        max_iterations: int
            Maximum optimization iterations.
        charge: int or None
            Molecular charge for xTB.
        uhf: int or None
            Number of unpaired electrons for xTB.

        Returns
        -------
        tuple
            Lowest-energy conformer identifier and energy.
        """
        conformer_ids = self.conformers(
            n_conformers=n_conformers,
            seed=seed,
            prune_threshold=prune_threshold,
        )
        results = []

        for conf_id in conformer_ids:
            energy, _, _ = self.optimize(
                conf_id=conf_id,
                method=method,
                max_iterations=max_iterations,
                charge=charge,
                uhf=uhf,
            )
            results.append((conf_id, energy))

        lowest_conf_id, lowest_energy = min(results, key=lambda result: result[1])
        self.lowest_energy_conf_id = lowest_conf_id
        self.lowest_energy_method = method
        self.energy = lowest_energy
        self.energy_method = method
        return lowest_conf_id, lowest_energy

    def get_lowest_energy_conformer(self, method: EnergyMethod | None = None) -> tuple[int, float]:
        """
        Return the cached lowest-energy conformer.

        Parameters
        ----------
        method: EnergyMethod | None
            Energy method used for conformer selection.

        Returns
        -------
        tuple
            Lowest-energy conformer identifier and energy.
        """
        selected_method = method or self.lowest_energy_method

        if selected_method is None:
            raise ValueError("No lowest-energy conformer is available. Call lowest_energy_conformer() first.")

        energies = [
            (conf_id, energy) for (conf_id, energy_method), energy in self._energies.items()
            if energy_method == selected_method
        ]

        if not energies:
            raise ValueError(f"No cached < {selected_method} > conformer energies are available.")

        lowest_conf_id, lowest_energy = min(energies, key=lambda result: result[1])
        self.lowest_energy_conf_id = lowest_conf_id
        self.lowest_energy_method = selected_method
        return lowest_conf_id, lowest_energy

    def retain_lowest_energy_conformer(self, method: EnergyMethod = "MMFF94", *,
                                       keep_global_charge_sets: bool = True) -> tuple[int, float]:
        """
        Retain only the lowest-energy conformer.

        Parameters
        ----------
        method: EnergyMethod
            Energy method used for conformer selection.
        keep_global_charge_sets: bool
            Whether to retain conformer-independent charges.

        Returns
        -------
        tuple
            Retained conformer identifier and energy.
        """
        conf_id, energy = self.get_lowest_energy_conformer(method=method)
        selected_method = method or self.lowest_energy_method

        if selected_method is None:
            raise RuntimeError("Could not resolve the lowest-energy method.")

        for conformer in list(self.mol.GetConformers()):
            if conformer.GetId() != conf_id:
                self.mol.RemoveConformer(conformer.GetId())

        self._energies = {
            (cached_conf_id, cached_method): cached_energy
            for (cached_conf_id, cached_method), cached_energy in self._energies.items()
            if cached_conf_id == conf_id and cached_method == selected_method
        }
        self._descriptors = {
            (cached_conf_id, cached_method): descriptors
            for (cached_conf_id, cached_method), descriptors in self._descriptors.items()
            if cached_conf_id == conf_id and cached_method == selected_method
        }
        self._charge_sets = {
            (cached_conf_id, charge_method): charges
            for (cached_conf_id, charge_method), charges in self._charge_sets.items()
            if cached_conf_id == conf_id or (keep_global_charge_sets and cached_conf_id is None)
        }
        self.lowest_energy_conf_id = conf_id
        self.lowest_energy_method = selected_method
        self.energy = float(energy)
        self.energy_method = selected_method

        active_charges = None

        if self._active_charge_method is not None:
            active_charges = self._charge_sets.get(
                (conf_id, self._active_charge_method),
                self._charge_sets.get((None, self._active_charge_method)),
            )

        if active_charges is None:
            self._active_charge_method = None

            for atom in self.mol.GetAtoms():
                for property_name in ("partial_charge", "partial_charge_method"):
                    if atom.HasProp(property_name):
                        atom.ClearProp(property_name)

            if self.mol.HasProp("partial_charge_method"):
                self.mol.ClearProp("partial_charge_method")
        else:
            for atom, partial_charge in zip(self.mol.GetAtoms(), active_charges):
                atom.SetDoubleProp("partial_charge", float(partial_charge))
                atom.SetProp("partial_charge_method", self._active_charge_method)

            self.mol.SetProp("partial_charge_method", self._active_charge_method)

        return conf_id, float(energy)

    def optimize_mmff(self, conf_id: int = -1, variant: MMFFVariant = "MMFF94",
                      max_iterations: int = 1024) -> tuple[float, bool]:
        """
        Optimize a conformer with MMFF.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        variant: MMFFVariant
            MMFF force-field variant.
        max_iterations: int
            Maximum optimization iterations.

        Returns
        -------
        tuple
            Energy and convergence status.
        """
        conf_id = self._resolve_conf_id(conf_id)
        properties = MMFFGetMoleculeProperties(self.mol, mmffVariant=variant)

        if properties is None:
            raise ValueError(f"MMFF parameters are unavailable for < {self.smiles!r} >.")

        self._clear_conformer_results(conf_id)
        status = MMFFOptimizeMolecule(self.mol, mmffVariant=variant, maxIters=max_iterations, confId=conf_id)
        energy = self.get_mmff_energy(conf_id=conf_id, variant=variant)
        self.n_optimized += 1
        return energy, status == 0

    def optimize_uff(self, conf_id: int = -1, max_iterations: int = 1024) -> tuple[float, bool]:
        """
        Optimize a conformer with UFF.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        max_iterations: int
            Maximum optimization iterations.

        Returns
        -------
        tuple
            Energy and convergence status.
        """
        conf_id = self._resolve_conf_id(conf_id)

        if not UFFHasAllMoleculeParams(self.mol):
            raise ValueError(f"UFF parameters are unavailable for < {self.smiles!r} >.")

        self._clear_conformer_results(conf_id)
        status = UFFOptimizeMolecule(self.mol, maxIters=max_iterations, confId=conf_id)
        energy = self.get_uff_energy(conf_id=conf_id)
        self.n_optimized += 1
        return energy, status == 0

    def optimize_xtb(self, conf_id: int = -1, charge: int | None = None, uhf: int | None = None,
                     method: str = "GFN2-xTB", max_steps: int = 256, fmax: float = 0.01) -> tuple[float, bool]:
        """
        Optimize a conformer with xTB.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        charge: int or None
            Molecular charge.
        uhf: int or None
            Number of unpaired electrons.
        method: str
            xTB method name.
        max_steps: int
            Maximum optimization steps.
        fmax: float
            Maximum force threshold.

        Returns
        -------
        tuple
            Energy and convergence status.
        """
        try:
            from ase.optimize import BFGS
        except ImportError as exc:
            raise ImportError("xTB optimization requires ASE and xtb-python.") from exc

        conf_id = self._resolve_conf_id(conf_id)
        atoms, charge, uhf = self._get_xtb_atoms(conf_id=conf_id, charge=charge, uhf=uhf, method=method)
        self._clear_conformer_results(conf_id)
        optimizer = BFGS(atoms, logfile=None)
        converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
        self._update_conformer_positions(conf_id=conf_id, positions=atoms.get_positions())
        energy = self.get_xtb_energy(conf_id=conf_id, charge=charge, uhf=uhf, method=method)
        self.n_optimized += 1
        return energy, converged

    def optimize(self, conf_id: int = -1, method: EnergyMethod = "MMFF94", max_iterations: int = 1024,
                 charge: int | None = None, uhf: int | None = None) -> tuple[float, EnergyMethod, bool]:
        """
        Optimize a conformer.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        method: EnergyMethod
            Optimization method.
        max_iterations: int
            Maximum optimization iterations.
        charge: int or None
            Molecular charge for xTB.
        uhf: int or None
            Number of unpaired electrons for xTB.

        Returns
        -------
        tuple
            Energy, method, and convergence status.
        """
        if method in ("MMFF94", "MMFF94s"):
            energy, converged = self.optimize_mmff(conf_id=conf_id, variant=method, max_iterations=max_iterations)
        elif method == "UFF":
            energy, converged = self.optimize_uff(conf_id=conf_id, max_iterations=max_iterations)
        elif method == "GFN2-xTB":
            energy, converged = self.optimize_xtb(conf_id=conf_id, charge=charge, uhf=uhf, max_steps=max_iterations)
        else:
            raise ValueError(f"Unknown optimization method: {method!r}.")

        return energy, method, converged

    def get_mmff_energy(self, conf_id: int = -1, variant: MMFFVariant = "MMFF94") -> float:
        """
        Calculate MMFF conformer energy.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        variant: MMFFVariant
            MMFF force-field variant.

        Returns
        -------
        float
            Conformer energy.
        """
        conf_id = self._resolve_conf_id(conf_id)
        properties = MMFFGetMoleculeProperties(self.mol, mmffVariant=variant)

        if properties is None:
            raise ValueError(f"MMFF parameters are unavailable for < {self.smiles!r} >.")

        force_field = MMFFGetMoleculeForceField(self.mol, properties, confId=conf_id)

        if force_field is None:
            raise RuntimeError(f"RDKit could not construct {variant} for < {self.smiles!r} >.")

        return self._store_energy(energy=force_field.CalcEnergy(), method=variant, conf_id=conf_id)

    def get_uff_energy(self, conf_id: int = -1) -> float:
        """
        Calculate UFF conformer energy.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.

        Returns
        -------
        float
            Conformer energy.
        """
        conf_id = self._resolve_conf_id(conf_id)

        if not UFFHasAllMoleculeParams(self.mol):
            raise ValueError(f"UFF parameters are unavailable for < {self.smiles!r} >.")

        force_field = UFFGetMoleculeForceField(self.mol, confId=conf_id)

        if force_field is None:
            raise RuntimeError(f"RDKit could not construct UFF for < {self.smiles!r} >.")

        return self._store_energy(energy=force_field.CalcEnergy(), method="UFF", conf_id=conf_id)

    def get_xtb_energy(self, conf_id: int = -1, charge: int | None = None, uhf: int | None = None,
                       method: str = "GFN2-xTB") -> float:
        """
        Calculate xTB conformer energy.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        charge: int or None
            Molecular charge.
        uhf: int or None
            Number of unpaired electrons.
        method: str
            xTB method name.

        Returns
        -------
        float
            Conformer energy.
        """
        conf_id = self._resolve_conf_id(conf_id)
        atoms, charge, uhf = self._get_xtb_atoms(conf_id=conf_id, charge=charge, uhf=uhf, method=method)
        energy = float(atoms.get_potential_energy())
        charges = np.asarray(atoms.get_charges(), dtype=np.float32)
        results = dict(atoms.calc.results)
        descriptors = {
            key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
            for key, value in results.items()
            if key not in ("energy", "charges")
        }
        self.charge = charge
        self.uhf = uhf
        self._store_charges(charges=charges, method="GFN2-xTB", conf_id=conf_id)
        return self._store_energy(
            energy=energy,
            method="GFN2-xTB",
            conf_id=conf_id,
            charge=charge,
            uhf=uhf,
            **descriptors,
        )

    def get_energy(self, method: EnergyMethod, conf_id: int = -1) -> float:
        """
        Calculate a conformer energy.

        Parameters
        ----------
        method: EnergyMethod
            Energy calculation method.
        conf_id: int
            RDKit conformer identifier.

        Returns
        -------
        float
            Conformer energy.
        """
        if method in ("MMFF94", "MMFF94s"):
            return self.get_mmff_energy(conf_id=conf_id, variant=method)

        if method == "UFF":
            return self.get_uff_energy(conf_id=conf_id)

        if method == "GFN2-xTB":
            return self.get_xtb_energy(conf_id=conf_id)

        raise ValueError(f"Unknown method: {method!r}.")

    def get_gasteiger_charges(self) -> npt.NDArray[np.float32]:
        """
        Calculate Gasteiger charges.

        Returns
        -------
        numpy.ndarray
            Partial charges.
        """
        self._prepare_for_charge_calculation()
        cached = self._get_cached_charges(method="Gasteiger", conf_id=None)

        if cached is not None:
            return cached

        rdPartialCharges.ComputeGasteigerCharges(self.mol)
        charges = np.asarray(
            [float(atom.GetProp("_GasteigerCharge")) for atom in self.mol.GetAtoms()], dtype=np.float32
        )
        return self._store_charges(charges=charges, method="Gasteiger")

    def get_mmff_charges(self, variant: MMFFVariant = "MMFF94") -> npt.NDArray[np.float32]:
        """
        Calculate MMFF charges.

        Parameters
        ----------
        variant: MMFFVariant
            MMFF parameterization variant.

        Returns
        -------
        numpy.ndarray
            Partial charges.
        """
        self._prepare_for_charge_calculation()
        cached = self._get_cached_charges(method=variant, conf_id=None)

        if cached is not None:
            return cached

        properties = MMFFGetMoleculeProperties(self.mol, mmffVariant=variant)

        if properties is None:
            raise ValueError(f"MMFF parameters are unavailable for < {self.smiles!r} >.")

        charges = np.asarray(
            [properties.GetMMFFPartialCharge(atom_index) for atom_index in range(self.mol.GetNumAtoms())],
            dtype=np.float32,
        )
        return self._store_charges(charges=charges, method=variant, conf_id=None)

    def get_xtb_charges(self, conf_id: int = -1, charge: int | None = None, uhf: int | None = None,
                        method: str = "GFN2-xTB") -> npt.NDArray[np.float32]:
        """
        Calculate xTB Mulliken charges.

        Parameters
        ----------
        conf_id: int
            RDKit conformer identifier.
        charge: int or None
            Molecular charge.
        uhf: int or None
            Number of unpaired electrons.
        method: str
            xTB method name.

        Returns
        -------
        numpy.ndarray
            Partial charges.
        """
        conf_id = self._resolve_conf_id(conf_id)
        cached = self._get_cached_charges(method="GFN2-xTB", conf_id=conf_id)

        if cached is not None:
            return cached

        self.get_xtb_energy(conf_id=conf_id, charge=charge, uhf=uhf, method=method)
        charges = self._get_cached_charges(method="GFN2-xTB", conf_id=conf_id)

        if charges is None:
            raise RuntimeError("xTB did not produce partial charges.")

        return charges

    def get_charges(self, method: ChargeMethod = "Gasteiger", conf_id: int = -1, heavy_only: bool = True,
                    combine_hydrogens: bool = False) -> npt.NDArray[np.float32]:
        """
        Return partial atomic charges.

        Parameters
        ----------
        method: ChargeMethod
            Charge calculation method.
        conf_id: int
            RDKit conformer identifier.
        heavy_only: bool
            Whether to return only non-hydrogen charges.
        combine_hydrogens: bool
            Whether to combine hydrogen charges with heavy atoms.

        Returns
        -------
        numpy.ndarray
            Partial charges.
        """
        if method == "Gasteiger":
            charges = self.get_gasteiger_charges()
        elif method in ("MMFF94", "MMFF94s"):
            charges = self.get_mmff_charges(variant=method)
        elif method == "GFN2-xTB":
            charges = self.get_xtb_charges(conf_id=conf_id)
        else:
            raise ValueError(f"Unknown charge method: {method!r}.")

        if combine_hydrogens:
            combined_charges = []

            for atom in self.get_atoms(heavy_only=True):
                atom_charge = charges[atom.GetIdx()]

                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        atom_charge += charges[neighbor.GetIdx()]

                combined_charges.append(atom_charge)

            return np.array(combined_charges, dtype=np.float32)

        if heavy_only:
            return charges[self.get_atom_indices(heavy_only=True)].copy()

        return charges.copy()

    def get_descriptors(self, method: EnergyMethod, conf_id: int = -1) -> dict[str, Any]:
        """
        Return cached calculation descriptors.

        Parameters
        ----------
        method: EnergyMethod
            Energy calculation method.
        conf_id: int
            RDKit conformer identifier.

        Returns
        -------
        dict
            Calculation descriptors.
        """
        conf_id = self._resolve_conf_id(conf_id)
        key = (conf_id, method)

        if key not in self._descriptors:
            self.get_energy(method=method, conf_id=conf_id)

        return {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in self._descriptors[key].items()
        }

    def save(self, path: str | Path) -> Path:
        """
        Serialize molecule state.

        Parameters
        ----------
        path: str or pathlib.Path
            Destination file path.

        Returns
        -------
        pathlib.Path
            Serialized molecule path.
        """
        output_path = Path(path).expanduser()
        state = {
            "version": 1,
            "smiles": self.smiles,
            "id": self.id,
            "mol_binary": self.mol.ToBinary(),
            "energy": self.energy,
            "energy_method": self.energy_method,
            "charge": self.charge,
            "uhf": self.uhf,
            "n_optimized": self.n_optimized,
            "lowest_energy_conf_id": self.lowest_energy_conf_id,
            "lowest_energy_method": self.lowest_energy_method,
            "energies": self._energies,
            "charge_sets": self._charge_sets,
            "descriptors": self._descriptors,
            "active_charge_method": self._active_charge_method,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> Molecule:
        """
        Restore molecule state.

        Parameters
        ----------
        path: str or pathlib.Path
            Serialized molecule path.

        Returns
        -------
        Molecule
            Restored molecule.
        """
        state = joblib.load(Path(path).expanduser())
        molecule = cls.__new__(cls)
        molecule.smiles = state["smiles"]
        molecule.id = state.get("id", state["smiles"])
        molecule.mol = Chem.Mol(state["mol_binary"])
        molecule.energy = state["energy"]
        molecule.energy_method = state["energy_method"]
        molecule.charge = state["charge"]
        molecule.uhf = state["uhf"]
        molecule.n_optimized = state["n_optimized"]
        molecule.lowest_energy_conf_id = state.get("lowest_energy_conf_id")
        molecule.lowest_energy_method = state.get("lowest_energy_method")
        molecule._energies = state["energies"]
        molecule._charge_sets = {
            key: np.array(values, dtype=np.float32)
            for key, values in state["charge_sets"].items()
        }
        molecule._descriptors = state["descriptors"]
        molecule._active_charge_method = state["active_charge_method"]
        return molecule


class DockMolecule(Molecule):
    """
    Molecule representation for docking and interaction fingerprints.
    """

    def __init__(self, smiles: str, id: str):
        """
        Initialize a docking molecule.

        Parameters
        ----------
        smiles: str
            SMILES representation used to construct the RDKit molecule.
        id: str
            Molecule identifier used in docking files.

        Returns
        -------
        None
        """
        super().__init__(smiles=smiles, id=id)
        self.dock_settings: dict[str, Any] = {}
        self.dock_score: float | None = None
        self.protein: str | None = None
        self.docked_file: str | None = None
        self.input_file: str | None = None
        self.h_file: str | None = None
        self.pose_files: list[str] = []
        self.interaction_df: Any = None

    @staticmethod
    def _executable(path: str) -> str:
        """
        Resolve an executable path.

        Parameters
        ----------
        path: str
            Executable command or file path.

        Returns
        -------
        str
            Resolved executable path.
        """
        resolved = shutil.which(path) if Path(path).parent == Path(".") else path

        if resolved is None or not Path(resolved).is_file():
            raise FileNotFoundError(f"Executable not found: {path}")

        return resolved

    def dock(self, settings: dict[str, Any], overwrite: bool = False) -> None:
        """
        Dock the current conformer with Smina.

        Parameters
        ----------
        settings: dict
            Docking settings.
        overwrite: bool
            Whether to replace existing docking files.

        Returns
        -------
        None
        """
        required = {
            "output_dir",
            "smina_path",
            "protein_path",
            "x_coord",
            "y_coord",
            "z_coord",
            "x_size",
            "y_size",
            "z_size",
            "exhaustiveness",
        }
        missing = required.difference(settings)

        if missing:
            raise KeyError(f"Missing docking settings: {', '.join(sorted(missing))}")

        if self.mol.GetNumConformers() == 0:
            raise ValueError("Generate a 3D conformer before docking.")

        output = Path(settings["output_dir"]).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        smina = self._executable(str(settings["smina_path"]))
        obabel = self._executable(str(settings.get("obabel_path", "obabel")))
        protein = Path(settings["protein_path"]).expanduser().resolve()

        if not protein.is_file():
            raise FileNotFoundError(f"Protein file not found: {protein}")

        mol_file = output / f"mol_{self.id}.mol"
        mol2_file = output / f"mol_{self.id}.mol2"
        out_file = output / f"mol_{self.id}_out.mol2"
        log_file = output / f"mol_{self.id}_out.log"

        if out_file.is_file() and not overwrite:
            self.read_dock(str(output))
            return

        Chem.MolToMolFile(self.mol, str(mol_file))
        subprocess.run([obabel, "-imol", str(mol_file), "-omol2", "-O", str(mol2_file)], check=True)
        subprocess.run(
            [
                smina,
                "-r",
                str(protein),
                "-l",
                str(mol2_file),
                "--center_x",
                str(settings["x_coord"]),
                "--center_y",
                str(settings["y_coord"]),
                "--center_z",
                str(settings["z_coord"]),
                "--size_x",
                str(settings["x_size"]),
                "--size_y",
                str(settings["y_size"]),
                "--size_z",
                str(settings["z_size"]),
                "--exhaustiveness",
                str(settings["exhaustiveness"]),
                "--out",
                str(out_file),
            ],
            check=True,
        )
        subprocess.run(
            [smina, "-r", str(protein), "-l", str(out_file), "--score_only", "--log", str(log_file)],
            check=True,
        )

        with log_file.open("a", encoding="utf-8") as file:
            for name, value in settings.items():
                file.write(f"< {name} > : < {value} >\n")

        self.read_dock(str(output))

    def read_dock(self, output_dir: str | Path) -> None:
        """
        Load an existing Smina docking result.

        Parameters
        ----------
        output_dir: str or pathlib.Path
            Directory containing molecule-specific docking output.

        Returns
        -------
        None
        """
        output = Path(output_dir).expanduser().resolve()
        log_file = output / f"mol_{self.id}_out.log"
        out_file = output / f"mol_{self.id}_out.mol2"
        mol_file = output / f"mol_{self.id}.mol"

        if not log_file.is_file() or not out_file.is_file():
            self.dock_score = np.nan
            return

        text = log_file.read_text(encoding="utf-8", errors="replace")
        scores = [
            float(value)
            for value in re.findall(r"Affinity:\s*([-+]?\d*\.?\d+)\s+\(kcal/mol\)", text)
        ]

        if not scores:
            raise RuntimeError(f"No Smina affinity was found in {log_file}.")

        self.dock_score = min(scores)
        self.docked_file = str(out_file)
        self.input_file = str(mol_file) if mol_file.is_file() else None
        self.dock_settings = dict(
            re.findall(r"^<\s*(.+?)\s*>\s*:\s*<\s*(.*?)\s*>$", text, re.MULTILINE)
        )
        self.protein = self.dock_settings.get("protein_path")

    def interactions(self, count: bool = False, file: str | None = None) -> Any:
        """
        Calculate protein-ligand interaction fingerprints.

        Parameters
        ----------
        count: bool
            Whether to return interaction counts.
        file: str or None
            Optional hydrogenated ligand MOL file.

        Returns
        -------
        polars.DataFrame
            Long-format interaction fingerprint table.
        """
        try:
            import MDAnalysis as mda
            import polars as pl
            import prolif as plf
        except ImportError as exc:
            raise ImportError(f"Method < interactions > requires MDAnalysis, Polars, and ProLIF:\n{exc}") from exc

        if self.protein is None or self.docked_file is None:
            raise RuntimeError("Dock the molecule or load docking output before calculating interactions.")

        protein = Path(self.protein)
        docked = Path(self.docked_file)

        if protein.suffix.lower() != ".pdb" or not protein.is_file():
            raise ValueError("protein must reference an existing PDB file.")

        if docked.suffix.lower() != ".mol2" or not docked.is_file():
            raise ValueError("docked_file must reference an existing MOL2 file.")

        temporary: Path | None = None

        if file is None:
            obabel = self._executable("obabel")
            mol_file = docked.with_suffix(".mol")
            h_mol = docked.with_name(f"{docked.stem}_H.mol")
            subprocess.run([obabel, "-imol2", str(docked), "-omol", "-O", str(mol_file)], check=True)
            subprocess.run([obabel, "-imol", str(mol_file), "-omol", "-O", str(h_mol), "-p", "7.4"], check=True)
            mol_file.unlink(missing_ok=True)
            temporary = h_mol
        else:
            h_mol = Path(file).expanduser().resolve()

            if h_mol.suffix.lower() != ".mol" or not h_mol.is_file():
                raise ValueError("file must reference an existing hydrogenated MOL file.")

        self.h_file = str(h_mol)
        content = h_mol.read_text(encoding="utf-8", errors="replace")
        blocks = [block.strip() + "\n" for block in content.split("$$$$") if block.strip()] or [content]
        receptor = plf.Molecule.from_mda(mda.Universe(str(protein)))
        frames: list[Any] = []
        self.pose_files = []

        try:
            for index, block in enumerate(blocks):
                pose = h_mol.with_name(f"{h_mol.stem}_{index}.mol")
                pose.write_text(block, encoding="utf-8")
                self.pose_files.append(str(pose))

                try:
                    ligand = Chem.MolFromMolFile(str(pose), removeHs=False)

                    if ligand is None:
                        continue

                    fingerprint = plf.Fingerprint()
                    fingerprint.run_from_iterable([plf.Molecule.from_rdkit(ligand)], receptor)
                    frame = fingerprint.to_dataframe(index_col="Pose", drop_empty=False, count=count)
                    frames.append(pl.from_pandas(frame).unpivot().with_columns(pl.lit(index).alias("dock_number")))
                finally:
                    pose.unlink(missing_ok=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        if not frames:
            raise RuntimeError("No interaction fingerprint could be calculated from the docked poses.")

        result = pl.concat(frames, how="vertical_relaxed")

        if "ligand" in result.columns:
            result = result.drop("ligand")

        self.interaction_df = result.with_columns(
            pl.lit(self.id).alias("Mol_ID"),
            pl.lit(self.smiles).alias("SMILES"),
        ).rename({"protein": "residue"})
        return self.interaction_df
