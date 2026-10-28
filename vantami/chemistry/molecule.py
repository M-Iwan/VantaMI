"""Small-molecule preparation, docking, and interaction-fingerprint utilities."""

from __future__ import annotations

import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from rdkit import Chem
from rdkit.Chem import Draw, rdPartialCharges
from rdkit.Chem.AllChem import (ETKDGv3, EmbedMolecule, EmbedMultipleConfs,
                                MMFFGetMoleculeForceField, MMFFGetMoleculeProperties, MMFFOptimizeMolecule,
                                UFFGetMoleculeForceField, UFFHasAllMoleculeParams, UFFOptimizeMolecule
                                )


class Molecule:
    """
    Represent a canonical small molecule and its prepared 3D state.

    Parameters
    ----------
    smiles: str
        Valid, canonical SMILES encoding all intended information.
    id: str
        Molecule identifier used in files and output tables.

    Notes
    -----
    Updated in vantami v.0.5.2
    """

    def __init__(self, smiles: str, id: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        self.smiles = smiles
        self.id = str(id)
        self.mol = mol
        self.energy: Optional[float] = None
        self.energy_method: Optional[str] = None
        self.charges: Optional[np.ndarray] = None
        self.charge_method: Optional[str] = None
        self.dock_settings: dict = {}
        self.dock_score: Optional[float] = None
        self.protein: Optional[str] = None
        self.docked_file: Optional[str] = None
        self.input_file: Optional[str] = None
        self.h_file: Optional[str] = None
        self.pose_files: list[str] = []
        self.interaction_df: Optional[pl.DataFrame] = None
        self.charge: Optional[int] = None
        self.uhf: Optional[int] = None
        self.n_optimized: int = 0

    def __str__(self) -> str:
        return f"Molecule object of < {self.smiles} >"

    def __repr__(self) -> str:
        return f"Molecule object of < {self.smiles} > with id < {self.id} >"

    @staticmethod
    def _with_hs(mol: Chem.Mol, keep_coords: bool = False) -> Chem.Mol:
        """
        Copy a molecule with explicit hydrogen atoms.

        Parameters
        ----------
        mol: rdkit.Chem.Mol
            Molecule to copy.
        keep_coords: bool, default=False
            Preserve a present conformer when adding hydrogen atoms.

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
    def _ff_energy(mol: Chem.Mol, conf_id: int) -> tuple[float, str]:
        """
        Optimize a conformer and return its MMFF94s or UFF energy.

        Parameters
        ----------
        mol: rdkit.Chem.Mol
            Molecule containing the conformer.
        conf_id: int
            Conformer identifier.

        Returns
        -------
        tuple[float, str]
            Final force-field energy and its method name.
        """
        props = MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is not None:
            status = MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=1024, confId=conf_id)
            if status == 1:
                MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=4096, confId=conf_id)
            field = MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            return float(field.CalcEnergy()), "MMFF94s"

        if UFFHasAllMoleculeParams(mol):
            status = UFFOptimizeMolecule(mol, maxIters=1024, confId=conf_id)
            if status == 1:
                UFFOptimizeMolecule(mol, maxIters=4096, confId=conf_id)
            field = UFFGetMoleculeForceField(mol, confId=conf_id)
            return float(field.CalcEnergy()), "UFF"

        return np.nan, "none"

    @classmethod
    def _best_ff_conformer(cls, mol: Chem.Mol, n: int = 24, seed: int = 42) -> tuple[Chem.Mol, float, str]:
        """
        Generate and select the lowest force-field-energy conformer.

        Parameters
        ----------
        mol: rdkit.Chem.Mol
            Molecule to embed.
        n: int, default=24
            Number of ETKDG conformers to generate.
        seed: int, default=42
            Random seed for ETKDG embedding.

        Returns
        -------
        tuple[rdkit.Chem.Mol, float, str]
            Molecule containing one selected conformer, its energy, and the force-field method used.
        """
        prepared = cls._with_hs(mol)
        params = ETKDGv3()
        params.randomSeed = seed
        params.pruneRmsThresh = 0.25
        params.useSmallRingTorsions = True
        conf_ids = list(EmbedMultipleConfs(prepared, numConfs=n, params=params))

        if not conf_ids:
            conf_id = EmbedMolecule(prepared, maxAttempts=1000, useRandomCoords=True, randomSeed=seed)
            if conf_id < 0:
                raise RuntimeError("RDKit could not generate a 3D conformer.")
            conf_ids = [conf_id]

        results = []
        for conf_id in conf_ids:
            try:
                energy, method = cls._ff_energy(prepared, conf_id)
            except Exception:
                energy, method = np.nan, "none"
            results.append((energy, method, conf_id))

        valid = [result for result in results if np.isfinite(result[0])]
        if valid:
            energy, method, conf_id = min(valid, key=lambda result: result[0])
        else:
            energy, method, conf_id = np.nan, "unoptimized", conf_ids[0]

        selected = Chem.Mol(prepared)
        conformer = Chem.Conformer(prepared.GetConformer(conf_id))
        selected.RemoveAllConformers()
        selected.AddConformer(conformer, assignId=True)
        return selected, float(energy), method

    @staticmethod
    def _store_charges(mol: Chem.Mol, charges: np.ndarray, method: str) -> None:
        """
        Store atomic partial charges on an RDKit molecule.

        Parameters
        ----------
        mol: rdkit.Chem.Mol
            Molecule corresponding to the charges.
        charges: numpy.ndarray
            Partial charges in RDKit atom-index order.
        method: str
            Charge-model name.

        Returns
        -------
        None
        """
        if len(charges) != mol.GetNumAtoms():
            raise ValueError("The number of charges does not match the number of atoms.")
        if not np.all(np.isfinite(charges)):
            raise RuntimeError(f"{method} produced non-finite partial charges.")
        for atom, charge in zip(mol.GetAtoms(), charges):
            atom.SetDoubleProp("PARTIAL_CHARGE", float(charge))
            atom.SetProp("PARTIAL_CHARGE_METHOD", method)
        mol.SetProp("PARTIAL_CHARGE_METHOD", method)

    def validate(self) -> None:
        """
        Validate the current molecule.
        """
        Chem.SanitizeMol(self.mol)

    def show(self):
        """
        Render the current molecule.

        Returns
        -------
        PIL.Image.Image
            RDKit molecular depiction.
        """
        return Draw.MolToImage(self.mol, size=(512, 512))

    def embed(self, n: int = 16, seed: int = 42) -> Chem.Mol:
        """
        Generate the lowest-energy force-field conformer.

        Parameters
        ----------
        n: int, default=16
            Number of ETKDG conformers to sample.
        seed: int, default=42
            Random seed for conformer generation.

        Returns
        -------
        rdkit.Chem.Mol
            Current molecule with explicit hydrogen atoms and one selected conformer.
        """
        self.mol, self.energy, self.energy_method = self._best_ff_conformer(self.mol, n=n, seed=seed)
        self.charges = None
        self.charge_method = None
        return self.mol

    def conformers(self, n: int = 32, charge: Optional[int] = None, uhf: Optional[int] = None,
                   fmax: float = 0.05, steps: int = 256, seed: int = 42, method: str = "xtb"):
        """
        Select the lowest-energy conformer of the input molecular state.

        Parameters
        ----------
        n: int, default=32
            Number of initial ETKDG conformers.
        charge: int or None, default=None
            Total charge for xTB. Formal charges are summed when ``None``.
        uhf: int or None, default=None
            Number of unpaired electrons for xTB. Radical electrons are summed when ``None``.
        fmax: float, default=0.05
            xTB force tolerance in eV/Angstrom.
        steps: int, default=256
            Maximum BFGS optimization steps per conformer.
        seed: int, default=42
            Random seed for ETKDG embedding.
        method: {"xtb", "ff"}, default="xtb"
            "xtb" performs GFN2-xTB optimization. "ff" performs MMFF94s optimization with a UFF fallback.

        Returns
        -------
        rdkit.Chem.Mol
            Current molecule with explicit hydrogen atoms and one selected conformer.
        """
        method = method.lower()
        if method not in {"xtb", "ff"}:
            raise ValueError("Method must be either 'xtb' or 'ff'.")
        if n < 1:
            raise ValueError("N must be at least 1.")

        if method == "ff":
            return self.embed(n=n, seed=seed)

        try:
            from ase import Atoms
            from ase.optimize import BFGS
            from xtb.ase.calculator import XTB
        except ImportError as exc:
            raise ImportError(f"Method < conformers > with method='xtb' requires ASE and xtb-python:\n{exc}") from exc

        mol = self._with_hs(self.mol)
        if charge is None:
            charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        if uhf is None:
            uhf = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())

        params = ETKDGv3()
        params.randomSeed = seed
        params.pruneRmsThresh = 0.25
        params.useSmallRingTorsions = True
        conf_ids = list(EmbedMultipleConfs(mol, numConfs=n, params=params))
        if not conf_ids:
            raise RuntimeError(f"RDKit could not generate 3D conformers for < {self.smiles} >.")

        for conf_id in conf_ids:
            try:
                self._ff_energy(mol, conf_id)
            except Exception:
                pass

        numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        best_energy = np.inf
        best_positions = None
        success = 0
        for conf_id in conf_ids:
            conformer = mol.GetConformer(conf_id)
            atoms = Atoms(
                numbers=numbers,
                positions=np.asarray(conformer.GetPositions(), dtype=float),
                pbc=False,
            )
            atoms.calc = XTB(method="GFN2-xTB", charge=charge, uhf=uhf)
            try:
                BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
                energy = float(atoms.get_potential_energy())
                if energy < best_energy:
                    best_energy = energy
                    best_positions = atoms.get_positions().copy()
                success += 1
            except Exception:
                continue

        if best_positions is None:
            raise RuntimeError("GFN2-xTB optimization failed for all generated conformers.")

        mol.RemoveAllConformers()
        optimized = Chem.Conformer(mol.GetNumAtoms())
        for index, position in enumerate(best_positions):
            optimized.SetAtomPosition(index, tuple(map(float, position)))
        mol.AddConformer(optimized, assignId=True)
        mol.SetProp("ENERGY_METHOD", "GFN2-xTB")
        mol.SetProp("ENERGY_EV", f"{best_energy:.8f}")
        mol.SetProp("TOTAL_CHARGE", str(charge))
        mol.SetProp("UHF", str(uhf))

        self.mol = mol
        self.energy = best_energy
        self.energy_method = "GFN2-xTB"
        self.charge = charge
        self.uhf = uhf
        self.n_optimized = success
        self.charges = None
        self.charge_method = None
        return self.mol

    def gasteiger_charges(self) -> np.ndarray:
        """
        Assign RDKit Gasteiger partial charges.

        Returns
        -------
        numpy.ndarray
            Gasteiger charges in RDKit atom-index order.
        """
        self.mol = self._with_hs(self.mol, keep_coords=True)
        rdPartialCharges.ComputeGasteigerCharges(self.mol)
        charges = np.asarray(
            [float(atom.GetProp("_GasteigerCharge")) for atom in self.mol.GetAtoms()],
            dtype=float,
        )
        self._store_charges(self.mol, charges, "GASTEIGER")
        self.charges = charges
        self.charge_method = "GASTEIGER"
        return charges

    def mmff_charges(self, variant: str = "MMFF94s") -> np.ndarray:
        """
        Assign MMFF partial charges.

        Parameters
        ----------
        variant : {"MMFF94", "MMFF94s"}, default="MMFF94s"
            RDKit MMFF parameterization variant.

        Returns
        -------
        numpy.ndarray
            MMFF charges in RDKit atom-index order.
        """
        self.mol = self._with_hs(self.mol, keep_coords=True)
        props = MMFFGetMoleculeProperties(self.mol, mmffVariant=variant)
        if props is None:
            raise ValueError(f"MMFF parameters are unavailable for < {self.smiles} >.")
        charges = np.asarray(
            [props.GetMMFFPartialCharge(i) for i in range(self.mol.GetNumAtoms())],
            dtype=float,
        )
        method = f"{variant}_PARTIAL_CHARGE"
        self._store_charges(self.mol, charges, method)
        self.charges = charges
        self.charge_method = method
        return charges

    def xtb_charges(self, charge: Optional[int] = None, uhf: Optional[int] = None, method: str = "GFN2-xTB"):
        """
        Calculate xTB Mulliken partial charges for the current conformer.

        Parameters
        ----------
        charge: int or None, default=None
            Total molecular charge. Formal charges are summed when ``None``.
        uhf: int or None, default=None
            Number of unpaired electrons. Radical electrons are summed when
            ``None``.
        method: str, default="GFN2-xTB"
            xTB method accepted by ``xtb-python``.

        Returns
        -------
        numpy.ndarray
            xTB Mulliken charges in RDKit atom-index order.
        """
        try:
            from ase import Atoms
            from xtb.ase.calculator import XTB
        except ImportError as exc:
            raise ImportError(f"Method < xtb_charges > requires ASE and xtb-python:\n{exc}") from exc

        if self.mol.GetNumConformers() == 0:
            raise ValueError("Generate a 3D conformer before assigning xTB charges.")
        if charge is None:
            charge = sum(atom.GetFormalCharge() for atom in self.mol.GetAtoms())
        if uhf is None:
            uhf = sum(atom.GetNumRadicalElectrons() for atom in self.mol.GetAtoms())

        conformer = self.mol.GetConformer()
        atoms = Atoms(
            numbers=[atom.GetAtomicNum() for atom in self.mol.GetAtoms()],
            positions=np.asarray(conformer.GetPositions(), dtype=float),
            pbc=False,
        )
        atoms.calc = XTB(method=method, charge=charge, uhf=uhf)
        atoms.get_potential_energy()
        charges = np.asarray(atoms.get_charges(), dtype=float)
        self._store_charges(self.mol, charges, "XTB_MULLIKEN")
        self.charges = charges
        self.charge_method = "XTB_MULLIKEN"
        return charges

    @staticmethod
    def _executable(path: str) -> str:
        """
        Resolve an executable path.

        Parameters
        ----------
        path : str
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

    def dock(self, settings: dict, overwrite: bool = False) -> None:
        """
        Dock the current 3D conformer with Smina.

        Parameters
        ----------
        settings : dict
            Requires ``output_dir``, ``smina_path``, ``protein_path``, docking
            box centers and sizes, and ``exhaustiveness``. ``obabel_path`` is
            optional.
        overwrite : bool, default=False
            Replace prior docking files when present.
        """

        required = {
            "output_dir", "smina_path", "protein_path", "x_coord", "y_coord", "z_coord",
            "x_size", "y_size", "z_size", "exhaustiveness",
        }

        missing = required.difference(settings)
        if missing:
            raise KeyError(f"Missing docking settings: {', '.join(sorted(missing))}")

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

        if self.mol.GetNumConformers() == 0:
            self.embed()

        Chem.MolToMolFile(self.mol, str(mol_file))
        subprocess.run([obabel, "-imol", str(mol_file), "-omol2", "-O", str(mol2_file)], check=True)

        subprocess.run([
            smina, "-r", str(protein), "-l", str(mol2_file),
            "--center_x", str(settings["x_coord"]), "--center_y", str(settings["y_coord"]),
            "--center_z", str(settings["z_coord"]), "--size_x", str(settings["x_size"]),
            "--size_y", str(settings["y_size"]), "--size_z", str(settings["z_size"]),
            "--exhaustiveness", str(settings["exhaustiveness"]), "--out", str(out_file),
        ], check=True)

        subprocess.run([smina, "-r", str(protein), "-l", str(out_file), "--score_only", "--log", str(log_file)], check=True)

        with log_file.open("a", encoding="utf-8") as file:
            for name, value in settings.items():
                file.write(f"< {name} > : < {value} >\n")

        self.read_dock(str(output))

    def read_dock(self, output_dir: str) -> None:
        """
        Load an existing Smina docking result.

        Parameters
        ----------
        output_dir: str
            Directory containing molecule-specific Smina output.
        """
        output = Path(output_dir).expanduser().resolve()
        log_file = output / f"mol_{self.id}_out.log"
        out_file = output / f"mol_{self.id}_out.mol2"
        mol_file = output / f"mol_{self.id}.mol"

        if not log_file.is_file() or not out_file.is_file():
            self.dock_score = np.nan
            return

        text = log_file.read_text(encoding="utf-8", errors="replace")
        scores = [float(value) for value in re.findall(r"Affinity:\s*([-+]?\d*\.?\d+)\s+\(kcal/mol\)", text)]

        if not scores:
            raise RuntimeError(f"No Smina affinity was found in {log_file}.")

        self.dock_score = min(scores)
        self.docked_file = str(out_file)
        self.input_file = str(mol_file) if mol_file.is_file() else None
        self.dock_settings = dict(re.findall(r"^<\s*(.+?)\s*>\s*:\s*<\s*(.*?)\s*>$", text, re.MULTILINE))
        self.protein = self.dock_settings.get("protein_path")

    def interactions(self, count: bool = False, file: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate ProLIF protein-ligand interaction fingerprints.

        Parameters
        ----------
        count: bool, default=False
            Return interaction counts instead of bit fingerprints.
        file: str or None, default=None
            Hydrogenated ligand MOL file. When ``None``, it is prepared from
            the docked MOL2 output with Open Babel.

        Returns
        -------
        pl.DataFrame
            Long-format interaction fingerprint table for every docked pose.
        """
        try:
            import MDAnalysis as mda
            import prolif as plf
        except ImportError as exc:
            raise ImportError(f"Method < interactions > requires MDAnalysis and ProLIF:\n{exc}") from exc

        if self.protein is None or self.docked_file is None:
            raise RuntimeError("Dock the molecule or load docking output before calculating interactions.")

        protein = Path(self.protein)
        docked = Path(self.docked_file)

        if protein.suffix.lower() != ".pdb" or not protein.is_file():
            raise ValueError("protein must reference an existing PDB file.")

        if docked.suffix.lower() != ".mol2" or not docked.is_file():
            raise ValueError("docked_file must reference an existing MOL2 file.")

        temporary = None

        if file is None:
            obabel = self._executable("obabel")
            mol = docked.with_suffix(".mol")
            h_mol = docked.with_name(f"{docked.stem}_H.mol")
            subprocess.run([obabel, "-imol2", str(docked), "-omol", "-O", str(mol)], check=True)
            subprocess.run([obabel, "-imol", str(mol), "-omol", "-O", str(h_mol), "-p", "7.4"], check=True)
            mol.unlink(missing_ok=True)
            temporary = h_mol
        else:
            h_mol = Path(file).expanduser().resolve()
            if h_mol.suffix.lower() != ".mol" or not h_mol.is_file():
                raise ValueError("file must reference an existing hydrogenated MOL file.")

        self.h_file = str(h_mol)
        content = h_mol.read_text(encoding="utf-8", errors="replace")
        blocks = [block.strip() + "\n" for block in content.split("$$$$") if block.strip()] or [content]
        receptor = plf.Molecule.from_mda(mda.Universe(str(protein)))
        frames = []
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
        result = pl.concat(frames, how="vertical_relaxed").drop("ligand")
        result["Mol_ID"] = self.id
        result["SMILES"] = self.smiles
        self.interaction_df = result.rename(columns={"protein": "residue"})
        return self.interaction_df

    def save(self, path: str | Path):
        """
        Save this Molecule instance as paired SDF and Parquet files.

        Parameters
        ----------
        path : str or pathlib.Path
            File prefix. A supplied suffix is ignored.
        """
        base = Path(path).expanduser()
        if base.suffix:
            base = base.with_suffix("")

        base.parent.mkdir(parents=True, exist_ok=True)

        sdf_file = base.with_suffix(".sdf")
        parquet_file = base.with_suffix(".parquet")

        rdkit_properties = {
            name: self.mol.GetProp(name)
            for name in self.mol.GetPropNames(includePrivate=False, includeComputed=False)
        }

        metadata = pl.DataFrame({
                "id": [self.id],
                "smiles": [self.smiles],
                "energy": [self.energy],
                "energy_method": [self.energy_method],
                "charges": [self.charges.tolist() if self.charges is not None else None],
                "charge_method": [self.charge_method],
                "charge": [self.charge],
                "uhf": [self.uhf],
                "n_optimized": [self.n_optimized],
                "rdkit_properties": [json.dumps(rdkit_properties, separators=(",", ":"))],
            })

        writer = Chem.SDWriter(str(sdf_temp))
        writer.write(self.mol)
        writer.close()

        metadata.write_parquet(parquet_temp, compression="zstd")

    @classmethod
    def load(cls, path: str | Path) -> "Molecule":
        """
        Load a Molecule instance saved with ``Molecule.save()``.

        Parameters
        ----------
        path : str or pathlib.Path
            File prefix originally passed to ``save()``.

        Returns
        -------
        Molecule
        """
        base = Path(path).expanduser()
        if base.suffix:
            base = base.with_suffix("")

        sdf_file = base.with_suffix(".sdf")
        parquet_file = base.with_suffix(".parquet")

        if not sdf_file.is_file():
            raise FileNotFoundError(f"SDF file not found: {sdf_file}")
        if not parquet_file.is_file():
            raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

        metadata = pl.read_parquet(parquet_file)
        if metadata.height != 1:
            raise ValueError(f"Expected one record in file, found {metadata.height}")

        data = metadata.row(0, named=True)

        supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
        mol = next((molecule for molecule in supplier if molecule is not None), None)
        if mol is None:
            raise ValueError(f"Could not load a valid molecule from {sdf_file}.")

        molecule = cls(smiles=data["smiles"], id=data["id"])
        molecule.mol = mol
        molecule.energy = data["energy"]
        molecule.energy_method = data["energy_method"]
        molecule.charge_method = data["charge_method"]
        molecule.charge = data["charge"]
        molecule.uhf = data["uhf"]
        molecule.n_optimized = int(data["n_optimized"] or 0)

        if data["charges"] is not None:
            molecule.charges = np.asarray(data["charges"], dtype=float)
            molecule._store_charges(
                molecule.mol,
                molecule.charges,
                molecule.charge_method or "STORED",
            )

        for name, value in json.loads(data["rdkit_properties"] or "{}").items():
            molecule.mol.SetProp(name, str(value))

        return molecule
