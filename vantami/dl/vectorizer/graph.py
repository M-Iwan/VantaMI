from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

try:
    from rdkit import Chem, RDLogger
    import torch
    from torch_geometric.data import Data as Graph
    from vantami.chem.molecule import DeepMolecule
except ImportError as exc:
    raise ImportError(f"GraphVectorizer requires rdkit, torch, and torch_geometric:\n{exc} ")


class GraphVectorizer:
    """
    Turn SMILES or prepared DeepMolecule objects into torch_geometric Data.

    Node features contain atom descriptors and a partial-charge feature. Edge features contain bond descriptors,
    3D bond distance, and a binary flag indicating whether 3D geometry was available.
    """

    def __init__(self, atom_encoding: dict = None, bond_encoding: dict = None, suppress: bool = True):

        if atom_encoding is None:
            self.atom_encoding = {
                "C": 0, "N": 1, "O": 2, "S": 3, "F": 4, "P": 5, "Cl": 6, "Mg": 7, "Na": 8, "Br": 9, "Fe": 10, "Ca": 11,
                "Cu": 12, "Mc": 13, "Pd": 14, "Pb": 15, "K": 16, "I": 17, "Al": 18, "Ni": 19, "Mn": 20,
            }
        else:
            self.atom_encoding = atom_encoding

        self.groups = {
            0: ['H', 'C', 'N', 'O', 'P', 'S'],  # non_metals
            1: ['Li', 'Na', 'K', 'Rb', ' Cs', 'Fr'],  # alkaline metals
            2: ['Be', 'Mg', 'Ca', 'Sr', ' Ba', 'Ra'],  # alkaline earth metals
            3: ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Age', 'Cd',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Rf', 'Db', 'Sg', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'],  # transition metals
            4: ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Nh', 'Fl', 'Mc', 'Lv'],  # metals
            5: ['B', 'Si', 'Ge', 'As', 'Sb', 'Te', 'Po'],  # metalloids
            6: ['F', 'Cl', 'Br', 'I', 'At', 'Ts'],  # halogens
            7: ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og'],  # noble gases
            8: ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],  # lanthanide
            9: ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']  # actinides
        }

        self.group_encoding = {}
        for group, elements in self.groups.items():
            for element in elements:
                self.group_encoding[element] = group

        self.atom_encoding_size = len(self.atom_encoding) + 1
        self.group_encoding_size = len(self.groups) + 1

        if bond_encoding is None:
            self.bond_encoding = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        else:
            self.bond_encoding = bond_encoding

        self.bond_encoding_size = len(self.bond_encoding) + 1

        if suppress:
            RDLogger.DisableLog("rdApp.*")

    def encode_atom(self, atom: Chem.rdchem.Atom, partial_charge: float = 0.0) -> npt.NDArray:
        """
        Encode one atom and append its partial-charge feature.
        """
        type_enc = np.zeros(shape=(self.atom_encoding_size), dtype=np.float32)
        type_enc[self.atom_encoding.get(atom.GetSymbol(), -1)] = 1.0

        group_enc = np.zeros(shape=(self.group_encoding_size), dtype=np.float32)
        group_enc[self.group_encoding.get(atom.GetSymbol(), -1)] = 1.0

        prop_enc = np.asarray(
            [
                atom.GetFormalCharge(),
                atom.GetHybridization().real,
                atom.GetIsAromatic(),
                atom.GetNumExplicitHs(),
                atom.GetDegree(),
                atom.IsInRing(),
                partial_charge,
            ],
            dtype=np.float32,
        )

        return np.hstack((type_enc, group_enc, prop_enc))

    def encode_mol_atoms(self, mol: Chem.rdchem.Mol, charges: Optional[np.ndarray] = None,
                         heavy_only: bool = True) -> npt.NDArray:
        """
        Encode molecular atoms.
        """
        atoms = [atom for atom in mol.GetAtoms() if not heavy_only or atom.GetAtomicNum() > 1]

        if charges is None:
            charges = np.zeros(len(atoms), dtype=np.float32)

        atom_list = [self.encode_atom(atom, partial_charge=float(charge)) for atom, charge in zip(atoms, charges)]

        return np.vstack(atom_list).astype(np.float32)

    def encode_bond(self, bond: Chem.rdchem.Bond, atom_indices: Optional[dict[int, int]] = None,
                    distance: float = 0.0) -> Tuple[list[list[int]], np.ndarray]:
        """
        Encode one bond and its interatomic distance.
        """
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if atom_indices is not None:
            start_idx = atom_indices[start_idx]
            end_idx = atom_indices[end_idx]

        bond_edges = [[start_idx, end_idx], [end_idx, start_idx]]

        bond_type = str(bond.GetBondType())
        type_enc = np.zeros(shape=(self.bond_encoding_size), dtype=np.float32)
        type_enc[self.bond_encoding.get(bond_type, -1)] = 1.0

        prop_enc = np.asarray(
            [
                bond.GetIsAromatic(),
                bond.GetIsConjugated(),
                bond.IsInRing(),
                distance,
            ],
            dtype=np.float32,
        )

        return bond_edges, np.hstack((type_enc, prop_enc))

    def encode_mol_bonds(self, mol: Chem.rdchem.Mol, positions: Optional[npt.NDArray] = None,
                         heavy_only: bool = False) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Encode bonds.
        """
        atoms = [atom for atom in mol.GetAtoms() if not heavy_only or atom.GetAtomicNum() > 1]
        atom_indices = {atom.GetIdx(): graph_index for graph_index, atom in enumerate(atoms)}

        bonds = [
            bond for bond in mol.GetBonds() if not heavy_only
            or (bond.GetBeginAtom().GetAtomicNum() > 1 and bond.GetEndAtom().GetAtomicNum() > 1)
        ]

        if not bonds:
            return (
                np.array([0, 0], dtype=np.int64).reshape(2, 1),
                np.zeros(shape=(1, self.bond_encoding_size + 4), dtype=np.float32,
                ),
            )

        edges, encoding = [], []

        for bond in bonds:
            distance = 0.0

            if positions is not None:
                start_idx = atom_indices[bond.GetBeginAtomIdx()]
                end_idx = atom_indices[bond.GetEndAtomIdx()]
                distance = float(np.linalg.norm(positions[start_idx] - positions[end_idx]))

            bond_edges, bond_enc = self.encode_bond(bond=bond, atom_indices=atom_indices, distance=distance)

            edges.extend(bond_edges)
            encoding.extend([bond_enc, bond_enc])

        return (np.asarray(edges, dtype=np.int64).T, np.vstack(encoding).astype(np.float32))

    def from_smiles(self, smiles: str, heavy_only: bool = True) -> Graph:
        """
        Build one graph from a SMILES string.

        The partial-charge and bond-distance are set to zero.
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=True)

        atoms_encoding_array = self.encode_mol_atoms(mol)
        edges_array, bonds_encoding_array = self.encode_mol_bonds(mol, heavy_only=heavy_only)

        graph_data = {
            "x": torch.from_numpy(atoms_encoding_array),
            "edge_index": torch.from_numpy(edges_array),
            "edge_attr": torch.from_numpy(bonds_encoding_array),
        }

        return Graph(**graph_data)

    def from_deepmol(self, deepmol: vantami.chem.molecule.DeepMolecule, charge_method: str = "Gasteiger",
                     energy_method: Optional[str] = None, heavy_only: bool = True, combine_hydrogens: bool = False) -> Graph:
        """
        Build a heavy-atom graph from a prepared DeepMolecule.
        """
        conf_id, _ = deepmol.get_lowest_energy_conformer(method=energy_method)

        charges = deepmol.get_charges(method=charge_method, conf_id=conf_id, heavy_only=heavy_only,
                                      combine_hydrogens=combine_hydrogens)

        positions = deepmol.get_conformer_positions(conf_id=conf_id, heavy_only=heavy_only)

        atoms_encoding_array = self.encode_mol_atoms(mol=deepmol.mol, charges=charges, heavy_only=heavy_only)

        edges_array, bonds_encoding_array = self.encode_mol_bonds(mol=deepmol.mol, positions=positions,
                                                                  heavy_only=heavy_only)

        graph_data = {
            "x": torch.from_numpy(atoms_encoding_array),
            "edge_index": torch.from_numpy(edges_array),
            "edge_attr": torch.from_numpy(bonds_encoding_array),
        }

        return Graph(**graph_data)

    def encode_smiles(self, smiles: Union[str, Iterable[str]]) -> List[Graph]:
        """
        Vectorize one SMILES or an iterable of SMILES strings.
        """
        if isinstance(smiles, str):
            return [self.from_smiles(smiles)]

        elif hasattr(smiles, "__iter__") and all(isinstance(item, str) for item in smiles):
            return [self.from_smiles(smi) for smi in smiles]

        else:
            raise TypeError(f"Expected < smiles > to be either str or Iterable[str], got {type(smiles)} instead.")


    def encode_deepmol(self, deepmol: Union[DeepMolecule, Iterable[DeepMolecule]]) -> List[Graph]:
        """
        Vectorize one DeepMolecule or an iterable of DeepMolecules.
        """
        try:
            from vantami.chem.molecule import DeepMolecule
        except ImportError as exc:
            raise ImportError(f"Method < encode_deepmol > requires < vantami > 0.5.2 or higher.")
        if isinstance(deepmol, DeepMolecule):
            return [self.from_deepmol(deepmol)]

        elif hasattr(deepmol, "__iter__") and all (isinstance(item, DeepMolecule) for item in deepmol):
            return [self.from_deepmol(dm) for dm in deepmol]

        else:
            raise TypeError(f"Expected < deepmol > to be either DeepMolecule or Iterable[DeepMolecule],"
                            f"got {type(deepmol)} instead.")