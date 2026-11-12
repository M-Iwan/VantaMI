from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Literal, Dict

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from itertools import chain

try:
    from rdkit import Chem, RDLogger
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import radius_graph
    from vantami.chem.molecule import Molecule
    from vantami.data.partition import batch_data
except ImportError as exc:
    raise ImportError(f"GraphVectorizer requires rdkit, torch, torch_geometric, and torch_cluster") from exc

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


MMFFVariant = Literal["MMFF94", "MMFF94s"]
EnergyMethod = Literal["MMFF94", "MMFF94s", "UFF", "GFN2-xTB"]
ChargeMethod = Literal["Gasteiger", "MMFF94", "MMFF94s", "GFN2-xTB"]


class GaussianRBF(torch.nn.Module):
    """
    Gaussian radial-basis encoding for paired distances.
    """
    def __init__(self, r_cut: float = 6.0, n_rbf: int = 32, eta: float | None = None):
        super().__init__()

        if r_cut <= 0:
            raise ValueError("< r_cut > must be a positive float")
        if n_rbf <= 1:
            raise ValueError("< n_rbf > must be at least 2")

        centers = torch.linspace(0.0, r_cut, n_rbf)  # Centers for Gaussian Functions
        self.register_buffer("centers", centers)

        spacing = r_cut / (n_rbf - 1)
        if eta is None:
            eta = 1.0 / (spacing ** 2)
        if eta <= 0:
            raise ValueError("< eta > must be a positive float")

        self.register_buffer("eta", torch.tensor(float(eta)))

    def forward(self, distances: torch.Tensor):
        """
        Gaussian radial-basis encoding for scalar interatomic distances.

        Parameters
        ----------
        distances: torch.Tensor
            Tensor of shape [E] containing interatomic distances

        Returns
        -------
        torch.Tensor
            Gaussian RBF features with shape [E, n_rbf]
        """
        if distances.ndim != 1:
            raise ValueError(f"Expected distances with shape [E], got {distances.shape} instead.")

        centers = self.centers.to(dtype=distances.dtype)
        eta = self.eta.to(dtype=distances.dtype)

        return torch.exp(
            -eta * (distances.unsqueeze(-1) - centers).square()
        )

class GaussianABF(torch.nn.Module):
    """
    Gaussian angular-basis encoding for triplet angles.
    """

    def __init__(self, n_abf: int = 16, eta: float | None = None):
        super().__init__()

        if n_abf < 2:
            raise ValueError("< n_abf > must be at least 2.")

        centers = torch.linspace(0.0, torch.pi, n_abf)
        self.register_buffer("centers", centers)

        spacing = torch.pi / (n_abf - 1)

        if eta is None:
            eta = 1.0 / (spacing**2)

        if eta <= 0:
            raise ValueError("< eta > must be a positive float")

        self.register_buffer("eta", torch.tensor(float(eta)))

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Calculate Gaussian angular-basis features.

        Parameters
        ----------
        angles:
            Tensor of shape [num_triplets] containing angles in radians.

        Returns
        -------
        torch.Tensor
            Tensor of shape [num_triplets, n_abf].
        """
        if angles.ndim != 1:
            raise ValueError(f"Expected angles with shape [num_triplets], got {tuple(angles.shape)}.")

        centers = self.centers.to(dtype=angles.dtype)
        eta = self.eta.to(dtype=angles.dtype)

        return torch.exp(
            -eta * (angles.unsqueeze(-1) - centers).square()
        )


class GraphVectorizer:
    """
    Turn SMILES or prepared Molecule objects into torch_geometric Data.

    Node features contain atom descriptors, periodic table group, and a partial-charge feature.
    Edge features contain bond descriptors and 3D bond distance.

    Updated in Vantami 0.5.3rc3
    """

    def __init__(self, atom_encoding: Dict[str, int] | None = None, bond_encoding: Dict[str, int] | None = None,
                 group_encoding: Dict[int, List[str]] | None = None, suppress: bool = True):
        """
        Initialize the graph vectorizer with atom and bond encoding schemes.

        Parameters
        ----------
        atom_encoding: Dict[str, int] | None
            Mapping of atom symbols to integer encoding indices. If `None`, the default atom encoding is used.
        bond_encoding: Dict[str, int] | None
            Mapping of bond-type names to integer encoding indices. If `None`, the default bond encoding is used.
        group_encoding: Dict[int, List[str]] | None
            Mapping of groups to Lists of atom symbols. If `None`, the default group encoding is used.
        suppress: bool
            Whether to suppress RDKit log messages.
        """

        if atom_encoding is None:
            self.atom_encoding = {
                "C": 0, "N": 1, "O": 2, "S": 3, "F": 4, "P": 5, "Cl": 6, "Mg": 7, "Na": 8, "Br": 9, "Fe": 10, "Ca": 11,
                "Cu": 12, "Mc": 13, "Pd": 14, "Pb": 15, "K": 16, "I": 17, "Al": 18, "Ni": 19, "Mn": 20,
            }
        else:
            self.atom_encoding = atom_encoding

        self.hybrid_encoding = {
            "S": 0, "SP": 1, "SP2": 2, "SP2D": 3, "SP3": 4, "SP3D": 5, "SP3D2": 6
        }

        if bond_encoding is None:
            self.bond_encoding = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        else:
            self.bond_encoding = bond_encoding

        if group_encoding is None:
            self.groups = {
                0: ["H", "C", "N", "O", "P", "S"],  # non_metals
                1: ["Li", "Na", "K", "Rb", "Cs", "Fr"],  # alkaline metals
                2: ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"],  # alkaline earth metals
                3: ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
                    "Rf", "Db", "Sg", "Hs", "Mt", "Ds", "Rg", "Cn"],  # transition metals
                4: ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Nh", "Fl", "Mc", "Lv"],  # metals
                5: ["B", "Si", "Ge", "As", "Sb", "Te", "Po"],  # metalloids
                6: ["F", "Cl", "Br", "I", "At", "Ts"],  # halogens
                7: ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"],  # noble gases
                8: ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],  # lanthanide
                9: ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]  # actinides
            }
        else:
            self.groups = group_encoding

        self.group_encoding = {}
        for group, elements in self.groups.items():
            for element in elements:
                self.group_encoding[element] = group

        self.atom_encoding_size = len(self.atom_encoding) + 1
        self.hybrid_encoding_size = len(self.hybrid_encoding) + 1
        self.group_encoding_size = len(self.groups) + 1
        self.bond_encoding_size = len(self.bond_encoding) + 1

        self.unk_atom_idx = len(self.atom_encoding)
        self.unk_hybrid_idx = len(self.hybrid_encoding)
        self.unk_group_idx = len(self.groups)
        self.unk_bond_idx = len(self.bond_encoding)

        self.gaussian_rbf: GaussianRBF | None = None
        self.gaussian_abf: GaussianABF | None = None
        self._rbf_args: Dict[str, float | int] | None = None
        self._abf_args: Dict[str, float | int] | None = None

        if suppress:
            RDLogger.DisableLog("rdApp.*")

    @staticmethod
    def get_atom_encoding(smiles: List[str]) -> Dict[str, int]:
        """
        Determine all atom types present in a dataset and return the corresponding atom encoding.

        Parameters
        ----------
        smiles: List[str]
            List of valid SMILES strings
        """
        atom_encoding = set()
        for smi in smiles:
            if (mol := Chem.MolFromSmiles(smi)) is None:
                continue
            atom_set = set([(atom.GetSymbol(), atom.GetAtomicNum()) for atom in mol.GetAtoms()])
            atom_encoding.update(atom_set)

        atom_encoding = dict(sorted(list(atom_encoding), key=lambda x: x[1]))
        return atom_encoding

    def encode_atom(self, atom: Chem.rdchem.Atom, partial_charge: float = 0.0) -> npt.NDArray:
        """
        Encode an RDKit atom as a numerical feature vector.

        Parameters
        ----------
        atom: Chem.rdchem.Atom
            RDKit atom to encode.
        partial_charge: float
            Partial atomic charge to include in the atom features.

        Returns
        -------
        npt.NDArray
        """

        type_enc = np.zeros(shape=self.atom_encoding_size, dtype=np.float32)
        type_enc[self.atom_encoding.get(atom.GetSymbol(), self.unk_atom_idx)] = 1.0

        group_enc = np.zeros(shape=self.group_encoding_size, dtype=np.float32)
        group_enc[self.group_encoding.get(atom.GetSymbol(), self.unk_group_idx)] = 1.0

        hybrid_enc = np.zeros(shape=self.hybrid_encoding_size, dtype=np.float32)
        hybrid_enc[self.hybrid_encoding.get(str(atom.GetHybridization()), self.unk_hybrid_idx)] = 1.0

        prop_enc = np.asarray(
            [
                atom.GetFormalCharge(),
                atom.GetIsAromatic(),
                atom.GetNumExplicitHs(),
                atom.GetDegree(),
                atom.IsInRing(),
                partial_charge,
            ],
            dtype=np.float32,
        )

        return np.hstack((type_enc, group_enc, hybrid_enc, prop_enc))

    def encode_mol_atoms(self, mol: Chem.rdchem.Mol, charges: npt.NDArray | None = None,
                         heavy_only: bool = True) -> npt.NDArray:
        """
        Encode the atoms of an RDKit molecule as a feature matrix.

        Parameters
        ----------
        mol: Chem.rdchem.Mol
            RDKit molecule containing the atoms to encode.
        charges: npt.NDArray | None
            Partial charges corresponding to the selected atoms. If `None`, zero-valued charges are used.
        heavy_only: bool
            Whether to include only heavy atoms, excluding hydrogen atoms.

        Returns
        -------
        npt.NDArray
        """

        atoms = [atom for atom in mol.GetAtoms() if not heavy_only or atom.GetAtomicNum() > 1]

        if charges is None:
            charges = np.zeros(len(atoms), dtype=np.float32)

        atom_list = [self.encode_atom(atom, partial_charge=float(charge)) for atom, charge in zip(atoms, charges)]

        return np.vstack(atom_list).astype(np.float32)

    def encode_bond(self, bond: Chem.rdchem.Bond, atom_indices: Optional[dict[int, int]] = None,
                    distance: float = 0.0) -> Tuple[list[list[int]], npt.NDArray]:
        """
        Encode an RDKit bond as bidirectional edges and a bond feature vector.

        Parameters
        ----------
        bond: Chem.rdchem.Bond
            RDKit bond to encode.
        atom_indices: Optional[dict[int, int]]
            Mapping from RDKit atom indices to graph-node indices. If `None`, the original atom indices are used.
        distance: float
            Interatomic distance to include in the bond features.

        Returns
        -------
        Tuple[list[list[int]], npt.NDArray]
        """

        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if atom_indices is not None:
            start_idx = atom_indices[start_idx]
            end_idx = atom_indices[end_idx]

        bond_edges = [[start_idx, end_idx], [end_idx, start_idx]]

        bond_type = str(bond.GetBondType())
        type_enc = np.zeros(shape=self.bond_encoding_size, dtype=np.float32)
        type_enc[self.bond_encoding.get(bond_type, self.unk_bond_idx)] = 1.0

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

    def encode_mol_bonds(self, mol: Chem.rdchem.Mol, positions: npt.NDArray | None = None,
                         heavy_only: bool = False) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Encode the bonds of an RDKit molecule as graph edges and edge features.

        Parameters
        ----------
        mol: Chem.rdchem.Mol
            RDKit molecule containing the bonds to encode.
        positions: npt.NDArray | None
            Cartesian coordinates of the selected atoms, used to calculate bond distances.
            If `None`, bond distances are set to zero.
        heavy_only: bool
            Whether to include only bonds whose endpoint atoms are both heavy atoms.

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray]
        """

        atoms = [atom for atom in mol.GetAtoms() if not heavy_only or atom.GetAtomicNum() > 1]
        atom_indices = {atom.GetIdx(): graph_index for graph_index, atom in enumerate(atoms)}

        bonds = [
            bond for bond in mol.GetBonds() if not heavy_only
            or (bond.GetBeginAtom().GetAtomicNum() > 1 and bond.GetEndAtom().GetAtomicNum() > 1)
        ]

        if not bonds:
            return (
                np.empty((2, 0), dtype=np.int64),
                np.empty((0, self.bond_encoding_size + 4), dtype=np.float32)
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

        return np.asarray(edges, dtype=np.int64).T, np.vstack(encoding).astype(np.float32)

    def from_smiles(self, smiles: str, heavy_only: bool = True) -> Iterable:
        """
        Generate a PyTorch Geometric graph from a SMILES string.

        Parameters
        ----------
        smiles: str
            SMILES representation of the molecule to encode.
        heavy_only: bool
            Whether to include only bonds between heavy atoms.

        Returns
        -------
        Iterable
        """

        mol = Chem.MolFromSmiles(smiles, sanitize=True)

        atoms_encoding_array = self.encode_mol_atoms(mol, heavy_only=heavy_only)
        edges_array, bonds_encoding_array = self.encode_mol_bonds(mol, heavy_only=heavy_only)

        graph_data = {
            "x": torch.from_numpy(atoms_encoding_array),
            "edge_index": torch.from_numpy(edges_array),
            "edge_attr": torch.from_numpy(bonds_encoding_array),
        }

        return Data(**graph_data)

    def from_molecule(self, molecule: Molecule, charge_method: ChargeMethod = "Gasteiger",
                      energy_method: EnergyMethod = "MMFF94", heavy_only: bool = True, combine_hydrogens: bool = False,
                      include_pos: bool = True) -> Data:
        """
        Generate a PyTorch Geometric graph from a prepared Molecule object.

        Parameters
        ----------
        molecule: Molecule
            Prepared molecule containing conformers and charge information.
        charge_method: ChargeMethod
            Method used to calculate or retrieve partial atomic charges.
        energy_method: EnergyMethod
            Method used to select the lowest-energy conformer. If `None`, the default
            conformer-selection method is used.
        heavy_only: bool
            Whether to include only heavy atoms and bonds between heavy atoms.
        combine_hydrogens: bool
            Whether to combine hydrogen charges with their associated heavy atoms.
        include_pos: bool
            Whether to include atom positions in the Data object.

        Returns
        -------
        Data
        """
        conf_id, _ = molecule.get_lowest_energy_conformer(method=energy_method)

        charges = molecule.get_charges(method=charge_method, conf_id=conf_id, heavy_only=heavy_only,
                                      combine_hydrogens=combine_hydrogens)

        positions = molecule.get_conformer_positions(conf_id=conf_id, heavy_only=heavy_only)

        atoms_encoding_array = self.encode_mol_atoms(mol=molecule.mol, charges=charges, heavy_only=heavy_only)

        edges_array, bonds_encoding_array = self.encode_mol_bonds(mol=molecule.mol, positions=positions,
                                                                  heavy_only=heavy_only)

        graph_data = {
            "x": torch.from_numpy(atoms_encoding_array),
            "edge_index": torch.from_numpy(edges_array),
            "edge_attr": torch.from_numpy(bonds_encoding_array),
        }

        if include_pos:
            graph_data["pos"] = torch.from_numpy(positions).to(torch.float32)

        return Data(**graph_data)

    def encode_smiles(self, smiles: str | Iterable[str], heavy_only: bool = True) -> List[Iterable]:
        """
        Convert one or multiple SMILES strings into PyTorch Geometric graphs.

        Parameters
        ----------
        smiles: str | Iterable[str]
            An Iterable of SMILES strings to encode.
        heavy_only: bool
            Whether to include only bonds between heavy atoms.

        Returns
        -------
        List[Iterable]
        """
        if isinstance(smiles, str):
            return [self.from_smiles(smiles, heavy_only=heavy_only)]

        elif hasattr(smiles, "__iter__") and all(isinstance(item, str) for item in smiles):
            return [self.from_smiles(smi, heavy_only=heavy_only) for smi in smiles]

        else:
            raise TypeError(f"Expected < smiles > to be either str or Iterable[str], got {type(smiles)} instead.")

    def encode_many_smiles(self, smiles: Iterable[str], heavy_only: bool = True, n_jobs: int = 1,
                           batch_size: int = 256, timeout: int | None = None) -> List[Data]:
        """
        Convert multiple SMILES strings into PyTorch Geometric graphs in parallel.

        Parameters
        ----------
        smiles: Iterable[str]
            An Iterable of SMILES strings to encode.
        heavy_only: bool
            Whether to include only bonds between heavy atoms.
        n_jobs: int, optional
            Number of cores to use for calculations.
        batch_size: int, optional
            Number of SMILES per batch.
        timeout: int | None, optional
            Timeout per worker. Defaults to None, i.e., no timeout

        Returns
        -------
        List[Data]
        """

        smiles, n_jobs, smiles_batches = batch_data(data=smiles, n_jobs=n_jobs, batch_size=batch_size)
        if not smiles:
            return []

        out = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
            delayed(self.encode_smiles)(smiles=smi, heavy_only=heavy_only) for smi in smiles_batches
        )

        return list(chain.from_iterable(out))

    def encode_molecules(self, molecule: Molecule | Iterable[Molecule], charge_method: ChargeMethod = "Gasteiger",
                         energy_method: EnergyMethod = "MMFF94", heavy_only: bool = True, combine_hydrogens: bool = False,
                         include_pos: bool = True) -> List[Data]:
        """
        Convert one or more prepared Molecule objects into graph representations.

        Parameters
        ----------
        molecule: Molecule | Iterable[Molecule]
            A single prepared Molecule object or an iterable of Molecule objects to encode.
        charge_method: ChargeMethod
            Method used to calculate or retrieve partial atomic charges.
        energy_method: EnergyMethod
            Method used to select the lowest-energy conformer. If `None`, the default conformer-selection method is used.
        heavy_only: bool
            Whether to include only heavy atoms and bonds between heavy atoms.
        combine_hydrogens: bool
            Whether to combine hydrogen charges with their associated heavy atoms.
        include_pos: bool
            Whether to include atom positions in the Data object.

        Returns
        -------
        List[Data]
        """

        try:
            from vantami.chem.molecule import Molecule
        except ImportError as exc:
            raise ImportError(f"Method < encode_molecules > requires < vantami > 0.5.2 or higher.") from exc
        if isinstance(molecule, Molecule):
            return [
                self.from_molecule(molecule, charge_method=charge_method, energy_method=energy_method,
                                   heavy_only=heavy_only, combine_hydrogens=combine_hydrogens,
                                   include_pos=include_pos),
            ]

        elif hasattr(molecule, "__iter__") and all(isinstance(item, Molecule) for item in molecule):
            return [
                self.from_molecule(dm, charge_method=charge_method, energy_method=energy_method,
                                   heavy_only=heavy_only, combine_hydrogens=combine_hydrogens,
                                   include_pos=include_pos)
                for dm in molecule
            ]

        else:
            raise TypeError(f"Expected < molecule > to be either Molecule or Iterable[Molecule],"
                            f"got {type(molecule)} instead.")

    def encode_many_molecules(self, molecule: Iterable[Molecule], charge_method: ChargeMethod = "Gasteiger",
                              energy_method: EnergyMethod = "MMFF94", heavy_only: bool = True, combine_hydrogens: bool = False,
                              include_pos: bool = True, n_jobs: int = 1, batch_size: int = 256,
                              timeout: int | None = None) -> List[Data]:
        """
        Convert multiple Molecules into PyTorch Geometric graphs in parallel.

        Parameters
        ----------
        molecule: Iterable[Molecule]
            Prepared Molecule objects.
        charge_method: ChargeMethod
            Method used to calculate or retrieve partial atomic charges.
        energy_method: EnergyMethod
            Method used to select the lowest-energy conformer. If `None`, the default conformer-selection method is used.
        heavy_only: bool
            Whether to include only heavy atoms and bonds between heavy atoms.
        combine_hydrogens: bool
            Whether to combine hydrogen charges with their associated heavy atoms.
        include_pos: bool
            Whether to include atom positions in the Data object.
        n_jobs: int, optional
            Number of cores to use for calculations.
        batch_size: int, optional
            Number of SMILES per batch.
        timeout: int | None
            Timeout per worker. Defaults to None, i.e., no timeout

        Returns
        -------
        List[Data]
        """

        molecules, n_jobs, molecule_batches = batch_data(data=molecule, n_jobs=n_jobs, batch_size=batch_size)
        if not molecules:
            return []

        out = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
            delayed(self.encode_molecules)(
                molecule=dm, charge_method=charge_method, energy_method=energy_method, heavy_only=heavy_only,
                combine_hydrogens=combine_hydrogens, include_pos=include_pos
            ) for dm in molecule_batches
        )

        return list(chain.from_iterable(out))

    @staticmethod
    def _cosine_smooth(distances: torch.Tensor, r_cut: float = 6.0):
        """
        Smooth distances using a cosine function
        """
        scaled_distances = distances / r_cut
        weights = 0.5 * (torch.cos(torch.pi * scaled_distances) + 1.0)
        return weights * (distances <= r_cut).to(distances.dtype)

    @staticmethod
    def _build_angles(edge_index: torch.Tensor, n_nodes: int):
        """
        Construct unique angular triplets j -> i <- k from directed spatial edges.
        """
        _, target = edge_index

        triplet_ij, triplet_ik = [], []

        for center_idx in range(n_nodes):
            incoming_edges = torch.where(target == center_idx)[0]

            if incoming_edges.numel() < 2:
                continue

            edge_pairs = torch.combinations(incoming_edges, r=2)
            triplet_ij.append(edge_pairs[:, 0])
            triplet_ik.append(edge_pairs[:, 1])

        if not triplet_ij:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty

        return (
            torch.cat(triplet_ij, dim=0),
            torch.cat(triplet_ik, dim=0),
        )

    @staticmethod
    def _validate_graph_positions(graph: Data) -> torch.Tensor | None:
        """
        Validate and return atom coordinates from a PyG graph.
        """
        if not hasattr(graph, "pos") or graph.pos is None:
            raise ValueError("Spatial encoding requires graph.pos.")

        pos = graph.pos

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"Expected graph.pos with shape [num_atoms, 3], got {tuple(pos.shape)}.")

        return pos

    def set_rbf(self, r_cut: float = 6.0, n_rbf: int = 32):
        """
        Set the parameters for Gaussian RBF kernel.
        """
        args = {
            "r_cut": r_cut,
            "n_rbf": n_rbf,
        }

        if self._rbf_args is None:
            self._rbf_args = args
            self.gaussian_rbf = GaussianRBF(r_cut=r_cut, n_rbf=n_rbf)
        else:
            if args != self._rbf_args:
                warnings.warn(f"Changing RBF arguments to:\n{args}", UserWarning)
                self._rbf_args = args
                self.gaussian_rbf = GaussianRBF(r_cut=r_cut, n_rbf=n_rbf)

    def set_abf(self, n_abf: int = 32):
        """
        Set the parameters for Gaussian ABF kernel.
        """
        args = {"n_abf": n_abf}

        if self._abf_args is None:
            self._abf_args = args
            self.gaussian_abf = GaussianABF(n_abf=n_abf)
        else:
            if args != self._abf_args:
                warnings.warn(f"Changing ABF arguments to:\n{args}", UserWarning)
                self._abf_args = args
                self.gaussian_abf = GaussianABF(n_abf=n_abf)

    def rbf_from_graph(self, graph: Data, r_cut: float = 6.0, n_rbf: int = 32, max_neighbors: int = 64) -> Data:
        """
        Add radius-graph connectivity and Gaussian radial-basis features.

        Parameters
        ----------
        graph: Data
            PyG graph with `pos` attribute
        r_cut: float
            Cutoff for spatial edges
        n_rbf: int
            Number of Gaussian radial-basis functions to use for encoding
        max_neighbors: int
            Maximum number of neighbors to encode per atom

        Returns
        -------
        Data with added:
            spatial_edge_index:
                Long tensor with shape [2, num_spatial_edges].
            spatial_edge_attr:
                Float tensor with shape [num_spatial_edges, n_rbf].
            atom_radial_basis:
                Float tensor with shape [num_nodes, n_rbf].
        """
        pos = self._validate_graph_positions(graph)

        if r_cut <= 0:
            raise ValueError("< r_cut > must be a positive float.")

        if n_rbf < 2:
            raise ValueError("< n_rbf > must be at least 2.")

        if max_neighbors < 1:
            raise ValueError("< max_neighbors > must be at least 1.")

        n_nodes = pos.shape[0]
        dtype = pos.dtype

        self.set_rbf(r_cut=r_cut, n_rbf=n_rbf)
        assert self.gaussian_rbf is not None

        edge_index = radius_graph(
            x=pos,
            r=r_cut,
            batch=None,
            loop=False,
            max_num_neighbors=max_neighbors,
        )

        graph.spatial_edge_index = edge_index
        graph.atom_radial_basis = pos.new_zeros((n_nodes, n_rbf))

        if edge_index.numel() == 0:
            graph.spatial_edge_attr = pos.new_zeros((0, n_rbf))
            return graph

        source, target = edge_index

        edge_vec = pos[source] - pos[target]
        edge_dist = torch.linalg.vector_norm(edge_vec, dim=-1)

        eps = torch.finfo(dtype).eps

        if torch.any(edge_dist <= eps):
            raise ValueError("Spatial graph contains coincident atoms or near-zero interatomic distances.")

        radial_weights = self._cosine_smooth(distances=edge_dist, r_cut=r_cut)

        edge_radial_basis = self.gaussian_rbf(edge_dist) * radial_weights.unsqueeze(-1)
        graph.spatial_edge_attr = edge_radial_basis

        graph.atom_radial_basis.index_add_(0, target, edge_radial_basis)

        return graph

    def abf_from_graph(self, graph: Data, r_cut: float = 6.0, n_abf: int = 32) -> Data:
        """
        Add Gaussian angular-basis features using an existing spatial radius graph.

        Parameters
        ----------
        graph: Data
            PyG graph with `pos` and `spatial_edge_index` attributes
        r_cut: float
            Cutoff for spatial edges
        n_abf: int
            Number of Gaussian angular-basis functions to use for encoding

        Returns
        -------
        Data with added:
            spatial_angle_index:
                Long tensor with shape [3, num_spatial_triplets].
            spatial_angle_attr:
                Float tensor with shape [num_spatial_triplets, n_abf].
            atom_angular_basis:
                Float tensor with shape [num_nodes, n_abf].
        """
        pos = self._validate_graph_positions(graph)

        if not hasattr(graph, "spatial_edge_index"):
            raise ValueError("Method < abf_from_graph > requires graph.spatial_edge_index.")

        if n_abf < 2:
            raise ValueError("< n_abf > must be at least 2.")

        if r_cut <= 0:
            raise ValueError("< r_cut > must be a positive float.")

        edge_index = graph.spatial_edge_index

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Expected graph.spatial_edge_index with shape [2, num_spatial_edges],"
                             f" got {tuple(edge_index.shape)}.")

        n_nodes = pos.shape[0]
        dtype = pos.dtype

        self.set_abf(n_abf=n_abf)
        assert self.gaussian_abf is not None

        graph.atom_angular_basis = pos.new_zeros((n_nodes, n_abf))

        if edge_index.numel() == 0:
            graph.spatial_angle_index = torch.empty((3, 0), dtype=torch.long)
            graph.spatial_angle_attr = pos.new_zeros((0, n_abf))
            return graph

        source, target = edge_index

        edge_vec = pos[source] - pos[target]
        edge_dist = torch.linalg.vector_norm(edge_vec, dim=-1)

        eps = torch.finfo(dtype).eps

        if torch.any(edge_dist <= eps):
            raise ValueError("Spatial graph contains coincident atoms or near-zero interatomic distances.")

        edge_ij_idx, edge_ik_idx = self._build_angles(edge_index=edge_index, n_nodes=n_nodes)

        if edge_ij_idx.numel() == 0:
            graph.spatial_angle_index = torch.empty((3, 0), dtype=torch.long)
            graph.spatial_angle_attr = pos.new_zeros((0, n_abf))
            return graph

        edge_vec_ij = edge_vec[edge_ij_idx]
        edge_vec_ik = edge_vec[edge_ik_idx]

        edge_dist_ij = edge_dist[edge_ij_idx]
        edge_dist_ik = edge_dist[edge_ik_idx]

        cos_theta = (
                (edge_vec_ij * edge_vec_ik).sum(dim=-1)
                / (edge_dist_ij * edge_dist_ik).clamp_min(eps)
        ).clamp(min=-1.0 + eps, max=1.0 - eps)

        angles = torch.acos(cos_theta)

        radial_weights = self._cosine_smooth(distances=edge_dist, r_cut=r_cut)

        angular_weights = (radial_weights[edge_ij_idx] * radial_weights[edge_ik_idx])

        triplet_angular_basis = self.gaussian_abf(angles) * angular_weights.unsqueeze(-1)

        graph.spatial_angle_index = torch.stack(
            (source[edge_ij_idx], target[edge_ij_idx], source[edge_ik_idx]),
            dim=0,
        )

        graph.spatial_angle_attr = triplet_angular_basis
        graph.atom_angular_basis.index_add_(0, target[edge_ij_idx], triplet_angular_basis)

        return graph

    def from_graph(self, graph: Data, r_cut: float = 6.0, n_rbf: int = 32, n_abf: int | None = None,
                   max_neighbors: int = 64) -> Data:
        """
        Encode multiple 3D non-covalent spatial edges from graph.pos.

        Parameters
        ----------
        graph: Iterable[Data]
            Data objects prepared from `meth`:encode_molecules that includes `pos`.
        r_cut: float, optional
            Cutoff distance for cosine smoothing.
        n_rbf: int, optional
            Number of Gaussian Radial Basis Functions to use for encoding distances
        n_abf: int | None
            Number of Gaussian Angular Basis Functions to use for encoding angles.
            If `None`, angles are not encoded.
        max_neighbors: int
            Maximum number of encoded neighbors

        Returns
        -------
        List[Data] with added:
            spatial_edge_index:
                Long tensor with shape [2, num_spatial_edges].
            spatial_edge_attr:
                Float tensor with shape [num_spatial_edges, n_rbf].
            atom_radial_basis:
                Float tensor with shape [num_nodes, n_rbf].
            spatial_angle_index:
                Long tensor with shape [3, num_spatial_triplets].
            spatial_angle_attr:
                Float tensor with shape [num_spatial_triplets, n_abf].
            atom_angular_basis:
                Float tensor with shape [num_nodes, n_abf].
        """
        graph = self.rbf_from_graph(graph=graph, r_cut=r_cut, n_rbf=n_rbf, max_neighbors=max_neighbors)
        if n_abf is not None:
            graph = self.abf_from_graph(graph=graph, r_cut=r_cut, n_abf=n_abf)
        return graph

    def encode_spatial(self, graph: Iterable[Data], r_cut: float, n_rbf: int = 32, n_abf: int | None = None,
                       max_neighbors: int = 64) -> List[Data]:
        """
        Encode multiple 3D non-covalent spatial edges from graph.pos.

        Parameters
        ----------
        graph: Iterable[Data]
            Data objects prepared from `meth`:encode_molecules that includes `pos`.
        r_cut: float, optional
            Cutoff distance for cosine smoothing.
        n_rbf: int, optional
            Number of Gaussian Radial Basis Functions to use for encoding distances
        n_abf: int, optional
            Number of Gaussian Angular Basis Functions to use for encoding angles.
            If `None`, angles are not encoded.
        max_neighbors: int
            Maximum number of encoded neighbors

        Returns
        -------
        List[Data] with added:
            spatial_edge_index:
                Long tensor with shape [2, num_spatial_edges].
            spatial_edge_attr:
                Float tensor with shape [num_spatial_edges, n_rbf].
            atom_radial_basis:
                Float tensor with shape [num_nodes, n_rbf].
            spatial_angle_index:
                Long tensor with shape [3, num_spatial_triplets].
            spatial_angle_attr:
                Float tensor with shape [num_spatial_triplets, n_abf].
            atom_angular_basis:
                Float tensor with shape [num_nodes, n_abf].
        """

        if isinstance(graph, Data):
            return [self.from_graph(graph=graph, r_cut=r_cut, n_rbf=n_rbf, n_abf=n_abf, max_neighbors=max_neighbors)]

        elif hasattr(graph, "__iter__") and all(isinstance(item, Data) for item in graph):
            return [
                self.from_graph(graph=gr, r_cut=r_cut, n_rbf=n_rbf, n_abf=n_abf, max_neighbors=max_neighbors)
                for gr in graph
            ]

        else:
            raise TypeError(f"Expected < graph > to be either Data or Iterable[Data], got {type(graph)} instead.")

    def encode_many_spatial(self, graph: Iterable[Data], r_cut: float = 6.0, n_rbf: int = 32, n_abf: int | None= None,
                            max_neighbors: int = 64, n_jobs: int = 1, batch_size: int = 256,
                            timeout: int | None = None) -> List[Data]:
        """
        Encode 3D non-covalent spatial edges from graph.pos in parallel

        Parameters
        ----------
        graph: Data
            Data object prepared from `meth`:encode_molecules that includes `pos`.
        r_cut: float, optional
            Cutoff distance for cosine smoothing.
        n_rbf: int, optional
            Number of Gaussian Radial Basis Functions to use for encoding distances
        n_abf: int | None
            Number of Gaussian Angular Basis Functions to use for encoding angles
        max_neighbors: int
            Maximum number of encoded neighbors
        n_jobs: int
            Number of cores to use for calculations.
        batch_size: int
            Number of SMILES per batch.
        timeout: int | None
            Timeout per worker. Defaults to None, i.e., no timeout

        Returns
        -------
        List[Data] with added:
            spatial_edge_index:
                Long tensor with shape [2, num_spatial_edges].
            spatial_edge_attr:
                Float tensor with shape [num_spatial_edges, n_rbf].
            atom_radial_basis:
                Float tensor with shape [num_nodes, n_rbf].
            spatial_angle_index:
                Long tensor with shape [3, num_spatial_triplets].
            spatial_angle_attr:
                Float tensor with shape [num_spatial_triplets, n_abf].
            atom_angular_basis:
                Float tensor with shape [num_nodes, n_abf].
        """
        graphs, n_jobs, graph_batches = batch_data(data=graph, n_jobs=n_jobs, batch_size=batch_size)

        if not graphs:
            return []

        out = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
            delayed(self.encode_spatial)(
                graph=gr, r_cut=r_cut, n_rbf=n_rbf, n_abf=n_abf, max_neighbors=max_neighbors) for gr in graph_batches
        )

        return list(chain.from_iterable(out))
