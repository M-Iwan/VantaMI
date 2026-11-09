from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from itertools import chain

try:
    from rdkit import Chem, RDLogger
    import torch
    from torch_geometric.data import Data
    from torch.geometric.nn import radius_graph
    from vantami.chem.molecule import Molecule, MMFFVariant, EnergyMethod, ChargeMethod
    from vantami.data.partition import batch_data
except ImportError as exc:
    raise ImportError(f"GraphVectorizer requires rdkit, torch, torch_geometric, and torch_cluster") from exc

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class GaussianRBF(torch.nn.Module):
    def __init__(self, r_off: float, n_rbf: int = 32):
        super().__init__()

        centers = torch.linspace(0.0, r_off, n_rbf)
        spacing = r_off / (n_rbf - 1)

        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (spacing ** 2)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers).pow(2))


class GraphVectorizer:
    """
    Turn SMILES or prepared Molecule objects into torch_geometric Data.

    Node features contain atom descriptors, periodic table group, and a partial-charge feature.
    Edge features contain bond descriptors and 3D bond distance.

    Updated in Vantami 0.5.3rc2
    """

    def __init__(self, atom_encoding: dict | None = None, bond_encoding: dict | None = None, suppress: bool = True):
        """
        Initialize the graph vectorizer with atom and bond encoding schemes.

        Parameters
        ----------
        atom_encoding: dict | None
            Mapping of atom symbols to integer encoding indices. If `None`, the default atom encoding is used.
        bond_encoding: dict | None
            Mapping of bond-type names to integer encoding indices. If `None`, the default bond encoding is used.
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

        self.group_encoding = {}
        for group, elements in self.groups.items():
            for element in elements:
                self.group_encoding[element] = group

        if bond_encoding is None:
            self.bond_encoding = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        else:
            self.bond_encoding = bond_encoding

        self.atom_encoding_size = len(self.atom_encoding) + 1
        self.group_encoding_size = len(self.groups) + 1
        self.bond_encoding_size = len(self.bond_encoding) + 1

        self.unk_atom_idx = len(self.atom_encoding)
        self.unk_group_idx = len(self.groups)
        self.unk_bond_idx = len(self.bond_encoding)

        self.gaussian_rbf: GaussianRBF | None = None

        if suppress:
            RDLogger.DisableLog("rdApp.*")

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
                np.array([0, 0], dtype=np.int64).reshape(2, 1),
                np.zeros(shape=(1, self.bond_encoding_size + 4), dtype=np.float32),
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
                      include_pos: bool = False) -> Data:
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
                         include_pos: bool = False) -> List[Data]:
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
                              include_pos: bool = False, n_jobs: int = 1, batch_size: int = 256,
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
    def _cosine_smooth(distances: torch.Tensor, r_on: float = 4.0, r_off: float = 6.0):
        weights = torch.zeros_like(distances)

        near = distances <= r_on
        transition = (distances > r_on) & (distances < r_off)

        weights[near] = 1.0

        scaled_distance = (distances[transition] - r_on) / (r_off - r_on)
        weights[transition] = 0.5 * (torch.cos(torch.pi * scaled_distance) + 1.0)

        return weights

    def from_graph(self, graph: Data, cutoff: float | Tuple[float, float], n_rbf: int = 32,
                   max_neighbors: int = 64) -> Data:
        """
        Encode 3D non-covalent spatial edges from graph.pos

        Parameters
        ----------
        graph: Data
            Data object prepared from `meth`:encode_molecules that includes `pos`.
        cutoff: float | Tuple[float, float]
            If float: hard outer cutoff. All selected spatial edges get weight 1.0.
            If tuple (r_on, r_off): Smooth transition region. Edges are generated up to r_off.
                Edge weights are 1.0 up to r_on and smoothly decay to 0.0 at r_off.
        n_rbf: int, optional
            Number of Gaussian Radial Basis Functions to use for encoding distances
        max_neighbors: int
            Maximum number of encoded neighbors

        Adds
        ----
        graph.spatial_edge_index:
            Long tensor of shape [2, E_spatial].
        graph.spatial_edge_dist:
            Float tensor of shape [E_spatial].
        graph.spatial_edge_weight:
            Float tensor of shape [E_spatial].
        graph.spatial_edge_attr:
            Float tensor of shape [E_spatial].
        """

        if not hasattr(graph, "pos") or graph.pos is None:
            raise ValueError("< .encode_spatial > requires graph.pos.")

        pos = graph.pos

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"Expected graph.pos with shape [num_atoms, 3], got {tuple(pos.shape)}.")

        if isinstance(cutoff, tuple):
            r_on, r_off = map(float, cutoff)
            if not 0.0 <= r_on < r_off:
                raise ValueError("cutoff must satisfy 0 <= r_on < r_off.")
            use_smooth_weights = True
        else:
            r_on, r_off = None, float(cutoff)
            if r_off <= 0:
                raise ValueError(f"cutoff must be positive, got {r_off}.")
            use_smooth_weights = False

        if self.gaussian_rbf is None:
            self.gaussian_rbf = GaussianRBF(
                r_off=r_off,
                n_rbf=n_rbf,
            )

        candidate_edge_index = radius_graph(
            x=pos,
            r=r_off,
            batch=None,
            loop=False,
            max_num_neighbors=max_neighbors,
        )

        if candidate_edge_index.numel() == 0:
            graph.spatial_edge_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
            graph.spatial_edge_dist = torch.empty((0,), dtype=torch.float32, device=pos.device)
            graph.spatial_edge_weight = torch.empty((0,), dtype=torch.float32, device=pos.device)
            graph.spatial_edge_attr = torch.empty((0, n_rbf), dtype=torch.float32, device=pos.device)
            return graph

        src, dst = candidate_edge_index

        if hasattr(graph, "edge_index") and graph.edge_index is not None:
            num_atoms = pos.size(0)
            covalent_adjacency = torch.zeros(
                (num_atoms, num_atoms),
                dtype=torch.bool,
                device=pos.device,
            )

            cov_src = graph.edge_index[0].to(device=pos.device, dtype=torch.long)
            cov_dst = graph.edge_index[1].to(device=pos.device, dtype=torch.long)

            covalent_adjacency[cov_src, cov_dst] = True
            covalent_adjacency[cov_dst, cov_src] = True

            keep = ~covalent_adjacency[src, dst]
            spatial_edge_index = candidate_edge_index[:, keep]
        else:
            spatial_edge_index = candidate_edge_index

        if spatial_pairs.numel() == 0:
            graph.spatial_edge_index = torch.empty((2, 0), dtype=torch.long)
            graph.spatial_edge_dist = torch.empty((0,), dtype=torch.float32)
            graph.spatial_edge_weight = torch.empty((0,), dtype=torch.float32)
            graph.spatial_edge_attr = torch.empty((0, n_rbf), dtype=torch.float32)
            return graph

        src, dst = spatial_edge_index
        spatial_edge_dist = (pos[src] - pos[dst]).norm(dim=-1)

        if use_smooth_weights:
            spatial_edge_weight = self._cosine_smooth(distances=spatial_edge_dist, r_on=r_on, r_off=r_off)
        else:
            spatial_edge_weight = torch.ones_like(spatial_edge_dist)

        graph.spatial_edge_index = spatial_edge_index
        graph.spatial_edge_dist = spatial_edge_dist
        graph.spatial_edge_weight = spatial_edge_weight
        graph.spatial_edge_attr = self.gaussian_rbf(spatial_edge_dist)

        return graph

    def encode_spatial(self, graph: Iterable[Data], cutoff: float | Tuple[float, float],
                       n_rbf: int = 32) -> List[Data]:
        """
        Encode multiple 3D non-covalent spatial edges from graph.pos.

        Parameters
        ----------
        graph: Iterable[Data]
            Data objects prepared from `meth`:encode_molecules that includes `pos`.
        cutoff: float | Tuple[float, float]
            If float: hard outer cutoff. All selected spatial edges get weight 1.0.
            If tuple (r_on, r_off): Smooth transition region. Edges are generated up to r_off.
                Edge weights are 1.0 up to r_on and smoothly decay to 0.0 at r_off.
        n_rbf: int, optional
            Number of Gaussian Radial Basis Functions to use for encoding distances

        Adds
        ----
        graph.spatial_edge_index:
            Long tensor of shape [2, E_spatial].
        graph.spatial_edge_dist:
            Float tensor of shape [E_spatial].
        graph.spatial_edge_weight:
            Float tensor of shape [E_spatial].
        graph.spatial_edge_attr:
            Float tensor of shape [E_spatial].
        """

        if isinstance(graph, Data):
            return [self.from_graph(graph, cutoff=cutoff, n_rbf=n_rbf)]

        elif hasattr(graph, "__iter__") and all(isinstance(item, Data) for item in graph):
            return [self.from_graph(gr, cutoff=cutoff, n_rbf=n_rbf) for gr in graph]

        else:
            raise TypeError(f"Expected < graph > to be either Data or Iterable[Data], got {type(graph)} instead.")

    def encode_many_spatial(self, graph: Iterable[Data], cutoff: float | Tuple[float, float] = 4.0, n_rbf: int = 5,
                            n_jobs: int = 1, batch_size: int = 256, timeout: int | None = None) -> List[Data]:
        """
        Encode 3D non-covalent spatial edges from graph.pos in parallel

        Parameters
        ----------
        graph: Data
            Data object prepared from `meth`:encode_molecules that includes `pos`.
        cutoff: float | Tuple[float, float]
            If float: hard outer cutoff. All selected spatial edges get weight 1.0.
            If tuple (r_on, r_off): Smooth transition region. Edges are generated up to r_off.
                Edge weights are 1.0 up to r_on and smoothly decay to 0.0 at r_off.
        n_rbf: int, optional
            Number of Gaussian Radial Basis Functions to use for encoding distances
        n_jobs: int
            Number of cores to use for calculations.
        batch_size: int
            Number of SMILES per batch.
        timeout: int | None
            Timeout per worker. Defaults to None, i.e., no timeout

        Adds
        ----
        graph.spatial_edge_index:
            Long tensor of shape [2, E_spatial].
        graph.spatial_edge_dist:
            Float tensor of shape [E_spatial].
        graph.spatial_edge_weight:
            Float tensor of shape [E_spatial].
        graph.spatial_edge_attr:
            Float tensor of shape [E_spatial].
        """
        graphs, n_jobs, graph_batches = batch_data(data=graph, n_jobs=n_jobs, batch_size=batch_size)

        if not graphs:
            return []

        out = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
            delayed(self.encode_spatial)(graph=gr, cutoff=cutoff, n_rbf=n_rbf) for gr in graph_batches)

        return list(chain.from_iterable(out))
