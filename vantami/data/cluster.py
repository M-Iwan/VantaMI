"""
TODO: clustering: dbscan, kmean, hdbscan, optics
"""
from typing import Union
from joblib import Parallel, delayed

import numpy as np
import polars as pl
from rdkit import DataStructs, Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from vantami.data.manipulate import embeddings_to_rdkit
from vantami.data.distance import distance_matrix


def butina_cluster(df: pl.DataFrame, fp_col: str = "ECFP", threshold: float = 0.3, metric: str = "jaccard",
                   n_jobs: int = 1, batch_size: int = 1024):
    """
    Perform Butina clustering from a thresholded pairwise neighbourhood graph.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    fp_col : str, optional
        Name of the column containing one embedding per row. Default is ECFP.
    threshold : float, optional
        Distance threshold for defining neighbours. Two entries are neighbours
        when their distance is less than or equal to this value. Default is 0.3.
    metric : str, optional
        Metric passed to distance_matrix. Default is "jaccard"
    n_jobs : int, optional
        Number of parallel jobs used for distance calculation.
    batch_size : int, optional
        Number of rows assigned to one distance-calculation batch. Default is 1024.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an added integer "Cluster" column.
    """

    if fp_col not in df.columns:
        raise ValueError(
            f"Fingerprint or embedding column {fp_col} was not found."
        )

    n_samples = len(df)
    resolved_n_jobs = effective_n_jobs(n_jobs)

    embeddings = [array.reshape(-1) for array in df.get_column(fp_col).to_numpy()]

    upper_triu = ds_matrix(
        array_1=embeddings,
        array_2=None,
        metric=metric,
        n_jobs=resolved_n_jobs,
        batch_size=batch_size,
        return_triu=True,
        threshold=threshold,
    )

    neighbour_mask = (
        upper_triu.astype(bool, copy=False) | upper_triu.T.astype(bool, copy=False)
    )

    seen = np.zeros(n_samples, dtype=bool)
    cluster_ids = np.full(n_samples, -1, dtype=np.int64)

    neighbour_counts = neighbour_mask.sum(axis=1, dtype=np.int64)

    current_cluster_id = 0

    while not np.all(seen):
        unassigned_indices = np.flatnonzero(~seen)

        candidate_counts = neighbour_counts.copy()
        candidate_counts[seen] = -1

        most_neighbours_idx = int(np.argmax(candidate_counts))

        if candidate_counts[most_neighbours_idx] == 0:
            cluster_ids[unassigned_indices] = np.arange(
                current_cluster_id,
                current_cluster_id + len(unassigned_indices),
                dtype=np.int64,
            )
            break

        cluster_members = np.flatnonzero(
            neighbour_mask[most_neighbours_idx] & ~seen
        )
        cluster_members = np.unique(np.append(cluster_members, most_neighbours_idx))

        cluster_ids[cluster_members] = current_cluster_id
        seen[cluster_members] = True

        removed_edge_counts = neighbour_mask[:, cluster_members].sum(
            axis=1,
            dtype=np.int64,
        )

        neighbour_counts -= removed_edge_counts
        neighbour_counts[seen] = -1

        current_cluster_id += 1

    result = df.with_columns(
        pl.Series("Cluster", cluster_ids)
    )

    return result


def murcko_cluster(df: pl.DataFrame, smiles_col: str = 'SMILES', generic: bool = False):
    """
    Cluster molecules based on their (generic) Murcko Scaffolds. Molecules for which a scaffold cannot be generated
    are assigned to a single cluster.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    smiles_col: str
        A name of the column with SMILES strings.
    generic: bool
        A flag to make Murcko scaffolds generic, i.e. only carbon atoms and single bonds

    Returns
    -------
    df: pl.DataFrame
    """

    def smiles_2_scaffold(smiles: str) -> str:
        """
        Calculate (generic) Murcko Scaffold from SMILES

        Parameters
        ----------
        smiles: str
         A SMILES string

        Returns
        -------
        str
        """

        nonlocal generic
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 'InvalidMolecule'
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if generic:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            return scaffold_smiles
        except Exception as e:
            print(f"Exception during calculating the scaffold of < {smiles} >\n{e}")
            return "ScaffoldNotGenerated"

    df = df.with_columns(pl.col(smiles_col).map_elements(
        smiles_2_scaffold, return_dtype=pl.String).alias('Cluster'))

    return df


def cc_cluster(df: pl.DataFrame, features_col: str = "ECFP", metric: str = 'jaccard',
               threshold: float = 0.3, n_jobs: int = 1):
    """
    Cluster molecules using connected components graphs.

    Parameters
    ----------

    df: pl.DataFrame
        A polars DataFrame
    features_col: str
        Name of the column with fingerprints. Default is ECFP
    metric: str, optional
        The distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    threshold: float
        Distance threshold. Edge is made if distance <= threshold. Default is 0.3
    n_jobs: int
        Number of cores to use. Default is 1 (i.e. use one core)

    Returns
    -------
    df: pl.DataFrame
    """

    array = np.vstack(df[features_col].to_numpy())
    dist_matrix = distance_matrix(array_1=array, array_2=array, metric=metric, n_jobs=n_jobs)
    adj_sparse = csr_matrix((dist_matrix <= threshold).astype(int))
    n_components, labels = connected_components(adj_sparse, directed=False, return_labels=True)

    df = df.with_columns(pl.Series('Cluster', labels))

    return df
