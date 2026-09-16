from __future__ import annotations
import os
import math
import json
import joblib
from itertools import chain
from typing import Union, List
from importlib.resources import files
from joblib import Parallel, delayed, effective_n_jobs

import numpy as np
import numpy.typing as npt
import polars as pl
from rdkit import Chem, RDLogger


def _to_mol(smiles: object):
    """
    Check if a valid mol object can be generated from a passed string.
    """
    if not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def _prepare_batches(df: pl.DataFrame, smiles_col: str, n_jobs: int, batch_size: int):
    """
    Prepare batches of SMILES strings for downstream processing.
    """
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size} instead.")

    smiles = df[smiles_col].unique(maintain_order=True).to_list()
    if not smiles:
        return [], 0, []

    n_jobs = min(effective_n_jobs(n_jobs), len(smiles))

    n_batches = max(n_jobs, math.ceil(len(smiles) / batch_size))
    smiles_batches = np.array_split(smiles, n_batches)

    return smiles, n_jobs, smiles_batches


def _to_murcko(smiles: str, generic: bool, mod):
    """
    Convert a single SMILES to Murcko scaffold.
    """
    if (mol := _to_mol(smiles)) is None:
        print(f"Unable to construct valid molecule from {smiles}")
        return None

    scaffold = mod.GetScaffoldForMol(mol)

    if generic:
        scaffold = mod.MakeScaffoldGeneric(scaffold)

    scaffold = Chem.MolToSmiles(scaffold)

    return None if scaffold == "" else scaffold


def smiles_2_murcko(smiles: Union[str, List[str], npt.NDArray[str]], generic: bool = False):
    """
    Convert SMILES to the corresponding Murcko scaffold.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A SMILES string or a list/array of SMILES strings.
    generic: bool, optional
        Whether to convert all bonds to single and all atoms to carbon.

    Returns
    -------
    Union[str, npt.NDArray]
    """
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise ImportError(f"Function < smiles_2_murcko > requires RDKit:\n{exc}")

    if isinstance(smiles, str):
        return _to_murcko(smiles, generic=generic, mod=MurckoScaffold)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_to_murcko(smi, generic=generic, mod=MurckoScaffold) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str or List[str], got {type(smiles)} instead")


def dataframe_2_murcko(df: pl.DataFrame, smiles_col: str = "SMILES", output_col: str = "Murcko",
                       generic: bool = False, n_jobs: int = 1, batch_size: int = 512):

    """
    Convert SMILES string(s) in a DataFrame to Murcko scaffolds.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame.
    smiles_col: str, optional
        Name of column with SMILES.
    output_col: str, optional
        Name of column for the output.
    generic: bool, optional
        Whether to convert all bonds to single and all atoms to carbon.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added Murcko column.
    """
    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_murcko)(smiles=smi, generic=generic) for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')
    return df


def _to_inchi(smiles: str):
    """
    Convert a single SMILES to InChI.
    """
    if (mol := _to_mol(smiles)) is None:
        print(f"Unable to construct valid molecule from {smiles}")
        return None

    return Chem.MolToInchi(mol)


def smiles_2_inchi(smiles: Union[str, List[str], npt.NDArray[str]]):
    """
    Convert SMILES to the corresponding InChI representation.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A SMILES string or a list/array of SMILES strings.

    Returns
    -------
    Union[str, npt.NDArray]
    """
    RDLogger.DisableLog("rdApp.warning")

    if isinstance(smiles, str):
        return _to_inchi(smiles)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_to_inchi(smi) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str or List[str], got {type(smiles)} instead")


def dataframe_2_inchi(df: pl.DataFrame, smiles_col: str = "SMILES", output_col: str = "InChI",
                      n_jobs: int = 1, batch_size: int = 512):

    """
    Convert SMILES string(s) in a DataFrame to InChI representation.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str, optional
        Name of column with SMILES.
    output_col : str, optional
        Name of column for the output.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added InChI column.
    """

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_inchi)(smiles=smi) for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def _to_inchi_key(smiles: str):
    """
    Convert a single SMILES to InChIKey.
    """
    if (mol := _to_mol(smiles)) is None:
        print(f"Unable to construct valid molecule from {smiles}")
        return None

    return Chem.MolToInchiKey(mol)


def smiles_2_inchi_key(smiles: Union[str, List[str], npt.NDArray[str]]):
    """
    Convert SMILES to the corresponding InChI Key representation.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A SMILES string or a list/array of SMILES strings.

    Returns
    -------
    Union[str, npt.NDArray]
    """
    RDLogger.DisableLog("rdApp.warning")

    if isinstance(smiles, str):
        return _to_inchi_key(smiles)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_to_inchi_key(smi) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str or List[str], got {type(smiles)} instead")


def dataframe_2_inchi_key(df: pl.DataFrame, smiles_col: str = "SMILES", output_col: str = "InChIKey",
                          n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES string(s) in a DataFrame to InChI Key representation.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str, optional
        Name of column with SMILES.
    output_col : str, optional
        Name of column for the output.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added InChIKey column.
    """

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_inchi_key)(smiles=smi) for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def _to_fp(smiles: str, generator, count: bool):
    """
    Convert a single SMILES to a (Count) Extended Connectivity / Daylight / AtomPairs Fingerprint
    """
    if (mol := _to_mol(smiles)) is None:
        print(f"Unable to construct valid molecule from {smiles}")
        return None

    if count:
        fp = np.asarray(generator.GetCountFingerprintAsNumPy(mol), dtype=np.uint32)
    else:
        fp =  np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=np.uint8)

    return fp


def smiles_2_ecfp(smiles: Union[str, List[str], npt.NDArray[str]], radius: int = 2, nbits: int = 1024, count: bool = False):
    """
    Convert SMILES string(s) to a (Count) Extended Connectivity Fingerprint.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A SMILES string or a list of SMILES strings.
    radius: int, optional
        The radius parameter for ECFP calculation. Default is 2.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of ECFP. Default is False.

    Returns
    -------
    Union[npt.NDArray, List[npt.NDArray]]
    """
    try:
        from rdkit.Chem import rdFingerprintGenerator as fpgen
    except ImportError as exc:
        raise ImportError(f"Function < smiles_2_ecfp > requires RDKit:\n{exc}")

    gen = fpgen.GetMorganGenerator(radius=radius, fpSize=nbits)

    if isinstance(smiles, str):
        return _to_fp(smiles=smiles, generator=gen, count=count)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_to_fp(smiles=smi, generator=gen, count=count) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str, List[str] or npt.NDArray[str], got {type(smiles)} instead")


def dataframe_2_ecfp(df: pl.DataFrame, smiles_col: str = 'SMILES', output_col: str = None, radius: int = 2,
                     nbits: int = 1024, count: bool = False, n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES string(s) in a DataFrame to an Extended Connectivity (Count) Fingerprint.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str, optional
        Name of column with SMILES.
    output_col : str, optional
        Name of column for the output.
    radius: int, optional
        The radius parameter for ECFP calculation. Default is 2.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of ECFP. Default is False.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added ECFP/ECFPCount column.
    """

    if output_col is None:
        output_col = "ECFP" if not count else "ECFPCount"

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_ecfp)(smiles=smi, radius=radius, nbits=nbits, count=count) for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def smiles_2_daylight(smiles: Union[str, List[str], npt.NDArray[str]], min_path: int = 1, max_path: int = 7,
                      nbits: int = 1024, count: bool = False):
    """
    Convert SMILES in a DataFrame to Daylight (Count) Fingerprints.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A SMILES string or a list of SMILES strings.
    min_path: int
        Smallest path length to consider. Default is 1.
    max_path: int
        Biggest path length to consider. Default is 7.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of DaylightFP. Default is False.

    Returns
    -------
    Union[npt.NDArray, List[npt.NDArray]]
    """
    try:
        from rdkit.Chem import rdFingerprintGenerator as fpgen
    except ImportError as exc:
        raise ImportError(f"Function < smiles_2_daylight > requires RDKit:\n{exc}")

    gen = fpgen.GetRDKitFPGenerator(minPath=min_path, maxPath=max_path, fpSize=nbits)

    if isinstance(smiles, str):
        return _to_fp(smiles=smiles, generator=gen, count=count)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_to_fp(smiles=smi, generator=gen, count=count) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str, List[str] or npt.NDArray[str], got {type(smiles)} instead")


def dataframe_2_daylight(df: pl.DataFrame, smiles_col: str = 'SMILES', output_col: str = None, min_path: int = 1,
                         max_path: int = 7, nbits: int = 1024, count: bool = False, n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to Daylight (Count) Fingerprints.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str, optional
        Name of column with SMILES.
    output_col : str, optional
        Name of column for the output.
    min_path: int
        Smallest path length to consider. Default is 1.
    max_path: int
        Biggest path length to consider. Default is 7.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of DaylightFP. Default is False.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added Daylight/DaylightCount column.
    """

    if output_col is None:
        output_col = "Daylight" if not count else "DaylightCount"

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_daylight)(smiles=smi, min_path=min_path, max_path=max_path, nbits=nbits, count=count) for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def smiles_2_atompair(smiles: Union[str, List[str], npt.NDArray[str]], min_distance: int = 1, max_distance: int = 7,
                      nbits: int = 1024, count: bool = False):
    """
    Convert SMILES in to AtomPair (Count) Fingerprints.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A SMILES string or a list of SMILES strings.
    min_distance: int
        Smallest distance between two atoms to consider. Default is 1.
    max_distance: int
        Largest distance between two atoms to consider. Default is 30.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of AtomPairFP. Default is False.

    Returns
    -------
    Union[npt.NDArray, List[npt.NDArray]]
    """
    try:
        from rdkit.Chem import rdFingerprintGenerator as fpgen
    except ImportError as exc:
        raise ImportError(f"Function < smiles_2_atompair > requires RDKit:\n{exc}")

    gen = fpgen.GetAtomPairGenerator(minDistance=min_distance, maxDistance=max_distance, fpSize=nbits)

    if isinstance(smiles, str):
        return _to_fp(smiles=smiles, generator=gen, count=count)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_to_fp(smiles=smi, generator=gen, count=count) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str, List[str] or npt.NDArray[str], got {type(smiles)} instead")


def dataframe_2_atompair(df: pl.DataFrame, smiles_col: str = 'SMILES', output_col: str = None, min_distance: int = 1,
                         max_distance: int = 7, nbits: int = 1024, count: bool = False, n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to AtomPair (Count) Fingerprints.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str, optional
        Name of column with SMILES.
    output_col : str, optional
        Name of column for the output.
    min_distance: int
        Smallest distance between two atoms to consider. Default is 1.
    max_distance: int
        Largest distance between two atoms to consider. Default is 30.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of AtomPairFP. Default is False.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added AtomPair/AtomPairCount column.
    """

    if output_col is None:
        output_col = "AtomPair" if not count else "AtomPairCount"

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_atompair)(smiles=smi, min_distance=min_distance, max_distance=max_distance, nbits=nbits, count=count)
        for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def _to_maccs(smiles: str, mod):
    """
    Convert a single SMILES to MACCS fingerprint.
    """
    if (mol := _to_mol(smiles)) is None:
        print(f"Unable to construct valid molecule from {smiles}")
        return None

    return np.array(mod.GetMACCSKeysFingerprint(mol), dtype=np.uint8)


def smiles_2_maccs(smiles: Union[str, List[str], npt.NDArray[str]]):
    """
    Convert SMILES to MACCS Fingerprints.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A SMILES or list of SMILES strings.

    Returns
    Union[npt.NDArray, List[npt.NDArray]]
    """

    try:
        from rdkit.Chem import rdMolDescriptors
    except ImportError as exc:
        raise ImportError(f"Function < smiles_2_maccs > requires RDKit:\n{exc}")

    if isinstance(smiles, str):
        return _to_maccs(smiles=smiles, mod=rdMolDescriptors)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_to_maccs(smiles=smi, mod=rdMolDescriptors) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str, List[str] or npt.NDArray[str], got {type(smiles)} instead")


def dataframe_2_maccs(df: pl.DataFrame, smiles_col: str = 'SMILES', output_col: str = 'MACCS',
                      n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to MACCS Fingerprints.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    output_col : str, optional
        Name of column for the output.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added MACCS fingerprints for given SMILES.
    """

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_maccs)(smiles=smi) for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def _fp_from_smarts(smiles: str, smarts: List):
    """
    Convert a single SMILES to fingerprint based on SMARTS patterns matching.
    """
    if (mol := _to_mol(smiles)) is None:
        print(f"Unable to construct valid molecule from {smiles}")
        return None

    return np.array([1 if mol.HasSubstructMatch(sm) else 0 for sm in smarts], dtype=np.uint8)


def smiles_2_klek(smiles: Union[str, List[str], npt.NDArray[str]]):
    """
    Convert SMILES to Klekota&Roth Fingerprints.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A SMILES or list of SMILES strings.

    Returns
    -------
    Union[npt.NDArray, List[npt.NDArray]]
    """

    klekota_smarts = joblib.load(files('vantami.files').joinpath('klekota_roth.joblib'))

    if isinstance(smiles, str):
        return _fp_from_smarts(smiles, smarts=klekota_smarts)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_fp_from_smarts(smiles=smi, smarts=klekota_smarts) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str, List[str] or npt.NDArray[str], got {type(smiles)} instead")


def dataframe_2_klek(df: pl.DataFrame, smiles_col: str = 'SMILES', output_col: str = 'Klek',
                     n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to Klekota&Roth Fingerprints.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame with data.
    smiles_col: str, optional
        Name of column with SMILES.
    output_col : str, optional
        Name of column for the output.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added Klek column.
    """

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_klek)(smiles=smi) for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def _to_rdkit(smiles: str, decimals: int, mod) -> np.array:
    """
    Convert a single SMILES to RDKit physicochemical descriptors.
    """
    if (mol := _to_mol(smiles)) is None:
        print(f"Unable to construct valid molecule from {smiles}")
        return None

    desc = mod.CalcMolDescriptors(mol, silent=False, missingVal=np.nan).values()
    return np.round(np.fromiter(desc, dtype=np.float64), decimals)


def smiles_2_rdkit(smiles: Union[str, List[str], npt.NDArray[str]], decimals: int = 5):
    """
    Convert SMILES to RDKit descriptors.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A SMILES or list of SMILES strings.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
    """
    try:
        from rdkit.Chem import Descriptors
    except ImportError as exc:
        raise ImportError(f"Function < smiles_2_rdkit > requires RDKit:\n{exc}")

    if isinstance(smiles, str):
        return _to_rdkit(smiles=smiles, decimals=decimals, mod=Descriptors)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_to_rdkit(smiles=smi, decimals=decimals, mod=Descriptors) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str, List[str] or npt.NDArray[str], got {type(smiles)} instead")


def dataframe_2_rdkit(df: pl.DataFrame, smiles_col: str = 'SMILES', output_col: str = 'RDKit',
                      decimals: int = 5, n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to RDKit descriptors.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    output_col : str, optional
        Name of column for the output.
    decimals: int
        Number of decimals to keep.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added RDKit column.
    """

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_rdkit)(smiles=smi, decimals=decimals) for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df



def _to_common_rdkit(smiles: str, decimals: int, mod):
    """
    Convert a single SMILES to selected RDKit physicochemical descriptors.
    """
    keys = ["MolWt", "LogP", "MolMR", "NHeavy", "NHetero", "NHBA", "NHBD", "NRotBonds", "NRings", "TPSA"]
    if (mol := _to_mol(smiles)) is None:
        print(f"Unable to construct valid molecule from {smiles}")
        return {key: None for key in keys}

    crippen = mod.CalcCrippenDescriptors(mol)
    return {
        "MolWt": np.round(mod.CalcExactMolWt(mol), decimals),
        "LogP": np.round(crippen[0], decimals),
        "MolMR": np.round(crippen[1], decimals),
        "NHeavy": mod.CalcNumHeavyAtoms(mol),
        "NHetero": mod.CalcNumHeteroatoms(mol),
        "NHBA": mod.CalcNumHBA(mol),
        "NHBD": mod.CalcNumHBD(mol),
        "NRotBonds": mod.CalcNumRotatableBonds(mol),
        "NRings": mod.CalcNumRings(mol),
        "TPSA": np.round(mod.CalcTPSA(mol), decimals),
    }


def smiles_2_common_rdkit(smiles: Union[str, List[str], npt.NDArray[str]], decimals: int = 5):
    """
    Convert SMILES to common RDKit descriptors.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A SMILES or list of SMILES strings.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    Union[dict, List[dict]]
        A dict (single SMILES) or list of dicts (multiple SMILES), with:
        MolWt, LogP, MolMR, NHeavy, NHetero, NHBA, NHBD, NRotBonds, NRings, TPSA.
    """
    try:
        from rdkit.Chem import rdMolDescriptors
    except ImportError as exc:
        raise ImportError(f"Function < smiles_2_common_rdkit > requires RDKit:\n{exc}")

    if isinstance(smiles, str):
        return _to_common_rdkit(smiles=smiles, decimals=decimals, mod=rdMolDescriptors)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_to_common_rdkit(smiles=smi, decimals=decimals, mod=rdMolDescriptors) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str, List[str] or npt.NDArray[str], got {type(smiles)} instead")


def dataframe_2_common_rdkit(df: pl.DataFrame, smiles_col: str = 'SMILES', decimals: int = 5,
                             n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to common RDKit descriptors.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    decimals: int
        Number of decimals to keep.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars DataFrame with added columns:
        MolWt, LogP, MolMR, NHeavy, NHetero, NHBA, NHBD, NRotBonds, NRings, TPSA.
    """

    keys = ["MolWt", "LogP", "MolMR", "NHeavy", "NHetero", "NHBA", "NHBD", "NRotBonds", "NRings", "TPSA"]

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns([pl.lit(None).alias(o_col) for o_col in keys])

    out_batches = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_common_rdkit)(smiles=smi, decimals=decimals) for smi in smiles_batches
    )

    out = list(chain.from_iterable(out_batches))

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        **{key: [o[key] for o in out] for key in keys},
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def _from_hf(smiles: str, model, tokenizer, torch, decimals):
    """
    Convert a single SMILES to embeddings from HuggingFace models.
    """
    tokens = tokenizer(smiles, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    with torch.no_grad():
        emb = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy().reshape(-1)
    return np.round(emb, decimals)


def smiles_2_chemberta(smiles: Union[str, List[str], npt.NDArray[str]], decimals: int = 5):
    """
    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A valid SMILES or list of valid SMILES strings.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"Function < smiles_2_chemberta > requires PyTorch:\n{exc}")

    try:
        from transformers import AutoTokenizer, AutoModel, logging
        from vantami.cache import get_chemberta_model_path, get_chemberta_tokenizer_path
    except ImportError as exc:
        raise ImportError(f"Function < smiles_2_chemberta > requires < transformers > library:\n{exc}")

    logging.set_verbosity_error()
    torch.set_num_threads(1)

    model_path = get_chemberta_model_path()
    tokenizer_path = get_chemberta_tokenizer_path()

    if not model_path.is_file() or not tokenizer_path.is_file():
        get_chemberta()

    model = joblib.load(model_path)
    model.eval()
    tokenizer = joblib.load(tokenizer_path)

    if isinstance(smiles, str):
        return _from_hf(smiles=smiles, model=model, tokenizer=tokenizer, torch=torch, decimals=decimals)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_from_hf(smiles=smi, model=model, tokenizer=tokenizer, torch=torch, decimals=decimals) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str, List[str] or npt.NDArray[str], got {type(smiles)} instead")


def dataframe_2_chemberta(df: pl.DataFrame, smiles_col: str = 'SMILES', output_col: str = 'ChemBERTa',
                          decimals: int = 5, n_jobs: int = 1, batch_size: int = 512 ):
    """
    Convert SMILES in a polars DataFrame to ChemBERTa embeddings.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str
        Name of column with SMILES.
    output_col : str, optional
        Name of column for the output.
    decimals: int
        Number of decimals to keep.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added ChemBERTa column.
    """

    try:
        from vantami.cache import get_chemberta_model_path, get_chemberta_tokenizer_path
    except ImportError as exc:
        raise ImportError(f"Function < dataframe_2_chemberta > requires the ChemBERTa cache utilities:\n{exc}")

    if not get_chemberta_model_path().is_file() or not get_chemberta_tokenizer_path().is_file():
        get_chemberta()

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_chemberta)(smiles=smi, decimals=decimals) for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def get_chemberta():
    """
    Download the ChemBERTa model and tokenizer and save them in .cache.
    """
    try:
        from transformers import AutoTokenizer, AutoModel, logging
        from vantami.cache import get_chemberta_model_path, get_chemberta_tokenizer_path
    except ImportError as exc:
        raise ImportError(f"Function < get_chemberta > requires < transformers > library:\n{exc}")

    model = AutoModel.from_pretrained("DeepChem/ChemBERTa-100M-MLM")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-100M-MLM")

    joblib.dump(model, get_chemberta_model_path())
    joblib.dump(tokenizer, get_chemberta_tokenizer_path())

    return {
        "model": str(get_chemberta_model_path()),
        "tokenizer": str(get_chemberta_tokenizer_path())
    }


def _to_mapc(smiles: str, radius: int, nbits: int, fn):
    """
    Convert a single SMILES to MAPC descriptors.
    """
    if (mol := _to_mol(smiles)) is None:
        print(f"Unable to construct valid molecule from {smiles}")
        return None

    return fn(mol, max_radius=radius, n_permutations=nbits)


def smiles_2_mapc(smiles: Union[str, List[str], npt.NDArray[str]], radius: int = 2, nbits: int = 1024):
    """
    Convert SMILES to MAPC descriptors.

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[str]]
        A valid SMILES or list of valid SMILES strings.
    radius: int, optional
        The radius parameter for MAPC calculation. Default is 2.
    nbits: int, optional
        The length of the FP. Default is 1024.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
    """
    try:
        from mapchiral.mapchiral import encode
    except ImportError as exc:
        raise ImportError(f"Function < smiles_2_mapc > requires < mapchiral > library:\n{exc}")

    if isinstance(smiles, str):
        return _to_mapc(smiles=smiles, radius=radius, nbits=nbits, fn=encode)

    elif isinstance(smiles, (list, np.ndarray)):
        return [_to_mapc(smiles=smi, radius=radius, nbits=nbits, fn=encode) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be str or Union[List[str], npt.NDArray[str]], got {type(smiles)} instead")


def dataframe_2_mapc(df: pl.DataFrame, smiles_col: str = 'SMILES', output_col: str = 'MAPC',
                     radius: int = 2, nbits: int = 1024, n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to MAPC (MAP Chiral) fingerprints.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str, optional
        Name of the column containing SMILES strings.
    output_col : str, optional
        Name of column for the output.
    radius : int, optional
        The maximum radius for MAPC calculation. Default is 2.
    nbits : int, optional
        The number of permutations (fingerprint length). Default is 1024.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added MAPC column.
    """

    smiles, n_jobs, smiles_batches = _prepare_batches(
        df=df, smiles_col=smiles_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not smiles:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_mapc)(smiles=smi, radius=radius, nbits=nbits) for smi in smiles_batches
    )

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


# ============ #
# Pain section #
# ============ #

def dataframe_2_mordred(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str = 'SMILES', desc_col: str = 'Mordred',
                        path_source: str = f'src/src_files/mordred_paths.json', decimals: int = 5):
    """
    Convert SMILES in dataframe to Mordred descriptors.
    TODO: Update similar to other functions once the osmordred package works

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars DataFrame with data.
    smiles_col : str
        Name of column with SMILES.
    desc_col : str
        Name of column to which add calculated descriptors.
    path_source : str
        Path to mordred_paths.json file.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars Dataframe with added column holding Mordred descriptors for given SMILES.
    """
    try:
        from vantami import read_pd, write_pd
        import pandas as pd
    except ImportError as exc:
        raise ImportError(f"Error during import:\n{exc}")

    def postprocess(entry, decimals_):
        if not isinstance(entry, (np.ndarray, list)):
            return np.nan
        if isinstance(entry, np.ndarray):
            return np.round(entry.reshape(-1), decimals_)
        if isinstance(entry, list):
            return [np.round(array.reshape(-1), decimals_) for array in entry]
        else:
            raise TypeError(f"Expected numpy array or list, got {type(entry)} instead.")

    with open(path_source, 'r') as file:
        paths = json.load(file)

    command = (f"{paths['python']} {paths['wrapper']} --input {paths['input']} --output {paths['output']} "
               f"--smiles_col {smiles_col} --descriptor_col {desc_col}")

    if is_polars := isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # pack the lists to strings so that they don't get broken -,-
    df.loc[:, smiles_col] = df[smiles_col].apply(lambda entry: ' : '.join(entry) if isinstance(entry, list) else entry)
    write_pd(df, paths['input'])

    os.system(command)

    df = read_pd(paths['output'])
    df.loc[:, desc_col] = df[desc_col].apply(postprocess, decimals_=decimals)

    if is_polars:
        df = pl.from_pandas(df)

    return df


def dataframe_2_cddd(df: Union[pd.DataFrame, pl.DataFrame], cddd_paths: str, smiles_col: str = 'SMILES',
                     descriptor_col: str = 'CDDD', n_cpus: int = 4, decimals: int = 5):
    """
    Convert SMILES in a DataFrame to CDDD descriptors.
    TODO: This already

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars DataFrame with data.
    cddd_paths : str
        Path to a JSON file containing paths to CDDD-related files. Generated using CDDD_conf.sh
    smiles_col : str
        Name of column with SMILES.
    descriptor_col : str
        Name of column to which add calculated descriptors.
    n_cpus : int
        Number of CPUs to use during processing
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars Dataframe with added column holding CDDD descriptors for given SMILES.
    """
    try:
        from vantami import read_pd, write_pd
        import pandas as pd
    except ImportError as exc:
        raise ImportError(f"Error during import:\n{exc}")

    def postprocess(entry, decimals_):
        if not isinstance(entry, (np.ndarray, list)):
            return np.nan
        if isinstance(entry, np.ndarray):
            return np.round(entry.reshape(-1), decimals_)
        if isinstance(entry, list):
            return [np.round(array.reshape(-1), decimals_) for array in entry]
        else:
            raise TypeError(f"Expected numpy array or list, got {type(entry)} instead.")

    with open(cddd_paths, 'r') as file:
        paths = json.load(file)

    command = (f"{paths['python']} {paths['wrapper']} --input {paths['input']} --output {paths['output']} "
               f"--smiles_col {smiles_col} --descriptor_col {descriptor_col} --n_cpu {n_cpus} --model_dir {paths['model']}")

    if is_polars := isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # pack the lists to strings so that they don't get broken;
    df.loc[:, smiles_col] = df[smiles_col].apply(lambda entry: ' : '.join(entry) if isinstance(entry, list) else entry)
    df.to_csv(paths['input'], sep='\t', header=True, index=False)

    os.system(command)

    df = joblib.load(paths['output'])
    df.loc[:, descriptor_col] = df[descriptor_col].apply(postprocess, decimals_=decimals)

    if is_polars:
        df = pl.from_pandas(df)

    return df
