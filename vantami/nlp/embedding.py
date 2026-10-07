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


def _from_hf(string: str, model, torch, decimals):
    """
    Convert a single string to embeddings from HuggingFace models.
    """
    with torch.no_grad():
        emb = model.encode(string, normalize_embeddings=True)
    return np.round(emb, decimals)


def _prepare_batches(df: pl.DataFrame, string_col: str, n_jobs: int, batch_size: int):
    """
    Prepare batches of strings for downstream processing.
    """
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size} instead.")

    strings = df[string_col].unique(maintain_order=True).to_list()
    if not strings:
        return [], 0, []

    n_jobs = min(effective_n_jobs(n_jobs), len(strings))

    n_batches = max(n_jobs, math.ceil(len(strings) / batch_size))
    string_batches = np.array_split(strings, n_batches)

    return strings, n_jobs, string_batches


def string_2_minilm(string: Union[str, List[str], npt.NDArray[str]], decimals: int = 5):
    """
    Parameters
    ----------
    string: Union[str, List[str], npt.NDArray[str]]
        An English sentence.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"Function < string_2_minilm > requires PyTorch:\n{exc}")

    try:
        from sentence_transformers import SentenceTransformer
        from vantami.cache import get_minilm_model_path
    except ImportError as exc:
        raise ImportError(f"Function < string_2_minilm > requires < sentence_transformers > library:\n{exc}")

    torch.set_num_threads(1)

    model_path = get_minilm_model_path()

    if not model_path.is_file():
        get_minilm()

    model = joblib.load(model_path)
    model.eval()

    if isinstance(string, str):
        return _from_hf(string=string, model=model, torch=torch, decimals=decimals)

    elif isinstance(string, (list, np.ndarray)):
        return [_from_hf(string=st, model=model, torch=torch, decimals=decimals) for st in string]

    else:
        raise TypeError(f"Expected string to be str, List[str] or npt.NDArray[str], got {type(string)} instead")


def dataframe_2_minilm(df: pl.DataFrame, string_col: str = 'String', output_col: str = 'MiniLM',
                           decimals: int = 5, n_jobs: int = 1, batch_size: int = 512 ):
    """
    Convert strings in a polars DataFrame to MiniLM embeddings.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    string_col : str
        Name of column with strings to convert.
    output_col : str, optional
        Name of column for the output.
    decimals: int
        Number of decimals to keep.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of strings per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added MiniLM column.
    """

    try:
        from vantami.cache import get_minilm_model_path
    except ImportError as exc:
        raise ImportError(f"Function < dataframe_2_minilm > requires the MiniLM cache utilities:\n{exc}")

    if not get_minilm_model_path().is_file():
        get_minilm()

    strings, n_jobs, string_batches = _prepare_batches(
        df=df, string_col=string_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not strings:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(string_2_minilm)(string=st, decimals=decimals) for st in string_batches
    )

    string_df = pl.DataFrame({
        string_col: strings,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(string_df, on=string_col, how='left')

    return df


def get_minilm():
    """
    Download the all-MiniLM-L6-v2 model and save it in .cache.
    """
    try:
        import truststore
        from sentence_transformers import SentenceTransformer
        from vantami.cache import get_minilm_model_path
    except ImportError as exc:
        raise ImportError(f"Function < get_minilm > requires sentence_transformers and truststore libraries:\n{exc}")

    truststore.inject_into_ssl()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    joblib.dump(model, get_minilm_model_path())

    return {
        "model": str(get_minilm_model_path()),
    }


def string_2_qwen3(string: Union[str, List[str], npt.NDArray[str]], decimals: int = 5):
    """
    Parameters
    ----------
    string: Union[str, List[str], npt.NDArray[str]]
        An English sentence.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"Function < string_2_qwen3 > requires PyTorch:\n{exc}")

    try:
        from sentence_transformers import SentenceTransformer
        from vantami.cache import get_qwen3_model_path
    except ImportError as exc:
        raise ImportError(f"Function < string_2_qwen3 > requires < sentence_transformers > library:\n{exc}")

    torch.set_num_threads(1)

    model_path = get_qwen3_model_path()

    if not model_path.is_file():
        get_qwen3()

    model = joblib.load(model_path)
    model.eval()

    if isinstance(string, str):
        return _from_hf(string=string, model=model, torch=torch, decimals=decimals)

    elif isinstance(string, (list, np.ndarray)):
        return [_from_hf(string=st, model=model, torch=torch, decimals=decimals) for st in string]

    else:
        raise TypeError(f"Expected string to be str, List[str] or npt.NDArray[str], got {type(string)} instead")


def dataframe_2_qwen3(df: pl.DataFrame, string_col: str = 'String', output_col: str = 'Qwen3',
                      decimals: int = 5, n_jobs: int = 1, batch_size: int = 256):
    """
    Convert strings in a polars DataFrame to Qwen3 embeddings.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    string_col : str
        Name of column with strings to convert.
    output_col : str, optional
        Name of column for the output.
    decimals: int
        Number of decimals to keep.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of strings per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added Qwen3 column.
    """

    try:
        from vantami.cache import get_qwen3_model_path
    except ImportError as exc:
        raise ImportError(f"Function < dataframe_2_qwen3 > requires the Qwen3 cache utilities:\n{exc}")

    if not get_qwen3_model_path().is_file():
        get_qwen3()

    strings, n_jobs, string_batches = _prepare_batches(
        df=df, string_col=string_col, n_jobs=n_jobs, batch_size=batch_size
    )
    if not strings:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=120, backend='loky')(
        delayed(string_2_qwen3)(string=st, decimals=decimals) for st in string_batches
    )

    string_df = pl.DataFrame({
        string_col: strings,
        output_col: list(chain.from_iterable(out))
    })

    df = df.join(string_df, on=string_col, how='left')

    return df


def get_qwen3():
    """
    Download the Qwen3-Embedding model (0.6B) and save it in .cache.
    """
    try:
        import truststore
        from sentence_transformers import SentenceTransformer
        from vantami.cache import get_qwen3_model_path
    except ImportError as exc:
        raise ImportError(f"Function < get_qwen3 > requires sentence_transformers and truststore libraries:\n{exc}")

    truststore.inject_into_ssl()

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    joblib.dump(model, get_qwen3_model_path())

    return {
        "model": str(get_qwen3_model_path()),
    }