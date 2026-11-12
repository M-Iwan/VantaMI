from __future__ import annotations

import joblib
from itertools import chain
from typing import List
from joblib import Parallel, delayed

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


def string_2_minilm(string: str | List[str] | npt.NDArray[np.str_], decimals: int = 5):
    """
    Parameters
    ----------
    string: str | List[str] | npt.NDArray[str]
        An English sentence.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    npt.NDArray | List[npt.NDArray]
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


def dataframe_2_minilm(df: pl.DataFrame, string_col: str = "String", output_col: str = "MiniLM",
                       decimals: int = 5, n_jobs: int = 1, batch_size: int = 512, timeout: int | None = None):
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
    timeout: int | None
        Per-worker timeout for joblib parallelism

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added MiniLM column.
    """

    try:
        from vantami.cache import get_minilm_model_path
        from vantami.data.descriptors import batch_data
    except ImportError as exc:
        raise ImportError(f"Function < dataframe_2_minilm > requires the MiniLM cache utilities:\n{exc}")

    if not get_minilm_model_path().is_file():
        get_minilm()

    strings = df[string_col].drop_nulls().unique(maintain_order=True).to_list()

    strings, n_jobs, string_batches = batch_data(
        data=strings, n_jobs=n_jobs, batch_size=batch_size
    )
    if not strings:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(string_2_minilm)(string=st, decimals=decimals) for st in string_batches
    )

    string_df = pl.DataFrame(
        {
            string_col: strings,
            output_col: list(chain.from_iterable(out))
        },
        schema={
            string_col: pl.String,
            output_col: pl.Array(pl.Float64, shape=384)
        }
    )

    df = df.join(string_df, on=string_col, how="left")

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


def string_2_qwen3(string:str | List[str] | npt.NDArray[np.str_], decimals: int = 5):
    """
    Parameters
    ----------
    string: str | List[str] | npt.NDArray[str]
        An English sentence.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    np.ndarray | List[np.ndarray]
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


def dataframe_2_qwen3(df: pl.DataFrame, string_col: str = "String", output_col: str = "Qwen3",
                      decimals: int = 5, n_jobs: int = 1, batch_size: int = 256, timeout: int | None = None):
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
    timeout: int | None
        Per-worker timeout for joblib parallelism

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added Qwen3 column.
    """

    try:
        from vantami.cache import get_qwen3_model_path
        from vantami.data.descriptors import batch_data
    except ImportError as exc:
        raise ImportError(f"Function < dataframe_2_qwen3 > requires the Qwen3 cache utilities:\n{exc}")

    if not get_qwen3_model_path().is_file():
        get_qwen3()

    strings = df[string_col].drop_nulls().unique(maintain_order=True).to_list()

    strings, n_jobs, string_batches = batch_data(
        data=strings, n_jobs=n_jobs, batch_size=batch_size
    )
    if not strings:
        return df.with_columns(pl.lit(None).alias(output_col))

    out = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(string_2_qwen3)(string=st, decimals=decimals) for st in string_batches
    )

    string_df = pl.DataFrame(
        {
            string_col: strings,
            output_col: list(chain.from_iterable(out))
        },
        schema={
            string_col: pl.String,
            output_col: pl.Array(pl.Float64, shape=1024)
        }
    )

    df = df.join(string_df, on=string_col, how="left")

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