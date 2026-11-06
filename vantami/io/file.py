import os
import io, gzip, lzma, bz2, pickle, zipfile, joblib
import warnings

from pathlib import Path
from typing import Optional, Dict, Union, Tuple

import pandas as pd
import polars as pl


_COMPRESSION_SUFFIXES = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
    ".zip": "zip"
}

def _get_extensions(path: Path) -> Tuple[str, Optional[str]]:
    """
    Return the data extension and optional compression type.
    """
    suffixes = [suffix.lower() for suffix in path.suffixes]

    if not suffixes:
        raise ValueError(f"No extension found for path: {path}")

    compression = None

    if suffixes[-1] in _COMPRESSION_SUFFIXES:
        compression = _COMPRESSION_SUFFIXES[suffixes[-1]]

        if len(suffixes) < 2:
            raise ValueError(f"Compressed file must include a data extension: {path}")

        ext = suffixes[-2]
    else:
        ext = suffixes[-1]

    return ext, compression


def _read_compressed_bytes(path: Path, compression: str) -> bytes:
    """
    Read a compressed file into bytes.
    """
    if compression == "gzip":
        with gzip.open(path, "rb") as file:
            return file.read()

    if compression == "xz":
        with lzma.open(path, "rb") as file:
            return file.read()

    if compression == "bz2":
        with bz2.open(path, "rb") as file:
            return file.read()

    if compression == "zip":
        with zipfile.ZipFile(path, "r") as archive:
            files = [name for name in archive.namelist() if not name.endswith("/")]

            if len(files) != 1:
                raise ValueError("ZIP archives must contain exactly one data file.")

            return archive.read(files[0])

    raise ValueError(f"Compression < {compression} > is not supported.")


def _write_compressed_bytes(data: bytes, path: Path, compression: str) -> None:
    """
    Write bytes using the requested compression format.
    """
    if compression == "gzip":
        with gzip.open(path, "wb") as file:
            file.write(data)
        return

    if compression == "bz2":
        with bz2.open(path, "wb") as file:
            file.write(data)
        return

    if compression == "xz":
        with lzma.open(path, "wb") as file:
            file.write(data)
        return

    if compression == "zip":
        archive_name = path.with_suffix("").name

        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(archive_name, data)

        return

    raise ValueError(f"Compression < {compression} > is not supported.")


def _validate_compression(ext: str, compression: Optional[str]) -> None:
    """
    Validate compression format.
    """
    supported_extensions = {".csv", ".tsv", ".pkl", ".joblib"}

    if compression is not None and ext not in supported_extensions:
        raise ValueError(f"Compression not supported for < {ext} > files. Supported are: {supported_extensions}")


def read_pd(path: Union[str, Path], kwargs: Optional[Dict] = None) -> pd.DataFrame:
    """
    Automatically detect extension and read a pandas DataFrame.

    Supports compressed CSV, TSV and pickle files based on the path suffixes.
    Examples: data.csv.xz, data.tsv.zip, data.pkl.gz.

    Parameters
    ----------
    path: Union[str, Path]
        Path to the file.
    kwargs: dict
        Additional keywords passed to a corresponding read function.
    """
    if kwargs is None:
        kwargs = {}

    if isinstance(path, str):
        path = Path(path)

    ext, compression = _get_extensions(path)

    _validate_compression(ext, compression)

    if ext == ".xlsx":
        df = pd.read_excel(path, **kwargs)
    elif ext == ".csv":
        df = pd.read_csv(path, compression=compression, **kwargs)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", compression=compression, **kwargs)
    elif ext == ".parquet":
        df = pd.read_parquet(path, **kwargs)
    elif ext == ".pkl":
        if compression is None:
            with open(path, "rb") as file:
                df = pickle.load(file)
        else:
            df = pickle.loads(_read_compressed_bytes(path, compression))
    elif ext == ".joblib":
        if compression is None:
            df = joblib.load(path)
        else:
            data = _read_compressed_bytes(path, compression)
            df = joblib.load(io.BytesIO(data))
    else:
        raise ValueError(f"Extension < {ext} > is not supported.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)} instead.")

    return df


def write_pd(df: pd.DataFrame, path: Union[str, Path], kwargs: Optional[Dict] = None) -> None:
    """
    Automatically detect extension and write a pandas DataFrame.

    Supports compressed CSV, TSV and pickle files based on the path suffixes.
    Examples: data.csv.xz, data.tsv.zip, data.pkl.gz.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to write.
    path: Union[str, Path]
        Path to the file.
    kwargs: dict
        Additional keywords passed to a corresponding write function.
    """
    if kwargs is None:
        kwargs = {}

    if isinstance(path, str):
        path = Path(path)

    ext, compression = _get_extensions(path)
    _validate_compression(ext, compression)

    if not os.path.isdir(path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        warnings.warn(f"Saving an empty DataFrame with columns: {df.columns}")

    if ext == ".xlsx":
        df.to_excel(path, index=False, **kwargs)
    elif ext == ".csv":
        df.to_csv(path, index=False, header=True, compression=compression, **kwargs)
    elif ext == ".tsv":
        df.to_csv(path, index=False, header=True, sep="\t", compression=compression, **kwargs)
    elif ext == ".parquet":
        df.to_parquet(path, **kwargs)
    elif ext == ".pkl":
        data = pickle.dumps(df)

        if compression is None:
            with open(path, "wb") as file:
                file.write(data)
        else:
            _write_compressed_bytes(data, path, compression)
    elif ext == ".joblib":
        buffer = io.BytesIO()
        joblib.dump(df, buffer)
        data = buffer.getvalue()

        if compression is None:
            with open(path, "wb") as file:
                file.write(data)
        else:
            _write_compressed_bytes(data, path, compression)
    else:
        raise ValueError(f"Extension < {ext} > is not supported.")


def read_pl(path: Union[str, Path], kwargs: Optional[Dict] = None) -> pl.DataFrame:
    """
    Automatically detect extension and read a Polars DataFrame.

    Supports compressed CSV, TSV, pickle and joblib files based on path
    suffixes. Examples: data.csv.xz, data.tsv.zip, data.pkl.gz.

    Parameters
    ----------
    path: Union[str, Path]
        Path to the file.
    kwargs: dict
        Additional keywords passed to a corresponding read function.
    """
    if kwargs is None:
        kwargs = {}

    if isinstance(path, str):
        path = Path(path)

    ext, compression = _get_extensions(path)

    _validate_compression(ext, compression)

    if ext == ".xlsx":
        df = pl.read_excel(path, **kwargs)
    elif ext == ".csv":
        if compression is None:
            df = pl.read_csv(path, **kwargs)
        else:
            data = _read_compressed_bytes(path, compression)
            df = pl.read_csv(io.BytesIO(data), **kwargs)
    elif ext == ".tsv":
        if compression is None:
            df = pl.read_csv(path, separator="\t", **kwargs)
        else:
            data = _read_compressed_bytes(path, compression)
            df = pl.read_csv(io.BytesIO(data), separator="\t", **kwargs)
    elif ext == ".parquet":
        df = pl.read_parquet(path, **kwargs)
    elif ext == ".pkl":
        if compression is None:
            with open(path, "rb") as file:
                df = pickle.load(file)
        else:
            df = pickle.loads(_read_compressed_bytes(path, compression))
    elif ext == ".joblib":
        if compression is None:
            df = joblib.load(path)
        else:
            data = _read_compressed_bytes(path, compression)
            df = joblib.load(io.BytesIO(data))
    else:
        raise ValueError(f"Extension < {ext} > is not supported.")

    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected Polars or pandas DataFrame, got {type(df)} instead.")

    if df.is_empty():
        warnings.warn(f"Loaded an empty DataFrame with columns: {df.columns}")

    return df


def write_pl(df: pl.DataFrame, path: Union[str, Path], kwargs: Optional[Dict] = None) -> None:
    """
    Automatically detect extension and write a Polars DataFrame.

    Parameters
    ----------
    df: pl.DataFrame
        DataFrame to write.
    path: Union[str, Path]
        Path to the file.
    kwargs: dict
        Additional keywords passed to a corresponding save function.
    """
    if kwargs is None:
        kwargs = {}

    if isinstance(path, str):
        path = Path(path)

    ext, compression = _get_extensions(path)
    _validate_compression(ext, compression)

    if not os.path.isdir(path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)

    if df.is_empty():
        warnings.warn(f"Saving an empty DataFrame with columns: {df.columns}")

    if ext == ".xlsx":
        df.write_excel(path, **kwargs)
    elif ext == ".csv":
        if compression is None:
            df.write_csv(path, **kwargs)
        else:
            data = df.write_csv(None, **kwargs).encode("utf-8")
            _write_compressed_bytes(data, path, compression)
    elif ext == ".tsv":
        if compression is None:
            df.write_csv(path, separator="\t", **kwargs)
        else:
            data = df.write_csv(None, separator="\t", **kwargs).encode("utf-8")
            _write_compressed_bytes(data, path, compression)
    elif ext == ".parquet":
        parquet_kwargs = {"compression": "zstd", "compression_level": 6}

        if kwargs:
            parquet_kwargs.update(kwargs)

        df.write_parquet(path, **parquet_kwargs)
    elif ext == ".pkl":
        data = pickle.dumps(df)

        if compression is None:
            with open(path, "wb") as file:
                file.write(data)
        else:
            _write_compressed_bytes(data, path, compression)
    elif ext == ".joblib":
        buffer = io.BytesIO()
        joblib.dump(df, buffer)
        data = buffer.getvalue()

        if compression is None:
            with open(path, "wb") as file:
                file.write(data)
        else:
            _write_compressed_bytes(data, path, compression)
    else:
        raise ValueError(f"Extension < {ext} > is not supported.")
