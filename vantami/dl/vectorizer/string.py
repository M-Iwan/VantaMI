import re
from typing import Iterable, List, Tuple, Literal
from collections import Counter


import numpy as np

try:
    from rdkit import Chem, RDLogger
    import torch
    import selfies as sf
    import deepsmiles as ds
except ImportError as exc:
    raise ImportError(f"Class < StringVectorizer > requires rdkit, torch, selfies, and deepsmiles") from exc


ALPHABET_TYPE = Literal["smiles", "deepsmiles", "selfies"]


class StringVectorizer:
    """
    Tokenize SMILES, DeepSMILES, or SELFIES into integer indices for embedding layers.

    Call prepare_alphabet on a corpus before from_smiles if alphabet is None.

    Parameters
    ----------
    alphabet: tuple, optional
        Token strings in vocabulary order; if None, set later via prepare_alphabet.
    alphabet_type: ALPHABET_TYPE, optional
        One of smiles, deepsmiles, selfies. Default is smiles.
    max_length: int, optional
        Maximum token count after split; longer strings raise ValueError.
    padding: bool, optional
        If True, pad sequences to max_length with pad token when alphabet is built.
    suppress: bool, optional
        If True, silence RDKit logs. Default is True.
    """
    def __init__(self, alphabet: Tuple = None, alphabet_type: ALPHABET_TYPE = "smiles", max_length: int = None,
                 padding: bool = True, suppress: bool = True):

        self.alphabet = alphabet
        self.alphabet_type = alphabet_type
        if self.alphabet_type not in {"smiles", "deepsmiles", "selfies"}:
            raise ValueError("Allowed options for alphabet are: smiles, deepsmiles, selfies")
        self.max_length = max_length
        self.padding = padding
        self.ds_converter = ds.Converter(branches=True, rings=True)
        if suppress:
            RDLogger.DisableLog("rdApp.*")

        r_atoms = r"Cl|Br|Si|Se|Na|Ca|Li|Mg|Zn|Fe|Cu|Mn|Hg|Sn|As|Bi|Cd|se|Cr|Sb"

        self.regex_patterns = {
            "smiles": re.compile(rf"(\[|]|{r_atoms}|[A-Z]|[a-z]|[=#/\\().+\-:]|\d)"),
            "deepsmiles": re.compile(rf"(\[|]|{r_atoms}|[A-Z]|[a-z]|[=#/\\().+\-:]|\)+|\(+|\d)"),
            "selfies": re.compile(r"\[.*?]")
        }
        self.char2idx = {char: idx for idx, char in enumerate(self.alphabet)} if self.alphabet is not None else None
        self.idx2char = {idx: char for idx, char in enumerate(self.alphabet)} if self.alphabet is not None else None

    @staticmethod
    def _prepare_batches(data: Iterable[str], n_jobs: int = 1, batch_size: int = 256):
        """
        Prepare batches of SMILES strings for downstream processing.
        """

        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {batch_size} instead.")

        n_jobs = min(effective_n_jobs(n_jobs), len(data))
        n_batches = max(n_jobs, math.ceil(len(data) / batch_size))
        data_batches = np.array_split(data, n_batches)

        return data, n_jobs, data_batches

    def from_smiles(self, smiles: str):
        """
        Encode one string to (int32 token tensor, int length).

        Parameters
        ----------
        smiles: str
            Input in the format given by alphabet_type.

        Returns
        -------
        tensor: torch.Tensor
            int32, shape (seq_len,) or (max_length,) if padding.
        length: int
            True token count before padding.
        """
        if self.char2idx is None:
            raise RuntimeError("Alphabet not initialized. Call prepare_alphabet to obtain it.")

        string = self.convert(smiles)
        tokens, length = self.split(string)

        if self.padding:
            tokens = self.pad(tokens)

        unk_idx = self.char2idx.get("<unk>", -1)
        array = np.array([self.char2idx.get(token, unk_idx) for token in tokens])
        tensor = torch.from_numpy(array).to(torch.int32).reshape(-1)
        return tensor, length

    def encode_smiles(self, smiles: str | Iterable[str]):
        """
        Same as from_smiles but accepts a str or iterable of str; always returns a list.

        Parameters
        ----------
        smiles: str | Iterable[str]

        Returns
        -------
        List[Tuple[Tensor, int]]
        """
        if isinstance(smiles, str):
            return [self.from_smiles(smiles)]
        else:
            if hasattr(smiles, "__iter__") and all(isinstance(item, str) for item in smiles):
                return [self.from_smiles(item) for item in smiles]
            else:
                raise ValueError("Unsupported datatype passed. Expected smiles to be either string"
                                 "or iterable of strings")

    def encode_many_smiles(self, smiles: Iterable[str]):
        """
        Convert multiple SMILES strings into their embedded format in parallel.

        Parameters
        ----------
        smiles: Iterable[str]
            Iterable of SMILES strings.
        n_jobs: int, optional
            Number of cores to use for calculations.
        batch_size: int, optional
            Number of SMILES per batch.
        timeout: int | None, optional
            Timeout per worker. Defaults to None, i.e., no timeout

        Returns
        -------
        List[Tuple[Tensor, int]]
        """
        smiles, n_jobs, smiles_batches = self._prepare_batches(
            smiles, n_jobs, batch_size
        )
        if not smiles:
            return []

        out = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
            delayed(self.encode_smiles)(smiles=smi) for smi in smiles_batches
        )

        return list(chain.from_iterable(out))


    def decode(self, indices: List[int]):
        """
        Map token indices back to a string

        Parameters
        ----------
        indices: List[int]
            Token indices.

        Returns
        -------
        str
            Concatenated characters; unknown indices become unk.
        """
        return "".join(self.idx2char.get(i, "<unk>") for i in indices)

    def convert(self, smiles: str):
        """
        Normalize input to the token string form used by split (SMILES as-is, etc.).

        Parameters
        ----------
        smiles : str

        Returns
        -------
        str
            SMILES, DeepSMILES, or SELFIES string depending on alphabet_type.
        """
        if self.alphabet_type == "smiles":
            return smiles
        elif self.alphabet_type == "deepsmiles":
            return self.ds_converter.encode(smiles)
        elif self.alphabet_type == "selfies":
            return sf.encoder(smiles)
        else:
            raise ValueError(f"Unsupported alphabet type: {self.alphabet_type}")

    def split(self, string):
        """
        Regex tokenization and length; enforces max_length when set.

        Parameters
        ----------
        string : str
            Already converted (e.g. via convert).

        Returns
        -------
        tokens : list
            Token strings.
        length : int
            len(tokens).
        """
        split_string = self.regex_patterns[self.alphabet_type].findall(string)
        length = len(split_string)

        if (self.max_length is not None) and (length > self.max_length):
            raise ValueError(f"Number of tokens in < {string} > [{len(split_string)}] exceeds allowed.")

        return split_string, length

    def pad(self, tokens: List[str]):
        """
        Append pad tokens so token list length equals max_length.

        Parameters
        ----------
        tokens: List[str]
            Token list after split.
        length: int
            Original length before padding.

        Returns
        -------
        List[str]
        """
        return tokens + ["<pad>"] * (self.max_length - len(tokens))

    def prepare_alphabet(self, smiles: List[str]):
        """
        Build vocabulary from corpus token frequency (most common first).

        Parameters
        ----------
        smiles : List[str]
            Corpus of SMILES (or converted strings) for counting.

        Returns
        -------
        alphabet : List
            Token list with unk and optional pad prepended when padding is True.
        """
        token_counter = Counter()

        for smi in smiles:
            string = self.convert(smi)
            tokens, _ = self.split(string)
            token_counter.update(tokens)

        alphabet = [token for token, _ in token_counter.most_common()] + ["<unk>"]

        if self.padding:
            alphabet = ["<pad>"] + alphabet

        return alphabet