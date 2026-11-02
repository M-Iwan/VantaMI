import re
import numpy as np
from collections import Counter

try:
    from rdkit import Chem, RDLogger
    import torch
    import selfies as sf
    import deepsmiles as ds
except ImportError as exc:
    raise ImportError(f"Class < StringVectorizer > requires rdkit, torch, selfies, and deepsmiles:\n{exc}")


class StringVectorizer:
    """
    Tokenize SMILES, DeepSMILES, or SELFIES into integer indices for embedding layers.

    Call prepare_alphabet on a corpus before from_smiles if alphabet is None.

    Parameters
    ----------
    alphabet : tuple, optional
        Token strings in vocabulary order; if None, set later via prepare_alphabet.
    alphabet_type : str, optional
        One of smiles, deepsmiles, selfies. Default is smiles.
    max_length : int, optional
        Maximum token count after split; longer strings raise ValueError.
    padding : bool, optional
        If True, pad sequences to max_length with pad token when alphabet is built.
    suppress : bool, optional
        If True, silence RDKit logs. Default is True.
    """
    def __init__(self, alphabet: tuple = None, alphabet_type: str = 'smiles', max_length: int = None,
                 padding: bool = True, suppress: bool = True):

        self.alphabet = alphabet
        self.alphabet_type = alphabet_type
        if self.alphabet_type not in ['smiles', 'deepsmiles', 'selfies']:
            raise ValueError('Allowed options for alphabet are: smiles, deepsmiles, selfies')
        self.max_length = max_length
        self.padding = padding
        self.ds_converter = ds.Converter(branches=True, rings=True)
        if suppress:
            RDLogger.DisableLog('rdApp.*')

        r_atoms = r"Cl|Br|Si|Se|Na|Ca|Li|Mg|Zn|Fe|Cu|Mn|Hg|Sn|As|Bi|Cd|se|Cr|Sb"

        self.regex_patterns = {
            'smiles': re.compile(rf"(\[|]|{r_atoms}|[A-Z]|[a-z]|[=#/\\().+\-:]|\d)"),
            'deepsmiles': re.compile(rf"(\[|]|{r_atoms}|[A-Z]|[a-z]|[=#/\\().+\-:]|\)+|\(+|\d)"),
            'selfies': re.compile(r"\[.*?]")
        }
        self.char2idx = {char: idx for idx, char in enumerate(self.alphabet)} if self.alphabet is not None else None
        self.idx2char = {idx: char for idx, char in enumerate(self.alphabet)} if self.alphabet is not None else None

    def from_smiles(self, smiles: str):
        """
        Encode one string to (int32 token tensor, int length).

        Parameters
        ----------
        smiles : str
            Input in the format given by alphabet_type.

        Returns
        -------
        tensor : torch.Tensor
            int32, shape (seq_len,) or (max_length,) if padding.
        length : int
            True token count before padding.
        """
        if self.char2idx is None:
            raise RuntimeError("Alphabet not initialized. Call prepare_alphabet to obtain it.")

        string = self.convert(smiles)
        string, length = self.split(string)

        if self.padding:
            string = self.pad(string, length)

        unk_idx = self.char2idx.get('<unk>', -1)
        array = np.array([self.char2idx.get(token, unk_idx) for token in string])
        tensor = torch.from_numpy(array).to(torch.int32).reshape(-1)
        return tensor, length

    def encode(self, smiles: Union[str, Iterable[str]]):
        """
        Same as from_smiles but accepts a str or iterable of str; always returns a list.

        Parameters
        ----------
        smiles : str or iterable of str

        Returns
        -------
        list [tensor, length]
        """
        if isinstance(smiles, str):
            return [self.from_smiles(smiles)]
        else:
            if hasattr(smiles, "__iter__") and all(isinstance(item, str) for item in smiles):
                return [self.from_smiles(item) for item in smiles]
            else:
                raise ValueError("Unsupported datatype passed. Expected smiles to be either string"
                                 "or iterable of strings")

    def decode(self, indices: List[int]):
        """
        Map token indices back to a string (best-effort for SMILES-like alphabets).

        Parameters
        ----------
        indices : list of int
            Token indices.

        Returns
        -------
        str
            Concatenated characters; unknown indices become unk.
        """
        return ''.join(self.idx2char.get(i, '<unk>') for i in indices)

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
        if self.alphabet_type == 'smiles':
            return smiles
        elif self.alphabet_type == 'deepsmiles':
            return self.ds_converter.encode(smiles)
        elif self.alphabet_type == 'selfies':
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
            raise ValueError(f'Number of tokens in < {string} > [{len(split_string)}] exceeds allowed.')

        return split_string, length

    def pad(self, string, length):
        """
        Append pad tokens so token list length equals max_length.

        Parameters
        ----------
        string : list
            Token list after split.
        length : int
            Original length before padding.

        Returns
        -------
        list
            Padded token list.
        """
        return string + ['<pad>'] * (self.max_length - length)

    def prepare_alphabet(self, smiles: List[str]):
        """
        Build vocabulary from corpus token frequency (most common first).

        Parameters
        ----------
        smiles : list of str
            Corpus of SMILES (or converted strings) for counting.

        Returns
        -------
        alphabet : list
            Token list with unk and optional pad prepended when padding is True.
        """
        token_counter = Counter()

        for smi in smiles:
            string = self.convert(smi)
            tokens, _ = self.split(string)
            token_counter.update(tokens)

        alphabet = [token for token, _ in token_counter.most_common()] + ['<unk>']

        if self.padding:
            alphabet = ['<pad>'] + alphabet

        return alphabet