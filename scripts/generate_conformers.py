"""
Generate conformers using DeepMolecule class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vantami.chem.manager import DeepMoleculeManager
from vantami.io import read_pl

parser = argparse.ArgumentParser("ConformerGeneration")

parser.add_argument("--data-path", type=Path, help="Input dataset path readable by vantami.io.read_pl")
parser.add_argument("--save-path", type=Path, help="Output path for the saved DeepMoleculeManager.")
parser.add_argument("--smiles-col", type=str, default="SMILES", help="Column containing SMILES strings.")
parser.add_argument("--n-conformers", type=int, default=32, help="Number of conformers to generate per molecule.")
parser.add_argument("--energy-method", type=str, default="MMFF94", help="Energy method used for conformer generation.")
parser.add_argument("--n-jobs", type=int, default=12, help="Number of parallel jobs.")
parser.add_argument("--batch-size", type=int, default=256, help="Number of molecules processed per batch.")

args = parser.parse_args()

if args.n_conformers < 1:
    parser.error("--n-conformers must be greater than zero.")
if args.n_jobs < 1:
    parser.error("--n-jobs must be greater than zero.")
if args.batch_size < 1:
    parser.error("--batch-size must be greater than zero.")
if args.prune_threshold < 0:
    parser.error("--prune-threshold must be non-negative.")
if args.max_iterations < 1:
    parser.error("--max-iterations must be greater than zero.")

def main(args) -> None:
    """
    Run conformer generation and save the molecule manager.
    """

    df = read_pl(args.data_path)

    if args.smiles_col not in df.columns:
        raise KeyError(f"Column {args.smiles_col!r} was not found. Available columns: {', '.join(df.columns)}")

    smiles = df[args.smiles_col].unique().drop_nulls().to_list()

    if not smiles:
        raise ValueError(f"No non-null SMILES values found in column {args.smiles_col!r}.")

    conformer_settings = {
        "n_conformers": args.n_conformers,
        "method": args.energy_method,
        "seed": args.seed,
        "prune_threshold": args.prune_threshold,
        "max_iterations": args.max_iterations,
        "charge": None,
        "uhf": None,
    }

    manager = DeepMoleculeManager(conformer_settings)

    manager.add_many(
        [(f"mol_{index:06d}", smiles_value) for index, smiles_value in enumerate(smiles)]
    )

    manager.search_conformers_many(
        smiles=None,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
    )

    print(manager.summary())

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    manager.save(args.save_path)

    print(f"\nSaved manager to: {args.save_path}")

if __name__ == "__main__":
    main()
