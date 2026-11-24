![](files/Vantami.png)

### What is this repository?

Vantami is a collection of ML and AI tools for chemistry applications, developed during my PhD. It includes code for
molecular representation, descriptor calculation, data processing, machine-learning workflows, deep learning,
statistics, and visualization.

The repository is still actively evolving. Some parts are relatively stable and reusable, while others are experimental,
project-specific, or retained for backwards compatibility.

### Installation

The base package can be installed from PyPI:

`pip install vantami`

For an installation including (the most important) optional dependencies:

`pip install "vantami[full]"`

### Versioning
Current version: 0.6.1

Versioning follows standard conventions where practical. Minor releases usually reflect larger updates to a particular
module or group of related features.

### Repository Structure

```
vantami/
├── deprecated/                    Older code kept for reference
├── dev/                           Development utilities and experimental code
├── dist/                          PyPI distribution files
├── environments/                  Environments for specialised workflows (e.g. CDDD)
├── vantami/
│   ├── api/
│   │   ├── convert.py             IUPAC-to-SMILES conversion using OPSIN
│   │   └── resolve.py             Name-to-SMILES resolution via PubChem, CIR, and experimental CAS support
│   ├── chemistry/
│   │   ├── manager.py             Tools for managing conformer searches
│   │   └── molecule.py            Conformer generation, optimisation, docking, and PLIF utilities
│   ├── cli/
│   │   ├── CDDD.py                CLI wrapper for CDDD descriptors
│   │   ├── CIR.py                 CLI wrapper for the CIR name resolver
│   │   ├── Dock.py                CLI wrapper for molecular docking workflows
│   │   ├── MolStandardizer.py     Molecular standardisation using RDKit
│   │   ├── Mordred.py             CLI wrapper for Mordred descriptors
│   │   └── OptunaOptimization.py  Older Optuna optimisation code
│   ├── data/
│   │   ├── cluster.py             Wrappers around many scikit-learn-based clustering methods
│   │   ├── descriptors.py         ECFP, MACCS, RDKit, Mordred, CDDD, ChemBERTa, MAPC, and other descriptors
│   │   ├── distance.py            Parallel distance-matrix and nearest-neighbour calculations
│   │   ├── embed.py               PCA, t-SNE, and UMAP embeddings
│   │   ├── filter.py              Outlier filtering based on molecular properties
│   │   ├── manager.py             Data management for training and inference
│   │   ├── partition.py           Dataset partitioning and scikit-learn convenience wrappers
│   │   └── transform.py           Data normalisation and preprocessing
│   ├── dl/
│   │   ├── dataset.py             Polars- and LMDB-backed multimodal datasets
│   │   ├── loader.py              Data loaders for multimodal datasets
│   │   ├── models.py              Models based on the MMUnit class
│   │   ├── modules.py             GNN, CNN, RNN, and linear-network building blocks
│   │   ├── storage.py             Utilities for storing Polars DataFrames in LMDB format
│   │   ├── utils.py               MMUnit base functionality and smaller helpers
│   │   └── vectorizer/
│   │       ├── graph.py           GraphVectorizer
│   │       └── string.py          StringVectorizer
│   ├── io/
│   │   ├── database.py            Preprocessing tools for ChEMBL and BindingDB files
│   │   └── file.py                Input/output utilities for several formats and Pandas/Polars DataFrames
│   ├── metrics/
│   │   └── modellability.py       MODI index
│   ├── ml/
│   │   ├── augood.py              AU-GOOD framework for model-performance evaluation
│   │   ├── evaluate.py            Utilities for evaluating models and ensembles
│   │   ├── models.py              Scikit-learn-compatible models and ensembles
│   │   ├── optimize.py            Hyperparameter optimisation
│   │   ├── params.py              Predefined Optuna parameter spaces
│   │   ├── score.py               Additional performance metrics
│   │   ├── select.py              Older sequential feature-selection code
│   │   └── utils.py               Helpers for building units from model and descriptor names
│   ├── nlp/
│   │   ├── article.py             Metadata retrieval from DOIs and article names
│   │   ├── cluster.py             Abstract clustering with latent Dirichlet allocation
│   │   ├── embedding.py           Sentence embeddings using MiniLM and Qwen3
│   │   └── tokenize.py            Word and document tokenisers
│   ├── standardize/
│   │   ├── clean.py               RDKit-based SMILES standardisation
│   │   ├── duplicates.py          Duplicate processing using median absolute deviation
│   │   ├── filter.py              Filtering based on selected features
│   │   └── validate.py            Structure validation and error detection
│   ├── stats/
│   │   ├── lmm.py                 Linear mixed-model tools
│   │   ├── tests.py               Statistical tests
│   │   └── utils.py               Statistical helper functions
│   ├── visualize/
│   │   ├── ecdf.py                Empirical cumulative distribution functions for molecular distances
│   │   ├── embedding.py           t-SNE and UMAP visualisation
│   │   ├── performance.py         Work-in-progress AU-GOOD plots
│   │   ├── predictions.py         Plots for assessing regression-model performance
│   │   ├── properties.py          Tools for comparing molecular properties across datasets
│   │   └── utils.py               Visualisation helpers and custom colour palette
│   └── paths.py                   Paths for locally saved models and data
├── vantami-r/
│   └── R/
│       └── lmm.R                  Linear mixed-model code in R
├── projects/
│   ├── cddd_setup/                Files for setting up a CDDD environment
│   └── qcg_template/              QCG PilotJob template for larger single-node training runs
├── temp/
├── tests/
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── README.md
└── setup.py
```

### Notes

This is primarily a research repository, so the level of maturity varies between modules. Code under deprecated/, 
as well as features labeled as experimental, work in progress, or older code, may require adaptation before use 
in a new project.

### Citation

If Vantami contributes to your work, please cite this repository and include the version used.

### License

This project is distributed under the terms described in [LICENSE](LICENSE).