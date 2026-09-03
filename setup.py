from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = ROOT / "README.md"

package_data: dict[str, list[str]] = {
    "novami": ["files/klekota_roth.joblib"],
}

setup(
    name="novami",
    version="0.4.3",
    author="Mateusz Iwan",
    author_email="mateusz.iwan@hotmail.com",
    description="A collection of ML/AI tools for chemistry applications.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/MateuszIwan/NovaMI",
    project_urls={
        "Source": "https://github.com/M-Iwan/NovaMI",
        "Bug Tracker": "https://github.com/M-Iwan/NovaMI/issues",
    },
    packages=find_packages(),
    include_package_data=False,
    package_data=package_data,
    install_requires=[
        "numpy >= 2.4",
        "pandas >= 3.0",
        "polars >= 1.40",
        "matplotlib >= 3.10",
        "seaborn >= 0.13",
        "scipy >= 1.17",
    ],
    extras_require={
        "full": [
            "scikit-learn >= 1.8",
            "rdkit >= 2026.03",
            "transformers >= 5.5",
        ],
    },
    python_requires=">=3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
