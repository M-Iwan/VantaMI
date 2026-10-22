from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import shutil


def setup_unipka_cli() -> dict[str, Path]:
    """
    Copy packaged Uni-pKa CLI files to ~/.vantami/cli.
    """
    cli_dir = Path.home() / ".vantami" / "cli"
    cli_dir.mkdir(parents=True, exist_ok=True)

    uni_pka_clis = [
        "build_unipka.sh",
        "test_unipka.sh",
        "run_unipka.sh",
    ]

    installed: dict[str, Path] = {}

    for filename in uni_pka_clis:
        source = files("vantami").joinpath("files", "unipka", filename)
        destination = cli_dir / filename

        with source.open("rb") as reader, destination.open("wb") as writer:
            shutil.copyfileobj(reader, writer)

        destination.chmod(0o755)
        installed[filename] = destination

    print("Uni-pKa CLI files installed:")
    for name, path in installed.items():
        print(f"  {name}: {path}")

    print()
    print("Run this once to set up the required Docker image and checkpoints:")
    print(f" {cli_dir / 'build_unipka.sh'}")
    print()
    print("This operation requires Docker, the Hugging Face CLI (`hf`), git, and working CUDA.")

    return installed
