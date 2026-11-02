from pathlib import Path
import subprocess


def get_vantami_cache_dir() -> Path:
    cache_dir = Path.home() / ".vantami" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_vantami_data_dir() -> Path:
    data_dir = Path.home() / ".vantami" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_python_path(env: str) -> Path:
    result = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", env, "which", "python"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def get_chemberta_model_path() -> Path:
    return get_vantami_cache_dir() / "ChemBERTa-model.joblib"


def get_chemberta_tokenizer_path() -> Path:
    return get_vantami_cache_dir() / "ChemBERTa-tokenizer.joblib"


def get_molencoder_model_path() -> Path:
    return get_vantami_cache_dir() / "MolEncoder-model.joblib"


def get_molencoder_tokenizer_path() -> Path:
    return get_vantami_cache_dir() / "MolEncoder-tokenizer.joblib"


def get_minilm_model_path() -> Path:
    return get_vantami_cache_dir() / "MiniLM-model.joblib"


def get_qwen3_model_path() -> Path:
    return get_vantami_cache_dir() / "Qwen3-model.joblib"


def get_unipka_input_path() -> Path:
    return get_vantami_data_dir() / "input.csv"


def get_unipka_output_path() -> Path:
    return get_vantami_data_dir() / "output.csv"


