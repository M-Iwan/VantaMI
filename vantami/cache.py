from pathlib import Path

def get_novami_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "novami"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_chemberta_model_path() -> Path:
    return get_novami_cache_dir() / "ChemBERTa-model.joblib"

def get_chemberta_tokenizer_path() -> Path:
    return get_novami_cache_dir() / "ChemBERTa-tokenizer.joblib"

