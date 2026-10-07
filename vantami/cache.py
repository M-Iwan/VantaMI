from pathlib import Path

def get_vantami_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "vantami"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

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