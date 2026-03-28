"""Central configuration for UEdu project."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths (relative to project root)
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = Path(__file__).parent.parent / "data"
    zenodo_dir: Path = Path(__file__).parent.parent / "data" / "raw" / "zenodo"
    kaggle_dir: Path = Path(__file__).parent.parent / "data" / "raw" / "kaggle"
    processed_dir: Path = Path(__file__).parent.parent / "data" / "processed"
    results_dir: Path = Path(__file__).parent.parent / "results"

    # API keys (all optional -- set via environment variables or .env file)
    # Only needed for LLM feature extraction (models M7-M9)
    gemini_api_key: str = ""
    # Optional second Gemini key for parallel extraction
    gemini_api_key_2: str = ""
    # Only needed for synthetic text generation experiments
    openai_api_key: str = ""

    # Model settings
    random_seed: int = 42
    n_folds: int = 5
    test_size: float = 0.15
    val_size: float = 0.15

    # LLM settings (for LLM feature extraction, models M7-M9)
    llm_model: str = "gemini-2.0-flash-lite"
    llm_sample_size: int = 2000
    llm_workers: int = 20
    llm_checkpoint_every: int = 5000

    # Feature dimensions
    tfidf_max_features: int = 256
    n_psycholinguistic: int = 40
    n_llm_features: int = 8

    # Zenodo files
    zenodo_files: dict[str, str] = {
        "depression": "depression_2019_features_tfidf_256.csv",
        "anxiety": "anxiety_2019_features_tfidf_256.csv",
        "suicidewatch": "suicidewatch_2019_features_tfidf_256.csv",
    }

    # Mental health subreddits (positive class)
    mh_subreddits: list[str] = [
        "depression",
        "anxiety",
        "SuicideWatch",
    ]

    # Control subreddits (negative class) -- from Zenodo full dataset
    control_subreddits: list[str] = [
        "fitness",
        "jokes",
        "relationships",
        "teaching",
        "personalfinance",
    ]

    model_config = {"env_file": ".env", "env_prefix": "UEDU_", "extra": "ignore"}


settings = Settings()
