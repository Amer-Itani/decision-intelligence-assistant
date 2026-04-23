from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Decision Intelligence Assistant API"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level: str = "INFO"

    groq_api_key: str = Field(default="")
    groq_model: str = "llama-3.1-8b-instant"

    chroma_persist_directory: str = str(PROJECT_ROOT / "artifacts" / "chroma")
    retrieval_index_path: str = str(PROJECT_ROOT / "artifacts" / "retrieval_index.joblib")
    top_k_results: int = 5

    model_artifact_path: str = str(PROJECT_ROOT / "artifacts" / "priority_model.joblib")
    vectorizer_artifact_path: str = str(PROJECT_ROOT / "artifacts" / "vectorizer.joblib")
    label_encoder_artifact_path: str = str(PROJECT_ROOT / "artifacts" / "label_encoder.joblib")
    model_metadata_artifact_path: str = str(PROJECT_ROOT / "artifacts" / "model_metadata.json")

    dataset_path: str = str(PROJECT_ROOT / "data" / "sample" / "customer_support_sample.csv")
    log_path: str = str(PROJECT_ROOT / "logs" / "requests.jsonl")
    llm_input_cost_per_1m_tokens: float = 0.05
    llm_output_cost_per_1m_tokens: float = 0.08


@lru_cache
def get_settings() -> Settings:
    return Settings()
