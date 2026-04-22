from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    chroma_persist_directory: str = "/app/chroma"
    top_k_results: int = 5

    model_artifact_path: str = "/app/artifacts/priority_model.joblib"
    vectorizer_artifact_path: str = "/app/artifacts/vectorizer.joblib"
    label_encoder_artifact_path: str = "/app/artifacts/label_encoder.joblib"

    dataset_path: str = "/app/data/sample/customer_support_sample.csv"


@lru_cache
def get_settings() -> Settings:
    return Settings()

