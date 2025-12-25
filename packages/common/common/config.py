from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    mock_mode: bool = Field(True, alias="MOCK_MODE")

    default_provider: str = Field("ollama", alias="DEFAULT_PROVIDER")
    ollama_model: str = Field("llama3", alias="OLLAMA_MODEL")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")

    orchestrator_url: str = Field("http://orchestrator:8001", alias="ORCHESTRATOR_URL")
    retrieval_url: str = Field("http://retrieval:8002", alias="RETRIEVAL_URL")
    evaluator_url: str = Field("http://evaluator:8003", alias="EVALUATOR_URL")

    chroma_persist_dir: str = Field("./data/chroma", alias="CHROMA_PERSIST_DIR")
    chroma_host: str = Field("chroma", alias="CHROMA_HOST")
    chroma_port: int = Field(8000, alias="CHROMA_PORT")

    allowed_origins: str = Field(
        "http://localhost:3000,http://127.0.0.1:3000", alias="ALLOWED_ORIGINS"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]

