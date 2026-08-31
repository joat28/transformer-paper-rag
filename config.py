from pydantic_settings import BaseSettings

class Settings (BaseSettings) :
    openai_api_key : str
    chunk_size: int = 1000
    chunk_overlap: int = 150
    model: str = "gpt-5.4"
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"
    class Config:
        env_file = ".env"

settings = Settings()