from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class MemoryServiceConfig(BaseSettings):
    # Infinity 配置
    INFINITY_HOST: str = "localhost"
    INFINITY_PORT: int = 23817

    # 向量服务配置
    EMBEDDING_SERVICE_TYPE: str = "openai"
    EMBEDDING_SERVICE_URL: str = "https://openai.linktre.cc/v1/embeddings"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536

    # 数据库配置
    DEFAULT_DATABASE: str = "memory_store"
    TABLE_PREFIX: str = "memories_"

    # HNSW索引配置
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200

    # 使用新的配置方式
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )