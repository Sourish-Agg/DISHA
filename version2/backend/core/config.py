# backend/core/config.py
# Centralised configuration — all env vars are read here once.

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    # MongoDB
    MONGO_URL: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "disha_db"

    # JWT
    JWT_SECRET: str = "change_me_in_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 10080  # 7 days

    # CORS
    CORS_ORIGINS: str = "http://localhost:5500,http://127.0.0.1:5500"

    # App
    APP_ENV: str = "development"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS origins as a Python list."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]


# Singleton — import this everywhere
settings = Settings()
