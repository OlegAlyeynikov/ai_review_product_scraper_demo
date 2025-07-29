#!/usr/bin/env python3
import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings with PostgreSQL and Pinecone configuration"""

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # PostgreSQL Database (PRIMARY DATABASE)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://reviews_user:reviews_pass@localhost:5432/reviews_db"
    )

    # Vector DB Configuration - PINECONE
    VECTOR_DB_TYPE: str = "pinecone"
    VECTOR_DB_PATH: str = "./vector_db"  # For local backups/cache

    # Pinecone Configuration (Cloud Vector Database)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: str = "us-west1-gcp-free"
    PINECONE_INDEX_NAME: str = "product-reviews"

    # Google Cloud Storage Configuration
    STORAGE_MODE: str = "mock"  # "mock" or "gcs"
    GOOGLE_CLOUD_BUCKET: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    LOCAL_STORAGE_PATH: str = "./local_storage"

    # AI APIs Configuration
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None  # Alternative LLM
    CLAUDE_API_KEY: Optional[str] = None  # Alternative LLM

    # Production Settings
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Use ConfigDict for Pydantic V2
    model_config = ConfigDict(env_file=".env")

# Global settings instance
settings = Settings()