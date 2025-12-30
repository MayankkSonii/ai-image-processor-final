from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Image Optimizer"
    REDIS_URL: str = "redis://localhost:6379/0"
    DATA_DIR: str = "/data"
    UPLOAD_DIR: str = "/data/uploads"
    PROCESSED_DIR: str = "/data/processed"
    ZIP_DIR: str = "/data/zips"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)
        os.makedirs(self.ZIP_DIR, exist_ok=True)

    class Config:
        env_case_sensitive = True

settings = Settings()
