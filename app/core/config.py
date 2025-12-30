from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Image Optimizer"
    REDIS_URL: str = "redis://localhost:6379/0"
    DATA_DIR: str = "/data"
    UPLOAD_DIR: str = ""
    PROCESSED_DIR: str = ""
    ZIP_DIR: str = ""
    WORKER_CONCURRENCY: int = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # If running natively (not in Docker), use a local 'data' folder
        if not os.path.exists("/data") or not os.access("/", os.W_OK):
             self.DATA_DIR = os.path.join(os.getcwd(), "data")
             
        self.UPLOAD_DIR = os.path.join(self.DATA_DIR, "uploads")
        self.PROCESSED_DIR = os.path.join(self.DATA_DIR, "processed")
        self.ZIP_DIR = os.path.join(self.DATA_DIR, "zips")
        
        # Ensure directories exist
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)
        os.makedirs(self.ZIP_DIR, exist_ok=True)

    class Config:
        env_case_sensitive = True

settings = Settings()
