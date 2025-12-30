from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingConfig(BaseModel):
    upscale_factor: int = 4
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    output_format: str = "PNG"
    quality: int = 90
    upscale_enabled: bool = True
    model_name: str = "RealESRGAN_x4plus"
    ultra_sharpen: bool = True

class JobCreate(BaseModel):
    config: ProcessingConfig

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    input_files: List[str]
    # output_files: List[str] = []
    error: Optional[str] = None
    progress: int = 0
    processed_files: List[dict] = []
