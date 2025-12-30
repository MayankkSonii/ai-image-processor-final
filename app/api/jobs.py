from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
import uuid
import os
import shutil
import json
from rq.job import Job
from rq.exceptions import NoSuchJobError
from app.models.job import JobResponse, JobStatus, ProcessingConfig
from app.core.config import settings
from app.utils.redis import queue, redis_conn
from app.services.pipeline import process_job_task 
from app.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

@router.post("/create", response_model=JobResponse)
async def create_job(
    files: List[UploadFile] = File(...),
    upscale_enabled: bool = Form(True),
    upscale_factor: int = Form(4),
    resize_width: Optional[int] = Form(None),
    resize_height: Optional[int] = Form(None),
    output_format: str = Form("PNG"),
    quality: int = Form(90),
    model_name: str = Form("RealESRGAN_x4plus"),
    ultra_sharpen: bool = Form(False)
):
    job_id = str(uuid.uuid4())
    logger.info(f"Received job creation request for job_id: {job_id}. Configuration: upscale={upscale_enabled}, factor={upscale_factor}, model={model_name}")
    job_upload_dir = os.path.join(settings.UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        file_path = os.path.join(job_upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)
    
    config = ProcessingConfig(
        upscale_enabled=upscale_enabled,
        upscale_factor=upscale_factor,
        resize_width=resize_width,
        resize_height=resize_height,
        output_format=output_format,
        quality=quality,
        model_name=model_name,
        ultra_sharpen=ultra_sharpen
    )
    
    # Enqueue Job
    # Enqueue Job
    # We pass args positionally to avoid conflict with RQ's 'job_id' kwarg
    rq_job = queue.enqueue(
        process_job_task,
        args=(job_id, saved_files, config.model_dump()),
        job_id=job_id,
        job_timeout='10m',
        result_ttl=86400
    )
    logger.info(f"Job {job_id} successfully queued.")
    
    return JobResponse(
        job_id=rq_job.id,
        status=JobStatus.QUEUED,
        created_at=rq_job.created_at,
        input_files=[f.filename for f in files]
    )

class JobResponseExtended(JobResponse):
    processed_files: List[dict] = []

@router.get("/{job_id}", response_model=JobResponseExtended)
async def get_job_status(job_id: str):
    logger.debug(f"Status check requested for job_id: {job_id}")
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        logger.warning(f"Job {job_id} not found in queue.")
        raise HTTPException(status_code=404, detail="Job not found")
        
    status = JobStatus.QUEUED
    error = None
    
    if job.is_started: status = JobStatus.PROCESSING
    if job.is_finished: status = JobStatus.COMPLETED
    if job.is_failed: 
        status = JobStatus.FAILED
        error = str(job.exc_info)
        logger.error(f"Job {job_id} failed: {error}")
        
    progress = job.meta.get('progress', 0)
    processed_files = job.meta.get('processed_files', [])
    if status == JobStatus.COMPLETED: 
        progress = 100
    else:
        logger.debug(f"Job {job_id} current status: {status}, progress: {progress}%")
        
    return {
        "job_id": job_id,
        "status": status,
        "created_at": job.created_at,
        "input_files": [],
        "error": error,
        "progress": progress,
        "processed_files": processed_files
    }

@router.get("/{job_id}/file/{filename}")
async def get_processed_file(job_id: str, filename: str):
    logger.info(f"Request for processed file {filename} from job {job_id}")
    file_path = os.path.join(settings.PROCESSED_DIR, job_id, filename)
    if not os.path.exists(file_path):
        logger.warning(f"Processed file {filename} for job {job_id} not found at {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    from fastapi.responses import FileResponse
    return FileResponse(file_path)

@router.get("/{job_id}/download")
async def download_job(job_id: str):
    logger.info(f"Download request for full package for job_id: {job_id}")
    zip_path = os.path.join(settings.ZIP_DIR, f"{job_id}.zip")
    if not os.path.exists(zip_path):
        logger.warning(f"Zip archive for job {job_id} not found at {zip_path}")
        raise HTTPException(status_code=404, detail="File not found or processing not complete")
        
    from fastapi.responses import FileResponse
    return FileResponse(zip_path, filename=f"processed_{job_id}.zip")
