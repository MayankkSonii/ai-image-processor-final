# 🚀 PixelForge AI: Enterprise Image Optimization Platform

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)
![Docker](https://img.shields.io/badge/Docker-First-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

A scalable, async image processing pipeline built with **FastAPI**, **Redis**, **RQ**, and **Real-ESRGAN**. Designed for high-performance, stateless operation.

## 🏗️ Architecture

```mermaid
graph TD
    User[User / Client] -->|Upload| Frontend[Streamlit UI]
    Frontend -->|HTTP POST| API[FastAPI Service]
    API -->|Enqueue Job| Redis[(Redis Queue)]
    API -->|Store File| Volume[(Shared Storage)]
    
    Worker[Python Worker] -->|Consume Job| Redis
    Worker -->|Read File| Volume
    Worker -->|1. Resize| Pipeline
    Worker -->|2. AI Upscale (x2/x4)| Pipeline
    Worker -->|3. Zip| Volume
    
    User -->|Download| API
```

## ✨ Key Features
- **Async Processing**: Jobs are offloaded to background workers via Redis.
- **AI Upscaling**: 
    - **Real-ESRGAN x4**: Professional-grade restoration and upscaling (PyTorch).
    - **FSRCNN (Fast CPU)**: Specialized for 2x/3x/4x scaling on standard hardware (5-10x faster).
    - **Standard Mode (Instant)**: Traditional high-quality Lanczos scaling for rapid asset creation.
- **🔥 Ultra Sharpening**: Multi-stage edge and text enhancement (Enabled by default) for maximum clarity in blurred screenshots and documents.
- **Bulk Operations**: Upload hundreds of images; the system handles the queue and provides a consolidated ZIP package.
- **Smart Optimization**: Auto-tiling to prevent memory crashes on CPU.
- **Stateless API**: Fully decoupled from workers.
- **Docker-First**: One command to launch the entire stack.

## 🛠️ Tech Stack
- **Backend**: FastAPI, Pydantic v2
- **Worker**: Python 3.10, RQ (Redis Queue), Pillow, OpenCV
- **AI Engine**: Real-ESRGAN (NCNN/PyTorch)
- **Infra**: Docker Compose, Redis 7
- **Frontend**: Streamlit

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose

### Deployment
1. **Start the Stack**
   ```bash
   ./docker-compose up --build -d
   ```
   *(Note: Use `./docker-compose` if you don't have the global command installed)*

2. **Access Interfaces**
   - **Frontend**: [http://localhost:8501](http://localhost:8501)
   - **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

3. **Stop**
   ```bash
   ./docker-compose down
   ```

## 🔌 API Reference

### Create Job
`POST /jobs/create`
- **Multipart Form**: `files` (List of images)
- **Fields**: 
  - `upscale_enabled` (bool)
  - `upscale_factor` (int: 2 or 4)
  - `model_name`: `RealESRGAN_x4plus` or `RealESRGAN_x4plus_anime_6B`
  - `resize_width` / `resize_height` (int)
  - `output_format` (PNG/JPEG/WEBP)

### Check Status
`GET /jobs/{job_id}`
- Returns: `{"status": "queued/processing/completed", "progress": 50, "processed_files": [...]}`

### Download
`GET /jobs/{job_id}/download`
- Returns: `application/zip`

## 📦 Directory Structure
```
.
├── app/            # FastAPI & Core Logic
├── worker/         # RQ Worker Logic
├── frontend/       # Streamlit App
├── docker/         # Dockerfiles
├── docker-compose.yml
└── requirements.txt
```

## 🔍 Troubleshooting & Logs

### Viewing Logs
To see what the AI is doing (and why it might be slow), view the **Worker Logs**:
```bash
./docker-compose logs -f worker
```
Look for lines like `Processing image.png` or `Downloading weights`.

### Parallel Processing (Multi-Job Execution)
By default, the worker processes jobs sequentially. To process multiple jobs in **parallel**, you can scale the workers using Docker Compose:
```bash
./docker-compose up -d --scale worker=2
```
*Note: Increasing the scale to 2 or more will allow multiple images to be upscaled simultaneously, but requires significant CPU and RAM. Each worker consumes about 2-4GB of RAM during AI upscaling.*

### Viewing Logs
To see the process status for each job (with Job ID tags), run:
```bash
./docker-compose logs -f worker
```
You will see logs like:
`[Job 123] Step 1/4: Loading Image...`
`[Job 123] Step 2/4: Starting AI Upscale...`

### Performance & Speed
-   **CPU vs GPU**: This app runs on **CPU** by default (for compatibility).
    -   **Fast CPU Mode (FSRCNN)**: Takes ~5-15 seconds per image. Recommended for Intel Macs.
    -   **Real-ESRGAN**: Takes ~1-5 minutes per image depending on scale and hardware.
-   **MPS (Mac M1/M2/M3)**: If running natively, the app will automatically use the Apple Silicon GPU for a massive speed boost.

### Running Natively (For 10x Speed)
To use your Mac GPU (Metal), run outside Docker:
1. `pip install -r requirements.txt`
2. `python main.py` and `python worker.py` (Refer to `setup-native.md` for full details).

### Common Issues
-   **"AI Engine Failed"**: Run `./docker-compose build --no-cache worker` to fix dependencies.
-   **Git Permission Denied**: Check your `git config user.name` and ensure you have write access to the repository.

---
Generated by Antigravity Agents.
