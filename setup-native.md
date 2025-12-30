# Native Setup for Mac (GPU Acceleration)

To get the best performance (up to 10x faster) using your Mac's M1/M2/M3 GPU, run the application natively instead of inside Docker.

## Prerequisites
- **Python 3.10+** installed
- **Redis** installed (`brew install redis`)

## Steps

### 1. Start Redis
```bash
brew services start redis
```

### 2. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Backend (Terminal 1)
```bash
source venv/bin/activate
export DATA_DIR="./data"
export REDIS_URL="redis://localhost:6379/0"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Start Worker (Terminal 2)
```bash
source venv/bin/activate
export DATA_DIR="./data"
export REDIS_URL="redis://localhost:6379/0"
python worker/worker.py
```

### 5. Start Frontend (Terminal 3)
```bash
source venv/bin/activate
export API_URL="http://localhost:8000"
streamlit run frontend/app.py
```

Now the worker will detect `mps` and use your GPU!
