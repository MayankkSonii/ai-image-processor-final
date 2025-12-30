import os
import shutil
import zipfile
from PIL import Image, ImageEnhance, ImageFilter
from rq import get_current_job
import concurrent.futures
from app.core.config import settings
from app.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Try importing RealESRGAN
HAS_AI = False
try:
    import torch
    import functools
    # PATCH: PyTorch 2.6+ defaults weights_only=True, breaking legacy RealESRGAN loading
    # We force weights_only=False safely here for this specific internal use
    torch.load = functools.partial(torch.load, weights_only=False)
    
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan.utils import RealESRGANer
    import numpy as np
    import cv2
    HAS_AI = True
    logger.info("AI upscaling core (RealESRGAN) successfully loaded.")
    
    class FastSuperRes:
        def __init__(self, model_path, scale):
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(model_path)
            self.sr.setModel("fsrcnn", scale)
            self.scale = scale
            
        def enhance(self, img, outscale=None):
            # OpenCV SuperRes upsamples to the model's trained scale
            result = self.sr.upsample(img)
            return result, None

except ImportError as e:
    # This is expected in the API container, as it doesnt need AI libs to queue jobs.
    logger.debug(f"AI libraries (torch/realesrgan) not found: {e}. Worker functionality will be limited to 'Standard' mode.")
    pass
except Exception as e:
    logger.warning(f"Unexpected error during AI library initialization: {e}")
    pass

def download_weights(url, dest):
    # Check if exists AND has reasonable size (> 1MB)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 1024:
        return

    logger.info(f"Downloading weights from {url} to {dest}")
    import requests
    temp_dest = dest + ".tmp"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(temp_dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        os.rename(temp_dest, dest)
    except Exception as e:
        if os.path.exists(temp_dest):
            os.remove(temp_dest)
        raise Exception(f"Failed to download weights: {e}")

def process_job_task(job_id, file_paths, config):
    job = get_current_job()
    logger.info(f"[{job_id}] Received processing task with {len(file_paths)} files.")
    
    processed_dir = os.path.join(settings.PROCESSED_DIR, job_id)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Weights Dir
    # Weights Dir (Persistent across container restarts)
    weights_dir = os.path.join(settings.DATA_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    total = len(file_paths)
    
    # Initialize AI model if needed
    upsampler = None
    if config.get('upscale_enabled'):
        model_type_sel = config.get('model_name', 'RealESRGAN_x4plus')
        target_scale = config.get('upscale_factor', 4)
        
        if not HAS_AI and model_type_sel != 'Standard':
            error_msg = f"AI libraries (torch/opencv-dnn) are missing. Cannot run {model_type_sel}."
            logger.error(f"[{job_id}] {error_msg}")
            raise Exception(error_msg)
            
        if model_type_sel == 'Standard':
             logger.info(f"[{job_id}] Standard scaling selected. Skipping AI for maximum speed.")
             upsampler = None

        elif model_type_sel.startswith('FSRCNN'):
             # OpenCV DNN Super Resolution (Extremely fast on CPU)
             if target_scale <= 2: scale = 2
             elif target_scale == 3: scale = 3
             else: scale = 4
             
             model_file = f"FSRCNN_x{scale}.pb"
             model_path = os.path.join(weights_dir, model_file)
             url = f"https://github.com/fannymonori/TF-ESPCN/raw/master/export/FSRCNN_x{scale}.pb"
             logger.info(f"[{job_id}] Ensuring Fast CPU weights for {model_file}...")
             download_weights(url, model_path)
             upsampler = FastSuperRes(model_path, scale)
             logger.info(f"[{job_id}] Fast CPU upsampler (FSRCNN x{scale}) ready.")

        elif 'RealESRGAN' in model_type_sel:
             # Map selection to file/config
             if target_scale == 2:
                 model_name = 'RealESRGAN_x2plus'
                 scale = 2
                 url_tag = 'v0.2.1'
             else:
                 model_name = 'RealESRGAN_x4plus'
                 scale = 4
                 url_tag = 'v0.1.0'
                 
             model_file = f"{model_name}.pth"
             model_path = os.path.join(weights_dir, model_file)
             url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/{url_tag}/{model_file}"
             logger.info(f"[{job_id}] Ensuring model weights for {model_name}...")
             download_weights(url, model_path)
             
             device = 'mps' if torch.backends.mps.is_available() else 'cpu'
             logger.info(f"[{job_id}] Using device: {device.upper()}")
             
             # RRDBNet Configs
             if model_name == 'RealESRGAN_x4plus':
                 model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
             elif model_name == 'RealESRGAN_x2plus':
                 model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
             
             # CPU can handle larger tiles than minimal VRAM GPUs. 1000px is a good balance.
             tile_size = 1000 if device == 'cpu' else 800
             logger.info(f"[{job_id}] Initializing RealESRGANer (tile={tile_size})")
             upsampler = RealESRGANer(
                 scale=scale, 
                 model_path=model_path, 
                 model=model, 
                 tile=tile_size, 
                 tile_pad=10, 
                 pre_pad=0, 
                 half=False, 
                 device=device
             )
             logger.info(f"[{job_id}] AI upsampler ready.")
    
    # Concurrency limit: 
    # AI models (Real-ESRGAN/FSRCNN) are NOT thread-safe and are RAM heavy.
    # We must use 1 thread for AI. For Standard mode, we can use the env setting.
    is_standard = config.get('model_name') == 'Standard'
    max_workers = settings.WORKER_CONCURRENCY if is_standard else 1
    
    logger.info(f"[{job_id}] Processing {total} files using {max_workers} thread(s). (AI: {not is_standard})")
    
    def process_single_file(index_and_path):
        i, file_path = index_and_path
        filename = os.path.basename(file_path)
        try:
            logger.info(f"[{job_id}] Thread started for file {i+1}/{total}: {filename}")
            img = Image.open(file_path).convert('RGB')
            
            # 1. Resize
            rw = config.get('resize_width')
            rh = config.get('resize_height')
            if rw or rh:
                orig_w, orig_h = img.size
                if not rw: rw = int(orig_w * (rh / orig_h))
                if not rh: rh = int(orig_h * (rw / orig_w))
                img = img.resize((rw, rh), Image.LANCZOS)
                
            # 2. Upscale
            if config.get('upscale_enabled') and upsampler is not None:
                img_np = np.array(img)
                img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                output, _ = upsampler.enhance(img_cv2, outscale=config.get('upscale_factor', 4))
                img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            elif config.get('upscale_enabled') and upsampler is None:
                target_factor = config.get('upscale_factor', 4)
                w, h = img.size
                img = img.resize((w * target_factor, h * target_factor), Image.LANCZOS)
            
            # 3. Sharpening
            if config.get('ultra_sharpen'):
                img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=250, threshold=0))
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(2.0)
                img = ImageEnhance.Contrast(img).enhance(1.2)
            else:
                img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
            
            # 4. Save
            out_format = config.get('output_format', 'PNG')
            base, _ = os.path.splitext(filename)
            out_name = f"{base}.{out_format.lower()}"
            save_path = os.path.join(processed_dir, out_name)
            
            save_kwargs = {'quality': config.get('quality', 90)} if out_format in ['JPEG', 'WEBP'] else {}
            img.save(save_path, format=out_format, **save_kwargs)
            
            logger.info(f"[{job_id}] File {i+1}/{total} completed: {out_name}")
            
            # Update Meta
            if job:
                with job.connection.lock(f"lock:job:meta:{job_id}"):
                    processed = job.meta.get('processed_files', [])
                    processed.append({'filename': out_name, 'url': f"/jobs/{job_id}/file/{out_name}"})
                    job.meta['processed_files'] = processed
                    progress = int(((len(processed)) / total) * 100)
                    job.meta['progress'] = progress
                    job.save_meta()
            return True
        except Exception as e:
            logger.error(f"[{job_id}] Thread FAILED for file {filename}: {e}")
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(process_single_file, enumerate(file_paths)))
            
    logger.info(f"[{job_id}] All files processed. Bundling into zip...")
    zip_path = os.path.join(settings.ZIP_DIR, f"{job_id}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for root, dirs, files in os.walk(processed_dir):
            for file in files:
                zf.write(os.path.join(root, file), file)
                
    logger.info(f"[{job_id}] SECURE BUNDLE CREATED at {zip_path}. Distribution ready.")
    return {"status": "completed", "zip": zip_path}
