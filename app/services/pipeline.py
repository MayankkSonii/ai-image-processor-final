import os
import shutil
import zipfile
from PIL import Image, ImageEnhance, ImageFilter
from rq import get_current_job
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

except Exception as e:
    logger.error(f"Failed to import RealESRGAN/Torch: {e}")
    pass

def download_weights(url, dest):
    if not os.path.exists(dest):
        logger.info(f"Downloading weights from {url} to {dest}")
        import requests
        response = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def process_job_task(job_id, file_paths, config):
    job = get_current_job()
    logger.info(f"[{job_id}] Received processing task with {len(file_paths)} files.")
    
    processed_dir = os.path.join(settings.PROCESSED_DIR, job_id)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Weights Dir
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    total = len(file_paths)
    
    # Initialize AI model if needed
    upsampler = None
    if config.get('upscale_enabled') and HAS_AI:
        model_type_sel = config.get('model_name', 'RealESRGAN_x4plus')
        target_scale = config.get('upscale_factor', 4)

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
             if model_type_sel == 'RealESRGAN_x4plus_anime_6B':
                 model_name = 'RealESRGAN_x4plus_anime_6B'
                 scale = 4
                 url_tag = 'v0.1.0'
             elif target_scale == 2:
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
             elif model_name == 'RealESRGAN_x4plus_anime_6B':
                 model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
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
    
    for i, file_path in enumerate(file_paths):
        out_name = None # Initialize scope
        try:
            filename = os.path.basename(file_path)
            logger.info(f"[Job {job_id}] Processing file {i+1}/{total}: {filename}")
            
            img = Image.open(file_path).convert('RGB')
            logger.info(f"[Job {job_id}] Image loaded: {filename} (Size: {img.size})")
            
            # 1. Resize (Pre-Upscale)
            rw = config.get('resize_width')
            rh = config.get('resize_height')
            if rw or rh:
                orig_w, orig_h = img.size
                if not rw: rw = int(orig_w * (rh / orig_h))
                if not rh: rh = int(orig_h * (rw / orig_w))
                logger.info(f"[Job {job_id}] Resizing {filename} to {rw}x{rh}")
                img = img.resize((rw, rh), Image.LANCZOS)
            # 2. AI Upscale (or Standard Resize)
            if config.get('upscale_enabled') and upsampler is not None:
                if not HAS_AI:
                    raise Exception("AI Engine (Real-ESRGAN/FSRCNN) failed to load.")
                
                # Convert PIL RGB to OpenCV BGR
                img_np = np.array(img)
                img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Enhance
                logger.info(f"[Job {job_id}] Starting AI Upscale for {filename} (Factor: {config.get('upscale_factor', 4)})")
                output, _ = upsampler.enhance(img_cv2, outscale=config.get('upscale_factor', 4))
                logger.info(f"[Job {job_id}] AI Upscale completed for {filename}")
                
                # Convert Back to PIL RGB
                img_np_out = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_np_out)
            elif config.get('upscale_enabled') and upsampler is None:
                # Standard Resize to target scale
                target_factor = config.get('upscale_factor', 4)
                w, h = img.size
                logger.info(f"[Job {job_id}] Applying standard {target_factor}x scaling for {filename}")
                img = img.resize((w * target_factor, h * target_factor), Image.LANCZOS)
            
            # 3. Post-Processing (Production Grade)
            from PIL import ImageEnhance, ImageFilter
            
            if config.get('ultra_sharpen'):
                logger.info(f"[{job_id}] Applying ULTRA SHARPENING for maximum clarity.")
                # Stage 1: Strong UnsharpMask
                img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=250, threshold=0))
                # Stage 2: Software Sharpness Boost
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(2.0)
                # Stage 3: Contrast Boost for Text Clarity
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
            else:
                # Standard Sharpen
                img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
                # Slight Contrast Boost
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.05)
             
            # 4. Save
            out_format = config.get('output_format', 'PNG')
            base, _ = os.path.splitext(filename)
            out_name = f"{base}.{out_format.lower()}"
            save_path = os.path.join(processed_dir, out_name)
            
            save_kwargs = {}
            if out_format in ['JPEG', 'WEBP']:
                save_kwargs['quality'] = config.get('quality', 90)
            
            logger.info(f"[{job_id}] Step 4/4: Saving processed image: {out_name}")
            img.save(save_path, format=out_format, **save_kwargs)
            
            # Update Meta (Success Case)
            if job:
                if 'processed_files' not in job.meta:
                    job.meta['processed_files'] = []
                
                job.meta['processed_files'].append({
                    'filename': out_name,
                    'url': f"/jobs/{job_id}/file/{out_name}"
                })
                job.save_meta()
            
        except Exception as e:
            logger.error(f"[{job_id}] Failed to process {filename}: {e}", exc_info=True)
            # Continue to next file
            
        # Update Overall Progress
        if job:
            progress = int(((i + 1) / total) * 100)
            job.meta['progress'] = progress
            job.save_meta()
            logger.info(f"[{job_id}] Progress: {progress}% ({i+1}/{total} files)")
            
    logger.info(f"[{job_id}] All files processed. Bundling into zip...")
    zip_path = os.path.join(settings.ZIP_DIR, f"{job_id}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for root, dirs, files in os.walk(processed_dir):
            for file in files:
                zf.write(os.path.join(root, file), file)
                
    logger.info(f"[{job_id}] Job Completed successfully. Package: {zip_path}")
    return {"status": "completed", "zip": zip_path}
