import streamlit as st
import requests
import time
import os

# Env
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="PixelForge AI", layout="wide")

st.title("PixelForge AI: Professional Image Optimizer")
st.markdown("### Docker-Powered • Async Processing • Real-ESRGAN")

with st.sidebar:
    st.header("⚙️ Configuration")
    upscale = st.toggle("Enable AI Upscaling", value=True)
    scale = st.selectbox("Upscale Factor", [2, 3, 4], index=2, help="2x is much faster. 3x/4x provides higher resolution.")
    model_type = st.selectbox("AI Model", [
        "RealESRGAN_x4plus (General)", 
        "FSRCNN (Fast CPU Optimized)",
        "Standard (High Speed / No AI)"
    ], index=2, help="FSRCNN is 5-10x faster. Standard is instantaneous.")
    
    st.info("💡 **Performance Tip**: AI upscaling on Intel CPUs is slow. For bulk business workflows, use **FSRCNN** or **Standard** mode.")
    
    st.divider()
    
    st.subheader("Resize (Optional)")

    width = st.number_input("Width (px)", min_value=0, value=0)
    height = st.number_input("Height (px)", min_value=0, value=0)
    
    st.divider()
    
    fmt = st.selectbox("Output Format", ["PNG", "JPEG", "WEBP"])
    quality = st.slider("Compression Quality (JPEG/WEBP only)", 10, 100, 90, help="Higher is better quality but larger file size. Ignored for PNG.")
    ultra_sharpen = st.toggle("🔥 Ultra Sharpening", value=True, help="Strongly sharpens edges and text. High performance but can add noise to photos.")

uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'webp'])

# Session State Initialization
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'job_status' not in st.session_state:
    st.session_state.job_status = None
if 'job_progress' not in st.session_state:
    st.session_state.job_progress = 0
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'logs' not in st.session_state:
    st.session_state.logs = []

def add_log(msg):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {msg}")
    if len(st.session_state.logs) > 50:
        st.session_state.logs.pop(0)

# Action: Start Processing
if st.button("Start Processing 🚀", disabled=(not uploaded_files or (st.session_state.job_status in ["queued", "processing"]))):
    files_payload = [('files', (f.name, f, f.type)) for f in uploaded_files]
    data_payload = {
        'upscale_enabled': str(upscale).lower(),
        'upscale_factor': scale,
        'model_name': model_type.split(" ")[0],
        'resize_width': width if width > 0 else None,
        'resize_height': height if height > 0 else None,
        'output_format': fmt,
        'quality': quality,
        'ultra_sharpen': str(ultra_sharpen).lower()
    }
    data_payload = {k: v for k, v in data_payload.items() if v is not None}
    
    with st.spinner("Uploading and creating job..."):
        try:
            res = requests.post(f"{API_URL}/jobs/create", files=files_payload, data=data_payload)
            if res.status_code == 200:
                job_data = res.json()
                st.session_state.job_id = job_data['job_id']
                st.session_state.job_status = "queued"
                st.session_state.job_progress = 0
                st.session_state.processed_files = []
                add_log(f"Job created: {st.session_state.job_id}")
                st.rerun()
            else:
                st.error(f"Error creating job: {res.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# Job Progress and Results Display
if st.session_state.job_id:
    job_id = st.session_state.job_id
    st.divider()
    st.subheader(f"Job: {job_id}")
    
    # Refresh/Polling Logic (Only if not completed/failed)
    if st.session_state.job_status in ["queued", "processing"]:
        try:
            status_res = requests.get(f"{API_URL}/jobs/{job_id}")
            if status_res.status_code == 200:
                status_data = status_res.json()
                if status_data['status'] != st.session_state.job_status:
                    add_log(f"Status changed: {status_data['status']}")
                
                # Check for new files to log
                new_files = [f['filename'] for f in status_data.get('processed_files', []) if f['filename'] not in [pf['filename'] for pf in st.session_state.processed_files]]
                for nf in new_files:
                    add_log(f"File processed: {nf}")

                st.session_state.job_status = status_data['status']
                st.session_state.job_progress = status_data.get('progress', 0)
                st.session_state.processed_files = status_data.get('processed_files', [])
                
                if st.session_state.job_status not in ["completed", "failed"]:
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("Failed to check status")
                st.session_state.job_status = "error"
        except Exception as e:
            st.error(f"Polling Error: {e}")
            st.session_state.job_status = "error"

    # Show Progress Bar
    st.progress(st.session_state.job_progress / 100.0)
    st.text(f"Status: {st.session_state.job_status.upper()} ({st.session_state.job_progress}%)")
    
    with st.expander("📄 Activity Log", expanded=True):
        log_text = "\n".join(st.session_state.logs[::-1])
        st.code(log_text, language="text")

    # Show Final Results (and intermediate if files exist)
    if st.session_state.processed_files:
        st.subheader("Results")
        files = st.session_state.processed_files
        cols = st.columns(3)
        for i, f in enumerate(files):
            with cols[i % 3]:
                import urllib.parse
                # Support both old and new URLs (strip /api if present)
                path = f['url']
                if path.startswith("/api/"):
                    path = path.replace("/api/", "/", 1)
                
                # Ensure the URL is properly quoted (handles spaces, commas, etc.)
                safe_path = urllib.parse.quote(path)
                file_url = f"{API_URL}{safe_path}"
                try:
                    # Proxy through backend if needed, but here we fetch bytes for download buttons
                    file_res = requests.get(file_url)
                    if file_res.status_code == 200:
                        file_bytes = file_res.content
                        st.image(file_bytes, caption=f['filename'], use_container_width=True)
                        
                        # Direct Download Link (Individual)
                        st.download_button(
                            label=f"⬇️ Download {f['filename']}",
                            data=file_bytes,
                            file_name=f['filename'],
                            mime=f"image/{f['filename'].split('.')[-1]}",
                            key=f"dl_{job_id}_{i}",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Failed to load {f['filename']}")
                except Exception as e:
                    st.error(f"Error displaying {f['filename']}: {e}")

    # Final Zip Download
    if st.session_state.job_status == "completed":
        st.success("All files processed!")
        try:
            download_res = requests.get(f"{API_URL}/jobs/{job_id}/download")
            if download_res.status_code == 200:
                st.download_button(
                    label="Download ALL (ZIP) 📦",
                    data=download_res.content,
                    file_name=f"processed_{job_id}.zip",
                    mime="application/zip",
                    key=f"zip_{job_id}"
                )
        except:
            st.error("Error fetching zip")
        
        if st.button("Start New Job 🆕"):
            st.session_state.job_id = None
            st.session_state.job_status = None
            st.session_state.processed_files = []
            st.session_state.job_progress = 0
            st.rerun()

    elif st.session_state.job_status == "failed":
        st.error("Job Failed. Check worker logs or Activity Log for details.")
        if st.button("Reset 🆕"):
            st.session_state.job_id = None
            st.session_state.job_status = None
            st.session_state.logs = []
            st.rerun()

