"""Upload project files to Hugging Face Spaces via API (bypasses git)."""
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "aryan6970/tri-modal-health-screening"
repo_type = "space"

# Project root
root = os.path.dirname(os.path.abspath(__file__))

# Files to upload (skip Model/ since user uploaded manually)
files_to_upload = [
    "app.py",
    "inference.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    ".dockerignore",
    "try-model-decription-image.jpg",
    "templates/index.html",
    "static/real_chest_xray.png",
    "static/screenshot.png",
]

for f in files_to_upload:
    local_path = os.path.join(root, f)
    if os.path.exists(local_path):
        print(f"Uploading {f}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"  ✓ {f} uploaded")
    else:
        print(f"  ✗ {f} not found, skipping")

print("\nAll files uploaded! Check your Space at:")
print(f"https://huggingface.co/spaces/{repo_id}")
