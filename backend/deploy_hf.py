"""
DeepTrace - Hugging Face Spaces Deployment Helper

This script packages the backend application (including model checkpoints via Git LFS)
and pushes it to your Hugging Face Space.

Prerequisites:
  1. A Hugging Face account: https://huggingface.co
  2. A new Space created with SDK: Docker: https://huggingface.co/new-space
  3. Git and Git LFS installed on your system.
  4. A Hugging Face Access Token (Write permissions): https://huggingface.co/settings/tokens

Usage:
  python deploy_hf.py --space-repo username/space-name [--token YOUR_HF_TOKEN]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cmd(cmd, cwd=None, env=None):
    print(f"--> Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error (code {res.returncode}):")
        if res.stdout:
            print("STDOUT:", res.stdout)
        if res.stderr:
            print("STDERR:", res.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    if res.stdout.strip():
        print(res.stdout.strip())
    return res


def main():
    parser = argparse.ArgumentParser(description="Deploy DeepTrace backend to Hugging Face Spaces")
    parser.add_argument(
        "--space-repo",
        required=True,
        help="Hugging Face Space in the format 'username/space-name' (e.g. adarsh/deeptrace-api)"
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face Write Token (or set HF_TOKEN env var)"
    )
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parent
    models_dir = backend_dir / "models"
    
    img_model = models_dir / "deepfake_efficientnet.pth"
    vid_model = models_dir / "deepfake_video_efficientnet_newdataset.pth"

    if not img_model.exists():
        sys.exit(f"Error: Model checkpoint not found at {img_model}")
    if not vid_model.exists():
        sys.exit(f"Error: Model checkpoint not found at {vid_model}")

    print(f"Deploying DeepTrace Backend to Hugging Face Space: {args.space_repo}")

    # Build auth URL if token is provided
    if args.token:
        remote_url = f"https://user:{args.token}@huggingface.co/spaces/{args.space_repo}"
    else:
        remote_url = f"https://huggingface.co/spaces/{args.space_repo}"

    with tempfile.TemporaryDirectory() as temp_dir:
        dest = Path(temp_dir)
        print(f"Staging files in temporary directory: {dest}")

        # Files to copy
        files_to_copy = [
            "app.py",
            "app_backup.py",
            "requirements.txt",
            "Dockerfile",
            ".dockerignore",
            "README.md",
        ]

        for fname in files_to_copy:
            src_f = backend_dir / fname
            if src_f.exists():
                shutil.copy2(src_f, dest / fname)

        # Copy models directory
        dest_models = dest / "models"
        dest_models.mkdir(exist_ok=True)
        shutil.copy2(img_model, dest_models / "deepfake_efficientnet.pth")
        shutil.copy2(vid_model, dest_models / "deepfake_video_efficientnet_newdataset.pth")
        if (models_dir / "README.md").exists():
            shutil.copy2(models_dir / "README.md", dest_models / "README.md")

        # Copy evaluation, training, dataset_tools, analysis if present
        for folder in ["evaluation", "training", "dataset_tools", "analysis"]:
            src_folder = backend_dir / folder
            if src_folder.exists():
                shutil.copytree(src_folder, dest / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        # Initialize Git in dest
        print("Configuring Git & Git LFS in staging directory...")
        run_cmd(["git", "init"], cwd=dest)
        run_cmd(["git", "lfs", "install"], cwd=dest)
        run_cmd(["git", "lfs", "track", "*.pth"], cwd=dest)
        
        # Git config local user if needed
        run_cmd(["git", "config", "user.name", "DeepTrace Deployer"], cwd=dest)
        run_cmd(["git", "config", "user.email", "deployer@deeptrace.local"], cwd=dest)

        run_cmd(["git", "add", ".gitattributes"], cwd=dest)
        run_cmd(["git", "add", "-A"], cwd=dest)
        run_cmd(["git", "commit", "-m", "Deploy DeepTrace backend to Hugging Face Spaces"], cwd=dest)

        print(f"Pushing to Hugging Face Space: https://huggingface.co/spaces/{args.space_repo}...")
        run_cmd(["git", "push", "--force", remote_url, "main"], cwd=dest)

    print("\n" + "="*60)
    print("SUCCESS: DeepTrace backend deployed to Hugging Face Spaces!")
    print(f"Space URL: https://huggingface.co/spaces/{args.space_repo}")
    user_part, space_part = args.space_repo.split("/", 1)
    direct_api_url = f"https://{user_part}-{space_part.replace('_', '-')}.hf.space"
    print(f"Direct API URL: {direct_api_url}")
    print("Use this Direct API URL as the VITE_API_URL environment variable on Vercel!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
