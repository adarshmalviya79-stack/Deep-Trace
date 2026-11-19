---
title: DeepTrace Backend API
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# DeepTrace — Python FastAPI Backend

Asynchronous REST backend for DeepTrace deepfake image and video classification, utilizing PyTorch, EfficientNet-B0 architectures, and OpenCV Haar Cascade face detection.

---

## 📁 Backend Directory Layout

```
backend/
├── app.py                     # Main FastAPI application and inference endpoints
├── app_backup.py              # Lightweight single-model backup API
├── requirements.txt           # Python dependencies (includes opencv-python-headless for Render)
├── README.md                  # Backend documentation (this file)
│
├── models/
│   ├── README.md              # Checkpoint specification & download guidance
│   ├── deepfake_efficientnet.pth                  (Excluded from Git)
│   └── deepfake_video_efficientnet_newdataset.pth  (Excluded from Git)
│
├── training/                  # Model training and fine-tuning scripts
│   ├── train.py               # Image EfficientNet-B0 training pipeline
│   ├── train_resnet.py        # Baseline ResNet18 training pipeline
│   ├── train_video_newdataset.py # Video frame EfficientNet-B0 training
│   └── finetune_video.py      # Video classifier fine-tuning
│
├── evaluation/                # Model evaluation and benchmark scripts
│   ├── evaluate.py            # Primary validation and metrics report
│   ├── evaluate_external.py   # External out-of-distribution benchmark
│   ├── evaluate_videos.py     # Video frame inference evaluator
│   ├── evaluate_face_crops.py # Face crop margin comparison
│   ├── evaluate_resnet.py     # ResNet baseline benchmark
│   └── threshold_analysis.py  # Precision-recall threshold sweep
│
├── dataset_tools/             # Data prep, facial extraction, and split tools
│   ├── extract_new_video_faces.py # Automated Haar Cascade face cropping
│   ├── prepare_splits.py      # Leakage-free subject/video train-test splits
│   ├── prepare_manifest.py    # CSV manifest builder
│   ├── create_video_split.py  # Video dataset partitioning
│   ├── prepare_video_faces.py # Frame extraction pipeline
│   ├── prepare_video_manifest.py # Video metadata manifest
│   ├── verify_video_split.py  # Leakage verification tool
│   ├── check_leakage.py       # Subject overlap detector
│   ├── check_video_dataset.py # Video dataset integrity validator
│   └── audit_face_dataset.py  # Face crop dimensions & corruption auditor
│
└── analysis/                  # Diagnostics and dataset inspection
    ├── analyze_source_groups.py # Source manipulation distribution
    ├── analyze_video_ids.py   # Subject ID collision checker
    ├── diagnose_video.py      # Video decoding and fps inspector
    ├── diagnose_face_crop.py  # Facial bounding box inspector
    ├── inspect_filenames.py   # Filename convention parser
    ├── inspect_new_dataset.py # Dataset partition stats
    └── inspect_video_metadata.py # Video resolution and bitrate profiler
```

---

## 🛠️ Environment Setup & Installation

### 1. Prerequisites
- Python 3.10 or 3.11
- Virtual environment tool (`venv`)
- NVIDIA GPU with CUDA drivers (optional, recommended for fast inference)

### 2. Create and Activate Virtual Environment

From within the `backend/` directory:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: `opencv-python-headless` is included in `requirements.txt` to enable cloud server deployments (e.g. Render) without requiring GUI/X11 system libraries.

---

## 💾 Model Checkpoint Placement

Due to file size constraints, `.pth` model weights are excluded from Git version control. Ensure the following files are placed in `backend/models/`:

1. `backend/models/deepfake_efficientnet.pth` (Image classification model)
2. `backend/models/deepfake_video_efficientnet_newdataset.pth` (Video classification model)

Model checkpoint paths are dynamically resolved relative to `app.py`:
```python
BASE_DIR = Path(__file__).resolve().parent
IMAGE_CHECKPOINT = BASE_DIR / "models" / "deepfake_efficientnet.pth"
VIDEO_CHECKPOINT = BASE_DIR / "models" / "deepfake_video_efficientnet_newdataset.pth"
```
You can also override these paths using environment variables:
- `IMAGE_CHECKPOINT_PATH`
- `VIDEO_CHECKPOINT_PATH`

---

## 🚀 Running the API Locally

From the `backend/` directory:
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Alternatively, from the repository root:
```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Interactive API documentation:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## ☁️ Hugging Face Spaces Deployment (Docker)

Hugging Face Spaces is the recommended cloud platform for DeepTrace backend deployment because it provides free compute (2 vCPU, 16GB RAM) and native Git LFS support for PyTorch model checkpoints.

### Automatic Deployment via CLI Helper:
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space):
   - **SDK**: Select **Docker** (Blank)
   - **License**: MIT
   - **Hardware**: CPU Basic (Free) or GPU
2. Generate an Access Token with **Write** permissions: [Hugging Face Tokens](https://huggingface.co/settings/tokens)
3. Deploy in one command:
```bash
python deploy_hf.py --space-repo YOUR_USERNAME/deeptrace-backend --token YOUR_HF_TOKEN
```
The script will package the application, track `.pth` files with Git LFS, and push to your Space.

Once built, access your API at:
```
https://YOUR_USERNAME-deeptrace-backend.hf.space
```

---

## ☁️ Render Deployment Configuration (Alternative)

When creating a Web Service on [Render](https://render.com):

- **Environment**: Python
- **Root Directory**: `backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

## 📡 API Reference

### 1. Health Check
- **Route**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "device": "cuda",
  "image_model_loaded": true,
  "video_model_loaded": true
}
```

### 2. Image Prediction
- **Route**: `POST /predict`
- **Body**: `multipart/form-data` with field `file`
- **Accepted Formats**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` (≤ 10 MB)
- **Response**:
```json
{
  "filename": "face.jpg",
  "prediction": "fake",
  "scores": {
    "fake": 0.9234,
    "real": 0.0766
  },
  "notice": "Model estimates are not proof of authenticity."
}
```

### 3. Video Prediction
- **Route**: `POST /predict-video`
- **Body**: `multipart/form-data` with field `file`
- **Accepted Formats**: `.mp4`, `.avi`, `.mov`, `.mkv` (≤ 100 MB)
- **Response**:
```json
{
  "filename": "sample.mp4",
  "prediction": "fake",
  "frames_analyzed": 4,
  "video_duration_seconds": 3.2,
  "average_scores": {
    "fake": 0.9123,
    "real": 0.0877
  },
  "frame_results": [
    {
      "time_seconds": 0.64,
      "prediction": "fake",
      "fake_score": 0.91,
      "real_score": 0.09
    },
    {
      "time_seconds": 1.28,
      "prediction": "fake",
      "fake_score": 0.93,
      "real_score": 0.07
    }
  ],
  "notice": "Prediction is an estimate based on sampled face crops."
}
```

---

## 🎓 Training & Evaluation Scripts

All training and evaluation scripts can be executed directly from the `backend/` directory:

```bash
# Train image model
python training/train.py --manifest data/split_manifest.csv --epochs 5 --batch-size 16

# Train video face model
python training/train_video_newdataset.py

# Run model evaluation on test images
python evaluation/evaluate.py

# Run external dataset validation
python evaluation/evaluate_external.py

# Run video dataset evaluation
python evaluation/evaluate_videos.py
```
