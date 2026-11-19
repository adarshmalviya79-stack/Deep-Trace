# DeepTrace: AI Deepfake Image & Video Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%2B%20TypeScript-61DAFB.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Bundler-Vite%205-646CFF.svg)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Haar%20Cascade-5C3EE8.svg)](https://opencv.org/)

**DeepTrace** is an end-to-end, research-grade deepfake detection platform engineered to identify synthetic visual artifacts and face manipulations across both high-resolution still images and video sequences. The platform couples custom convolutional neural network checkpoints with computer vision face localization pipelines and serves results through an interactive, minimal React dashboard.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Reported Model Performance & Evaluation](#reported-model-performance--evaluation)
3. [Key Features](#key-features)
4. [System Architecture](#system-architecture)
5. [Repository Structure](#repository-structure)
6. [Detection Workflows](#detection-workflows)
   - [Image Detection Pipeline](#image-detection-pipeline)
   - [Video Detection Pipeline](#video-detection-pipeline)
7. [Model Checkpoint Requirements](#model-checkpoint-requirements)
8. [Backend Setup (FastAPI)](#backend-setup-fastapi)
9. [Frontend Setup (React + TypeScript)](#frontend-setup-react--typescript)
10. [Environment Configuration](#environment-configuration)
11. [Running the Full Stack Locally](#running-the-full-stack-locally)
12. [Deployment Guide (Vercel & Render)](#deployment-guide-vercel--render)
13. [API Reference](#api-reference)
14. [Limitations](#limitations)
15. [Future Scope](#future-scope)

---

## 🔬 Project Overview

With the rapid acceleration of generative adversarial networks (GANs), diffusion-based inpainting, and facial reenactment models, distinguishing between authentic and synthesized digital media requires multi-stage verification. 

DeepTrace addresses this challenge through two dedicated **EfficientNet-B0** models:
- An **Image Deepfake Classifier** specialized in detecting subtle frequency inconsistencies, compression anomalies, and generative blending boundaries.
- A **Video Deepfake Classifier** operating on OpenCV Haar Cascade-isolated facial regions sampled evenly across temporal frames.

---

## 📊 Reported Model Performance & Evaluation

> **Note on Evaluation Methodology**: The image and video detectors utilize separate architectures, distinct preprocessing routines, and independently trained weights. Their evaluation accuracies reflect separate test distributions.

| Evaluation Metric | Image Detection Model | Video Detection Model |
| :--- | :--- | :--- |
| **Model Architecture** | EfficientNet-B0 (Spatial CNN) | EfficientNet-B0 (Face Crop Spatial) |
| **Input Format** | Full Image (224×224 RGB) | 4 Temporal Face Crops (+20% Pad) |
| **Reported Evaluation Accuracy** | **97.58%** *(External Evaluation)* | **79.53%** *(Video Test Split)* |
| **Primary Evaluation Focus** | Boundary & synthesis artifacts | Cross-frame facial manipulation |
| **Aggregator** | Direct Softmax Probability | Arithmetic Mean across sampled frames |

*The two models have distinct task difficulties: video detection requires robust temporal sampling and real-time facial cropping under varying lighting, head poses, and motion blur, leading to the reported 79.53% benchmark, while image evaluation achieves 97.58% on high-resolution external benchmark sets.*

---

## 🌟 Key Features

- **Dual Detection Engine**: Dedicated inference modes for single images and video clips.
- **Automated Facial ROI Extraction**: Automatic Haar Cascade frontal face localization with a 20% margin padding to preserve critical cheek and forehead blending seams.
- **Temporal Frame Sampling**: Evenly distributed 4-point frame extraction (20%, 40%, 60%, 80%) for rapid, representative video diagnostics without processing redundant static frames.
- **Transparent Softmax Scoring**: Unadulterated real vs. fake confidence distribution with scannable visual comparison bars.
- **Hardware Agnostic**: Seamless execution on NVIDIA CUDA GPUs with automatic CPU fallback.
- **Modern Minimal UI/UX**: Warm editorial palette, restrained shadows, drag-and-drop validation, and mobile responsiveness.

---

## 🏗️ System Architecture

```
[ User Browser / Client ]
           │
           │  (Multipart Form-Data: 'file')
           ▼
[ Vite Dev Server / Reverse Proxy ] (Port 5173)  OR  [ Vercel CDN ]
           │
           ▼
[ FastAPI Asynchronous Backend ] (Port 8000 / Render Cloud)
    ├── GET  /health           ── Returns hardware device & checkpoint status
    ├── POST /predict          ── Image Inference Pipeline
    └── POST /predict-video    ── Video Frame Extraction & Inference Pipeline
           │
           ├─── PyTorch (CUDA / CPU)
           ├─── EfficientNet-B0 Image Checkpoint (`backend/models/deepfake_efficientnet.pth`)
           ├─── EfficientNet-B0 Video Checkpoint (`backend/models/deepfake_video_efficientnet_newdataset.pth`)
           └─── OpenCV Haar Cascade Frontal Face Classifier
```

---

## 📂 Repository Structure

```
Deep-Trace/
│
├── backend/                                   # Python FastAPI Backend
│   ├── app.py                                 # Main FastAPI application and API routes
│   ├── app_backup.py                          # Single-model fallback API
│   ├── requirements.txt                       # Backend dependencies (with opencv-python-headless)
│   ├── README.md                              # Backend setup and script reference
│   ├── .env.example                           # Backend environment template
│   │
│   ├── models/                                # Model checkpoints (weights excluded from Git)
│   │   └── README.md                          # Checkpoint placement instructions
│   │
│   ├── training/                              # Training scripts
│   │   ├── train.py                           # Image EfficientNet-B0 trainer
│   │   ├── train_resnet.py                    # ResNet18 baseline trainer
│   │   ├── train_video_newdataset.py          # Video face trainer
│   │   └── finetune_video.py                  # Fine-tuning module
│   │
│   ├── evaluation/                            # Evaluation scripts
│   │   ├── evaluate.py                        # Primary image evaluator
│   │   ├── evaluate_external.py               # External benchmark evaluator
│   │   ├── evaluate_videos.py                 # Video inference evaluator
│   │   ├── evaluate_face_crops.py             # Face crop margin benchmark
│   │   ├── evaluate_resnet.py                 # ResNet baseline benchmark
│   │   └── threshold_analysis.py              # Threshold curve evaluation
│   │
│   ├── dataset_tools/                         # Data preparation & split tools
│   │   ├── extract_new_video_faces.py         # Automated Haar Cascade face cropping
│   │   ├── prepare_splits.py                  # Leakage-free subject partitioning
│   │   ├── prepare_manifest.py                # Image dataset manifest builder
│   │   ├── create_video_split.py              # Video dataset split builder
│   │   ├── prepare_video_faces.py             # Video face extraction pipeline
│   │   ├── prepare_video_manifest.py          # Video metadata manifest
│   │   ├── verify_video_split.py              # Subject leakage verifier
│   │   ├── check_leakage.py                   # Split collision checker
│   │   ├── check_video_dataset.py             # Video archive validator
│   │   └── audit_face_dataset.py              # Face dimensions and corruptions auditor
│   │
│   └── analysis/                              # Inspection and diagnostic scripts
│       ├── analyze_source_groups.py           # Source distribution inspector
│       ├── analyze_video_ids.py               # Subject ID collision inspector
│       ├── diagnose_video.py                  # Video stream decoder
│       ├── diagnose_face_crop.py              # Bounding box coordinates inspector
│       ├── inspect_filenames.py               # Filename parser
│       ├── inspect_new_dataset.py             # Partition statistics inspector
│       └── inspect_video_metadata.py          # Resolution and bitrate inspector
│
├── frontend/                                  # React + TypeScript + Vite Dashboard
│   ├── src/
│   │   ├── components/                        # Modular UI components
│   │   ├── pages/                             # Home, Detector, About views
│   │   ├── services/                          # Axios API service
│   │   ├── types/                             # TypeScript response schemas
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── public/                                # Public assets (favicon)
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.ts                         # Reverse-proxy configuration
│   └── tailwind.config.js
│
├── .gitignore                                 # Git exclusions (.venv, node_modules, .pth, data)
├── .env.example                               # Root environment template
├── package.json                               # Root workspace convenience scripts
└── README.md                                  # Root documentation (this file)
```

---

## 🔄 Detection Workflows

### Image Detection Pipeline
```
Image Upload (JPG, PNG, WEBP, BMP ≤ 10MB)
  └──> PIL Verification & RGB Conversion
        └──> Tensor Preprocessing (Resize to 224x224, ImageNet Normalization)
              └──> EfficientNet-B0 Feedforward
                    └──> Softmax Probability Computation
                          └──> JSON Result: Verdict (Real/Fake) + Raw Scores
```

### Video Detection Pipeline
```
Video Upload (MP4, AVI, MOV, MKV ≤ 100MB)
  └──> Tempfile Buffer & OpenCV VideoCapture Inspection
        └──> Proportional Frame Indices Calculation (20%, 40%, 60%, 80%)
              └──> Haar Cascade Face Detection (minSize 60x60)
                    └──> ROI Crop with +20% Spatial Padding
                          └──> Batch Tensor Formulation (N×3×224×224)
                                └──> Video EfficientNet-B0 Inference
                                      └──> Per-Frame Scores + Mean Aggregation
```

---

## 💾 Model Checkpoint Requirements

Due to GitHub binary storage limitations, model weights (`.pth`) are **excluded** from Git tracking. You must train or place pre-trained weights in the `backend/models/` folder:

| Filepath | Description |
| :--- | :--- |
| `backend/models/deepfake_efficientnet.pth` | Checkpoint for Image inference |
| `backend/models/deepfake_video_efficientnet_newdataset.pth` | Checkpoint for Video inference |

To train the models from scratch:
```powershell
cd backend
# Train image model
python training/train.py --manifest data/split_manifest.csv --epochs 10 --batch-size 32

# Train video face model
python training/train_video_newdataset.py --epochs 10 --batch-size 32
```

---

## ⚙️ Backend Setup (FastAPI)

### Prerequisites
- Python 3.10 or 3.11
- Virtual environment (`venv`)

### Installation & Execution
```powershell
# 1. Navigate to backend directory
cd backend

# 2. Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the FastAPI server
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
FastAPI documentation will be accessible at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 💻 Frontend Setup (React + TypeScript)

### Prerequisites
- Node.js v18+ (tested on Node v20/v24)
- npm or pnpm

### Installation & Execution
```powershell
cd frontend
npm install
npm run dev
```
Access the client dashboard at `http://localhost:5173`.

---

## 🌐 Environment Configuration

Safe configuration templates are provided. Do **not** commit local `.env` files.

### Frontend (`frontend/.env.example`)
```env
# URL to FastAPI Backend Engine
VITE_API_URL=http://127.0.0.1:8000
```

### Backend (`backend/.env.example`)
```env
PORT=8000
HOST=127.0.0.1
DEVICE=auto

# Optional override for model checkpoint locations (defaults to relative backend/models/)
# IMAGE_CHECKPOINT_PATH=models/deepfake_efficientnet.pth
# VIDEO_CHECKPOINT_PATH=models/deepfake_video_efficientnet_newdataset.pth
```

---

## 🚀 Running the Full Stack Locally

1. **Terminal 1 (Backend)**:
   ```powershell
   cd backend
   .\.venv\Scripts\Activate.ps1
   uvicorn app:app --reload --port 8000
   ```

2. **Terminal 2 (Frontend)**:
   ```powershell
   cd frontend
   npm run dev
   ```

3. Open **`http://localhost:5173`** in your browser.

*(Alternatively, from repository root, run `npm run dev` to launch the frontend).*

---

## ☁️ Deployment Guide (Hugging Face Spaces & Vercel)

### 1. Deploying the Backend to Hugging Face Spaces (Docker)

Hugging Face Spaces provides free hosting with Git LFS support for PyTorch model checkpoints (`.pth`).

#### Step A: Create a New Space
1. Log in to [Hugging Face](https://huggingface.co).
2. Go to **[New Space](https://huggingface.co/new-space)**.
3. Settings:
   - **Space name**: `deeptrace-backend` (or your preferred name)
   - **License**: `mit` or `apache-2.0`
   - **Space SDK**: Select **Docker** (Blank)
   - **Space hardware**: `CPU Basic · 2 vCPU · 16 GB · Free` (or GPU tier for faster inference)
   - **Visibility**: `Public`
4. Click **Create Space**.

#### Step B: Deploy with the Automated Script
From your local terminal, run the included deployment helper:
```powershell
python backend/deploy_hf.py --space-repo YOUR_USERNAME/deeptrace-backend --token YOUR_HF_TOKEN
```
*(Get a Write token from [Hugging Face Access Tokens](https://huggingface.co/settings/tokens)).*

This script automatically:
- Copies backend code, `Dockerfile`, and `requirements.txt`.
- Bundles `deepfake_efficientnet.pth` and `deepfake_video_efficientnet_newdataset.pth`.
- Configures Git LFS for binary `.pth` weights.
- Pushes directly to your Hugging Face Space repository.

#### Step C: Get Your Backend Direct API URL
Once Hugging Face finishes building the Docker container, your API is available at:
```
https://YOUR_USERNAME-deeptrace-backend.hf.space
```
Test it in your browser:
`https://YOUR_USERNAME-deeptrace-backend.hf.space/health`

---

### 2. Deploying the Frontend to Vercel

1. Create or log in to your account on [Vercel](https://vercel.com).
2. Click **Add New...** > **Project** and import this repository (`Deep-Trace`).
3. Configure Project Settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend` *(or leave `./` as root `vercel.json` is provided)*
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Add Environment Variable:
   - **Key**: `VITE_API_URL`
   - **Value**: Your Hugging Face Space Direct API URL (e.g., `https://YOUR_USERNAME-deeptrace-backend.hf.space`)
5. Click **Deploy**. Vercel will build the frontend and serve it with global CDN and automated SSL.

---

### Alternative: Deploying Backend to Render

If you prefer Render instead of Hugging Face Spaces:
1. Create a new **Web Service** on [Render](https://render.com) connected to this repository.
2. Settings:
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. Provide `.pth` weights via Render disk or external storage.

---

## 📡 API Reference

### 1. Health Check
`GET /health`
```json
{
  "status": "healthy",
  "device": "cuda",
  "image_model_loaded": true,
  "video_model_loaded": true
}
```

### 2. Predict Image
`POST /predict`
- **Request**: `multipart/form-data` with key `file` (JPG, PNG, WEBP, BMP ≤ 10MB)
- **Response**:
```json
{
  "filename": "sample.jpg",
  "prediction": "fake",
  "scores": {
    "fake": 0.9234,
    "real": 0.0766
  },
  "notice": "Model estimates are not proof of authenticity."
}
```

### 3. Predict Video
`POST /predict-video`
- **Request**: `multipart/form-data` with key `file` (MP4, AVI, MOV, MKV ≤ 100MB)
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

## ⚠️ Limitations

- **Facial Dependency in Video**: Video detection relies on Haar Cascade face localization. Videos with extreme profile angles, heavy occlusion, or absence of faces will return HTTP 422.
- **Adversarial Robustness**: Models are trained on specific synthesis generators; performance may vary when encountering novel, unseen generative diffusion pipelines or extreme downsampling compression.
- **Probabilistic Nature**: Model predictions are diagnostic indicators and do not constitute definitive legal proof.

---

## 🔮 Future Scope

- **Temporal Attention Modeling**: Integration of Vision Transformers (ViT) or LSTM temporal sequences to capture inter-frame flicker artifacts.
- **Grad-CAM Saliency Maps**: Visual heatmap overlays pinpointing manipulated pixel clusters.
- **Audio-Visual Multimodal Sync**: Evaluation of audio-visual lip synchrony (Wav2Lip detection).
- **Expanded Face Detectors**: Integration of MTCNN or RetinaFace for improved profile detection.

---

## 📄 License

This project is developed for educational, forensic, and research purposes.
