# DeepTrace: AI Deepfake Image & Video Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%2B%20TypeScript-61DAFB.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Bundler-Vite%205-646CFF.svg)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Haar%20Cascade-5C3EE8.svg)](https://opencv.org/)

**DeepTrace** is an end-to-end, research-grade deepfake detection system engineered to identify synthetic visual artifacts and face-manipulations across both high-resolution still images and video sequences. The platform couples custom convolutional neural network checkpoints with computer vision face localization pipelines and serves results through an interactive, minimal React dashboard.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Reported Model Performance & Evaluation](#reported-model-performance--evaluation)
3. [Key Features](#key-features)
4. [System Architecture](#system-architecture)
5. [Detection Workflows](#detection-workflows)
   - [Image Detection Pipeline](#image-detection-pipeline)
   - [Video Detection Pipeline](#video-detection-pipeline)
6. [Dataset Preparation & Training Overview](#dataset-preparation--training-overview)
7. [Model Checkpoint Requirements](#model-checkpoint-requirements)
8. [Backend Setup (FastAPI)](#backend-setup-fastapi)
9. [Frontend Setup (React + TypeScript)](#frontend-setup-react--typescript)
10. [Environment Configuration](#environment-configuration)
11. [Running the Full Stack Locally](#running-the-full-stack-locally)
12. [API Reference](#api-reference)
13. [Limitations](#limitations)
14. [Future Scope](#future-scope)

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
[ Vite Dev Server / Reverse Proxy ] (Port 5173)
           │
           ▼
[ FastAPI Asynchronous Backend ] (Port 8000)
    ├── GET  /health           ── Returns hardware device & checkpoint status
    ├── POST /predict          ── Image Inference Pipeline
    └── POST /predict-video    ── Video Frame Extraction & Inference Pipeline
           │
           ├─── PyTorch (CUDA / CPU)
           ├─── EfficientNet-B0 Image Checkpoint
           ├─── EfficientNet-B0 Video Checkpoint
           └─── OpenCV Haar Cascade Frontal Face Classifier
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
                          └──> JSON Result: Verdict (Real/Fake) + Scores
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

## 📁 Dataset Preparation & Training Overview

### 1. Image Training Layout
Prepare datasets under `data/`:
```text
data/
  train/
    real/    # Authentic facial photographs
    fake/    # Synthesized / Deepfake images
  val/
    real/
    fake/
  test/
    real/
    fake/
```

### 2. Video Dataset Layout
Videos are organized into `data/video_test/` and processed using the provided dataset utility scripts:
- `prepare_splits.py`: Establishes video ID splits to prevent subject leakage between train/val sets.
- `extract_new_video_faces.py`: Runs automated face extraction across video archives.
- `audit_face_dataset.py`: Audits image balance, corruption, and aspect ratios.

---

## 💾 Model Checkpoint Requirements

Due to GitHub binary storage limitations, model weights are **excluded** from Git tracking. You must train or place pre-trained weights in the `models/` folder:

| Filepath | Description |
| :--- | :--- |
| `models/deepfake_efficientnet.pth` | Checkpoint for Image inference |
| `models/deepfake_video_efficientnet_newdataset.pth` | Checkpoint for Video inference |

To train the models from scratch:
```powershell
# Train image model
python train.py --data-dir data --epochs 10 --batch-size 32

# Train video face model
python train_video_newdataset.py --epochs 10 --batch-size 32
```

---

## ⚙️ Backend Setup (FastAPI)

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA drivers (optional, recommended for fast video inference)

### Installation
```powershell
# 1. Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Running Backend Server
```powershell
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

### Installation
```powershell
cd frontend
npm install
```

### Running Frontend Development Server
```powershell
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

### Backend (`.env.example`)
```env
PORT=8000
HOST=127.0.0.1
DEVICE=auto
```

---

## 🚀 Running the Full Stack Locally

1. **Terminal 1 (Backend)**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   uvicorn app:app --reload --port 8000
   ```

2. **Terminal 2 (Frontend)**:
   ```powershell
   npm run dev
   # Or: cd frontend && npm run dev
   ```

3. Open **`http://localhost:5173`** in your browser.

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
