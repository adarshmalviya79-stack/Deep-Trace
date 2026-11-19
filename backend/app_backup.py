from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from torch import nn
import io
import cv2
import tempfile
import os

app = FastAPI(
    title="Deepfake Detector API",
    description="Research prototype for estimating whether an image resembles the training-set real/fake classes.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = Path(os.getenv("IMAGE_CHECKPOINT_PATH", str(BASE_DIR / "models" / "deepfake_efficientnet.pth")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.get("/")
def home():
    return {"message": "Deepfake Detector API is running", "docs": "/docs"}


def load_model():
    global model
    if model is not None:
        return model
    if not CHECKPOINT.exists():
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not found. Prepare the dataset and run train.py first.",
        )
    try:
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
        loaded_model = models.efficientnet_b0(weights=None)
        loaded_model.classifier[1] = nn.Linear(loaded_model.classifier[1].in_features, 2)
        loaded_model.load_state_dict(checkpoint["state_dict"])
        loaded_model.eval().to(DEVICE)
        model = loaded_model
        return model
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model checkpoint: {exc}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, WEBP, or BMP image.")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    active_model = load_model()
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(active_model(tensor), dim=1)[0].cpu()
    # ImageFolder class ordering: fake=0, real=1.
    fake_score = float(probabilities[0])
    real_score = float(probabilities[1])
    predicted = "fake" if fake_score >= real_score else "real"
    return {
        "filename": file.filename,
        "prediction": predicted,
        "scores": {
            "fake": round(fake_score, 4),
            "real": round(real_score, 4),
        },
        "notice": "Scores are model outputs, not proof of authenticity or calibrated probabilities.",
    }


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail="Upload an MP4, AVI, MOV, or MKV video."
        )

    raw = await file.read()

    if len(raw) > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Video must be 100 MB or smaller."
        )

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="The uploaded video is empty."
        )

    temp_path = None
    cap = None

    try:
        # Save the upload temporarily for OpenCV.
        with tempfile.NamedTemporaryFile(
            suffix=extension, delete=False
        ) as temp_file:
            temp_file.write(raw)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open the video. Try another video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not read video frames."
            )

        # Sample at most 30 frames evenly across the video.
        sample_count = min(30, total_frames)

        if sample_count == 1:
            frame_indices = [0]
        else:
            frame_indices = [
                round(i * (total_frames - 1) / (sample_count - 1))
                for i in range(sample_count)
            ]

        active_model = load_model()
        tensors = []
        frame_times = []

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()

            if not success:
                continue

            # OpenCV uses BGR; PIL expects RGB.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            tensors.append(preprocess(image))
            frame_times.append(round(frame_index / fps, 2))

        if not tensors:
            raise HTTPException(
                status_code=400,
                detail="No readable frames found in the video."
            )

        batch = torch.stack(tensors).to(DEVICE)

      
        with torch.no_grad():
            outputs = active_model(batch)

            print("Raw logits:", outputs[:5].detach().cpu().tolist())
            print("Logit range:", outputs.min().item(), outputs.max().item())

            probabilities = torch.softmax(outputs, dim=1).cpu()
            print("Probabilities:", probabilities[:5])


        # Class order: fake=0, real=1.
        fake_scores = probabilities[:, 0].tolist()
        real_scores = probabilities[:, 1].tolist()

        avg_fake = sum(fake_scores) / len(fake_scores)
        avg_real = sum(real_scores) / len(real_scores)

        prediction = "fake" if avg_fake >= avg_real else "real"

        frame_results = [
            {
                "time_seconds": timestamp,
                "prediction": (
                    "fake" if fake_score >= real_score else "real"
                ),
                "fake_score": round(fake_score, 8),
                "real_score": round(real_score, 8),
            }
            for timestamp, fake_score, real_score
            in zip(frame_times, fake_scores, real_scores)
        ]

        return {
            "filename": file.filename,
            "prediction": prediction,
            "frames_analyzed": len(tensors),
            "video_duration_seconds": round(total_frames / fps, 2),
            "average_scores": {
                "fake": round(avg_fake, 8),
                "real": round(avg_real, 8),
            },
            "frame_results": frame_results,
            "notice": (
                "This is an estimate based on sampled frames. "
                "It does not prove whether a video is authentic "
                "or manipulated."
            ),
        }

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

