
from pathlib import Path
import io
import os
import tempfile

import cv2
import torch
from torch import nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


app = FastAPI(
    title="Deepfake Detector API",
    description="AI-based image and video deepfake detection.",
    version="1.0.0"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_CHECKPOINT = Path("models/deepfake_efficientnet.pth")
VIDEO_CHECKPOINT = Path("models/deepfake_video_efficientnet_newdataset.pth")

image_model = None
video_model = None

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)


@app.get("/")
def home():
    return {
        "message": "Deepfake Detector API is running",
        "endpoints": ["/predict", "/predict-video"],
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "image_model_loaded": image_model is not None,
        "video_model_loaded": video_model is not None
    }


def load_checkpoint(checkpoint_path):
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True
        )

        loaded_model = models.efficientnet_b0(weights=None)
        loaded_model.classifier[1] = nn.Linear(
            loaded_model.classifier[1].in_features,
            2
        )

        loaded_model.load_state_dict(checkpoint["state_dict"])
        loaded_model.eval()
        loaded_model.to(DEVICE)

        return loaded_model

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model: {exc}"
        )


def load_image_model():
    global image_model

    if image_model is None:
        image_model = load_checkpoint(IMAGE_CHECKPOINT)

    return image_model


def load_video_model():
    global video_model

    if video_model is None:
        video_model = load_checkpoint(VIDEO_CHECKPOINT)

    return video_model


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp"
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, WEBP, or BMP image."
        )

    raw = await file.read()

    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller."
        )

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    active_model = load_image_model()

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(
            active_model(tensor),
            dim=1
        )[0].cpu()

    fake_score = float(probabilities[0])
    real_score = float(probabilities[1])

    prediction = "fake" if fake_score >= real_score else "real"

    return {
        "filename": file.filename,
        "prediction": prediction,
        "scores": {
            "fake": round(fake_score, 4),
            "real": round(real_score, 4)
        },
        "notice": "Model estimates are not proof of authenticity."
    }


def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    pad = int(max(w, h) * 0.20)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    return Image.fromarray(face)


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    }

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
            detail="Video file is empty."
        )

    temp_path = None
    cap = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:
            temp_file.write(raw)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0 or total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not read video frames."
            )

        frame_indices = [
            min(total_frames - 1, int(total_frames * ratio))
            for ratio in [0.20, 0.40, 0.60, 0.80]
        ]

        active_model = load_video_model()

        tensors = []
        frame_results = []

        for frame_index in frame_indices:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index
            )

            success, frame = cap.read()

            if not success:
                continue

            face_image = extract_face(frame)

            if face_image is None:
                continue

            tensors.append(preprocess(face_image))
            frame_results.append({
                "time_seconds": round(frame_index / fps, 2)
            })

        if not tensors:
            raise HTTPException(
                status_code=422,
                detail="No faces detected in the sampled video frames. Try a clearer video with visible faces."
            )

        batch = torch.stack(tensors).to(DEVICE)

        with torch.no_grad():
            outputs = active_model(batch)
            probabilities = torch.softmax(
                outputs,
                dim=1
            ).cpu()

        fake_scores = probabilities[:, 0].tolist()
        real_scores = probabilities[:, 1].tolist()

        for result, fake_score, real_score in zip(
            frame_results,
            fake_scores,
            real_scores
        ):
            result["prediction"] = (
                "fake" if fake_score >= real_score else "real"
            )
            result["fake_score"] = round(fake_score, 4)
            result["real_score"] = round(real_score, 4)

        avg_fake = sum(fake_scores) / len(fake_scores)
        avg_real = sum(real_scores) / len(real_scores)

        prediction = (
            "fake" if avg_fake >= avg_real else "real"
        )

        return {
            "filename": file.filename,
            "prediction": prediction,
            "frames_analyzed": len(tensors),
            "video_duration_seconds": round(
                total_frames / fps, 2
            ),
            "average_scores": {
                "fake": round(avg_fake, 4),
                "real": round(avg_real, 4)
            },
            "frame_results": frame_results,
            "notice": (
                "Prediction is an estimate based on sampled face crops. "
                "It is not proof of authenticity or manipulation."
            )
        }

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
