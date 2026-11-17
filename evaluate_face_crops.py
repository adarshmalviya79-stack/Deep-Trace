
import random
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from evaluate_videos import load_model, preprocess, DEVICE

DATA_DIR = Path("data/video_test")
OUTPUT = Path("results/face_crop_comparison.json")
MAX_FRAMES = 8
SAMPLES_PER_CLASS = 20
SEED = 42

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load face detector")

model = load_model()
random.seed(SEED)

def get_videos(label):
    folder = DATA_DIR / label
    videos = sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    if len(videos) < SAMPLES_PER_CLASS:
        raise RuntimeError(f"Not enough videos in {folder}")
    return random.sample(videos, SAMPLES_PER_CLASS)

def analyze_video(path):
    cap = cv2.VideoCapture(str(path))

    try:
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None

        count = min(MAX_FRAMES, total)
        indices = (
            [0] if count == 1 else [
                round(i * (total - 1) / (count - 1))
                for i in range(count)
            ]
        )

        full_frames = []
        face_frames = []
        no_face_frames = 0

        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            full_frames.append(
                preprocess(Image.fromarray(rgb))
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )

            if len(faces) == 0:
                no_face_frames += 1
                continue

            x, y, w, h = max(
                faces, key=lambda box: box[2] * box[3]
            )

            px, py = int(w * 0.25), int(h * 0.25)
            x1, y1 = max(0, x-px), max(0, y-py)
            x2 = min(frame.shape[1], x+w+px)
            y2 = min(frame.shape[0], y+h+py)

            crop = frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            face_frames.append(
                preprocess(Image.fromarray(crop))
            )

        if not full_frames or not face_frames:
            return None

        def score(frames):
            batch = torch.stack(frames).to(DEVICE)
            with torch.inference_mode():
                probs = torch.softmax(model(batch), dim=1)
            return probs.mean(dim=0).cpu().tolist()

        return {
            "full": score(full_frames),
            "face": score(face_frames),
            "frames_with_face": len(face_frames),
            "frames_without_face": no_face_frames,
        }

    finally:
        cap.release()

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "fake_precision": round(
            precision_score(y_true, y_pred, pos_label=0,
                            zero_division=0), 4
        ),
        "fake_recall": round(
            recall_score(y_true, y_pred, pos_label=0,
                         zero_division=0), 4
        ),
        "fake_f1": round(
            f1_score(y_true, y_pred, pos_label=0,
                     zero_division=0), 4
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }

records = []

for label, true_label in [("fake", 0), ("real", 1)]:
    videos = get_videos(label)
    print(f"\nTesting {len(videos)} {label} videos")

    for i, path in enumerate(videos, 1):
        result = analyze_video(path)

        if result is None:
            print("Skipped:", path)
            continue

        record = {
            "filename": str(path),
            "true_label": label,
            **result,
        }
        records.append(record)

        print(
            f"{i}/20 | {label} | "
            f"full fake={result['full'][0]:.4f} | "
            f"face fake={result['face'][0]:.4f}"
        )

# Use only videos where both methods were evaluated.
y_true, full_pred, face_pred = [], [], []

for r in records:
    y = 0 if r["true_label"] == "fake" else 1
    y_true.append(y)
    full_pred.append(0 if r["full"][0] >= 0.5 else 1)
    face_pred.append(0 if r["face"][0] >= 0.5 else 1)

if not records:
    raise RuntimeError("No videos could be evaluated")

report = {
    "seed": SEED,
    "videos_evaluated": len(records),
    "videos_skipped": 40 - len(records),
    "full_frame_metrics": calculate_metrics(y_true, full_pred),
    "face_crop_metrics": calculate_metrics(y_true, face_pred),
    "results": records,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n--- COMPARISON ---")
print("Videos evaluated:", len(records))
print("Full-frame:", report["full_frame_metrics"])
print("Face-crop:", report["face_crop_metrics"])
print("Saved to:", OUTPUT)
