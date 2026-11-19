
import argparse
import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Paths and device
DATA_DIR = Path("data/video_test")
CHECKPOINT = Path("models/deepfake_efficientnet.pth")
RESULTS_DIR = Path("results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same preprocessing used by the image model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    weights = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(weights)
    model.to(DEVICE)
    model.eval()

    return model


def analyze_video(model, video_path, max_frames=8):
    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            return None

        count = min(max_frames, total_frames)

        if count == 1:
            indices = [0]
        else:
            indices = [
                round(i * (total_frames - 1) / (count - 1))
                for i in range(count)
            ]

        tensors = []

        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            tensors.append(preprocess(image))

        if not tensors:
            return None

        batch = torch.stack(tensors).to(DEVICE)

        with torch.inference_mode():
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)

        # Training class order: fake=0, real=1.
        # Average sampled-frame probabilities for a baseline.
        avg_fake = probabilities[:, 0].mean().item()
        avg_real = probabilities[:, 1].mean().item()

        prediction = 0 if avg_fake >= avg_real else 1

        return {
            "prediction": prediction,
            "fake_score": avg_fake,
            "real_score": avg_real,
            "frames_analyzed": len(tensors),
        }

    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=10,
        help="Videos per class for initial test; use 0 for all videos.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="Maximum frames sampled per video.",
    )
    args = parser.parse_args()

    if args.limit_per_class < 0 or args.frames < 1:
        raise ValueError("Invalid limit or frame count.")

    print("Device:", DEVICE)

    model = load_model()
    y_true = []
    y_pred = []
    fake_scores = []
    records = []
    skipped = []

    # Label mapping: fake=0, real=1
    for class_name, true_label in [("fake", 0), ("real", 1)]:
        folder = DATA_DIR / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        videos = sorted(
            path for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {
                ".mp4", ".avi", ".mov", ".mkv"
            }
        )

        if args.limit_per_class > 0:
            videos = videos[:args.limit_per_class]

        print(f"\n{class_name}: {len(videos)} videos selected")

        for number, video_path in enumerate(videos, start=1):
            result = analyze_video(
                model, video_path, args.frames
            )

            if result is None:
                skipped.append(str(video_path))
                continue

            y_true.append(true_label)
            y_pred.append(result["prediction"])
            fake_scores.append(result["fake_score"])

            records.append({
                "filename": str(video_path),
                "true_label": class_name,
                "predicted_label": (
                    "fake" if result["prediction"] == 0 else "real"
                ),
                "fake_score": round(result["fake_score"], 8),
                "real_score": round(result["real_score"], 8),
                "frames_analyzed": result["frames_analyzed"],
            })

            if number % 10 == 0:
                print(f"Processed {number}/{len(videos)}")

    if not y_true:
        raise RuntimeError(
            "No videos were successfully processed."
        )

    labels = [0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "device": str(DEVICE),
        "frames_per_video_max": args.frames,
        "limit_per_class": args.limit_per_class,
        "videos_evaluated": len(y_true),
        "videos_skipped": len(skipped),
        "accuracy": accuracy_score(y_true, y_pred),
        "fake_precision": precision_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "fake_recall": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "fake_f1": f1_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "confusion_matrix_labels": ["fake", "real"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=["fake", "real"],
            output_dict=True,
            zero_division=0,
        ),
        "skipped_files": skipped,
        "video_results": records,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "video_test_metrics.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n--- VIDEO EVALUATION RESULTS ---")
    print(f"Videos evaluated: {len(y_true)}")
    print(f"Videos skipped: {len(skipped)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Fake precision: {metrics['fake_precision']:.4f}")
    print(f"Fake recall: {metrics['fake_recall']:.4f}")
    print(f"Fake F1-score: {metrics['fake_f1']:.4f}")
    print("Confusion matrix [true fake, true real]:")
    print(matrix)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
