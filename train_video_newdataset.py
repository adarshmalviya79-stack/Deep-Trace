import os
import csv
import copy
import random
from pathlib import Path
from collections import Counter, defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


SEED = 42
IMAGE_SIZE = 224
CLASSES = ["fake", "real"]

MANIFEST = "data/new_video_dataset/face_manifest.csv"
OUTPUT = "models/deepfake_video_efficientnet_newdataset.pth"


class VideoFaceDataset(Dataset):
    def __init__(self, split, transform):
        self.samples = []
        self.transform = transform

        with open(MANIFEST, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue

                if row["label"] not in CLASSES:
                    continue

                path = Path(row["image_path"])

                if not path.is_file():
                    raise FileNotFoundError(path)

                self.samples.append((
                    path,
                    CLASSES.index(row["label"]),
                    row["video_id"]
                ))

        if not self.samples:
            raise ValueError(f"No samples found for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, video_id = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        image = self.transform(image)

        return image, label, video_id


def calculate_metrics(labels, predictions):
    matrix = [[0, 0], [0, 0]]

    for actual, predicted in zip(labels, predictions):
        matrix[actual][predicted] += 1

    tp_fake = matrix[0][0]
    fp_fake = matrix[1][0]
    fn_fake = matrix[0][1]

    tp_real = matrix[1][1]
    fp_real = matrix[0][1]
    fn_real = matrix[1][0]

    precision_fake = tp_fake / max(tp_fake + fp_fake, 1)
    recall_fake = tp_fake / max(tp_fake + fn_fake, 1)

    precision_real = tp_real / max(tp_real + fp_real, 1)
    recall_real = tp_real / max(tp_real + fn_real, 1)

    f1_fake = (
        2 * precision_fake * recall_fake
        / max(precision_fake + recall_fake, 1e-12)
    )

    f1_real = (
        2 * precision_real * recall_real
        / max(precision_real + recall_real, 1e-12)
    )

    accuracy = sum(
        actual == predicted
        for actual, predicted in zip(labels, predictions)
    ) / max(len(labels), 1)

    return {
        "accuracy": accuracy,
        "macro_f1": (f1_fake + f1_real) / 2,
        "fake_precision": precision_fake,
        "fake_recall": recall_fake,
        "real_precision": precision_real,
        "real_recall": recall_real,
        "confusion_matrix": matrix
    }


def evaluate(model, loader, criterion, device):
    model.eval()

    video_scores = defaultdict(list)
    video_labels = {}

    total_loss = 0
    total_images = 0

    with torch.no_grad():
        for images, labels, video_ids in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * labels.size(0)
            total_images += labels.size(0)

            for i, video_id in enumerate(video_ids):
                label = int(labels[i].item())
                video_key = (label, video_id)

                video_scores[video_key].append(
                    probabilities[i].cpu()
                )

                video_labels[video_key] = label

    all_labels = []
    all_predictions = []

    for video_key, scores in video_scores.items():
        average_scores = torch.stack(scores).mean(dim=0)
        prediction = average_scores.argmax().item()

        all_labels.append(video_labels[video_key])
        all_predictions.append(prediction)

    metrics = calculate_metrics(all_labels, all_predictions)

    metrics["loss"] = total_loss / max(total_images, 1)
    metrics["videos"] = len(video_scores)

    return metrics


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_ds = VideoFaceDataset("train", train_transform)
    val_ds = VideoFaceDataset("val", eval_transform)
    test_ds = VideoFaceDataset("test", eval_transform)

    print("Training frames:", len(train_ds))
    print("Validation frames:", len(val_ds))
    print("Test frames:", len(test_ds))

    video_splits = {}

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video_id = row["video_id"]
            split = row["split"]

            if video_id in video_splits:
                if video_splits[video_id] != split:
                    raise RuntimeError(
                        f"Video leakage: {video_id} appears in multiple splits"
                    )

            video_splits[video_id] = split

    workers = 0

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-1].parameters():
        parameter.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    model = model.to(device)

    counts = Counter(
        label for _, label, _ in train_ds.samples
    )

    class_weights = torch.tensor(
        [
            len(train_ds) / (2 * counts[i])
            for i in range(2)
        ],
        dtype=torch.float32,
        device=device
    )

    print("Class mapping: fake=0, real=1")
    print("Training class counts:", dict(counts))
    print("Class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW([
        {
            "params": model.features[-1].parameters(),
            "lr": 1e-5
        },
        {
            "params": model.classifier.parameters(),
            "lr": 1e-4
        }
    ], weight_decay=1e-4)

    best_f1 = -1
    best_state = None
    epochs = 5

    for epoch in range(epochs):
        model.train()
        model.features[:-1].eval()

        total_loss = 0
        total = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = total_loss / max(total, 1)

        val_metrics = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"\nEpoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"video_accuracy={val_metrics['accuracy']:.4f} | "
            f"video_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        print(
            "Validation confusion matrix:",
            val_metrics["confusion_matrix"]
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

            print("New best video-level model!")

    if best_state is None:
        raise RuntimeError("No best model produced.")

    os.makedirs("models", exist_ok=True)

    torch.save({
        "state_dict": best_state,
        "classes": {"fake": 0, "real": 1},
        "image_size": IMAGE_SIZE,
        "best_val_macro_f1": best_f1,
        "architecture": "efficientnet_b0",
        "preprocessing": "haar_face_crop_imagenet_normalization",
        "evaluation_level": "video"
    }, OUTPUT)

    print("\nBest model saved:", OUTPUT)
    print("Best validation video macro-F1:", best_f1)

    model.load_state_dict(best_state)

    test_metrics = evaluate(
        model, test_loader, criterion, device
    )

    print("\nFINAL VIDEO-LEVEL TEST RESULTS")
    print("Videos:", test_metrics["videos"])
    print("Accuracy:", round(test_metrics["accuracy"], 4))
    print("Macro F1:", round(test_metrics["macro_f1"], 4))
    print("Fake precision:", round(test_metrics["fake_precision"], 4))
    print("Fake recall:", round(test_metrics["fake_recall"], 4))
    print("Real precision:", round(test_metrics["real_precision"], 4))
    print("Real recall:", round(test_metrics["real_recall"], 4))
    print("Confusion matrix:", test_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()