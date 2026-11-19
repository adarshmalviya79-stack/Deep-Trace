
import argparse
import copy
import csv
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


SEED = 42
IMAGE_SIZE = 224
CLASS_NAMES = ["fake", "real"]


class FaceDataset(Dataset):
    def __init__(self, manifest, split, transform=None):
        self.transform = transform
        self.samples = []

        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                if row["label"] not in CLASS_NAMES:
                    continue

                path = Path(row["image_path"])
                if not path.is_file():
                    raise FileNotFoundError(path)

                label = CLASS_NAMES.index(row["label"])
                self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No images found for split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_metrics(labels, predictions):
    # Class 0 = fake; class 1 = real
    tp_fake = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fn_fake = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    tp_real = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fn_real = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))

    precision_fake = tp_fake / max(tp_fake + fn_real, 1)
    recall_fake = tp_fake / max(tp_fake + fn_fake, 1)

    # For real-class precision, false positives are fake images predicted real.
    precision_real = tp_real / max(tp_real + fn_fake, 1)
    recall_real = tp_real / max(tp_real + fn_real, 1)

    f1_fake = (
        2 * precision_fake * recall_fake
        / max(precision_fake + recall_fake, 1e-12)
    )
    f1_real = (
        2 * precision_real * recall_real
        / max(precision_real + recall_real, 1e-12)
    )

    accuracy = sum(y == p for y, p in zip(labels, predictions)) / max(
        len(labels), 1
    )

    return {
        "accuracy": accuracy,
        "macro_f1": (f1_fake + f1_real) / 2,
        "fake_precision": precision_fake,
        "fake_recall": recall_fake,
        "real_precision": precision_real,
        "real_recall": recall_real,
        "fake_f1": f1_fake,
        "real_f1": f1_real,
        "tp_fake": tp_fake,
        "fn_fake": fn_fake,
        "tp_real": tp_real,
        "fn_real": fn_real,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels_all, predictions_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

    metrics = get_metrics(labels_all, predictions_all)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="data/split_manifest.csv"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output", default="models/deepfake_efficientnet.pth"
    )
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    train_ds = FaceDataset(manifest, "train", train_tfms)
    val_ds = FaceDataset(manifest, "val", eval_tfms)
    test_ds = FaceDataset(manifest, "test", eval_tfms)

    # Check that no video appears in more than one split.
    video_splits = {}
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video = row["video_id"]
            split = row["split"]
            if video in video_splits and video_splits[video] != split:
                raise SystemExit(
                    f"Video leakage detected: {video}"
                )
            video_splits[video] = split

    workers = 0  # Windows-safe default
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # Stage 1: freeze the feature extractor to reduce GPU memory use.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model = model.to(device)

    counts = Counter(label for _, label in train_ds.samples)
    class_weights = torch.tensor(
        [
            len(train_ds) / (2 * counts[i])
            for i in range(2)
        ],
        dtype=torch.float32,
        device=device,
    )

    print(f"Class mapping: fake=0, real=1")
    print(f"Training images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Test images: {len(test_ds)}")
    print(f"Training class counts: {dict(counts)}")
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=args.lr
    )

    best_f1 = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_metrics = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": best_state,
        "classes": {"fake": 0, "real": 1},
        "image_size": IMAGE_SIZE,
        "best_val_macro_f1": best_f1,
        "architecture": "efficientnet_b0",
    }, output_path)

    print(f"\nBest model saved to: {output_path.resolve()}")

    # Test set is used only after checkpoint selection is complete.
    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model, test_loader, criterion, device
    )

    print("\nHeld-out test results")
    print(f"Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:      {test_metrics['macro_f1']:.4f}")
    print(f"Fake precision:{test_metrics['fake_precision']:.4f}")
    print(f"Fake recall:   {test_metrics['fake_recall']:.4f}")
    print(f"Real precision:{test_metrics['real_precision']:.4f}")
    print(f"Real recall:   {test_metrics['real_recall']:.4f}")
    print(
        "Confusion matrix (actual rows: fake, real; "
        "predicted columns: fake, real):"
    )
    print([
        [test_metrics["tp_fake"], test_metrics["fn_fake"]],
        [test_metrics["fn_real"], test_metrics["tp_real"]],
    ])


if __name__ == "__main__":
    main()
