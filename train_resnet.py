import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "train").exists() or not (data_dir / "val").exists():
        raise SystemExit("Expected data/train and data/val folders.")

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)
    if train_ds.classes != ["fake", "real"] or val_ds.classes != train_ds.classes:
        raise SystemExit(f"Expected class folders fake/real. Train: {train_ds.classes}; val: {val_ds.classes}")
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("Training or validation dataset is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_acc = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    print(f"Device: {device}")
    print(f"Classes: {train_ds.class_to_idx}")
    print(f"Train images: {len(train_ds)} | Validation images: {len(val_ds)}")

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)
        train_loss = total_loss / max(total, 1)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device))
                val_correct += (outputs.argmax(1).cpu() == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / max(val_total, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={train_loss:.4f} "
              f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    output = Path("models/resnet18_deepfake.pth")
    output.parent.mkdir(exist_ok=True)
    torch.save({
        "state_dict": best_weights,
        "classes": train_ds.class_to_idx,
        "image_size": 224,
        "best_val_accuracy": best_acc,
    }, output)
    print(f"Saved checkpoint: {output}")
    print("Use the same untouched test set for the final comparison.")


if __name__ == "__main__":
    main()
