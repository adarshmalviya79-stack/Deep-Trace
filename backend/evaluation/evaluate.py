import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_DIR = Path("data/test")
CHECKPOINT = Path("models/deepfake_efficientnet.pth")
RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")
    if not DATA_DIR.exists():
        raise SystemExit(
            "Test folder not found. Create data/test/fake and data/test/real, "
            "then copy the original dataset's untouched test images into them."
        )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    if set(dataset.classes) != {"fake", "real"}:
        raise SystemExit(
            f"Expected folders named fake and real; found {dataset.classes}. "
            "Rename the class folders exactly to fake and real."
        )
    if not len(dataset):
        raise SystemExit("No images found in data/test/fake and data/test/real.")

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()

    y_true, y_pred, fake_scores = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(DEVICE)), dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(probabilities.argmax(dim=1).tolist())
            fake_scores.extend(probabilities[:, dataset.class_to_idx["fake"]].tolist())

    fake_idx, real_idx = dataset.class_to_idx["fake"], dataset.class_to_idx["real"]
    binary_true = [1 if label == fake_idx else 0 for label in y_true]
    metrics = {
        "test_images": len(dataset),
        "class_counts": {
            name: sum(1 for _, label in dataset.samples if label == idx)
            for name, idx in dataset.class_to_idx.items()
        },
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, pos_label=fake_idx, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, pos_label=fake_idx, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, pos_label=fake_idx, zero_division=0),
        "roc_auc_fake": roc_auc_score(binary_true, fake_scores) if len(set(y_true)) == 2 else None,
        "confusion_matrix_labels": ["fake", "real"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[fake_idx, real_idx]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[fake_idx, real_idx],
            target_names=["fake", "real"], zero_division=0, output_dict=True
        ),
        "note": "Only use a test set that was never used for training, validation, or model selection."
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / "test_metrics.json"
    output_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(dataset)}")
    print(f"Class counts: {metrics['class_counts']}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision (fake): {metrics['precision_fake']:.4f}")
    print(f"Recall (fake):    {metrics['recall_fake']:.4f}")
    print(f"F1-score (fake):  {metrics['f1_fake']:.4f}")
    if metrics["roc_auc_fake"] is not None:
        print(f"ROC-AUC (fake):   {metrics['roc_auc_fake']:.4f}")
    print("Confusion matrix (rows=true [fake, real], columns=predicted [fake, real]):")
    print(metrics["confusion_matrix"])
    print(f"Detailed report saved to {output_file}")


if __name__ == "__main__":
    main()
