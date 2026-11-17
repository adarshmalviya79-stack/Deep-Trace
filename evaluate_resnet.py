import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

DATA_DIR = Path("data/test")
CHECKPOINT = Path("models/resnet18_deepfake.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not CHECKPOINT.exists():
        raise SystemExit("Checkpoint missing. Run train_resnet.py first.")
    if not DATA_DIR.exists():
        raise SystemExit("Test dataset missing: expected data/test/fake and data/test/real.")

    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(DATA_DIR, transform=tfms)
    if ds.classes != ["fake", "real"] or not len(ds):
        raise SystemExit(f"Expected non-empty fake/real folders, found classes {ds.classes}.")

    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    true_labels, pred_labels, fake_scores = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(DEVICE)), dim=1).cpu()
            true_labels.extend(labels.tolist())
            pred_labels.extend(probs.argmax(1).tolist())
            fake_scores.extend(probs[:, ds.class_to_idx["fake"]].tolist())

    fake_idx, real_idx = ds.class_to_idx["fake"], ds.class_to_idx["real"]
    binary_true = [1 if y == fake_idx else 0 for y in true_labels]
    result = {
        "model": "ResNet18",
        "test_images": len(ds),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision_fake": precision_score(true_labels, pred_labels, pos_label=fake_idx, zero_division=0),
        "recall_fake": recall_score(true_labels, pred_labels, pos_label=fake_idx, zero_division=0),
        "f1_fake": f1_score(true_labels, pred_labels, pos_label=fake_idx, zero_division=0),
        "roc_auc_fake": roc_auc_score(binary_true, fake_scores) if len(set(true_labels)) == 2 else None,
        "confusion_matrix_labels": ["fake", "real"],
        "confusion_matrix": confusion_matrix(true_labels, pred_labels, labels=[fake_idx, real_idx]).tolist(),
        "note": "Use only an untouched test split. Compare against EfficientNet on exactly the same images."
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/resnet18_test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Device: {DEVICE}")
    print(f"Test images: {len(ds)}")
    print(f"Accuracy:         {result['accuracy']:.4f}")
    print(f"Precision (fake): {result['precision_fake']:.4f}")
    print(f"Recall (fake):    {result['recall_fake']:.4f}")
    print(f"F1-score (fake):  {result['f1_fake']:.4f}")
    if result["roc_auc_fake"] is not None:
        print(f"ROC-AUC (fake):   {result['roc_auc_fake']:.4f}")
    print("Confusion matrix (rows=true [fake, real], columns=predicted [fake, real]):")
    print(result["confusion_matrix"])
    print("Saved results/resnet18_test_metrics.json")


if __name__ == "__main__":
    main()
