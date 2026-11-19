
import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Paths
data_dir = "data/external_test"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("Classes:", dataset.class_to_idx)
print("External images:", len(dataset))

# Confirm labels
if dataset.class_to_idx != {"fake": 0, "real": 1}:
    raise ValueError(
        "Expected fake=0 and real=1. "
        f"Found: {dataset.class_to_idx}"
    )

# Model definitions

def load_efficientnet(path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )

    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )

    weights = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(weights)
    return model


def load_resnet(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )

    weights = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(weights)
    return model

# Evaluate one model
def evaluate(model, model_name):
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            y_scores.extend(probabilities[:, 0].cpu().numpy().tolist())

    metrics = {
        "model": model_name,
        "images": len(y_true),
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
        "fake_roc_auc": roc_auc_score(
            [1 if label == 0 else 0 for label in y_true],
            y_scores
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["fake", "real"],
            output_dict=True,
            zero_division=0
        )
    }

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Fake precision: {metrics['fake_precision']:.4f}")
    print(f"Fake recall:    {metrics['fake_recall']:.4f}")
    print(f"Fake F1-score:  {metrics['fake_f1']:.4f}")
    print(f"Fake ROC-AUC:   {metrics['fake_roc_auc']:.4f}")
    print("Confusion matrix [true fake, true real]:")
    print(metrics["confusion_matrix"])

    return metrics

# Load and evaluate both models
efficientnet = load_efficientnet(
    "models/deepfake_efficientnet.pth"
)
resnet = load_resnet(
    "models/resnet18_deepfake.pth"
)

results = {
    "dataset": "external_test",
    "class_to_idx": dataset.class_to_idx,
    "efficientnet": evaluate(efficientnet, "EfficientNet-B0"),
    "resnet18": evaluate(resnet, "ResNet18")
}

# Save results
output_path = os.path.join(
    results_dir, "external_test_metrics.json"
)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print(f"\nResults saved to: {output_path}")
