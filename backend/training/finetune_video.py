
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

# Ensure training directory and backend root are in import path
BASE_DIR = Path(__file__).resolve().parent.parent
for p in [str(Path(__file__).resolve().parent), str(BASE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from train import FaceDataset, evaluate
except ImportError:
    from training.train import FaceDataset, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    "models/deepfake_video_efficientnet.pth",
    map_location="cpu",
    weights_only=True
)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2
)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

dataset = FaceDataset(
    "data/split_manifest.csv",
    "test",
    transform
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

metrics = evaluate(
    model,
    loader,
    nn.CrossEntropyLoss(),
    device
)

print("\nFine-tuned held-out test results")
print("Best validation macro F1:", checkpoint["best_val_macro_f1"])

for key in [
    "accuracy", "macro_f1",
    "fake_precision", "fake_recall",
    "real_precision", "real_recall"
]:
    print(f"{key}: {metrics[key]:.4f}")

print("Confusion matrix (actual rows: fake, real):")
print(metrics["confusion"])
