
import csv
import random
from pathlib import Path

ROOT = Path(r"D:\deepfake-detection\deep-fake-dataset")
OUT = Path("results/video_manifest.csv")
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SEED = 42

random.seed(SEED)
records = []

# Keep YouTube-real separate for now until source relationships are checked.
sources = [
    (ROOT / "fake", "fake"),
    (ROOT / "real", "real"),
]

for folder, label in sources:
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    random.shuffle(files)

    n = len(files)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    for i, path in enumerate(files):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        records.append({
            "path": str(path.resolve()),
            "label": label,
            "split": split,
            "source_folder": folder.name,
        })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["path", "label", "split", "source_folder"]
    )
    writer.writeheader()
    writer.writerows(records)

print("Manifest saved:", OUT.resolve())
print("Total labeled videos:", len(records))

for split in ["train", "val", "test"]:
    for label in ["fake", "real"]:
        count = sum(
            r["split"] == split and r["label"] == label
            for r in records
        )
        print(f"{split:5} | {label:4} | {count}")

