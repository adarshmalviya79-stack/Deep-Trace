
import hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path("data")
SPLITS = ["train", "val", "test"]
image_hashes = defaultdict(list)

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for split in SPLITS:
    folder = ROOT / split

    if not folder.exists():
        print(f"Missing split: {folder}")
        continue

    count = 0

    for path in folder.rglob("*"):
        if path.suffix.lower() not in extensions:
            continue

        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            image_hashes[digest].append((split, str(path)))
            count += 1
        except OSError as error:
            print(f"Could not read {path}: {error}")

    print(f"{split}: scanned {count} images")

duplicates = 0

for locations in image_hashes.values():
    splits = {split for split, _ in locations}

    if len(splits) > 1:
        duplicates += 1
        print("\nDuplicate image across splits:")
        for split, path in locations:
            print(f"  {split}: {path}")

print(f"\nCross-split duplicate groups: {duplicates}")

if duplicates == 0:
    print("No exact cross-split duplicates found.")
else:
    print("Review duplicates before reporting research results.")
