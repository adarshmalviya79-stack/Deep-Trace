
import csv
import random
from pathlib import Path

SOURCE = Path("data/video_faces")
OUTPUT = Path("data/split_manifest.csv")

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    rng = random.Random(SEED)
    rows = []

    if not SOURCE.is_dir():
        raise SystemExit(f"Source not found: {SOURCE.resolve()}")

    for label in ("fake", "real"):
        class_dir = SOURCE / label

        if not class_dir.is_dir():
            raise SystemExit(f"Missing class directory: {class_dir}")

        videos = sorted(
            folder for folder in class_dir.iterdir()
            if folder.is_dir()
        )

        videos = [
            folder for folder in videos
            if any(
                file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
                for file in folder.rglob("*")
            )
        ]

        if len(videos) < 3:
            raise SystemExit(f"Not enough video folders for {label}")

        rng.shuffle(videos)

        n_train = int(len(videos) * 0.70)
        n_val = int(len(videos) * 0.15)

        assignments = [
            ("train", videos[:n_train]),
            ("val", videos[n_train:n_train + n_val]),
            ("test", videos[n_train + n_val:]),
        ]

        for split, video_folders in assignments:
            image_count = 0

            for video in video_folders:
                images = sorted(
                    file for file in video.rglob("*")
                    if file.is_file()
                    and file.suffix.lower() in IMAGE_EXTENSIONS
                )

                for image_path in images:
                    rows.append({
                        "split": split,
                        "label": label,
                        "video_id": f"{label}/{video.name}",
                        "image_path": str(image_path.resolve()),
                    })
                    image_count += len([image_path])

            print(
                f"{label}/{split}: "
                f"{len(video_folders)} videos, "
                f"{image_count} images"
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "label", "video_id", "image_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest saved: {OUTPUT.resolve()}")
    print(f"Total image entries: {len(rows)}")


if __name__ == "__main__":
    main()
