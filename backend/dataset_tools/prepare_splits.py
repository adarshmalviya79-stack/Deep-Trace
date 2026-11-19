
import random
import shutil
from pathlib import Path

SOURCE = Path("data/video_faces")
OUTPUT = Path("data/splits")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    if not SOURCE.is_dir():
        raise SystemExit(f"Source directory not found: {SOURCE}")

    if OUTPUT.exists():
        raise SystemExit(
            f"{OUTPUT} already exists. Rename or remove it "
            "after checking its contents before rerunning."
        )

    rng = random.Random(SEED)
    plan = []

    for class_name in ("fake", "real"):
        class_dir = SOURCE / class_name

        if not class_dir.is_dir():
            raise SystemExit(f"Missing class directory: {class_dir}")

        videos = sorted(
            p for p in class_dir.iterdir() if p.is_dir()
        )
        videos = [
            video for video in videos
            if any(
                f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                for f in video.rglob("*")
            )
        ]

        if len(videos) < 3:
            raise SystemExit(
                f"Not enough video folders in {class_dir}"
            )

        rng.shuffle(videos)

        n_train = int(len(videos) * TRAIN_RATIO)
        n_val = int(len(videos) * VAL_RATIO)

        if n_train == 0 or n_val == 0:
            raise SystemExit(f"Not enough videos for {class_name}")

        groups = {
            "train": videos[:n_train],
            "val": videos[n_train:n_train + n_val],
            "test": videos[n_train + n_val:],
        }

        for split_name, split_videos in groups.items():
            for video_dir in split_videos:
                plan.append(
                    (class_name, split_name, video_dir)
                )

        print(
            f"{class_name}: {len(videos)} videos | "
            f"train={len(groups['train'])}, "
            f"val={len(groups['val'])}, "
            f"test={len(groups['test'])}"
        )

    # Copy only after validating both classes.
    try:
        for class_name, split_name, video_dir in plan:
            destination = (
                OUTPUT / split_name / class_name / video_dir.name
            )

            for file in video_dir.rglob("*"):
                if not file.is_file():
                    continue
                if file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                relative_path = file.relative_to(video_dir)
                target = destination / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)

        print("\nDataset split completed.")
        print(f"Output: {OUTPUT.resolve()}")

    except Exception:
        print(
            "\nCopying failed. Inspect the partial output directory "
            "before rerunning this script."
        )
        raise


if __name__ == "__main__":
    main()
