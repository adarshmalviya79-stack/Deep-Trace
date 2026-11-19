
import csv
import cv2
from pathlib import Path

ROOT = Path(r"D:\deepfake-detection\deep-fake-dataset")
OUTPUT = Path("results/video_metadata.csv")

folders = {
    "fake": "fake",
    "real": "real",
    "YouTube-real": "real",
}

rows = []

for folder_name, label in folders.items():
    folder = ROOT / folder_name

    for path in sorted(folder.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            cap.release()
            continue

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration = frames / fps if fps and fps > 0 else 0

        rows.append({
            "filename": path.name,
            "folder": folder_name,
            "label": label,
            "frames": frames,
            "fps": round(fps, 2),
            "duration_seconds": round(duration, 2),
            "width": width,
            "height": height,
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Videos inspected:", len(rows))
print("Metadata saved:", OUTPUT.resolve())
print("\nSample:")
for row in rows[:10]:
    print(row)
