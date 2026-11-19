
import csv
import cv2
from pathlib import Path

ROOT = Path("data/video_test")
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

report = []
counts = {"fake": 0, "real": 0}
unreadable = 0

for label in ["fake", "real"]:
    folder = ROOT / label

    if not folder.exists():
        print(f"Missing folder: {folder}")
        continue

    videos = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    print(f"\nChecking {label}: {len(videos)} videos")

    for i, path in enumerate(videos, start=1):
        counts[label] += 1
        cap = cv2.VideoCapture(str(path))

        opened = cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0

        readable = opened and frame_count > 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0

        if readable:
            ok, frame = cap.read()
            readable = ok and frame is not None

        cap.release()

        if not readable:
            unreadable += 1

        report.append({
            "label": label,
            "path": str(path),
            "readable": readable,
            "frames": frame_count,
            "fps": round(fps, 2),
            "width": width,
            "height": height
        })

        if i % 500 == 0:
            print(f"Checked {i}/{len(videos)}")

with open("results/video_dataset_audit.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=report[0].keys() if report else [])
    if report:
        writer.writeheader()
        writer.writerows(report)

print("\n========== AUDIT SUMMARY ==========")
print("Fake videos:", counts["fake"])
print("Real videos:", counts["real"])
print("Total videos:", sum(counts.values()))
print("Unreadable videos:", unreadable)
print("Audit saved to results/video_dataset_audit.csv")
