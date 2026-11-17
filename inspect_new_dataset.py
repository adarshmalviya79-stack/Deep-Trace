
import cv2
import os
import random

base = "data/new_video_dataset"

for label in ["real", "fake"]:
    folder = os.path.join(base, label)
    videos = [f for f in os.listdir(folder) if f.endswith(".mp4")]

    print(f"\n{label.upper()} videos: {len(videos)}")

    samples = random.sample(videos, min(5, len(videos)))

    for name in samples:
        path = os.path.join(folder, name)
        cap = cv2.VideoCapture(path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        duration = frames / fps if fps > 0 else 0

        print(
            name,
            f"Resolution: {int(width)}x{int(height)}",
            f"FPS: {fps:.2f}",
            f"Duration: {duration:.2f}s"
        )

        cap.release()
