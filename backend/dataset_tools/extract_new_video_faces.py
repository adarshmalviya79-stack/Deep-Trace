
import os
import cv2
import csv

BASE = "data/new_video_dataset"
OUTPUT = "data/new_video_dataset/face_frames"
MANIFEST = "data/new_video_dataset/face_manifest.csv"

FRAMES_PER_VIDEO = 4

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load Haar Cascade")

with open(
    "data/new_video_dataset/split_manifest.csv",
    newline="",
    encoding="utf-8"
) as f:
    videos = list(csv.DictReader(f))

rows = []
failed = 0
processed = 0

for i, video in enumerate(videos, 1):
    path = video["video_path"]
    label = video["label"]
    split = video["split"]
    video_id = os.path.splitext(video["video_id"])[0]

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        failed += 1
        continue

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 1:
        cap.release()
        failed += 1
        continue

    positions = [
        int(total * p / (FRAMES_PER_VIDEO + 1))
        for p in range(1, FRAMES_PER_VIDEO + 1)
    ]

    saved = 0

    for frame_number in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            continue

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

        pad = int(max(w, h) * 0.20)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        relative_path = os.path.join(
            split, label, video_id, f"frame_{frame_number}.jpg"
        )

        output_path = os.path.join(OUTPUT, relative_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if cv2.imwrite(output_path, face):
            rows.append({
                "split": split,
                "label": label,
                "video_id": video["video_id"],
                "image_path": output_path,
                "frame_number": frame_number
            })
            saved += 1

    cap.release()
    processed += 1

    if i % 100 == 0:
        print(
            f"Processed {i}/{len(videos)} | "
            f"Saved frames: {len(rows)} | Failed videos: {failed}"
        )

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split", "label", "video_id",
            "image_path", "frame_number"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nEXTRACTION COMPLETE")
print("Videos processed:", processed)
print("Failed videos:", failed)
print("Faces saved:", len(rows))
print("Manifest:", MANIFEST)
