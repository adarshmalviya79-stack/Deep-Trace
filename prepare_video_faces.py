
import cv2
import random
from pathlib import Path

SOURCE = Path(r"D:\deepfake-detection\deep-fake-dataset")
OUTPUT = Path("data/video_faces")
FRAMES_PER_VIDEO = 4
SEED = 42

random.seed(SEED)

sources = [
    (SOURCE / "fake", "fake"),
    (SOURCE / "real", "real"),
    (SOURCE / "YouTube-real", "real"),
]

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load OpenCV face detector.")

for folder, label in sources:
    videos = sorted(folder.glob("*.mp4"))
    print(f"\n{label} | {folder.name}: {len(videos)} videos")

    for video_index, video_path in enumerate(videos):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            cap.release()
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            continue

        positions = sorted(set(
            int(x) for x in
            [frame_count * (i + 1) / (FRAMES_PER_VIDEO + 1)
             for i in range(FRAMES_PER_VIDEO)]
        ))

        saved = 0

        for frame_number in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) == 0:
                continue

            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            pad_x, pad_y = int(w * 0.20), int(h * 0.20)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            output_folder = OUTPUT / label / video_path.stem
            output_folder.mkdir(parents=True, exist_ok=True)

            output_path = output_folder / f"frame_{frame_number}.jpg"
            if cv2.imwrite(str(output_path), crop):
                saved += 1

        cap.release()

        if (video_index + 1) % 100 == 0:
            print(f"Processed {video_index + 1}/{len(videos)}")

print("\nFace extraction completed.")
print("Output:", OUTPUT.resolve())

for label in ["fake", "real"]:
    folders = list((OUTPUT / label).glob("*"))
    image_count = sum(
        1 for p in (OUTPUT / label).rglob("*.jpg")
    )
    print(f"{label}: {len(folders)} video folders, {image_count} face crops")
