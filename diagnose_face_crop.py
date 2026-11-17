
import cv2
import torch
from pathlib import Path
from PIL import Image

from evaluate_videos import load_model, preprocess, DEVICE

VIDEOS = {
    "fake": Path("data/video_test/fake/0.mp4"),
    "real": Path("data/video_test/real/0.mp4"),
}

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load OpenCV face detector")

model = load_model()
print("Device:", DEVICE)

for label, path in VIDEOS.items():
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        print("Could not open:", path)
        continue

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(set(
        round(i * (total - 1) / 7) for i in range(8)
    )) if total > 0 else []

    full_frames = []
    face_frames = []
    no_face = 0

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()

        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        full_frames.append(preprocess(Image.fromarray(rgb)))

        if len(faces) == 0:
            no_face += 1
            continue

        # Select the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add 25% padding around the face
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.25)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)

        crop = frame[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        face_frames.append(
            preprocess(Image.fromarray(crop))
        )

    cap.release()

    print(f"\n{label.upper()} video")
    print("Full frames:", len(full_frames))
    print("Face crops:", len(face_frames))
    print("Frames without detected faces:", no_face)

    def predict(frames):
        if not frames:
            return None

        batch = torch.stack(frames).to(DEVICE)

        with torch.inference_mode():
            probs = torch.softmax(model(batch), dim=1)

        avg = probs.mean(dim=0).cpu()
        return avg[0].item(), avg[1].item()

    full_result = predict(full_frames)
    crop_result = predict(face_frames)

    print("Full-frame scores (fake, real):", full_result)
    print("Face-crop scores (fake, real):", crop_result)
