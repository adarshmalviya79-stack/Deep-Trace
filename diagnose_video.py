
import cv2
import torch
from pathlib import Path
from PIL import Image

from evaluate_videos import load_model, preprocess, DEVICE

VIDEOS = {
    "fake": Path("data/video_test/fake/0.mp4"),
    "real": Path("data/video_test/real/0.mp4"),
}

OUTPUT = Path("results/diagnostic_frames")
OUTPUT.mkdir(parents=True, exist_ok=True)

model = load_model()
print("Device:", DEVICE)

for label, video_path in VIDEOS.items():
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open:", video_path)
        continue

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        print("No frames found:", video_path)
        cap.release()
        continue

    indices = [
        round(i * (total - 1) / 7)
        for i in range(8)
    ]

    print(f"\n{label.upper()} VIDEO: {video_path}")
    tensors = []
    saved_frames = []

    for i, index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()

        if not ok:
            continue

        # Save original frame for visual inspection
        image_path = OUTPUT / f"{label}_frame_{i}.jpg"
        cv2.imwrite(str(image_path), frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensors.append(preprocess(Image.fromarray(rgb)))
        saved_frames.append(image_path)

    cap.release()

    if not tensors:
        print("No frames could be read.")
        continue

    batch = torch.stack(tensors).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1).cpu()

    for i, probs in enumerate(probabilities):
        print(
            f"Frame {i}: "
            f"FAKE={probs[0].item():.8f}, "
            f"REAL={probs[1].item():.8f}, "
            f"saved={saved_frames[i]}"
        )

print("\nDiagnostic frames saved in:", OUTPUT)
