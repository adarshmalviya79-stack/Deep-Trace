
import csv
import os
from collections import defaultdict

path = "data/new_video_dataset/face_manifest.csv"

data = []
video_frames = defaultdict(int)
video_split = {}
video_label = {}

with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        data.append(row)

        video = row["video_id"]
        video_frames[video] += 1
        video_split[video] = row["split"]
        video_label[video] = row["label"]

print("TOTAL EXTRACTED FRAMES:", len(data))
print("UNIQUE VIDEOS WITH FACES:", len(video_frames))

print("\nFRAMES BY SPLIT AND CLASS:")

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        count = sum(
            1 for row in data
            if row["split"] == split and row["label"] == label
        )

        videos = sum(
            1 for v in video_frames
            if video_split[v] == split and video_label[v] == label
        )

        print(split, label, "frames:", count, "videos:", videos)

print("\nFRAMES PER VIDEO:")

counts = list(video_frames.values())

print("Minimum:", min(counts))
print("Maximum:", max(counts))
print("Average:", round(sum(counts) / len(counts), 2))

print("\nVIDEOS WITH FEWER THAN 4 FRAMES:",
      sum(count < 4 for count in counts))

print("VIDEOS WITH EXACTLY 4 FRAMES:",
      sum(count == 4 for count in counts))

print("\nFRAMES BY CLASS:")

for label in ["real", "fake"]:
    print(label, sum(row["label"] == label for row in data))

print("\nMISSING IMAGE FILES:")

missing = sum(
    not os.path.exists(row["image_path"])
    for row in data
)

print(missing)
