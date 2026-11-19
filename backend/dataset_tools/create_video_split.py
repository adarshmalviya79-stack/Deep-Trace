
import os
import re
import csv
import random
from collections import defaultdict

BASE = "data/new_video_dataset"
OUTPUT = "data/new_video_dataset/split_manifest.csv"

random.seed(42)

parent = {}

def find(x):
    if x not in parent:
        parent[x] = x
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra = find(a)
    rb = find(b)
    if ra != rb:
        parent[rb] = ra

# Connect c IDs referenced by fake videos
for name in os.listdir(os.path.join(BASE, "real")):
    stem = os.path.splitext(name)[0]
    if re.fullmatch(r"c\d+", stem):
        find(stem)

for name in os.listdir(os.path.join(BASE, "fake")):
    stem = os.path.splitext(name)[0]
    match = re.fullmatch(r"(c\d+)_(c\d+)", stem)
    if match:
        union(match.group(1), match.group(2))

# Build connected groups
groups = defaultdict(list)

for video_id in parent:
    groups[find(video_id)].append(video_id)

# Map each c ID to its group
id_to_group = {}

for root, ids in groups.items():
    for video_id in ids:
        id_to_group[video_id] = root

# Collect videos by group
group_videos = defaultdict(list)
independent_videos = []

for label in ["real", "fake"]:
    folder = os.path.join(BASE, label)

    for name in os.listdir(folder):
        if not name.lower().endswith(".mp4"):
            continue

        stem = os.path.splitext(name)[0]
        path = os.path.join(folder, name)

        if label == "real" and re.fullmatch(r"c\d+", stem):
            group_videos[id_to_group[stem]].append(
                (label, path, name)
            )

        elif label == "fake":
            match = re.fullmatch(r"(c\d+)_(c\d+)", stem)

            if match:
                root = id_to_group[match.group(1)]
                group_videos[root].append((label, path, name))
            else:
                independent_videos.append((label, path, name))

        else:
            independent_videos.append((label, path, name))

# Shuffle groups, then assign approximately 70/15/15
group_items = list(group_videos.items())
random.shuffle(group_items)

total = sum(len(videos) for _, videos in group_items)
targets = {
    "train": total * 0.70,
    "val": total * 0.15,
    "test": total * 0.15
}

counts = defaultdict(int)
assignments = {}

for root, videos in group_items:
    remaining = {
        split: targets[split] - counts[split]
        for split in targets
    }

    split = max(remaining, key=remaining.get)
    assignments[root] = split
    counts[split] += len(videos)

# Assign f-prefixed videos separately, distributing each class
for label in ["real", "fake"]:
    items = [
        item for item in independent_videos
        if item[0] == label
    ]
    random.shuffle(items)

    n = len(items)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    for i, item in enumerate(items):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        root = ("independent", label, item[2])
        assignments[root] = split

# Write manifest
rows = []

for root, videos in group_videos.items():
    split = assignments[root]

    for label, path, name in videos:
        rows.append({
            "split": split,
            "label": label,
            "video_id": name,
            "video_path": path
        })

for label, path, name in independent_videos:
    root = ("independent", label, name)
    rows.append({
        "split": assignments[root],
        "label": label,
        "video_id": name,
        "video_path": path
    })

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["split", "label", "video_id", "video_path"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Manifest saved:", OUTPUT)
print("Total videos:", len(rows))

for split in ["train", "val", "test"]:
    print("\n", split.upper())

    for label in ["real", "fake"]:
        count = sum(
            1 for row in rows
            if row["split"] == split and row["label"] == label
        )
        print(label, count)
