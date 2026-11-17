
import csv
import re
from collections import defaultdict

manifest = "data/new_video_dataset/split_manifest.csv"

c_splits = defaultdict(set)
f_splits = defaultdict(set)

with open(manifest, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

for row in rows:
    name = row["video_id"]
    split = row["split"]
    stem = name[:-4]

    if stem.startswith("c"):
        ids = re.fullmatch(r"(c\d+)(?:_(c\d+))?", stem)

        if ids:
            for video_id in ids.groups():
                if video_id:
                    c_splits[video_id].add(split)

    elif re.fullmatch(r"f\d+", stem):
        f_splits[stem].add(split)

c_leaks = {
    key: value for key, value in c_splits.items()
    if len(value) > 1
}

f_leaks = {
    key: value for key, value in f_splits.items()
    if len(value) > 1
}

print("TOTAL MANIFEST ROWS:", len(rows))
print("C SOURCE IDS CROSSING SPLITS:", len(c_leaks))
print("F IDs CROSSING SPLITS:", len(f_leaks))

print("\nC LEAK EXAMPLES:", list(c_leaks.items())[:10])
print("F LEAK EXAMPLES:", list(f_leaks.items())[:10])

print("\nSPLIT COUNTS:")

for split in ["train", "val", "test"]:
    selected = [r for r in rows if r["split"] == split]
    print(
        split,
        "real:", sum(r["label"] == "real" for r in selected),
        "fake:", sum(r["label"] == "fake" for r in selected)
    )
