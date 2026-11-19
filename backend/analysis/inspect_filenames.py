
import os
from collections import Counter

base = "data/new_video_dataset"

for label in ["real", "fake"]:
    folder = os.path.join(base, label)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".mp4")
    ]

    print(f"\n{label.upper()} FILENAME PATTERNS")

    patterns = Counter()

    for name in files:
        stem = os.path.splitext(name)[0]

        if stem.startswith("c") and "_" in stem:
            patterns["c_ID_c_ID"] += 1
        elif stem.startswith("c"):
            patterns["c_ID"] += 1
        elif stem.startswith("f"):
            patterns["f_ID"] += 1
        else:
            patterns["other"] += 1

    print(patterns)

    print("\nFirst 30 filenames:")
    for name in sorted(files)[:30]:
        print(name)
