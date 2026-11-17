
import os
import re
from collections import Counter

base = "data/new_video_dataset"

real_files = [
    f for f in os.listdir(os.path.join(base, "real"))
    if f.lower().endswith(".mp4")
]

fake_files = [
    f for f in os.listdir(os.path.join(base, "fake"))
    if f.lower().endswith(".mp4")
]

real_ids = set()
fake_ids = set()
invalid = []

for name in real_files:
    match = re.fullmatch(r"(c\d+)", os.path.splitext(name)[0])

    if match:
        real_ids.add(match.group(1))
    else:
        invalid.append(("real", name))

for name in fake_files:
    match = re.fullmatch(
        r"(c\d+)_(c\d+)", os.path.splitext(name)[0]
    )

    if match:
        fake_ids.update(match.groups())
    else:
        invalid.append(("fake", name))

print("REAL FILES:", len(real_files))
print("FAKE FILES:", len(fake_files))
print("UNIQUE REAL SOURCE IDs:", len(real_ids))
print("UNIQUE IDs REFERENCED BY FAKE:", len(fake_ids))
print("SOURCE IDs PRESENT IN BOTH:", len(real_ids & fake_ids))
print("INVALID FILENAMES:", len(invalid))

print("\nREAL ID EXAMPLES:", sorted(real_ids)[:10])
print("FAKE ID EXAMPLES:", sorted(fake_ids)[:10])

print("\nINVALID EXAMPLES:", invalid[:10])
