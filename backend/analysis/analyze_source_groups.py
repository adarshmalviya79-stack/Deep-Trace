
import os
import re
from collections import defaultdict

base = "data/new_video_dataset"

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

# Read real c IDs
for name in os.listdir(os.path.join(base, "real")):
    stem = os.path.splitext(name)[0]

    if re.fullmatch(r"c\d+", stem):
        find(stem)

# Connect IDs referenced by fake videos
for name in os.listdir(os.path.join(base, "fake")):
    stem = os.path.splitext(name)[0]

    match = re.fullmatch(r"(c\d+)_(c\d+)", stem)

    if match:
        union(match.group(1), match.group(2))

groups = defaultdict(list)

for video_id in parent:
    groups[find(video_id)].append(video_id)

sizes = sorted([len(x) for x in groups.values()], reverse=True)

print("Unique c IDs:", len(parent))
print("Connected source groups:", len(groups))
print("Largest group:", sizes[0])
print("Smallest group:", sizes[-1])
print("Groups containing multiple IDs:", sum(s > 1 for s in sizes))
print("Top 20 group sizes:", sizes[:20])
