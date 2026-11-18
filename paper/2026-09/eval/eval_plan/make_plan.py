"""Fixed evaluation plan shared by all representations (20 trials) + a placement map on the top-camera view.

Factors per trial: banana position (4 training clusters P1-P4; P5 = gap 70 px from any training start, P6 = right of the training x-range),
banana orientation in the top view (H = long axis left-right, V = long axis toward/away from the robot),
distractor (none / apple / carrot, placed at the fixed slot farther from the banana), target basket color.
Run order: sorted by position, then distractor, then orientation (fixed, not shuffled).
Pixel coordinates are in the top-camera image (reference_frames/top.jpg). Order is shuffled with a fixed seed.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE.parents[1] / "reference_frames" / "top.jpg"
TRAIN = json.load(open(HERE.parent / "train_banana_starts.json"))

POS = {  # id: (x, y, in_distribution)
    "P1": (232, 166, True), "P2": (354, 222, True), "P3": (127, 333, True), "P4": (315, 366, True),
    "P5": (100, 200, False), "P6": (420, 300, False),
}
SLOTS = {"S1": (70, 430), "S2": (150, 105)}  # distractor slots, left side, off the path to the baskets

# (position, orientation, distractor) - ID positions: 2 H + 2 V, 2 none + 1 apple + 1 carrot each
cells = []
for i, p in enumerate(["P1", "P2", "P3", "P4"]):
    a, b = ("apple", "carrot") if i % 2 == 0 else ("carrot", "apple")
    cells += [(p, "H", "none"), (p, "H", a), (p, "V", "none"), (p, "V", b)]
cells += [("P5", "H", "none"), ("P5", "V", "apple"), ("P6", "H", "carrot"), ("P6", "V", "none")]

rng = np.random.default_rng(2026)
colors = ["blue"] * 7 + ["yellow"] * 7 + ["green"] * 6
rng.shuffle(colors)
rng.permutation(len(cells))  # kept so the color assignment above stays identical
# Run order (2026-09-28 user decision): sorted by position, then distractor (none, apple, carrot), then orientation
DIS_RANK = {"none": 0, "apple": 1, "carrot": 2}
order = sorted(range(len(cells)), key=lambda k: (cells[k][0], DIS_RANK[cells[k][2]], cells[k][1]))

T = np.array([[r["x"], r["y"]] for r in TRAIN])
rows = []
for n, k in enumerate(order, start=1):
    p, ori, dis = cells[k]
    x, y, ind = POS[p]
    slot = max(SLOTS, key=lambda s: np.hypot(SLOTS[s][0] - x, SLOTS[s][1] - y)) if dis != "none" else ""
    rows.append({
        "trial": n, "position": p, "x": x, "y": y, "in_distribution": ind, "orientation": ori,
        "distractor": dis, "distractor_slot": slot, "color": colors[k],
        "task": f"pick the banana and place it in the {colors[k]} basket",
    })

with open(HERE / "plan.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

img = cv2.imread(str(REF))
for r in TRAIN:
    cv2.circle(img, (int(r["x"]), int(r["y"])), 2, (0, 0, 255), -1)
for p, (x, y, ind) in POS.items():
    col = (255, 0, 0) if ind else (0, 140, 255)
    cv2.circle(img, (x, y), 14, col, 2)
    cv2.line(img, (x - 24, y), (x + 24, y), col, 1)   # H guide
    cv2.line(img, (x, y - 24), (x, y + 24), col, 1)   # V guide
    near = int(np.sum(np.hypot(T[:, 0] - x, T[:, 1] - y) < 40))
    cv2.putText(img, f"{p} ({near})", (x + 16, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
for s, (x, y) in SLOTS.items():
    cv2.rectangle(img, (x - 14, y - 14), (x + 14, y + 14), (0, 160, 0), 2)
    cv2.putText(img, s, (x + 16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 160, 0), 2)
cv2.putText(img, "blue=in-dist  orange=out-of-dist  (n)=training starts within 40px  green=distractor slot",
            (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
cv2.imwrite(str(HERE / "position_map.jpg"), img)
print(open(HERE / "plan.csv").read())

# eval1 (2026-09-28): 3 positions x 3 baskets = 9 trials, banana horizontal, no distractor.
# Pilot data for planning the EE evaluation. Positions = the three training clusters with the most starts.
rows1 = []
for p in ["P1", "P3", "P4"]:
    x, y, ind = POS[p]
    for color in ["blue", "yellow", "green"]:
        rows1.append({
            "trial": len(rows1) + 1, "position": p, "x": x, "y": y, "in_distribution": ind, "orientation": "H",
            "distractor": "none", "distractor_slot": "", "color": color,
            "task": f"pick the banana and place it in the {color} basket",
        })
with open(HERE / "plan_eval1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows1[0]))
    w.writeheader()
    w.writerows(rows1)
print(open(HERE / "plan_eval1.csv").read())
