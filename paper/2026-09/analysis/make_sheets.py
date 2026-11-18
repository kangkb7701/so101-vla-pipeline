"""Contact sheets: first and last top/side frame of every evaluation episode (for outcome judgement)."""
import cv2, numpy as np
from lerobot.datasets import LeRobotDataset
from common import REC, load_all
A = load_all()
for r, eps in A.items():
    s = eps[0].session
    ds = LeRobotDataset(f"local/rollout_smolvla_{r}_{s}", root=REC / f"smolvla_{r}" / s)
    frm = ds.meta.episodes["dataset_from_index"]; to = ds.meta.episodes["dataset_to_index"]
    tiles = []
    for e in eps:
        row = []
        for i in (int(frm[e.ep]), int(to[e.ep]) - 1):
            it = ds[i]
            for k in ("observation.images.camera1", "observation.images.camera2"):
                img = cv2.cvtColor((it[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                row.append(cv2.resize(img, (320, 240)))
        strip = np.hstack(row)
        p = e.plan
        txt = f"ep{e.ep} trial{e.trial} {p.get('position','')} {p.get('orientation','')} {p.get('distractor','')} -> {p.get('color','')} | len {len(e.action)/10:.0f}s   [start top | start side | END top | END side]"
        cv2.rectangle(strip, (0, 0), (1280, 22), (255, 255, 255), -1)
        cv2.putText(strip, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(strip)
    for k in range(0, len(tiles), 5):
        cv2.imwrite(f"sheets/{r}_{k//5}.jpg", np.vstack(tiles[k:k+5]))
    print(r, len(tiles))
