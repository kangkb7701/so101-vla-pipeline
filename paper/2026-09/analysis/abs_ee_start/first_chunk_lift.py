# First chunk only (t = 0, arm at its start pose in every episode): commanded lift and realized lift
import numpy as np
from common import load_session, load_training, fk_pos
tr = load_training()
lab = np.array([(fk_pos(A[14:15]) - fk_pos(S[:1]))[0] * 100 for _, A, S in tr])
print("demo label, first chunk: dz median %+.2f cm, |d| median %.2f" % (np.median(lab[:, 2]), np.median(np.linalg.norm(lab, axis=1))))
for r in ("abs_joint", "abs_ee"):
    c, m = [], []
    for e in load_session(r, "eval2"):
        pm = fk_pos(e.state) * 100; pc = fk_pos(e.action) * 100
        c.append(pc[14] - pm[0]); m.append(pm[16] - pm[0])
    c, m = np.array(c), np.array(m)
    print(f"{r:9s} first chunk: commanded dz {np.median(c[:, 2]):+.2f} cm, |cmd| {np.median(np.linalg.norm(c, axis=1)):.2f} | "
          f"realized dz {np.median(m[:, 2]):+.2f}, |moved| {np.median(np.linalg.norm(m, axis=1)):.2f} | episodes lifted >0.5cm: {(m[:, 2] > 0.5).sum()}/20")
