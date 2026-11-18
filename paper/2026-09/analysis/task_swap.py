"""After the decisive close, does the language instruction still steer the arm?
For the two chunks after the close, the chunk is re-predicted with each of the three task strings (3 seeds).
language_effect_cm = mean pairwise distance between the chunk-end positions of different tasks (minus nothing);
seed_cm = mean distance between seeds under the same task (noise floor). Also: is the chunk end under task c
closer to basket c than to the others (share of correct steering)."""
import json
import numpy as np, pandas as pd
from replay_lib import Replayer
from common import REPRS, load_all, fk_pos

BASKET = {k: np.array(v) for k, v in json.load(open("basket_release_xyz.json")).items()}
TASK = {c: f"pick the banana and place it in the {c} basket" for c in BASKET}
ex = pd.read_csv("execution_check.csv")
A = load_all(); rows = []
for r in REPRS:
    R = Replayer(r)
    for e in A[r]:
        x = ex[(ex.repr == r) & (ex.ep == e.ep)].iloc[0]
        if np.isnan(x.t_close):
            continue
        tc = int(round(x.t_close * 10))
        for k, b in enumerate((15 * (tc // 15) + 15, 15 * (tc // 15) + 30)):
            if b + 15 > len(e.action):
                continue
            end = {c: [fk_pos(R.chunk_at(e.ep, b, seed=s, task=TASK[c])[-1:])[0] for s in range(3)] for c in BASKET}
            mean = {c: np.mean(v, 0) for c, v in end.items()}
            cs = list(BASKET)
            lang = np.mean([np.linalg.norm(mean[a][:2] - mean[c][:2]) for i, a in enumerate(cs) for c in cs[i + 1:]]) * 100
            seed = np.mean([np.linalg.norm(v[i][:2] - v[j][:2]) for v in end.values() for i, j in ((0, 1), (0, 2), (1, 2))]) * 100
            correct = np.mean([min(BASKET, key=lambda q: np.linalg.norm(mean[c][:2] - BASKET[q][:2])) == c for c in cs])
            rows.append({"repr": r, "ep": e.ep, "label": x.label, "chunk_after": k + 1, "held": bool((e.state[b, 5] - e.action[b, 5]) >= 6),
                         "language_cm": lang, "seed_cm": seed, "steer_correct": correct})
        print(r, e.ep, flush=True)
    R.close()
df = pd.DataFrame(rows); df.to_csv("task_swap.csv", index=False)
print(df.groupby(["repr", "held", "chunk_after"])[["language_cm", "seed_cm", "steer_correct"]].median().round(2).to_string())
print(df.groupby(["held", "chunk_after"])[["language_cm", "seed_cm", "steer_correct"]].median().round(2).to_string())
