"""Copy so101_pick_place_fruit_180 (hard links) with exact per-frame quantiles for action / observation.state.

LeRobot's episode-merge aggregation left q01..q99 far from the true distribution (e.g. wrist_roll q99 99 vs 62 deg).
pi0.5 normalizes state and action with quantiles, so its dataset gets exact np.percentile values;
mean/std/min/max and the data itself are unchanged. Run from 2026-09-paper/runs.
"""
import glob, json, os, shutil
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path("/home/aivlab/kkb_capstone/data/lerobot/so101_pick_place_fruit_180")
DST = SRC.with_name(SRC.name + "_q")
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, copy_function=os.link)
stats_path = DST / "meta" / "stats.json"
stats = json.loads(stats_path.read_text())
stats_path.unlink()  # break the hard link before rewriting
df = pd.concat([pd.read_parquet(f, columns=["action", "observation.state"])
                for f in sorted(glob.glob(str(DST / "data/**/*.parquet"), recursive=True))])
for key in ["action", "observation.state"]:
    arr = np.stack(df[key]).astype(np.float64)
    for q in (1, 10, 50, 90, 99):
        stats[key][f"q{q:02d}"] = np.percentile(arr, q, axis=0).tolist()
    lo, hi = np.array(stats[key]["q01"]), np.array(stats[key]["q99"])
    print(key, "outside q01~q99 %", np.round(((arr < lo) | (arr > hi)).mean(0) * 100, 2))
stats_path.write_text(json.dumps(stats, indent=4))
print("written", DST)
