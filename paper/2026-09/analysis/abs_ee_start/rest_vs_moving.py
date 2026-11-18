# Where do the small commanded chunks happen? (eval2, before the decisive close)
import numpy as np, pandas as pd
from common import REPRS, load_session, fk_pos
from e2_common import decisive_close
rows=[]
for r in REPRS:
    cpc=pd.read_csv(f"e2/chunks_per_chunk_{r}.csv")
    for e in load_session(r,"eval2"):
        pm=fk_pos(e.state); tc,_=decisive_close(e.action,e.state)
        moved=np.linalg.norm(pm-pm[0],axis=1)*100
        leave=np.argmax(moved>=3) if (moved>=3).any() else None
        for _,c in cpc[(cpc.ep==e.ep)&(cpc.phase=="before_close")].iterrows():
            t=int(round(c.t*10))
            rows.append({"repr":r,"ep":e.ep,"t":c.t,"at_rest":moved[t]<3,"cmd":c.cmd_travel_cm,"meas":c.meas_travel_cm,
                         "leave_s":None if leave is None else leave/10})
d=pd.DataFrame(rows)
g=d.groupby(["repr","at_rest"]).cmd.agg(["count","median",lambda x:(x<1).mean()]).rename(columns={"<lambda_0>":"frac<1cm"})
print(g.reindex(REPRS,level=0).round(2))
lv=d.groupby(["repr","ep"]).leave_s.first()
print("time to move 3cm from start (s), median & #never:")
for r in REPRS: x=lv[r]; print(r, x.dropna().median(), int(x.isna().sum()), "of", len(x))
tot=d.groupby("repr").cmd.agg(["count","median"]).reindex(REPRS); print(tot.round(2))
print("at-rest chunks: commanded vs actually moved (cm)")
for r in REPRS:
    x=d[(d.repr==r)&d.at_rest]
    print(f"{r:9s} n={len(x):3d} cmd median {x.cmd.median():.2f}  meas median {x.meas.median():.2f}  share cmd<0.5cm {(x.cmd<0.5).mean():.2f}")
print("real eval2, at-rest chunks: share commanding > 3 cm")
for r in REPRS:
    x=d[(d.repr==r)&d.at_rest]; print(f"{r:9s} {(x.cmd>3).mean():.2f} (n={len(x)})")
