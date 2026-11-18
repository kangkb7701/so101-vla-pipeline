import glob, numpy as np, pandas as pd
R=["abs_joint","rel_joint","abs_ee","rel_ee"]
c=pd.concat([pd.read_csv(f) for f in glob.glob("e2/sens_close_*.csv")]); s=pd.concat([pd.read_csv(f) for f in glob.glob("e2/sens_start_*.csv")])
c["grp"]=np.where(c.label.isin(["success","rim"]),"success",np.where(c.label.str.contains("never_approached"),"closed_before_approach",np.where(c.label.str.contains("stuck"),"stuck","failed_at_banana")))
pd.set_option("display.width",200)
print("A. share of seeds commanding a close at the decisive chunk:")
print(c.pivot_table(index=["repr","grp"],columns="cond",values="closes",aggfunc="mean").round(2).reindex(R,level=0).to_string())
print("   episodes:", c[c.cond=="base"].groupby(["repr","grp"]).ep.nunique().to_dict())
b=c[c.cond=="base"].groupby(["repr","ep"]).roll_end.mean()
for k,sg in (("roll+30",1),("roll-30",-1)):
    m=c[c.cond==k].groupby(["repr","ep"]).roll_end.mean(); g=((m-b)/(30*sg))
    print(f"   {k} gain median:", g.groupby("repr").median().reindex(R).round(2).to_dict())
print("B. start-up p_go (chunk moves gripper > 3 cm), by condition:")
s["go"]=s.disp_cm>3
print(s.pivot_table(index="repr",columns="cond",values="go",aggfunc="mean").round(2).reindex(R).to_string())
print("   n chunks:", s[s.cond=="base"].groupby("repr").size().to_dict())
