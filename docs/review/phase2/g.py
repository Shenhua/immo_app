from h import *
U1=prop("U1",100000,500,meuble=False,works=5000,energy=15000)
U2=prop("U2",100000,500,meuble=False,works=5000,energy=15000)
df,b=run(strat(U1,U2),3)
for r in b["fiscal_annuel"]: print({k:round(v) for k,v in r.items()})
print("exit", {k:round(v) for k,v in b["exit_tax"].items()})
print("pv details deficit_recapture", b["plus_value"].get("deficit_recapture"), "total_tax", round(b["plus_value"]["total_tax"]))
# micro-foncier w/ works: works not deducted; PV majoration real works at H>5
df,b=run(strat(prop("U",100000,500,meuble=False,works=30000)),10,{"regime_foncier":"micro"})
print("micro foncier works_maj", b["plus_value_par_bien"][0]["works_majoration"])
df,b=run(strat(prop("U",100000,500,meuble=False,works=30000)),10,{"regime_foncier":"reel"})
print("reel foncier works_maj", b["plus_value_par_bien"][0]["works_majoration"])
