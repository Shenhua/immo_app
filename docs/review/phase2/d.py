from h import *
# zero-rate loan, IRA on
Z=prop("Z",100000,600,rate=0.0)
df,b=run(strat(Z),10,{"apply_ira":True})
print("zero-rate: y1 interest", round(df.iloc[0]["Intérêts & Assurance"]), "dette y10", round(df.iloc[-1]["Dette"]), "exp", round(97200 - 97200*120/240), "ira", b["ira_total"], "tri", b["tri_annuel"])
# all-cash
C=prop("C",100000,600,loan=0.0)
df,b=run(strat(C),10)
print("cash: apport", strat(C)["apport_total"], "dette", df.iloc[-1]["Dette"], "tri", round(b["tri_annuel"],2), "wealth", round(b["enrichissement_net"]))
# all cash, loan duration 0
C2=prop("C2",100000,600,loan=0.0, years=0)
try:
    df,b=run(strat(C2),10); print("cash years0 ok", round(b["tri_annuel"],2))
except Exception as e: print("cash years0 ERR", type(e).__name__, e)
# negative appreciation -> PV floored at 0
N=prop("N",100000,600)
df,b=run(strat(N),10,market={"appreciation_bien_pct":-3})
print("neg apprec exit", b["exit_tax"]["total"], "wealth", round(b["enrichissement_net"]))
# zero rent
R0=prop("R0",100000,0)
df,b=run(strat(R0),5)
print("zero rent tax", df["Impôt Dû"].round(1).tolist(), b["tri_annuel"], b["tri_warning"])
df,b=run(strat(R0),5,{"regime_fiscal":"microbic"})
print("zero rent micro tax", df["Impôt Dû"].round(1).tolist())
# horizon 1
df,b=run(strat(Z),1,{"apply_ira":True}); print("H1", round(b["impot_pv"]), round(b["tri_annuel"],2), b["ira_total"])
# horizon > loan with IRA
df,b=run(strat(prop("L",100000,600,years=15)),20,{"apply_ira":True}); print("H20 loan15 dette", df.iloc[-1]["Dette"], b["ira_total"])
