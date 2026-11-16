from h import *
A=prop("A",200000,1100,meuble=False)
B=prop("B",150000,900,meuble=True,furn=5000)
for H in (10,20,35):
    df,b=run(strat(A,B),H,{"sci_is":True})
    da,ba=run(strat(A),H,{"sci_is":True}); db,bb=run(strat(B),H,{"sci_is":True})
    print(H, b["regimes"], "IS yearly", round(df["Impôt Dû"].sum()), "sep", round(da["Impôt Dû"].sum()+db["Impôt Dû"].sum()),
      "exit", {k:round(v) for k,v in b["exit_tax"].items()}, "sep exit", round(ba["impot_pv"]+bb["impot_pv"]))
    pv=b["plus_value"]; print("   vnc", round(pv["net_book_value"]), "pv", round(pv["plus_value_pro"]), "boni", round(pv["boni_de_liquidation"]), "wealth", round(b["enrichissement_net"]), "liq", round(b["liquidation_nette"]), "cumCF", round(df["Cash-Flow Net d'Impôt"].sum()))
