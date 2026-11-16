from h import *
# micro-BIC ceiling across properties: 3 x 2500/month = 90k > 83.6k
P=[prop(f"F{i}", 400000, 2500, furn=10000) for i in range(3)]
df,b=run(strat(*P),10,{"regime_fiscal":"microbic"})
print("B1", b["regimes"], b["tax_warnings"])
df,b=run(strat(*P[:2]),10,{"regime_fiscal":"microbic"})
print("B2", b["regimes"], b["tax_warnings"][:1], "y2 tax", round(df.iloc[1]["Impôt Dû"]), "expected", round(60000*0.5*(0.3+0.186)))
# micro-foncier ceiling: 2 x 700 = 16.8k > 15k
U=[prop(f"U{i}", 100000, 700, meuble=False) for i in range(2)]
df,b=run(strat(*U),10,{"regime_foncier":"micro"})
print("B3", b["regimes"], b["tax_warnings"])
# micro foncier below ceiling w/ vacancy: 2x650 with 5% vacancy => 14820 ok ; indexation 1.5% pushes over by year H
U=[prop(f"U{i}", 100000, 600, meuble=False, vac=5) for i in range(2)]
df,b=run(strat(*U),20,{"regime_foncier":"micro"},{"revalo_loyer_pct":1.5})
print("B4", b["regimes"], b["tax_warnings"])
# micro-BIC with works months & vacancy: base on receipts net
F=prop("F",150000,1000,furn=5000,works=20000,vac=5,mois=3)
df,b=run(strat(F),10,{"regime_fiscal":"microbic"})
print("B5 y1 rent",round(df.iloc[0]["Loyers Bruts"]), "exp", 1000*9*0.95, "tax", round(df.iloc[0]["Impôt Dû"]), "exp", round(8550*0.5*0.486), "y2 tax", round(df.iloc[1]["Impôt Dû"]), round(11400*.5*.486))
# low rent micro-BIC 305 min
F=prop("F",30000,50,furn=0)
df,b=run(strat(F),5,{"regime_fiscal":"microbic"})
print("B6 rent600 tax", round(df.iloc[0]["Impôt Dû"],2), "exp", round((600-305)*0.486,2))
