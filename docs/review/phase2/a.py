from h import *
F1 = prop("F1", 150000, 750, furn=5000)
F2 = prop("F2", 100000, 550, furn=4000, works=10000)
U1 = prop("U1", 120000, 600, meuble=False, works=30000)
for H in (8, 25, 35):
    df, b = run(strat(F1,F2,U1), H)
    dff, bf = run(strat(F1,F2), H)
    dfu, bu = run(strat(U1), H)
    tax_mixed = df["Impôt Dû"].sum(); tax_sep = dff["Impôt Dû"].sum()+dfu["Impôt Dû"].sum()
    print(H, "regimes", b["regimes"], "tax mixed", round(tax_mixed,2), "sep", round(tax_sep,2),
          "exit", round(b["impot_pv"],2), round(bf["impot_pv"]+bu["impot_pv"],2),
          "wealth", round(b["enrichissement_net"]), round(bf["enrichissement_net"]+bu["enrichissement_net"]),
          "tri", round(b["tri_annuel"],2))
    print("  y1 IR", round(df.iloc[0]["IR"],1), "PS", round(df.iloc[0]["PS"],1), "fiscal y1", {k:round(v) for k,v in b["fiscal_annuel"][0].items()})
    print("  exit", {k:round(v) for k,v in b["exit_tax"].items()})
    for pp in b["plus_value_par_bien"]: print("   ", pp["nom"], round(pp["plus_value_brute"]), round(pp["amortization_addback"]), round(pp["total_tax"]))
df,b=run(strat(F1,F2,U1),35)
print(df[["Année","Amortissements","Résultat Fiscal","Impôt Dû","Cash-Flow Net d'Impôt"]].round(0).to_string())
