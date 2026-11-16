import sys, json; sys.path.insert(0, "/Users/mperrier/immo_app")
import logging; logging.disable(logging.CRITICAL)
from src.application.run_context import RunContext
from src.application.services.brick_factory import build_financing_config, build_operating_config, create_investment_bricks
from src.application.services.simulation import simulate_long_term_strategy
from src.core.glossary import calculate_dscr_metric
arch=json.load(open("/Users/mperrier/immo_app/data/archetypes_2026_v1.json"))
ctx=RunContext()
print("works archetypes:", sum(1 for a in arch if (a.get("budget_travaux",0) or 0)+(a.get("renovation_energetique_cout",0) or 0)>0), "/", len(arch))
B=create_investment_bricks(arch, build_financing_config(ctx.to_credit_params()), build_operating_config(ctx.gestion_pct, ctx.vacance_pct, ctx.cfe_par_bien_ann, ctx.maintenance_m2_an, ctx.gli_pct, ctx.mois_travaux))
seen=set()
for b in B:
    d=b.model_dump()
    if d["nom_bien"] in seen or d["mois_travaux"]==0: continue
    seen.add(d["nom_bien"])
    d["credit_final"]=d["capital_emprunte"]
    s={"details":[d],"apport_total":d["apport_min"]}
    df,bl=simulate_long_term_strategy(s, 20, ctx.market_hypotheses, ctx.to_tax_params())
    print(f'{d["nom_bien"][:28]:28s} rent {d["loyer_mensuel_initial"]:6.0f} alloc cf0 {d["loyer_mensuel_initial"]-d["depenses_mensuelles_hors_credit_initial"]-d["pmt_total"]:7.0f} simCF y1 {df.iloc[0]["Cash-Flow Net d\'Impôt"]/12:7.0f} y2 {df.iloc[1]["Cash-Flow Net d\'Impôt"]/12:7.0f} dscr y1 {bl["dscr_y1"]:.2f} y2 {calculate_dscr_metric(df.iloc[1:2]):.2f}')
    if len(seen)>5: break
from src.application.services.allocator import PortfolioAllocator
from src.application.services.evaluator import StrategyEvaluator
from src.application.services.simulation import SimulationEngine, MarketHypotheses, TaxParams, IRACalculator
eng=SimulationEngine(MarketHypotheses(**{k:v for k,v in ctx.market_hypotheses.items()}), TaxParams(), IRACalculator())
alloc=PortfolioAllocator("min")
ev=StrategyEvaluator(eng, alloc, None)
b=[x for x in B if x.nom_bien.startswith("T4 pour Colocation")][0]
target=-1700
ok,det,cf0,ap=alloc.allocate([b], 200000, target, 50)
valid,m=ev.evaluate({"details":det,"apport_total":ap}, target, 50, 20)
print("alloc ok",ok,"cf0",round(cf0),"apport",round(ap),"| eval valid",valid,"cf_y1",round(m["cf_monthly_y1"]),"cf_avg5",round(m["cf_monthly_avg"]))
