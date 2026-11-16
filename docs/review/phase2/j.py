import sys, json; sys.path.insert(0, "/Users/mperrier/immo_app")
import logging; logging.disable(logging.CRITICAL)
from src.application.run_context import RunContext
from src.ui.app_controller import run_strategy_search, simulate_selected_strategy
arch=json.load(open("/Users/mperrier/immo_app/data/archetypes_2026_v1.json"))
for tf in (0.0, 3.0, 10.0):
    ctx=RunContext(apport=100000, cf_cible=-1500, max_properties=2, horizon_ans=20, tf_growth_pct=tf)
    S=run_strategy_search(arch, ctx); s=S[0]
    df=simulate_selected_strategy(s, ctx)
    print(tf, [d["nom_bien"][:15] for d in s["details"]], "card TRI", round(s["tri_annuel"],2), "card liq", round(s["liquidation_nette"]), "detail CF sum", round(df["Cash-Flow Net d'Impôt"].sum()))
