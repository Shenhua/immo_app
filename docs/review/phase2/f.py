import sys, json; sys.path.insert(0, "/Users/mperrier/immo_app")
import logging; logging.disable(logging.CRITICAL)
from src.application.run_context import RunContext
from src.ui.app_controller import run_strategy_search, simulate_selected_strategy
from src.application.services.simulation import simulate_long_term_strategy
arch=json.load(open("/Users/mperrier/immo_app/data/archetypes_2026_v1.json"))
for kw in [dict(), dict(regime_fiscal="microbic", regime_foncier="micro"), dict(sci_is=True)]:
    ctx=RunContext(apport=100000, cf_cible=-1500, max_properties=3, tf_growth_pct=2.0, horizon_ans=20, **kw)
    S=run_strategy_search(arch, ctx)
    print(kw, len(S))
    for s in S[:4]:
        df,b=simulate_long_term_strategy(s, ctx.horizon_ans, ctx.market_hypotheses, ctx.to_tax_params())
        print("  ", [d["nom_bien"][:18] for d in s["details"]], "search enrich", round(s["enrich_net"]), "detail", round(b["enrichissement_net"]),
              "tri", round(s["tri_annuel"],2), round(b["tri_annuel"],2), "cf_final", round(s["cash_flow_final"]), "cf_y1 search", round(s.get("cf_monthly_y1",0)), "detail", round(df.iloc[0]["Cash-Flow Net d'Impôt"]/12))
