import sys; sys.path.insert(0, "/Users/mperrier/immo_app")
from src.application.services.simulation import simulate_long_term_strategy
def prop(name, price, rent, meuble=True, loan=None, rate=3.5, years=20, works=0.0, energy=0.0, furn=0.0, land=15.0,
         copro=1200, tf=1000, mgmt=0.0, vac=0.0, mois=0, maint=0.0, gli=0.0, surface=40):
    notary = price*0.08
    cost = price+notary+works+energy+furn
    loan = cost*0.9 if loan is None else loan
    return dict(nom_bien=name, prix_achat_bien=price, frais_notaire=notary, budget_travaux=works, renovation_energetique=energy,
        mobilier=furn, ratio_terrain_pct=land, credit_final=loan, taux_pret=rate, duree_pret=years, assurance_ann_pct=0.3,
        loyer_mensuel_initial=rent, charges_const_mth0=copro/12, tf_const_mth0=tf/12, frais_gestion_pct=mgmt, provision_pct=vac,
        meuble=meuble, mois_travaux=mois, maintenance_m2_an=maint, gli_pct=gli, surface=surface), cost-loan
def strat(*ps):
    return {"details":[p for p,_ in ps], "apport_total": sum(a for _,a in ps)}
def run(s, H, tax=None, market=None):
    m = {"appreciation_bien_pct":2.0,"revalo_loyer_pct":0.0,"inflation_charges_pct":0.0, "tf_growth_pct":0.0}
    m.update(market or {})
    t = {"tmi_pct":30.0,"apply_ira":False,"cfe_par_bien_ann":300,"frais_vente_pct":5.0}
    t.update(tax or {})
    return simulate_long_term_strategy(s, H, m, t)
