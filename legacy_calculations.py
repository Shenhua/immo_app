# Fichier: financial_calculations.py
import numpy_financial as npf
import pandas as pd
from config import TAUX_IR_PLUS_VALUE, TAUX_PRELEVEMENTS_SOCIAUX


def mensualite_et_assurance(capital: float, taux_annuel_pct: float, duree_ans: int, assurance_ann_pct: float):
    if capital <= 0 or duree_ans <= 0: return 0.0, 0.0, 0.0
    r = (taux_annuel_pct / 100.0) / 12.0; n = duree_ans * 12
    pmt_base = (capital / n) if r == 0 else capital * r / (1.0 - (1.0 + r)**(-n))
    pmt_assur = capital * (assurance_ann_pct / 100.0) / 12.0
    # CORRECTION DE LA NameError ICI
    return pmt_base, pmt_assur, pmt_base + pmt_assur

def k_factor(taux_annuel_pct: float, duree_ans: int, assurance_ann_pct: float) -> float:
    r = (taux_annuel_pct / 100.0) / 12.0; n = max(1, duree_ans * 12)
    base = (1.0 / n) if r == 0 else r / (1.0 - (1.0 + r)**(-n))
    assur = (assurance_ann_pct / 100.0) / 12.0
    return base + assur

def echeancier_mensuel(capital: float, taux_annuel_pct: float, duree_ans: int, assurance_ann_pct: float):
    r = (taux_annuel_pct / 100.0) / 12.0; n = duree_ans * 12
    pmt_base, pmt_assur, pmt_total = mensualite_et_assurance(capital, taux_annuel_pct, duree_ans, assurance_ann_pct)
    interets, principals, balances = [], [], []; balance = capital
    for _ in range(n):
        interet = balance * r; principal = max(0.0, pmt_base - interet); balance = max(0.0, balance - principal)
        interets.append(interet); principals.append(principal); balances.append(balance)
    return {"pmt_base": pmt_base, "pmt_assur": pmt_assur, "pmt_total": pmt_total, "interets": interets, "principals": principals, "balances": balances, "nmois": n}

def _calculer_impot_plus_value(duree_detention: int, plus_value_brute: float) -> float:
    if plus_value_brute <= 0: return 0.0
    abattement_ir = 0
    if duree_detention > 5:
        abattement_ir += sum(6 for _ in range(6, min(duree_detention, 21) + 1))
        if duree_detention >= 22: abattement_ir += 4
    base_imposable_ir = plus_value_brute * (1 - min(100, abattement_ir) / 100.0)
    impot_ir = base_imposable_ir * (TAUX_IR_PLUS_VALUE / 100.0)
    abattement_ps = 0
    if duree_detention > 5:
        abattement_ps += sum(1.65 for _ in range(6, min(duree_detention, 21) + 1))
        if duree_detention >= 22: abattement_ps += 1.60
        if duree_detention > 22: abattement_ps += sum(9 for _ in range(23, min(duree_detention, 30) + 1))
    base_imposable_ps = plus_value_brute * (1 - min(100, abattement_ps) / 100.0)
    impot_ps = base_imposable_ps * (TAUX_PRELEVEMENTS_SOCIAUX / 100.0)
    return impot_ir + impot_ps

def simuler_strategie_long_terme(
    strategie: dict,
    duree_simulation_ans: int,
    hypotheses_marche: dict,
    tmi_pct: float,
    frais_vente_pct: float,
    cfe_par_bien_ann: float,
    micro_bic_abatt_pct: float = 50.0,
    regime_fiscal: str = "lmnp",
    amort_mobilier: bool = True,
    lmnp_amort_mob: bool = True,
    horizon_years: int = 25,
    apply_ira: bool = True,
    ira_cap_pct: float = 3.0,
):
    H = int(duree_simulation_ans if duree_simulation_ans is not None else (horizon_years if horizon_years is not None else 25))
    if not strategie or not strategie.get("details"): return pd.DataFrame(), {}
    projets = [p.copy() for p in strategie["details"]]
    schedules = [echeancier_mensuel(p["credit_final"], p["taux_pret"], p["duree_pret"], p["assurance_ann_pct"]) for p in projets]
    valeur_biens = sum(p["prix_achat_bien"] for p in projets)
    flux = [-float(strategie["apport_total"])]
    res, tot_imp, deficit_reportable = [], 0.0, 0.0

    for year in range(1, H + 1):
        if year > 1: valeur_biens *= (1.0 + hypotheses_marche["appreciation_bien_pct"] / 100.0)

        cf_ann, dette_fin, capital_rembourse_ann, interets_assurance_ann = 0.0, 0.0, 0.0, 0.0
        loyers_bruts_pf, charges_deductibles_pf, amortissements_pf = 0.0, 0.0, 0.0

        for i, p in enumerate(projets):
            if year == 1:
                p["loyer_mth_courant"], p["charges_fix_mth_courant"] = p["loyer_mensuel_initial"], p["charges_const_mth0"] + p["tf_const_mth0"]
            else:
                p["loyer_mth_courant"] *= (1.0 + hypotheses_marche["revalo_loyer_pct"] / 100.0)
                p["charges_fix_mth_courant"] *= (1.0 + hypotheses_marche["inflation_charges_pct"] / 100.0)

            loyer_ann = p["loyer_mth_courant"] * 12
            gest_ann, prov_ann = loyer_ann * (p["frais_gestion_pct"] / 100.0), loyer_ann * (p["provision_pct"] / 100.0)
            charges_fixes_ann = p["charges_fix_mth_courant"] * 12

            sch = schedules[i]; start, end = (year - 1) * 12, min(year * 12, sch["nmois"]); months = max(0, end - start)

            interets_emprunt_ann = sum(sch["interets"][start:end]) if months > 0 else 0.0
            capital_rembourse_ann += sum(sch["principals"][start:end]) if months > 0 else 0.0
            assurance_emprunt_ann = months * sch["pmt_assur"]
            debt_service_ann = months * sch["pmt_total"]
            solde_fin = sch["balances"][end - 1] if months > 0 else sch["balances"][-1] if sch["nmois"] > 0 and year * 12 > sch["nmois"] else (p["credit_final"] if sch["nmois"] == 0 else 0.0)

            loyers_bruts_pf += loyer_ann
            charges_deductibles_pf += charges_fixes_ann + gest_ann + prov_ann + interets_emprunt_ann + assurance_emprunt_ann + cfe_par_bien_ann

            amort_immo_ann = (p["prix_achat_bien"] + p["budget_travaux"] + p.get("renovation_energetique", 0.0)) / 30.0
            amort_mob_ann = p["mobilier"] / 10.0
            amort_immo_on = (regime_fiscal == "lmnp")  # LMNP réel: amortissement immo
            amortissements_pf += (amort_immo_ann if amort_immo_on else 0.0) + (amort_mob_ann if lmnp_amort_mob else 0.0)

            cf_ann += loyer_ann - (charges_fixes_ann + gest_ann + prov_ann) - debt_service_ann - cfe_par_bien_ann
            interets_assurance_ann += interets_emprunt_ann + assurance_emprunt_ann
            dette_fin += solde_fin

        if regime_fiscal == "microbic": base_imposable = max(0.0, loyers_bruts_pf * (1.0 - micro_bic_abatt_pct / 100.0))
        else:
            resultat_brut = loyers_bruts_pf - charges_deductibles_pf - amortissements_pf
            resultat_apres_report = resultat_brut + deficit_reportable
            base_imposable = max(0.0, resultat_apres_report); deficit_reportable = min(0.0, resultat_apres_report)

        impot_total = (base_imposable * (tmi_pct / 100.0)) + (base_imposable * (TAUX_PRELEVEMENTS_SOCIAUX / 100.0))
        tot_imp += impot_total; cf_ann -= impot_total; flux.append(cf_ann)

        res.append({
            "Année": year, "Valeur Biens": valeur_biens, "Dette": max(0.0, dette_fin), "Patrimoine Net": valeur_biens - dette_fin,
            "Résultat Fiscal": base_imposable, "Impôt Dû": impot_total, "Cash-Flow Net d'Impôt": cf_ann,
            "Capital Remboursé": capital_rembourse_ann, "Intérêts & Assurance": interets_assurance_ann,
            "Loyers Bruts": loyers_bruts_pf, "Charges Déductibles": -charges_deductibles_pf, "Amortissements": -amortissements_pf
        })

    frais_vente = valeur_biens * (frais_vente_pct / 100.0); dette_finale = res[-1]["Dette"] if res else 0.0
    prix_acquisition_brut = sum(p["prix_achat_bien"] for p in projets); frais_acquisition = sum(p["frais_notaire"] for p in projets); cout_travaux = sum(p["budget_travaux"] + p.get("renovation_energetique", 0.0) for p in projets)
    prix_acquisition_corrige = prix_acquisition_brut + frais_acquisition + cout_travaux
    plus_value_brute = max(0, valeur_biens - prix_acquisition_corrige); impot_pv = _calculer_impot_plus_value(H, plus_value_brute)
    liquidation_nette = valeur_biens - dette_finale - frais_vente - impot_pv

    # --- IRA at horizon (per-bien) ---
    ira_total = 0.0
    if apply_ira:
        for i, p in enumerate(projets):
            duree_pret = int(p.get("duree_pret", 0))
            if H < duree_pret:
                sch = schedules[i]
                nmois = int(sch.get("nmois", 0))
                mH = min(H * 12, nmois)
                # CRD at H years (month index mH-1); if H==0 fallback to initial capital
                crd_H = float(sch["balances"][mH - 1]) if mH > 0 else float(p.get("credit_final", 0.0))
                monthly_rate = (float(p.get("taux_pret", 0.0)) / 100.0) / 12.0
                cap_pct_fee = (float(ira_cap_pct) / 100.0) * crd_H
                six_months_interest_fee = 6.0 * monthly_rate * crd_H
                ira_i = min(cap_pct_fee, six_months_interest_fee)
                ira_total += max(0.0, ira_i)

    liquidation_nette -= ira_total
    if len(flux) >= 2: flux[-1] += liquidation_nette
    else: flux.append(liquidation_nette)

    tri_annuel = float(npf.irr(flux)) * 100.0 if npf.irr(flux) is not None else 0.0
    bilan = {
        "tri_annuel": tri_annuel,
        "liquidation_nette": liquidation_nette,
        "ira_total": ira_total,
    }

    return pd.DataFrame(res), bilan

def _accept_cf(cf_observe: float, cf_cible: float, tol: float, mode: str) -> bool:
    return (cf_observe >= (cf_cible - tol)) if mode == "min" else (abs(cf_observe - cf_cible) <= tol)
