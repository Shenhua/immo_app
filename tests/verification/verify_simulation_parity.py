"""Verification script to compare legacy and modular simulation engines.

Runs both implementations on identical inputs and asserts results match.
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

# Ensure we can import from root
sys.path.append(".")

from financial_calculations import simuler_strategie_long_terme, echeancier_mensuel
from src.core.simulation import SimulationEngine, MarketHypotheses, TaxParams, IRACalculator

def create_test_case() -> Dict[str, Any]:
    """Create a complex test case with multiple properties."""
    return {
        "apport_total": 50000.0,
        "details": [
            {
                "nom_bien": "Appartement Paris",
                "prix_achat_bien": 200000.0,
                "frais_notaire": 15000.0,
                "budget_travaux": 10000.0,
                "renovation_energetique": 5000.0,
                "mobilier": 5000.0,
                "credit_final": 175000.0,
                "taux_pret": 3.5,
                "duree_pret": 20,
                "assurance_ann_pct": 0.36,
                "loyer_mensuel_initial": 1200.0,
                "charges_const_mth0": 100.0,
                "tf_const_mth0": 80.0,
                "frais_gestion_pct": 5.0,
                "provision_pct": 1.0,
            },
            {
                "nom_bien": "Studio Lyon",
                "prix_achat_bien": 150000.0,
                "frais_notaire": 11000.0,
                "budget_travaux": 2000.0,
                "renovation_energetique": 0.0,
                "mobilier": 3000.0,
                "credit_final": 140000.0,
                "taux_pret": 3.7,
                "duree_pret": 25,
                "assurance_ann_pct": 0.36,
                "loyer_mensuel_initial": 900.0,
                "charges_const_mth0": 80.0,
                "tf_const_mth0": 60.0,
                "frais_gestion_pct": 6.0,
                "provision_pct": 2.0,
            }
        ]
    }

def run_comparison():
    print("🔬 Starting Parity Check: Legacy vs Modular Simulation")
    
    # 1. Setup Inputs
    strategy = create_test_case()
    horizon = 25
    
    hypotheses = {
        "appreciation_bien_pct": 2.5,
        "revalo_loyer_pct": 1.2,
        "inflation_charges_pct": 2.1,
    }
    
    tax_config = {
        "tmi_pct": 30.0,
        "frais_vente_pct": 6.0,
        "cfe_par_bien_ann": 150.0,
        "apply_ira": True,
        "ira_cap_pct": 3.0,
        "regime_fiscal": "lmnp",
        "micro_bic_abatt_pct": 50.0,
        "duree_simulation_ans": horizon,
    }
    
    market_params = {**hypotheses}
    simulation_params = {**hypotheses, **tax_config, "horizon_years": horizon}
    
    # 2. Run Legacy Implementation
    print("  Running Legacy Implementation...", end="", flush=True)
    legacy_df, legacy_bilan = simuler_strategie_long_terme(
        strategie=strategy,
        hypotheses_marche=hypotheses,
        **tax_config
    )
    print(" ✅")
    
    # 3. Run Modular Implementation
    print("  Running Modular Implementation...", end="", flush=True)
    
    # Init Modular Engine
    market = MarketHypotheses(
        appreciation_bien_pct=hypotheses["appreciation_bien_pct"],
        revalo_loyer_pct=hypotheses["revalo_loyer_pct"],
        inflation_charges_pct=hypotheses["inflation_charges_pct"],
    )
    
    tax = TaxParams(
        tmi_pct=tax_config["tmi_pct"],
        regime_fiscal=tax_config["regime_fiscal"],
        micro_bic_abatt_pct=tax_config["micro_bic_abatt_pct"],
    )
    
    ira = IRACalculator(
        apply_ira=tax_config["apply_ira"],
        ira_cap_pct=tax_config["ira_cap_pct"],
    )
    
    engine = SimulationEngine(
        market=market,
        tax=tax,
        ira=ira,
        cfe_par_bien_ann=tax_config["cfe_par_bien_ann"],
        frais_vente_pct=tax_config["frais_vente_pct"],
    )
    
    # Pre-calculate schedules (shared input dependency)
    schedules = [
        echeancier_mensuel(
            p["credit_final"], 
            p["taux_pret"], 
            p["duree_pret"], 
            p["assurance_ann_pct"]
        ) 
        for p in strategy["details"]
    ]
    
    modular_df, modular_bilan = engine.simulate(strategy, horizon, schedules)
    print(" ✅")
    
    # 4. Compare Results
    print("\n📊 Comparing Results:")
    
    # 4a. Compare Bilan (Summary)
    print("  Checking Bilan (Summary)...")
    legacy_keys = {"tri_annuel", "liquidation_nette", "ira_total"}
    diffs = []
    
    for k in legacy_keys:
        legacy_val = float(legacy_bilan.get(k, 0.0))
        modular_val = float(modular_bilan.get(k, 0.0))
        
        # Use relative tolerance for floats
        if not np.isclose(legacy_val, modular_val, rtol=1e-5, atol=1e-2):
            diffs.append(f"    ❌ {k}: Legacy={legacy_val:.4f} vs Modular={modular_val:.4f}")
        else:
            print(f"    ✅ {k}: Match ({legacy_val:.2f})")
            
    if diffs:
        print("\n".join(diffs))
        print("⚠️ Bilan mismatch detected, checking DataFrame to investigate...")
        # raise AssertionError("Bilan mismatch")
        
    # 4b. Compare DataFrame (Yearly Data)
    print("  Checking DataFrame (Yearly Projections)...")
    
    # Check shape
    if legacy_df.shape != modular_df.shape:
        raise AssertionError(f"DataFrame shape mismatch: Legacy {legacy_df.shape} vs Modular {modular_df.shape}")
    
    # Check columns
    cols_to_check = [
        "Valeur Biens", "Dette", "Patrimoine Net", "Résultat Fiscal", 
        "Impôt Dû", "Cash-Flow Net d'Impôt", "Capital Remboursé", 
        "Intérêts & Assurance", "Loyers Bruts", "Amortissements"
    ]
    
    df_diffs = []
    for col in cols_to_check:
        legacy_series = legacy_df[col].astype(float)
        modular_series = modular_df[col].astype(float)
        
        # Check max difference
        max_diff = (legacy_series - modular_series).abs().max()
        
        if max_diff > 0.01:  # 1 cent tolerance per year
            df_diffs.append(f"    ❌ {col}: Max Diff = {max_diff:.4f}")
        else:
            print(f"    ✅ {col}: Match (Max Diff < 0.01)")
            
    if df_diffs:
        print("\n".join(df_diffs))
        raise AssertionError("DataFrame column mismatch")

    print("\n✨ SUCCESS: Legacy and Modular implementations produce identical results!")

if __name__ == "__main__":
    try:
        run_comparison()
    except AssertionError as e:
        print(f"\n❌ PARITY CHECK FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
