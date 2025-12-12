"""Main Application Entry Point.

Refactored to use modular layered architecture (Phases 1-5).
Orchestrates UI components and services.
"""

import json
import os
import sys
from typing import Dict, Any, List

import streamlit as st
import pandas as pd

# Add src to path if not present (for running executing from root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.logging import get_logger
from src.ui.state import SessionManager, BaseParams
from src.ui.components.sidebar import render_objectives_section, render_credit_params_tab
from src.ui.components.filters import render_property_filters, filter_archetypes
from src.ui.pages.main import render_main_page
from src.services.brick_factory import create_investment_bricks, FinancingConfig, OperatingConfig
from src.services.strategy_finder import StrategyFinder
from src.models.archetype import ArchetypeV2


# --- Configuration & Setup ---

st.set_page_config(
    page_title="Simulateur Immo v27.6",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "src/ui/assets/style.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


log = get_logger(__name__)


def load_data() -> List[Dict[str, Any]]:
    """Charge les données des archétypes."""
    try:
        archetypes = SessionManager.get_archetypes()
        if not archetypes:
            path = os.path.join(os.path.dirname(__file__), "data/archetypes_recale_2025_v2.json")
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                archetypes = [ArchetypeV2(**item).model_dump() for item in raw_data]
            
            SessionManager.set_archetypes(archetypes)
            log.info("archetypes_loaded_from_disk", count=len(archetypes))
        else:
            log.info("archetypes_loaded_from_session", count=len(archetypes))
            
        return archetypes
    except Exception as e:
        log.error("data_load_failed", error=str(e))
        st.error(f"Erreur lors du chargement des données: {e}")
        return []


def apply_compliance(archetypes: list, apply_cap: bool = True) -> list:
    """Apply regulatory compliance (rent caps) to archetypes."""
    processed = []
    for item in archetypes:
        a = item.copy()
        if apply_cap and a.get("soumis_encadrement") and a.get("loyer_m2_max") is not None:
            try:
                cap = float(a["loyer_m2_max"])
                current = float(a.get("loyer_m2", 0.0))
                a["loyer_m2"] = min(current, cap)
            except (ValueError, TypeError):
                pass
        processed.append(a)
    return processed


def main():
    # 1. Initialization
    SessionManager.initialize()
    log.info("app_started")
    
    # 2. Sidebar Configuration
    with st.sidebar:
        st.title("⚙️ Paramètres")
        
        # Objectives
        apport, cf_cible, tolerance, mode_cf, qual_weight, horizon = render_objectives_section()
        
        # Store basic params to session
        # We update SessionManager/BaseParams implicitly via the render function? 
        # Actually render_objectives_section returns values, we should store them if needed for persistence across reruns 
        # or use them directly.
        # But SessionManager has setters.
        # SessionManager.set_horizon(horizon) # Removed: Widget with key handles state automatically
        
        # Credit & Costs
        with st.expander("Financement & Frais", expanded=False):
            # Tabs for Credit vs Costs
            t1, t2 = st.tabs(["Crédit", "Exploitation"])
            with t1:
                credit_params = render_credit_params_tab()
            with t2:
                cse_params = st.number_input("CFE (€/an)", 100, 2000, 500, 50)
                gestion = st.slider("Gestion (%)", 0.0, 15.0, 5.0, 0.5)
                vacance = st.slider("Vacance/Imp. (%)", 0.0, 10.0, 3.0, 0.5)
                frais_vente = st.slider("Frais Revente (%)", 0.0, 10.0, 6.0, 0.5)
                
        # Tax
        with st.expander("Fiscalité", expanded=False):
            tmi = st.slider("TMI (%)", 0, 45, 30, 1)
            regime = st.selectbox("Régime", ["LMNP", "SCI IS"], index=0).lower().replace(" ", "")
            if "lmnp" in regime:
                 regime = "lmnp" # Normalize
            
    # 3. Main Content - Filtering
    raw_archetypes = load_data()
    if not raw_archetypes:
        return

    # Filters
    selected_villes, selected_types, apply_cap = render_property_filters(raw_archetypes)
    
    # Apply filters & compliance
    filtered_archetypes = filter_archetypes(
        raw_archetypes, 
        selected_villes, 
        selected_types
    )
    compliant_archetypes = apply_compliance(filtered_archetypes, apply_cap)
    
    st.write(f"*{len(compliant_archetypes)} biens éligibles sélectionnés*")
    
    # 4. Action Button
    if st.sidebar.button("🚀 Lancer l'analyse", type="primary"):
        with st.spinner("Analyse des stratégies en cours..."):
            log.info("analysis_started", 
                     archetypes_count=len(compliant_archetypes),
                     horizon=horizon)
            
            # 1. Création des briques
            fin_config = FinancingConfig(
                credit_rates=credit_params["taux_credits"],
                frais_notaire_pct=credit_params["frais_notaire_pct"],
                apport_min_pct=10.0, # Defaulting hardcoded for now or use session? Legacy used formula.
                # Legacy used inputs from sidebar which are in credit_params? 
                # Sidebar credit_params doesn't assume apport_min_pct selection, usually calculated.
                # Legacy code: apport_min = frais_notaire + prix * (apport_min_pct_prix / 100)
                # Where is apport_min_pct provided? Default was usually 0 or user provided? 
                # Let's verify legacy 'trouver_top_strategies' signature or 'creer_briques'.
                # creer_briques took 'apport_min_pct_prix'. 
                # Usually it was hardcoded or env var. Let's use 0.0 (110% loan) or 10.0.
                assurance_ann_pct=credit_params["assurance_ann_pct"],
                frais_pret_pct=credit_params["frais_pret_pct"],
                inclure_travaux=credit_params["inclure_travaux"],
                inclure_reno_ener=credit_params["inclure_reno_ener"],
                inclure_mobilier=credit_params["inclure_mobilier"],
                financer_mobilier=credit_params["financer_mobilier"],
            )
            
            # Using 10% apport min for safety/realism
            fin_config.apport_min_pct = 0.0 # Legacy often assumed project cost full loan for calculation base
            
            op_config = OperatingConfig(
                frais_gestion_pct=gestion,
                provision_pct=vacance,
                cfe_par_bien_ann=cse_params,
            )
            
            bricks = create_investment_bricks(compliant_archetypes, fin_config, op_config)
            
            # B. Strategy Search
            finder = StrategyFinder(
                bricks=bricks,
                apport_disponible=apport,
                cash_flow_cible=cf_cible,
                tolerance=tolerance,
                qualite_weight=qual_weight,
                mode_cf=mode_cf,
            )
            
            eval_params = {
                "tmi_pct": tmi,
                "regime_fiscal": regime,
                "frais_vente_pct": frais_vente,
                "apply_ira": credit_params["apply_ira"],
                "ira_cap_pct": credit_params["ira_cap_pct"],
                "cfe_par_bien_ann": cse_params,
                # Simple hypotheses for now
                "hypotheses_marche": {
                    "appreciation_bien_pct": 2.5,
                    "revalo_loyer_pct": 1.5,
                    "inflation_charges_pct": 2.0,
                },
                "finance_preset_name": "Équilibré (défaut)", # Could add selector
            }
            
            strategies = finder.find_strategies(
                eval_params=eval_params,
                horizon_years=horizon,
            )
            
            # 3. Sauvegarde
            SessionManager.set_strategies(strategies)
            
            # Auto-save to results/
            try:
                from src.services.exporter import ResultExporter
                exporter = ResultExporter()
                exporter.save_results(
                    strategies, 
                    metadata={"horizon": horizon, "compliance": use_compliance}
                )
            except Exception as e:
                log.warning("autosave_failed", error=str(e))
            
            log.info("analysis_completed", 
                     strategies_found=len(strategies))
            st.rerun()

    # 5. Render Results
    strategies = SessionManager.get_strategies()
    
    # Selected Strategy Simulation Data (lazy load/sim)
    idx = SessionManager.get_selected_idx()
    df_sim = None
    if strategies and 0 <= idx < len(strategies):
        # We need to simulate the detailed yearly data for the selected strategy
        # StrategyFinder only returned metrics. Run simulation for charts.
        
        # Re-instantiate engine locally (lightweight)
        # Or better: check if strategy dict already has 'df'? No it doesn't.
        # We need to run simulation again for the specialized charts.
        # For efficiency, we can do it here.
        from src.core.simulation import SimulationEngine, MarketHypotheses, TaxParams, IRACalculator
        from src.core.financial import generate_amortization_schedule
        
        sel_strat = strategies[idx]
        
        # Re-construct params (should be saved in session? ideally yes)
        # For now, using defaults or simple re-sim
        mk = MarketHypotheses(appreciation_bien_pct=2.5, revalo_loyer_pct=1.5, inflation_charges_pct=2.0)
        tx = TaxParams(tmi_pct=tmi, regime_fiscal=regime)
        ir = IRACalculator(apply_ira=credit_params["apply_ira"], ira_cap_pct=credit_params["ira_cap_pct"])
        eng = SimulationEngine(market=mk, tax=tx, ira=ir, cfe_par_bien_ann=cse_params, frais_vente_pct=frais_vente)
        
        schedules = [
             generate_amortization_schedule(
                float(p["credit_final"]), 
                float(p["taux_pret"]), 
                int(p["duree_pret"]), 
                float(p["assurance_ann_pct"])
            ) for p in sel_strat["details"]
        ]
        
        df_sim, _ = eng.simulate(sel_strat, horizon, schedules)
        
    render_main_page(compliant_archetypes, strategies, df_sim)


if __name__ == "__main__":
    main()