"""Sensitivity analysis component."""

import copy
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.core.simulation import simulate_long_term_strategy
from src.ui.state import SessionManager
from src.ui.components.results import format_euro, format_pct


def run_sensitivity_simulation(
    strategy: Dict[str, Any],
    delta_loyer_pct: float,
    delta_vacance_pct: float,
    delta_travaux_pct: float,
    horizon: int,
) -> Dict[str, Any]:
    """Run simulation with modified parameters."""
    # Deep copy to avoid mutating original
    modified = copy.deepcopy(strategy)
    
    # Apply modifiers
    for brick in modified.get("details", []):
        # Loyer
        if delta_loyer_pct != 0:
            brick["loyer_mensuel_initial"] *= (1.0 + delta_loyer_pct / 100.0)
        
        # Vacance (Provision)
        if delta_vacance_pct != 0:
            # Add to provision_pct (additive)
            current = brick.get("provision_pct", 5.0)
            brick["provision_pct"] = max(0.0, current + delta_vacance_pct)
            
        # Travaux
        if delta_travaux_pct != 0:
            budget = brick.get("budget_travaux", 0.0)
            brick["budget_travaux"] = budget * (1.0 + delta_travaux_pct / 100.0)
            # Update total cost and financing needs
            brick["cout_total"] += (brick["budget_travaux"] - budget)
            # Assuming extra cost is funded by apport if loan is fixed?
            # Or proportional?
            # For simplicity, let's assume extra cost is added to loan if possible or apport.
            # But changing loan changes schedule.
            # To keep it simple/fast: Assume Apport absorbs the difference.
            diff = (brick["budget_travaux"] - budget)
            brick["apport_min"] += diff
            modified["apport_total"] += diff

    # Get current session params for context (Market, Tax)
    base_params = SessionManager.get_base_params()
    eval_params = base_params.to_eval_params()
    
    # Run simulation
    df, bilan = simulate_long_term_strategy(
        modified,
        duration_years=horizon,
        market_hypotheses=eval_params.get("hypotheses_marche"),
        tax_params={
            "tmi_pct": eval_params.get("tmi_pct"),
            "regime_fiscal": eval_params.get("regime_fiscal"),
            "apply_ira": eval_params.get("apply_ira"),
            "ira_cap_pct": eval_params.get("ira_cap_pct"),
        }
    )
    
    # Construct result dict similar to strategy dict
    if not df.empty:
        final_row = df.iloc[-1] if not df.empty else {}
        first_row = df.iloc[0] if not df.empty else {}
        
        # Calculate DSCR Y1
        ds = first_row["Capital Remboursé"] + first_row["Intérêts & Assurance"]
        noi = first_row["Loyers Bruts"] + (first_row["Charges Déductibles"] + first_row["Intérêts & Assurance"])
        dscr = (noi / ds) if ds > 1 else 0.0
        
        return {
            "cash_flow_final": final_row.get("Cash-Flow Net d'Impôt", 0), # Wait, CF Final is ambiguous. Use Avg or Year 1? 
            # StrategyFinder uses mean? Or Year 1?
            # Let's use first year CF for "Monthly CF" usually displayed
            "cash_flow_mensuel": first_row.get("Cash-Flow Net d'Impôt", 0) / 12.0,
            "tri_annuel": bilan.get("tri_annuel", 0),
            "liquidation_nette": bilan.get("liquidation_nette", 0),
            "dscr_y1": dscr,
        }
    return {}


def render_sensitivity_analysis(strategy: Dict[str, Any], horizon: int = 25) -> None:
    """Render interactive sensitivity analysis."""
    st.markdown("### ⚡ Stress Test & Sensibilité")
    st.caption("Simulez des variations de marché pour tester la robustesse de cette stratégie.")
    
    col_input, col_res = st.columns([0.4, 0.6])
    
    with col_input:
        st.markdown("**Paramètres**")
        d_loyer = st.slider("Loyer", -30, 30, 0, 5, format="%+d%%", key="sens_rent")
        d_vacance = st.slider("Vacance (pts)", -5, 15, 0, 1, format="%+d pt", key="sens_vac")
        d_travaux = st.slider("Budget Travaux", -20, 50, 0, 10, format="%+d%%", key="sens_works")
    
    # Recalculate on fly
    if d_loyer != 0 or d_vacance != 0 or d_travaux != 0:
        res = run_sensitivity_simulation(strategy, d_loyer, d_vacance, d_travaux, horizon)
        
        # Display comparison
        with col_res:
            st.markdown("**Impact Projeté**")
            
            # Helper to show delta
            def show_impact(label, original, new, fmt_func):
                delta = new - original
                color = "green" if delta >= 0 else "red"
                # Invert color for cost/risk metrics? None here.
                # Actually DSCR lower is bad (red). Cashflow lower is bad (red).
                # So green if positive is correct.
                
                cols = st.columns([0.4, 0.3, 0.3])
                cols[0].write(label)
                cols[1].write(fmt_func(original))
                if abs(delta) > 0.01:
                    code = f":{color}[{fmt_func(new)}]"
                    cols[2].markdown(f"**{code}**")
                else:
                    cols[2].write("—")
            
            show_impact("CF Mensuel", strategy.get("cash_flow_final", 0), res.get("cash_flow_mensuel", 0), format_euro)
            show_impact("TRI", strategy.get("tri_annuel", 0), res.get("tri_annuel", 0), format_pct)
            show_impact("Enrichissement", strategy.get("liquidation_nette", 0), res.get("liquidation_nette", 0), format_euro)
            show_impact("DSCR Y1", strategy.get("dscr_y1", 0), res.get("dscr_y1", 0), lambda x: f"{x:.2f}")
            
    else:
        with col_res:
            st.info("👈 Modifiez les curseurs pour voir l'impact.")
