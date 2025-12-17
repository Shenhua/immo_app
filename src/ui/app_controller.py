"""Application controller - orchestrates UI and business logic.

Extracted from app.py to reduce main() complexity and improve testability.
Each function represents a distinct phase of the application flow.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.core.financial import generate_amortization_schedule
from src.core.logging import get_logger
from src.core.simulation import IRACalculator, MarketHypotheses, SimulationEngine, TaxParams
from src.models.archetype import ArchetypeV2
from src.services.brick_factory import FinancingConfig, OperatingConfig, apply_rent_caps, create_investment_bricks
from src.services.strategy_finder import StrategyFinder
from src.ui.state import SessionManager

log = get_logger(__name__)


def load_archetypes_from_disk(data_path: str) -> list[dict[str, Any]]:
    """Load archetype data from JSON file.

    Args:
        data_path: Path to archetypes JSON file

    Returns:
        List of archetype dictionaries

    Raises:
        FileNotFoundError: If data file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    with open(data_path, encoding="utf-8") as f:
        raw_data = json.load(f)
        archetypes = [ArchetypeV2(**item).model_dump() for item in raw_data]
    
    # Validate rent caps at load time
    cap_violations = 0
    for a in archetypes:
        if a.get("soumis_encadrement") and a.get("loyer_m2_max"):
            if a.get("loyer_m2", 0) > a["loyer_m2_max"]:
                log.warning("archetype_rent_exceeds_cap",
                           nom=a.get("nom"),
                           loyer=a["loyer_m2"],
                           cap=a["loyer_m2_max"])
                cap_violations += 1
    
    if cap_violations > 0:
        log.warning("rent_cap_violations_found", count=cap_violations)
    
    log.info("archetypes_loaded_from_disk", count=len(archetypes))
    return archetypes


def load_archetypes() -> list[dict[str, Any]]:
    """Load archetypes from session or disk.

    Returns:
        List of archetype dictionaries
    """
    try:
        archetypes = SessionManager.get_archetypes()
        if not archetypes:
            path = os.path.join(os.path.dirname(__file__), "../../data/archetypes_recale_2025_v2.json")
            path = os.path.normpath(path)
            archetypes = load_archetypes_from_disk(path)
            SessionManager.set_archetypes(archetypes)
        else:
            log.debug("archetypes_loaded_from_session", count=len(archetypes))
        return archetypes
    except Exception as e:
        log.error("data_load_failed", error=str(e))
        st.error(f"Erreur lors du chargement des données: {e}")
        return []




def build_financing_config(credit_params: dict[str, Any]) -> FinancingConfig:
    """Build FinancingConfig from credit parameters.

    Args:
        credit_params: Dict from sidebar credit tab

    Returns:
        FinancingConfig instance
    """
    config = FinancingConfig(
        credit_rates=credit_params["taux_credits"],
        frais_notaire_pct=credit_params["frais_notaire_pct"],
        apport_min_pct=0.0,  # Legacy: project cost full loan for calculation base
        assurance_ann_pct=credit_params["assurance_ann_pct"],
        frais_pret_pct=credit_params["frais_pret_pct"],
        inclure_travaux=credit_params["inclure_travaux"],
        inclure_reno_ener=credit_params["inclure_reno_ener"],
        inclure_mobilier=credit_params["inclure_mobilier"],
        financer_mobilier=credit_params["financer_mobilier"],
    )
    log.info("financing_config_built", rates=config.credit_rates)
    return config


def build_operating_config(gestion: float, vacance: float, cfe: float) -> OperatingConfig:
    """Build OperatingConfig from sidebar values.

    Args:
        gestion: Management fee percentage
        vacance: Vacancy/unpaid provision percentage
        cfe: Annual CFE per property

    Returns:
        OperatingConfig instance
    """
    return OperatingConfig(
        frais_gestion_pct=gestion,
        provision_pct=vacance,
        cfe_par_bien_ann=cfe,
    )


def run_strategy_search(
    archetypes: list[dict[str, Any]],
    fin_config: FinancingConfig,
    op_config: OperatingConfig,
    apport: float,
    cf_cible: float,
    tolerance: float,
    qual_weight: float,
    mode_cf: str,
    eval_params: dict[str, Any],
    horizon_years: int,
    top_n: int = 100,
) -> list[dict[str, Any]]:
    """Execute the strategy search pipeline.

    This is the core business logic extracted from main().

    Args:
        archetypes: Filtered and compliant archetypes
        fin_config: Financing configuration
        op_config: Operating configuration
        apport: Available down payment
        cf_cible: Target cash flow
        tolerance: Cash flow tolerance
        qual_weight: Qualitative score weight
        mode_cf: Cash flow mode
        eval_params: Evaluation parameters for simulation
        horizon_years: Simulation horizon
        top_n: Max number of strategies to return (Phase 17.1)

    Returns:
        List of ranked strategies
    """
    log.info("analysis_started", archetypes_count=len(archetypes), horizon=horizon_years, top_n=top_n)
    log.debug(
        "run_strategy_search_called",
        apport=apport,
        cf_cible=cf_cible,
        mode_cf=mode_cf,
        max_props=eval_params.get("max_properties"),
        use_full_capital=eval_params.get("use_full_capital"),
    )

    # 1. Create investment bricks
    bricks = create_investment_bricks(archetypes, fin_config, op_config)

    # 2. Find strategies
    finder = StrategyFinder(
        bricks=bricks,
        apport_disponible=apport,
        cash_flow_cible=cf_cible,
        tolerance=tolerance,
        qualite_weight=qual_weight,
        mode_cf=mode_cf,
        max_properties=eval_params.get("max_properties", 3)
    )

    # Allow use_full_capital from params
    use_full = eval_params.get("use_full_capital", False)
    
    strategies = finder.find_strategies(
        eval_params=eval_params,
        horizon_years=horizon_years,
        use_full_capital_override=use_full,
        top_n=top_n
    )

    log.info("analysis_completed", strategies_found=len(strategies))

    # Phase 19: Automatic Debug Context
    try:
        from src.utils.debug import collect_debug_context, save_debug_context
        
        ctx_params = {
            "apport": apport,
            "cf_cible": cf_cible,
            "tolerance": tolerance,
            "qual_weight": qual_weight,
            "mode_cf": mode_cf,
            "horizon_years": horizon_years,
            "top_n": top_n,
            "eval_params": eval_params,
        }
        
        ctx = collect_debug_context(
            params=ctx_params,
            strategies=strategies,
            log_file_path="logs/app.log"
        )
        # Use timestamped name
        save_debug_context(ctx, filepath=f"results/debug_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        log.info("debug_context_saved")
    except Exception as e:
        log.warning("debug_context_save_failed", error=str(e))

    return strategies


def simulate_selected_strategy(
    strategy: dict[str, Any],
    horizon: int,
    credit_params: dict[str, Any],
    tmi: float,
    regime: str,
    cfe: float,
    frais_vente: float,
    market_hypo: dict[str, float],
) -> pd.DataFrame | None:
    """Run detailed simulation for a selected strategy.

    Creates yearly projection data for charts.

    Args:
        strategy: Strategy to simulate
        horizon: Simulation years
        credit_params: Credit parameters for IRA
        tmi: Marginal tax rate
        regime: Tax regime
        cfe: CFE per property
        frais_vente: Sale costs percentage
        market_hypo: Market hypotheses

    Returns:
        DataFrame with yearly projections or None on error
    """
    try:
        mk = MarketHypotheses(
            appreciation_bien_pct=market_hypo.get("appreciation_bien_pct", 2.5),
            revalo_loyer_pct=market_hypo.get("revalo_loyer_pct", 1.5),
            inflation_charges_pct=market_hypo.get("inflation_charges_pct", 2.0),
        )
        tx = TaxParams(tmi_pct=tmi, regime_fiscal=regime)
        ir = IRACalculator(
            apply_ira=credit_params.get("apply_ira", True),
            ira_cap_pct=credit_params.get("ira_cap_pct", 3.0),
        )
        eng = SimulationEngine(
            market=mk,
            tax=tx,
            ira=ir,
            cfe_par_bien_ann=cfe,
            frais_vente_pct=frais_vente,
        )

        schedules = [
            generate_amortization_schedule(
                float(p["credit_final"]),
                float(p["taux_pret"]),
                int(p["duree_pret"]) * 12,
                float(p["assurance_ann_pct"]),
            )
            for p in strategy["details"]
        ]

        df_sim, _ = eng.simulate(strategy, horizon, schedules)
        return df_sim

    except Exception as e:
        log.warning("simulation_failed", error=str(e))
        return None


def autosave_results(
    strategies: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """Auto-save results to disk.

    Args:
        strategies: Strategies to save
        metadata: Additional metadata
    """
    try:
        from src.services.exporter import ResultExporter
        exporter = ResultExporter()
        exporter.save_results(strategies, metadata=metadata)
    except Exception as e:
        log.warning("autosave_failed", error=str(e))
