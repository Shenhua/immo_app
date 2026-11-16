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

from src.application.run_context import RunContext
from src.application.services.simulation import simulate_long_term_strategy
from src.application.services.strategy_finder import StrategyFinder
from src.core import defaults
from src.core.exceptions import SimulationError
from src.core.logging import get_logger
from src.domain.models.archetype import ArchetypeV2
from src.ui.cached import cached_archetypes_from_file, cached_bricks, file_signature
from src.ui.state import SessionManager

log = get_logger(__name__)

# Disk side effects are opt-in. The env vars are read once, in src/core/defaults.py.
# Module-level names are kept so tests can monkeypatch them.
AUTOSAVE_ENABLED: bool = defaults.AUTOSAVE_ENABLED
RESULTS_DIR: str = defaults.RESULTS_DIR


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
            path = os.path.join(os.path.dirname(__file__), "../../data/archetypes_2026_v1.json")
            path = os.path.normpath(path)
            archetypes = cached_archetypes_from_file(*file_signature(path))
            SessionManager.set_archetypes(archetypes)
        else:
            log.debug("archetypes_loaded_from_session", count=len(archetypes))
        return archetypes
    except Exception as e:
        log.error("data_load_failed", error=str(e))
        st.error(f"Erreur lors du chargement des données: {e}")
        return []




def run_strategy_search(
    archetypes: list[dict[str, Any]],
    ctx: RunContext,
    progress_callback: Any = None,
    run_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute the strategy search pipeline.

    Every parameter comes from the RunContext snapshot.

    Args:
        archetypes: Filtered and compliant archetypes
        ctx: Parameters of this run
        progress_callback: Optional callback for progress updates
        run_info: Optional dict filled with facts about the run that the caller
            cannot see otherwise: ``bricks_count`` and ``solver_mode``

    Returns:
        List of ranked strategies
    """
    eval_params = ctx.to_eval_params()
    log.info("analysis_started", archetypes_count=len(archetypes), horizon=ctx.horizon_ans, top_n=ctx.top_n)
    log.debug(
        "run_strategy_search_called",
        apport=ctx.apport,
        cf_cible=ctx.cf_cible,
        mode_cf=ctx.mode_cf,
        max_props=ctx.max_properties,
        use_full_capital=ctx.use_full_capital,
    )

    # 1. Create investment bricks
    bricks = cached_bricks(archetypes, ctx)

    # 2. Find strategies
    finder = StrategyFinder(
        bricks=bricks,
        apport_disponible=ctx.apport,
        cash_flow_cible=ctx.cf_cible,
        tolerance=ctx.tolerance,
        qualite_weight=ctx.qualite_weight,
        mode_cf=ctx.mode_cf,
        max_properties=ctx.max_properties,
    )

    strategies = finder.find_strategies(
        eval_params=eval_params,
        horizon_years=ctx.horizon_ans,
        use_full_capital_override=ctx.use_full_capital,
        top_n=ctx.top_n,
        progress_callback=progress_callback,
    )

    if run_info is not None:
        run_info["bricks_count"] = len(bricks)
        run_info["solver_mode"] = finder.solver_mode

    log.info("analysis_completed", strategies_found=len(strategies))

    # Phase 19: Automatic Debug Context (disk write, opt-in via APP_AUTOSAVE=1)
    if AUTOSAVE_ENABLED:
        try:
            from src.utils.debug import collect_debug_context, save_debug_context

            ctx_params = {
                "apport": ctx.apport,
                "cf_cible": ctx.cf_cible,
                "tolerance": ctx.tolerance,
                "qual_weight": ctx.qualite_weight,
                "mode_cf": ctx.mode_cf,
                "horizon_years": ctx.horizon_ans,
                "top_n": ctx.top_n,
                "eval_params": eval_params,
                "run_context": ctx.to_dict(),
            }

            debug_ctx = collect_debug_context(
                params=ctx_params,
                strategies=strategies,
                log_file_path="logs/app.log"
            )
            # Use timestamped name
            debug_path = os.path.join(RESULTS_DIR, f"debug_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            save_debug_context(debug_ctx, filepath=debug_path)
            log.info("debug_context_saved")
        except Exception as e:
            log.warning("debug_context_save_failed", error=str(e))

    return strategies


def _strategy_key(strategy: dict[str, Any]) -> str:
    return json.dumps(strategy, sort_keys=True, default=str)


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_simulation(strategy_json: str, ctx_json: str) -> tuple[pd.DataFrame | None, str | None]:
    """Detail simulation keyed by (strategy, RunContext). Returns (frame, error message); copies on every hit."""
    strategy = json.loads(strategy_json)
    ctx = RunContext.model_validate_json(ctx_json)
    try:
        df_sim, bilan = simulate_long_term_strategy(
            strategy,
            duration_years=ctx.horizon_ans,
            market_hypotheses=ctx.market_hypotheses,
            tax_params=ctx.to_tax_params(),
        )
    except (ValueError, ZeroDivisionError, SimulationError) as e:
        # Expected failures (invalid inputs rejected by a regime or the schedule check). Any other
        # exception is a bug and propagates (and is not cached).
        log.warning("simulation_failed", error=str(e))
        return None, str(e)
    # Regime actually applied and why a choice was overridden (micro ceilings, LMP flag).
    df_sim.attrs["tax_regimes"] = bilan.get("regimes", {})
    df_sim.attrs["tax_warnings"] = bilan.get("tax_warnings", [])
    return df_sim, None


def simulate_selected_strategy(strategy: dict[str, Any], ctx: RunContext) -> pd.DataFrame | None:
    """Run detailed simulation for a selected strategy, with the parameters of its run.

    Cached by (strategy, ctx): reruns caused by row selection or tab switches do not recompute it.
    The returned frame is a private copy.

    Args:
        strategy: Strategy to simulate
        ctx: Parameters of the run that produced the strategy

    Returns:
        DataFrame with yearly projections or None on error
    """
    df_sim, error = _cached_simulation(_strategy_key(strategy), ctx.model_dump_json())
    if error is not None:
        st.warning(f"Simulation détaillée impossible : {error}")
    return df_sim


def autosave_results(
    strategies: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """Auto-save results to disk. No-op unless APP_AUTOSAVE=1.

    Args:
        strategies: Strategies to save
        metadata: Additional metadata
    """
    if not AUTOSAVE_ENABLED:
        return
    try:
        from src.application.services.exporter import ResultExporter
        exporter = ResultExporter(RESULTS_DIR)
        exporter.save_results(strategies, metadata=metadata)
    except Exception as e:
        log.warning("autosave_failed", error=str(e))
