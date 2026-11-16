"""Main Application Entry Point.

Refactored to use modular layered architecture.
Orchestrates UI components and services via app_controller.
"""

import os
import sys
from typing import Any

import streamlit as st

# Add src to path if not present (for running from root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.application.progress import SearchPhase, SearchProgress
from src.application.run_context import RunContext
from src.application.validation import cheapest_brick_apport, validate_durations, validate_search_inputs
from src.core.defaults import OPERATING, REALISM, SEARCH, TAX
from src.core.logging import get_logger
from src.ui.app_controller import (
    autosave_results,
    load_archetypes,
    run_strategy_search,
    simulate_selected_strategy,
)
from src.ui.cached import cached_bricks, filter_and_caps
from src.ui.components.filters import render_property_filters
from src.ui.components.sidebar import (
    render_credit_params_tab,
    render_market_hypotheses,
    render_objectives_section,
    render_scoring_preset,
    render_tax_section,
)
from src.ui.pages.main import render_main_page
from src.ui.state import SessionManager

# --- Configuration & Setup ---
# Note: Streamlit calls moved to main() to prevent blocking on import


def load_css() -> None:
    """Load custom CSS styles."""
    css_file = os.path.join(os.path.dirname(__file__), "src/ui/assets/style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar() -> dict[str, Any]:
    """Render sidebar and collect all parameters.

    Returns:
        Dictionary containing all sidebar parameters
    """
    with st.sidebar:
        st.title("⚙️ Paramètres")

        # Objectives
        objectives = render_objectives_section()

        # Credit
        with st.expander("🏦 Financement", expanded=False):
            credit_params = render_credit_params_tab()

        # Operating Costs (Exploitation)
        with st.expander("📉 Frais & Charges", expanded=False):
            cfe = st.number_input("CFE (€/an, meublé)", 0, 2000, value=int(OPERATING.cfe_par_bien_ann), step=50)
            gestion = st.slider("Gestion (%)", 0.0, 15.0, value=OPERATING.gestion_pct, step=0.5)
            vacance = st.slider(
                "Vacance minimale (%)", 0.0, 10.0, value=OPERATING.vacance_pct, step=0.5,
                help="Plancher de vacance : chaque bien utilise max(vacance propre au type de bien, ce curseur). "
                     "Le curseur ne peut donc que relever la vacance d'un bien, jamais la baisser.",
            )
            gli_pct = st.slider(
                "Loyers impayés / GLI (% loyers)", 0.0, 10.0, value=REALISM.gli_pct, step=0.5,
                help="Prime d'assurance loyers impayés ou provision pour impayés, en % des loyers encaissés.",
            )
            maintenance_m2_an = st.number_input(
                "Entretien / gros travaux (€/m²/an)", 0.0, 100.0, value=REALISM.maintenance_m2_an, step=1.0,
                help="Provision pour réparations et gros travaux : sortie de trésorerie, déductible au réel.",
            )
            mois_travaux = st.slider(
                "Mois sans loyer si travaux", 0, 12, value=REALISM.mois_travaux, step=1,
                help="Année 1 uniquement, pour les biens avec travaux : mois sans loyer.",
            )
            tf_growth_pct = st.slider(
                "Hausse taxe foncière (%/an)", -2.0, 10.0, value=REALISM.tf_growth_pct, step=0.5,
                help="Croissance annuelle de la taxe foncière (distincte de l'inflation des charges).",
            )
            frais_vente = st.slider("Frais Revente (%)", 0.0, 10.0, value=TAX.frais_vente_pct, step=0.5)

        # Tax
        with st.expander("⚖️ Fiscalité", expanded=False):
            tax_choice = render_tax_section()

        # Property Filters
        raw_archetypes = load_archetypes()
        # User requested folded (expanded=False)
        with st.expander("🏠 Sélection des Biens", expanded=False):
            if raw_archetypes:
                selected_villes, selected_types, apply_cap = render_property_filters(raw_archetypes)
            else:
                selected_villes, selected_types, apply_cap = [], [], True

        # Market Hypotheses
        market_hypo = render_market_hypotheses()

        # Scoring Preset
        finance_preset_name, finance_weights = render_scoring_preset()

        # Advanced Settings (Phase 17.1)
        with st.expander("⚙️ Avancé", expanded=False):
            top_n = st.slider(
                "Nombre max de résultats",
                min_value=10, max_value=500, value=SEARCH.top_n, step=10,
                help="Limite le nombre de stratégies affichées après optimisation."
            )

    return {
        "apport": objectives["apport"],
        "cf_cible": objectives["cf_cible"],
        "tolerance": objectives["tolerance"],
        "mode_cf": objectives["mode_cf"],
        "qual_weight": objectives["qualite_weight"],
        "horizon": objectives["horizon"],
        "max_properties": objectives.get("max_properties", SEARCH.max_properties_classique),
        "use_full_capital": objectives.get("use_full_capital", False),
        "credit_params": credit_params,
        "cfe": cfe,
        "gestion": gestion,
        "vacance": vacance,
        "gli_pct": gli_pct,
        "maintenance_m2_an": maintenance_m2_an,
        "mois_travaux": mois_travaux,
        "tf_growth_pct": tf_growth_pct,
        "frais_vente": frais_vente,
        "tmi": tax_choice["tmi"],
        "regime": tax_choice["regime"],
        "regime_foncier": tax_choice["regime_foncier"],
        "sci_is": tax_choice["sci_is"],
        "raw_archetypes": raw_archetypes,
        "selected_villes": selected_villes,
        "selected_types": selected_types,
        "apply_cap": apply_cap,
        "market_hypo": market_hypo,
        "finance_preset_name": finance_preset_name,
        "finance_weights": finance_weights,
        "top_n": top_n,
    }


def track_progress(tracker: dict[str, int], progress: SearchProgress) -> None:
    """Fold one progress event into the search stats tracker.

    Only the evaluation phase counts combinations. Scoring and deduplication
    report their own item totals, which must not overwrite the count.
    """
    if progress.phase == SearchPhase.EVALUATION and progress.items_total > 0:
        tracker["combos_evaluated"] = progress.items_total
    if progress.valid_count > 0:
        tracker["valid_strategies"] = progress.valid_count


def save_search_stats(
    duration: float,
    tracker: dict[str, int],
    strategies: list[dict[str, Any]],
    bricks_count: int,
    solver_mode: str,
    max_props: int
) -> None:
    """Save search statistics to session state.

    Args:
        bricks_count: Number of investment bricks actually searched
        solver_mode: Solver the finder used ("EXHAUSTIVE", "SUBSET_PROXY" or "")
    """
    from src.application.progress import SearchStats
    from src.ui.state import set_state

    stats = SearchStats(
        duration_seconds=duration,
        bricks_count=bricks_count,
        combos_evaluated=tracker["combos_evaluated"],
        valid_strategies=tracker["valid_strategies"],
        strategies_after_dedupe=len(strategies),
        mode=solver_mode,
        max_properties=max_props,
    )
    set_state("search_stats", {
        "timestamp": stats.timestamp,
        "duration_seconds": stats.duration_seconds,
        "bricks_count": stats.bricks_count,
        "combos_generated": stats.combos_generated,
        "combos_evaluated": stats.combos_evaluated,
        "valid_strategies": stats.valid_strategies,
        "strategies_after_dedupe": stats.strategies_after_dedupe,
        "mode": stats.mode,
        "max_properties": stats.max_properties,
    })


def handle_analysis(ctx: RunContext, archetypes: list[dict[str, Any]]) -> None:
    """Handle the analysis button click with progress display.

    Args:
        ctx: Snapshot of the sidebar parameters at click time
        archetypes: Filtered and compliant archetypes
    """
    import time

    from src.application.progress import SearchProgress
    from src.ui.components.progress_display import render_progress_display

    start_time = time.time()
    stats_tracker = {"combos_evaluated": 0, "valid_strategies": 0}
    run_info: dict[str, Any] = {}
    progress_placeholder = st.empty()

    def on_progress(progress: SearchProgress) -> None:
        with progress_placeholder.container():
            render_progress_display(progress)
        track_progress(stats_tracker, progress)

    strategies = run_strategy_search(archetypes, ctx, progress_callback=on_progress, run_info=run_info)

    # Save stats and results
    save_search_stats(
        time.time() - start_time,
        stats_tracker,
        strategies,
        run_info.get("bricks_count", 0),
        run_info.get("solver_mode", ""),
        ctx.max_properties,
    )
    progress_placeholder.empty()
    SessionManager.reset_selection()
    SessionManager.set_strategies(strategies)
    SessionManager.set_run_context(ctx)
    autosave_results(strategies, {"horizon": ctx.horizon_ans, "compliance": ctx.apply_cap, "run_context": ctx.to_dict()})
    st.rerun()


def collect_input_errors(ctx: RunContext, archetypes: list[dict[str, Any]]) -> list[str]:
    """Blocking validation errors of the current sidebar (T43). The search is not launched while any remain."""
    min_apport = None
    if archetypes and not validate_durations(ctx.taux_credits):
        min_apport = cheapest_brick_apport(cached_bricks(archetypes, ctx))
    return validate_search_inputs(
        apport=ctx.apport,
        tolerance=ctx.tolerance,
        mode_cf=ctx.mode_cf,
        max_properties=ctx.max_properties,
        min_brick_apport=min_apport,
        taux_credits=ctx.taux_credits,
        n_archetypes=len(archetypes),
    )


def main() -> None:
    """Main application entry point."""
    # Streamlit configuration (must be first Streamlit call)
    st.set_page_config(
        page_title="Simulateur Immo v27.6",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()
    log = get_logger(__name__)

    # 1. Initialize session
    SessionManager.initialize()
    log.info("app_started")

    # 2. Render sidebar and collect parameters
    params = render_sidebar()

    # 3. Exit early if no data
    if not params["raw_archetypes"]:
        return

    # 4. Apply filters & compliance
    compliant = filter_and_caps(
        params["raw_archetypes"], params["selected_villes"], params["selected_types"], params["apply_cap"]
    )

    # 5. Snapshot of the current sidebar. Used for the search and for the stale check only.
    current_ctx = RunContext.from_sidebar(params, compliant)
    errors = collect_input_errors(current_ctx, compliant)
    for message in errors:
        st.sidebar.error(message, icon="🚫")
    if st.sidebar.button(
        "🚀 Lancer l'analyse",
        type="primary",
        disabled=bool(errors),
        help="Corrigez les erreurs ci-dessus pour lancer la recherche." if errors else None,
    ) and not errors:
        handle_analysis(current_ctx, compliant)

    # 6. Results and the context they were computed with
    strategies = SessionManager.get_strategies()
    run_ctx = SessionManager.get_run_context()
    stale = run_ctx is not None and run_ctx != current_ctx

    # 7. Simulate selected strategy for charts, with the run's own parameters
    df_sim = None
    idx = SessionManager.get_selected_idx()
    if run_ctx is not None and strategies and 0 <= idx < len(strategies):
        df_sim = simulate_selected_strategy(strategies[idx], run_ctx)

    # 8. Render main page
    render_main_page(compliant, strategies, df_sim, run_ctx, stale)

    # 9. Render Debug Tools (at bottom of sidebar)
    from src.ui.components.sidebar import render_debug_section
    render_debug_section(params)


if __name__ == "__main__":
    main()
