"""Regression tests for the T40 results UI batch.

Covers: combos_evaluated counting (app.track_progress), the solver mode reported by
StrategyFinder, the selection reset and URL param for strategies, and the apport
shown on each bien in the details view.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

import app
from src.application.progress import SearchPhase, SearchProgress
from src.application.services.strategy_finder import StrategyFinder
from src.domain.models.brick import InvestmentBrick

# ---------------------------------------------------------------------------
# combos_evaluated: only evaluation events count
# ---------------------------------------------------------------------------


def test_scoring_and_dedupe_do_not_overwrite_combos_evaluated():
    tracker = {"combos_evaluated": 0, "valid_strategies": 0}
    app.track_progress(
        tracker,
        SearchProgress(
            phase=SearchPhase.EVALUATION,
            items_processed=500,
            items_total=500,
            valid_count=7,
        ),
    )
    app.track_progress(tracker, SearchProgress(phase=SearchPhase.SCORING, items_total=7))
    app.track_progress(tracker, SearchProgress(phase=SearchPhase.DEDUPLICATION, items_total=3))

    assert tracker["combos_evaluated"] == 500
    assert tracker["valid_strategies"] == 7


def test_brick_generation_does_not_count_as_evaluation():
    tracker = {"combos_evaluated": 0, "valid_strategies": 0}
    app.track_progress(
        tracker,
        SearchProgress(
            phase=SearchPhase.BRICK_GENERATION,
            items_processed=42,
            items_total=42,
        ),
    )
    assert tracker["combos_evaluated"] == 0


# ---------------------------------------------------------------------------
# Solver mode reported by the finder
# ---------------------------------------------------------------------------


# Fields the simulation engine needs (it no longer swallows a KeyError from an incomplete brick).
_SIM_FIELDS = {
    "charges_const_mth0": 50.0,
    "tf_const_mth0": 50.0,
    "frais_gestion_pct": 0.0,
    "budget_travaux": 0.0,
    "mobilier": 0.0,
    "frais_notaire": 7000.0,
    "taux_annuel_pct": 3.5,
    "duree_credit_mois": 240,
}


def test_finder_reports_exhaustive_solver_after_run():
    bricks = [
        InvestmentBrick(
            nom="A",
            apport_min=10000,
            prix_achat_bien=100000,
            capital_emprunte=90000,
            loyer_mensuel_initial=500,
            **_SIM_FIELDS,
        ),
        InvestmentBrick(
            nom="B",
            apport_min=20000,
            prix_achat_bien=200000,
            capital_emprunte=180000,
            loyer_mensuel_initial=1000,
            **_SIM_FIELDS,
        ),
    ]
    finder = StrategyFinder(bricks=bricks, apport_disponible=50000, cash_flow_cible=0.0)
    assert finder.solver_mode == ""

    finder.find_strategies(eval_params={}, horizon_years=10, top_n=10)

    assert finder.solver_mode == "EXHAUSTIVE"


# ---------------------------------------------------------------------------
# Selection reset and URL param (AppTest runs the script in a Streamlit session)
# ---------------------------------------------------------------------------


def _script_reset_selection():
    import streamlit as st

    from src.ui.state import SessionManager, set_state

    set_state("selected_strategy_idx", 3)
    set_state("show_details", True)
    st.query_params["strategy"] = "3"
    SessionManager.reset_selection()


def test_new_search_resets_selection_and_details_and_url():
    at = AppTest.from_function(_script_reset_selection)
    at.run()

    assert not at.exception
    assert at.session_state["selected_strategy_idx"] == 0
    assert at.session_state["show_details"] is False
    assert "strategy" not in at.query_params


def _script_select_strategy():
    from src.ui.state import SessionManager

    SessionManager.set_selected_idx(2)


def test_selecting_a_strategy_writes_url_param():
    at = AppTest.from_function(_script_select_strategy)
    at.run()

    assert not at.exception
    assert at.session_state["selected_strategy_idx"] == 2
    assert at.query_params["strategy"] == "2"


def _script_deep_link():
    from src.ui.state import SessionManager

    SessionManager.initialize()


def test_deep_link_param_selects_strategy_on_load():
    at = AppTest.from_function(_script_deep_link)
    at.query_params["strategy"] = "4"
    at.run()

    assert not at.exception
    assert at.session_state["selected_strategy_idx"] == 4


# ---------------------------------------------------------------------------
# Details view shows the final apport of each bien
# ---------------------------------------------------------------------------


def _script_details_apport():
    from src.ui.components.results import render_strategy_details

    strategy = {
        "details": [
            {
                "nom_bien": "Studio",
                "ville": "Lyon",
                "apport_min": 10000,
                "apport_final_bien": 25000,
                "credit_final": 90000,
            }
        ],
    }
    render_strategy_details(strategy, horizon=20)


def test_details_show_apport_final_bien_not_apport_min():
    from src.ui.helpers import format_euro

    at = AppTest.from_function(_script_details_apport)
    at.run()

    assert not at.exception
    texts = [m.value for m in at.markdown]
    text_blob = "\n".join(str(t) for t in texts)
    assert f"Apport: {format_euro(25000)}" in text_blob
    assert f"Apport: {format_euro(10000)}" not in text_blob
