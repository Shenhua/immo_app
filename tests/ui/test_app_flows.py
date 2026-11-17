"""T51: end-to-end UI flows on app.py with streamlit.testing.v1.AppTest.

Searches stay small (defaults: 54 bricks, at most 3 properties, a few dozen evaluations, serial) so the whole
module runs in a few seconds. Helpers live in ``tests/ui/conftest.py``.
"""

from __future__ import annotations

from src.application.run_context import RunContext
from src.application.scenario_io import decode_context, encode_context

from .conftest import (
    all_text,
    new_app,
    open_details,
    page_warnings,
    run_search,
    search_button,
    set_inputs,
    strategies,
)

MODE_RENTIER = "Rentier (Max Cash-Flow)"
MODE_EMPIRE = "Empire (Max Volume)"
NE_PAS_RENOVER = "Ne pas rénover (loyer nul dès l'interdiction)"


# 1. First load -------------------------------------------------------------------------------------------------
def test_first_load_shows_defaults_and_enabled_search_button():
    at = new_app()
    assert not at.exception
    defaults = RunContext()
    assert at.number_input(key="sb_apport").value == int(defaults.apport)
    assert at.number_input(key="sb_cf_cible").value == defaults.cf_cible
    assert at.slider(key="horizon_ans").value == defaults.horizon_ans
    assert at.radio(key="sb_mode").value == "Classique (Levier)"
    assert at.slider(key="sb_tmi").value == int(defaults.tmi_pct)
    assert not at.sidebar.error
    assert not search_button(at).disabled
    assert not at.session_state["strategies"]  # nothing searched yet


def test_first_load_shows_the_configure_prompt_not_the_empty_results_message():
    at = new_app()
    assert any("lancez l'analyse" in str(i.value) for i in at.info)
    assert not any("Aucune stratégie trouvée" in str(i.value) for i in at.info)


# 2. Search with defaults ---------------------------------------------------------------------------------------
def test_default_search_renders_cards_with_debt_ratio_and_pareto_badge():
    at = run_search(new_app())
    assert not at.exception
    found = strategies(at)
    assert len(found) >= 5
    assert any(f"{len(found)} stratégie(s) trouvée(s)" in str(s.value) for s in at.success)
    text = all_text(at)
    assert "Endettement" in text  # T60 debt ratio on the selected card
    assert any(s.get("pareto_rank") == 1 for s in found)
    assert "Pareto" in text  # T32 badge on a rank-1 card
    assert len(at.dataframe) == 1  # the comparison table lists every strategy
    assert at.session_state["search_stats"]["strategies_after_dedupe"] == len(found)


# 3. Strategy detail --------------------------------------------------------------------------------------------
def test_detail_view_renders_every_tab_and_report_downloads():
    at = open_details(run_search(new_app()))
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "🏠 Biens",
        "📈 Projections",
        "🎯 Radar",
        "⚡ Stress Test",
        "⚖️ Comparaison fiscale",
    ]
    assert len(at.get("plotly_chart")) >= 4
    expanders = [e.label for e in at.expander]
    assert any("Tornade" in e for e in expanders)  # sensitivity
    assert any("Scénarios de risque" in e for e in expanders)  # risk expander
    assert any("Détails des biens" in str(m.value) for m in at.markdown)  # tab "Biens"
    assert len(at.tabs[4].children) > 0  # fiscal comparison rendered something

    assert not [d for d in at.get("download_button") if "Rapport" in d.proto.label]
    at.button(key="report_prepare_0").click()
    at.run()
    assert not at.exception
    labels = [d.proto.label for d in at.get("download_button")]
    assert any("Excel" in lbl for lbl in labels) and any("PDF" in lbl for lbl in labels)


def test_selecting_another_strategy_renders_its_details():
    at = run_search(new_app())
    at.session_state["selected_strategy_idx"] = 1
    at.session_state["show_details"] = True
    at.run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert any("Stratégie #2" in str(m.value) for m in at.markdown)


# 4. Stale banner -----------------------------------------------------------------------------------------------
def test_changing_a_sidebar_value_after_search_shows_stale_banner():
    at = run_search(new_app())
    assert not any("Paramètres modifiés" in w for w in page_warnings(at))
    set_inputs(at, sb_apport=int(at.number_input(key="sb_apport").value) + 10_000)
    assert not at.exception
    assert any("Paramètres modifiés" in w for w in page_warnings(at))
    assert strategies(at)  # previous results stay displayed

    run_search(at)  # re-running clears it
    assert not any("Paramètres modifiés" in w for w in page_warnings(at))


# 5. Validation -------------------------------------------------------------------------------------------------
def test_down_payment_below_cheapest_brick_blocks_the_search():
    at = set_inputs(new_app(), sb_apport=1_000)
    assert not at.exception
    assert len(at.sidebar.error) == 1
    assert "apport minimum" in str(at.sidebar.error[0].value)
    assert search_button(at).disabled

    set_inputs(at, sb_apport=80_000)  # fixing it re-enables the search
    assert not at.sidebar.error
    assert not search_button(at).disabled


def test_negative_tolerance_cannot_be_entered():
    # The widget has min_value=0 and clamps: the invalid value never reaches the T43 validation.
    at = set_inputs(new_app(), sb_precise=True)
    set_inputs(at, sb_tolerance=-5)
    assert not at.exception
    assert at.number_input(key="sb_tolerance").value >= 0
    assert not search_button(at).disabled


# 6. Empty results and suggestions ------------------------------------------------------------------------------
def test_impossible_cash_flow_target_shows_diagnostics_and_suggestion_recovers():
    at = run_search(set_inputs(new_app(), sb_cf_cible=2000.0))
    assert not at.exception
    assert strategies(at) == []
    assert any("Aucune stratégie trouvée" in str(i.value) for i in at.info)
    assert any("Contrainte la plus éliminatoire" in w for w in page_warnings(at))

    apply_keys = [b.key for b in at.button if b.label == "Appliquer cette suggestion"]
    cf_key = next(k for k in apply_keys if k.endswith("_cf_cible"))
    at.button(key=cf_key).click()
    at.run()
    assert not at.exception
    assert at.number_input(key="sb_cf_cible").value < 2000.0  # the suggestion patched the sidebar input

    run_search(at)
    assert not at.exception
    assert len(strategies(at)) >= 1


# 7. Modes ------------------------------------------------------------------------------------------------------
def test_rentier_mode_runs():
    at = run_search(set_inputs(new_app(), sb_mode=MODE_RENTIER))
    assert not at.exception
    assert at.session_state["run_context"].use_full_capital is True
    assert len(strategies(at)) >= 1


def test_empire_mode_runs_with_a_reduced_archetype_set():
    at = set_inputs(new_app(), sb_mode=MODE_EMPIRE)
    assert at.slider(key="max_properties").value == 5
    set_inputs(at, sb_villes=["Grenoble", "Lyon", "Chambéry"])
    run_search(at)
    assert not at.exception
    assert at.session_state["run_context"].max_properties == 5
    assert len(strategies(at)) >= 1


# 8. Regimes ----------------------------------------------------------------------------------------------------
def test_micro_bic_regime_runs_and_shows_fiscal_comparison():
    at = set_inputs(new_app(), tax_regime_meuble="Micro-BIC")
    run_search(at)
    assert not at.exception
    assert at.session_state["run_context"].regime_fiscal == "microbic"
    open_details(at)
    assert not at.exception
    assert len(at.tabs) == 5


def test_sci_is_runs():
    at = set_inputs(new_app(), tax_sci_is=True)
    assert all(sb.disabled for sb in at.selectbox if sb.key in ("tax_regime_meuble", "tax_regime_nu"))
    run_search(at)
    assert not at.exception
    assert at.session_state["run_context"].sci_is is True
    assert len(strategies(at)) >= 1
    open_details(at)
    assert not at.exception


# 9. URL --------------------------------------------------------------------------------------------------------
def test_p_query_param_is_applied_to_the_sidebar():
    ctx = RunContext(apport=120_000, cf_cible=-250.0, tmi_pct=41.0, regime_fiscal="microbic", sci_is=False)
    at = new_app({"p": encode_context(ctx)})
    assert not at.exception
    assert at.number_input(key="sb_apport").value == 120_000
    assert at.number_input(key="sb_cf_cible").value == -250.0
    assert at.slider(key="sb_tmi").value == 41
    assert at.selectbox(key="tax_regime_meuble").value == "Micro-BIC"
    assert decode_context(at.query_params["p"]).apport == 120_000
    assert not search_button(at).disabled


# 10. DPE ban ---------------------------------------------------------------------------------------------------
def test_ne_pas_renover_policy_warns_on_a_dpe_e_property_within_horizon():
    # Chambéry T2 (DPE E, banned from 2034, inside the 25-year horizon) with the E->D renovation option off.
    at = set_inputs(
        new_app(),
        sb_villes=["Chambéry"],
        sb_types=["meuble_classique"],
        sb_cf_cible=-500.0,
        sb_inc_reno=False,
        sb_dpe_policy=NE_PAS_RENOVER,
    )
    run_search(at)
    assert not at.exception
    assert any(b.get("dpe_initial") == "E" for s in strategies(at) for b in s["details"])
    assert any("DPE E" in w and "2034" in w for w in page_warnings(at))


def test_default_search_has_no_dpe_warning():
    at = run_search(new_app())
    assert not any("DPE" in w for w in page_warnings(at))
