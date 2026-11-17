"""T23: regime choices flow from the sidebar to the simulation (TAX_SPEC 8)."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src.application.run_context import RunContext
from src.application.services.strategy_finder import EvaluationParams
from src.core.defaults import TAX


def _script_tax_section():
    import streamlit as st

    from src.ui.components.sidebar import render_tax_section
    from src.ui.components.sidebar_state import ensure_defaults

    ensure_defaults()  # widgets take their defaults from session state (T65)
    st.session_state["tax_choice"] = render_tax_section()


def test_tax_section_defaults_and_sci_override():
    at = AppTest.from_function(_script_tax_section)
    at.run()
    assert not at.exception
    choice = at.session_state["tax_choice"]
    assert choice == {"tmi": int(TAX.tmi_pct), "regime": "lmnp", "regime_foncier": "reel", "sci_is": False}
    assert len(at.selectbox) == 2 and not any(sb.disabled for sb in at.selectbox)

    at.selectbox[0].select("Micro-BIC")
    at.selectbox[1].select("Micro-foncier")
    at.run()
    choice = at.session_state["tax_choice"]
    assert choice["regime"] == "microbic" and choice["regime_foncier"] == "micro" and choice["sci_is"] is False

    at.checkbox[0].check()
    at.run()
    choice = at.session_state["tax_choice"]
    assert choice["sci_is"] is True
    assert all(sb.disabled for sb in at.selectbox)  # SCI IS overrides both category regimes
    assert choice["regime"] == "microbic"  # the category choices survive the toggle (stable widget keys)


def test_run_context_carries_regimes_to_services():
    ctx = RunContext(regime_fiscal="microbic", regime_foncier="micro", sci_is=True)
    for d in (ctx.to_tax_params(), ctx.to_eval_params()):
        assert d["regime_fiscal"] == "microbic" and d["regime_foncier"] == "micro" and d["sci_is"] is True
    ep = EvaluationParams.from_dict(ctx.to_eval_params())
    assert (ep.regime_fiscal, ep.regime_foncier, ep.sci_is) == ("microbic", "micro", True)
    assert RunContext.from_dict(ctx.to_dict()) == ctx


def test_run_context_defaults_from_defaults_module():
    ctx = RunContext()
    assert (ctx.regime_fiscal, ctx.regime_foncier, ctx.sci_is) == (TAX.regime_fiscal, TAX.regime_foncier, TAX.sci_is)
    assert EvaluationParams.from_dict({}).regime_foncier == TAX.regime_foncier
