"""T43: input validation (pure functions), custom loan durations, default values and a UI smoke run."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.application.run_context import RunContext
from src.application.services.brick_factory import (
    apply_rent_caps,
    build_financing_config,
    build_operating_config,
    create_investment_bricks,
)
from src.application.services.strategy_finder import StrategyFinder
from src.application.validation import cheapest_brick_apport, validate_durations, validate_search_inputs
from src.core.defaults import FINANCING
from src.ui.app_controller import load_archetypes_from_disk
from src.ui.pages.main import strategy_warnings

RATES = {15: 3.5, 20: 3.6, 25: 3.8}


def _ok(**kw):
    base = {"apport": 50_000, "tolerance": 100, "min_brick_apport": 4_000, "taux_credits": RATES}
    return validate_search_inputs(**{**base, **kw})


def test_valid_inputs_have_no_error() -> None:
    assert _ok() == []
    assert _ok(budget=60_000) == []


def test_apport_below_cheapest_brick_is_an_error() -> None:
    errors = _ok(apport=3_000)
    assert len(errors) == 1 and "Apport" in errors[0]
    assert _ok(apport=4_000) == []  # equal is fine


def test_negative_tolerance_is_an_error() -> None:
    errors = _ok(tolerance=-1)
    assert len(errors) == 1 and "Tolérance" in errors[0]
    assert _ok(tolerance=0) == []


def test_budget_below_apport_is_an_error_only_when_given() -> None:
    errors = _ok(budget=40_000)
    assert len(errors) == 1 and "Budget" in errors[0]
    assert _ok(budget=None) == []


def test_unknown_cheapest_brick_skips_that_rule() -> None:
    assert _ok(apport=0, min_brick_apport=None) == []


def test_max_properties_and_empty_filters() -> None:
    assert any("Nombre max" in e for e in _ok(max_properties=0))
    assert any("Nombre max" in e for e in _ok(max_properties=6))
    assert any("filtres" in e for e in _ok(n_archetypes=0))


def test_several_errors_are_all_reported() -> None:
    assert len(_ok(apport=1_000, tolerance=-5, budget=500)) == 3


def test_duration_validation() -> None:
    assert validate_durations(RATES) == []
    assert validate_durations({10: 3.4, 12: 3.45}) == []
    assert validate_durations({}) != []
    assert any("hors plage" in e for e in validate_durations({9: 3.0}))
    assert any("hors plage" in e for e in validate_durations({26: 3.0}))
    assert any("Taux" in e for e in validate_durations({15: 0.0}))
    assert any("maximum" in e for e in validate_durations(dict.fromkeys(range(10, 16), 3.0)))


def test_default_rate_matches_anchors_and_interpolates() -> None:
    for years, rate in RATES.items():
        assert FINANCING.default_rate(years) == pytest.approx(rate)
    assert FINANCING.default_rate(17) == pytest.approx(3.54)
    assert FINANCING.default_rate(10) < FINANCING.default_rate(15)


def test_cheapest_brick_apport() -> None:
    class B:
        def __init__(self, a: float) -> None:
            self.apport_min = a

    assert cheapest_brick_apport([B(5), B(3), B(9)]) == 3
    assert cheapest_brick_apport([]) is None


def test_strategy_warnings_collects_and_dedupes() -> None:
    assert strategy_warnings({}) == []
    assert strategy_warnings({"warnings": "a"}) == ["a"]
    assert strategy_warnings({"warnings": ["a", "b"], "tri_warning": "b"}) == ["a", "b"]
    assert strategy_warnings({"tri_warning": "t"}) == ["t"]


def _search(ctx: RunContext, archetypes) -> list:
    bricks = create_investment_bricks(
        archetypes,
        build_financing_config(ctx.to_credit_params()),
        build_operating_config(ctx.gestion_pct, ctx.vacance_pct, ctx.cfe_par_bien_ann),
    )
    finder = StrategyFinder(
        bricks=bricks, apport_disponible=ctx.apport, cash_flow_cible=ctx.cf_cible, tolerance=ctx.tolerance,
        qualite_weight=ctx.qualite_weight, mode_cf=ctx.mode_cf, max_properties=ctx.max_properties,
    )
    return finder.find_strategies(eval_params=ctx.to_eval_params(), horizon_years=ctx.horizon_ans, top_n=ctx.top_n)


@pytest.fixture(scope="module")
def archetypes():
    return apply_rent_caps(load_archetypes_from_disk("data/archetypes_2026_v1.json"))


def test_default_sidebar_values_give_a_meaningful_result_set(archetypes) -> None:
    assert len(_search(RunContext(), archetypes)) >= 5


def test_custom_durations_are_searched(archetypes) -> None:
    ctx = RunContext(taux_credits={10: 3.4, 12: 3.45, 25: 3.8})
    strategies = _search(ctx, archetypes)
    assert strategies
    years = {d["duree_pret"] for s in strategies for d in s["details"]}
    assert years <= {10, 12, 25}


def test_ui_smoke_blocks_search_on_invalid_input_and_hides_tolerance() -> None:
    at = AppTest.from_file("../../app.py", default_timeout=180)
    at.run()
    assert not at.exception
    labels = [n.label for n in at.sidebar.number_input]
    assert not any(label.startswith("Tolérance") for label in labels)  # hidden in "min" mode
    assert any(s.label == "Nombre max de biens" for s in at.sidebar.slider)
    launch = next(b for b in at.sidebar.button if "Lancer" in b.label)
    assert not launch.disabled

    next(n for n in at.sidebar.number_input if n.label.startswith("Apport")).set_value(0)
    at.run()
    assert not at.exception
    assert at.sidebar.error  # inline error
    launch = next(b for b in at.sidebar.button if "Lancer" in b.label)
    assert launch.disabled

    at.sidebar.toggle[0].set_value(True)  # Ciblage précis: tolerance appears
    at.run()
    assert any(n.label.startswith("Tolérance") for n in at.sidebar.number_input)
