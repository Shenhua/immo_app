"""T60: HCSF bankability (debt ratio, 70 % rent weighting, filter, diagnostics)."""

from __future__ import annotations

import pytest

from src.application.run_context import RunContext
from src.application.services.diagnostics import REASON_LABELS
from src.application.services.strategy_finder import StrategyFinder
from src.domain.hcsf import hcsf_check


def _prop(name="A", pmt=1000.0, rent=1000.0, credit=200_000.0, months=240):
    return {"nom": name, "pmt_total": pmt, "loyer_mensuel_initial": rent, "credit_final": credit,
            "duree_credit_mois": months}


def test_ratio_hand_computed():
    # (500 + 1000 + 800) / (4000 + 0.7 * (1000 + 800)) = 2300 / 5260 = 43.73 %
    r = hcsf_check([_prop("A", 1000, 1000), _prop("B", 800, 800)], 4000, 500)
    assert r.taux_endettement == pytest.approx(2300 / 5260 * 100)
    assert not r.hcsf_ok
    # income needed for exactly 35 %: 2300 / 0.35 - 1260
    assert r.revenu_requis == pytest.approx(2300 / 0.35 - 1260)


def test_rent_weighting_is_70_percent():
    base = hcsf_check([_prop(pmt=1000, rent=0)], 4000, 0).taux_endettement
    with_rent = hcsf_check([_prop(pmt=1000, rent=1000)], 4000, 0).taux_endettement
    assert base == pytest.approx(25.0)
    assert with_rent == pytest.approx(1000 / 4700 * 100)


def test_threshold_and_duration():
    assert hcsf_check([_prop(pmt=1400, rent=0)], 4000).hcsf_ok  # exactly 35 %
    assert not hcsf_check([_prop(pmt=1401, rent=0)], 4000).hcsf_ok
    r = hcsf_check([_prop(pmt=100, rent=0, months=26 * 12)], 4000)
    assert not r.duree_ok and not r.hcsf_ok


def test_full_cash_always_ok():
    r = hcsf_check([_prop(pmt=0, credit=0)], 1000, 900)  # existing debts alone above 35 %
    assert r.hcsf_ok


def _strategies():
    return [
        {"details": [_prop("OK", 500, 600)], "apport_total": 1.0},
        {"details": [_prop("KO", 3000, 600)], "apport_total": 1.0},
    ]


def _finder():
    return StrategyFinder(bricks=[], apport_disponible=1.0, cash_flow_cible=0.0)


def test_filter_on_and_off():
    cfg = {"revenu_net_mensuel": 4000, "charges_mensuelles": 0, "filtrer": True}
    f = _finder()
    kept = f._apply_hcsf(_strategies(), cfg)
    assert [s["details"][0]["nom"] for s in kept] == ["OK"]
    assert kept[0]["hcsf_ok"] and kept[0]["taux_endettement"] == pytest.approx(500 / 4420 * 100)

    f = _finder()
    kept = f._apply_hcsf(_strategies(), {**cfg, "filtrer": False})
    assert [s["hcsf_ok"] for s in kept] == [True, False]
    assert f.last_diagnostics.counts["hcsf"] == 0


def test_no_config_means_no_check():
    out = _finder()._apply_hcsf(_strategies(), {})
    assert all("hcsf_ok" not in s for s in out)


def test_subset_kept_if_any_assignment_bankable():
    strats = [{"details": [_prop("A", 3000, 600)]}, {"details": [_prop("A", 800, 600)]}]
    f = _finder()
    kept = f._apply_hcsf(strats, {"revenu_net_mensuel": 4000, "filtrer": True})
    assert len(kept) == 1 and f.last_diagnostics.counts["hcsf"] == 0


def test_diagnostics_reason_near_miss_and_suggestion():
    f = _finder()
    f._apply_hcsf(_strategies(), {"revenu_net_mensuel": 4000, "charges_mensuelles": 0, "filtrer": True})
    d = f.last_diagnostics
    assert d.counts["hcsf"] == 1
    assert "HCSF" in REASON_LABELS["hcsf"]
    near = d.near_misses["hcsf"][0]
    assert near.names == ("KO",) and near.cf == pytest.approx(3000 / 4420 * 100)
    assert "taux d'endettement" in near.describe()
    sug = [s for s in d.suggestions() if s.reason == "hcsf"][0]
    assert sug.param == "revenu_net_mensuel" and sug.suggested >= 3000 / 0.35 - 420


def test_run_context_carries_profile():
    ctx = RunContext(revenu_net_mensuel=5000, charges_mensuelles=300, filtrer_hcsf=False)
    assert ctx.to_eval_params()["hcsf"] == {"revenu_net_mensuel": 5000, "charges_mensuelles": 300, "filtrer": False}
