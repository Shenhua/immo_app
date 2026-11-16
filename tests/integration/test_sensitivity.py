"""T41: sensitivity computation (pure functions)."""

from __future__ import annotations

import copy

import pytest

from src.application.run_context import RunContext
from src.application.services.sensitivity import Deltas, impacts, run_sensitivity, tornado_data
from tests.integration.test_search_detail_parity import _run_search

CTX = RunContext(apport=80000, cf_cible=-500)


@pytest.fixture(scope="module")
def strategy():
    return _run_search(CTX)


def test_zero_delta_zero_impact_on_every_metric(strategy):
    base = run_sensitivity(strategy, Deltas(), CTX)
    again = run_sensitivity(strategy, Deltas(), CTX)
    assert all(v == 0.0 for v in impacts(base, again).values())
    assert base["tri_annuel"] == pytest.approx(strategy["tri_annuel"], abs=1e-9)
    assert base["enrichissement_net"] == pytest.approx(strategy["enrich_net"])


def test_signs(strategy):
    base = run_sensitivity(strategy, Deltas(), CTX)
    rate = impacts(base, run_sensitivity(strategy, Deltas(taux_pts=1.0), CTX))
    assert rate["cf_stab_mensuel"] < 0 and rate["tri_annuel"] < 0
    rent = impacts(base, run_sensitivity(strategy, Deltas(loyer_pct=10.0), CTX))
    assert rent["cf_stab_mensuel"] > 0 and rent["tri_annuel"] > 0
    vac = impacts(base, run_sensitivity(strategy, Deltas(vacance_pts=3.0), CTX))
    assert vac["cf_stab_mensuel"] < 0
    appr = impacts(base, run_sensitivity(strategy, Deltas(appreciation_pts=1.0), CTX))
    assert appr["tri_annuel"] > 0


def test_no_mutation(strategy):
    before = copy.deepcopy(strategy)
    run_sensitivity(strategy, Deltas(1.0, 2.0, 30.0, 1.0, 1.0), CTX)
    assert strategy == before


def test_tornado_shape(strategy):
    rows = tornado_data(strategy, CTX)
    assert len(rows) == 4
    assert all({"tri_down", "tri_up", "cf_down", "cf_up"} <= r.keys() for r in rows)
    assert rows[0]["cf_up"] < 0 < rows[0]["cf_down"]
