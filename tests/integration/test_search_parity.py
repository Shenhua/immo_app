"""T31: the rewritten search must return what the old exhaustive solver returned on small instances.

``search_parity_golden.json`` was captured from the pre-T31 ``ExhaustiveOptimizer`` (commit a496ffc, develop)
by running this module with ``REGEN_SEARCH_GOLDEN=1``. The old solver is deleted by T31, so the golden file is
the only copy of its output; do not regenerate it unless a later task changes the expected results on purpose.

Cases P1/P2 accept every combination at its minimum down payment on both cash-flow bases (pre-tax allocator and
after-tax evaluator), so they compare the full pipeline (enumeration, evaluation, scoring, dedupe, ranking)
exactly. Case P3 needs allocation: there the T31 fixes (no step truncation, after-tax acceptance) may legitimately
change the result set, and the test only checks the strategies both solvers accept.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.application.run_context import RunContext
from src.application.services.brick_factory import (
    build_financing_config,
    build_operating_config,
    create_investment_bricks,
)
from src.application.services.strategy_finder import StrategyFinder

GOLDEN = Path(__file__).with_name("search_parity_golden.json")
DATA = Path(__file__).resolve().parents[2] / "data" / "archetypes_2026_v1.json"
ARCHETYPE_IDX = [8, 10, 13, 7, 0, 2]  # 4 cheap furnished + 2 unfurnished (foncier tax path)

CASES: dict[str, dict] = {
    "P1_min_loose": {"apport": 100000, "cf_cible": -3000.0, "mode_cf": "min", "max_properties": 3},
    "P2_target_wide": {"apport": 80000, "cf_cible": -1500.0, "tolerance": 2500.0, "mode_cf": "target",
                       "max_properties": 2},
    "P3_min_allocating": {"apport": 120000, "cf_cible": -450.0, "mode_cf": "min", "max_properties": 2},
}
EXACT_CASES = ("P1_min_loose", "P2_target_wide")


def _archetypes() -> list[dict]:
    from src.domain.models.archetype import ArchetypeV2

    raw = json.loads(DATA.read_text(encoding="utf-8"))
    return [ArchetypeV2(**raw[i]).model_dump() for i in ARCHETYPE_IDX]


def _record(s: dict) -> dict:
    return {
        "sig": sorted([d["nom"], int(d["duree_credit_mois"])] for d in s["details"]),
        "apport_total": float(s["apport_total"]),
        "enrich_net": float(s["enrich_net"]),
        "tri_annuel": None if s.get("tri_annuel") is None else float(s["tri_annuel"]),
        "cf_monthly_stab": float(s["cf_monthly_stab"]),
        "balanced_score": float(s["balanced_score"]),
        "is_acceptable": bool(s.get("is_acceptable", True)),
    }


def _run(case: dict) -> list[dict]:
    ctx = RunContext(**case)
    fin = build_financing_config(ctx.to_credit_params())
    op = build_operating_config(
        ctx.gestion_pct, ctx.vacance_pct, ctx.cfe_par_bien_ann,
        maintenance_m2_an=ctx.maintenance_m2_an, gli_pct=ctx.gli_pct, mois_travaux=ctx.mois_travaux,
    )
    bricks = create_investment_bricks(_archetypes(), fin, op)
    finder = StrategyFinder(
        bricks=bricks, apport_disponible=ctx.apport, cash_flow_cible=ctx.cf_cible, tolerance=ctx.tolerance,
        qualite_weight=ctx.qualite_weight, mode_cf=ctx.mode_cf, max_properties=ctx.max_properties,
    )
    res = finder.find_strategies(eval_params=ctx.to_eval_params(), horizon_years=ctx.horizon_ans, top_n=ctx.top_n)
    return [_record(s) for s in res]


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


if os.environ.get("REGEN_SEARCH_GOLDEN") == "1":  # pragma: no cover - manual capture only
    GOLDEN.write_text(json.dumps({k: _run(c) for k, c in CASES.items()}, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")


def _approx(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return a == pytest.approx(b, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("name", EXACT_CASES)
def test_same_ranked_results_as_old_exhaustive(name):
    old = _golden()[name]
    new = _run(CASES[name])
    assert old, "the golden case must not be empty"
    assert [r["sig"] for r in new] == [r["sig"] for r in old]
    for o, n in zip(old, new, strict=True):
        for key in ("apport_total", "enrich_net", "tri_annuel", "cf_monthly_stab", "balanced_score"):
            assert _approx(n[key], o[key]), (o["sig"], key, o[key], n[key])


def test_allocating_case_agrees_where_both_solvers_accept():
    """Every old result whose after-tax CF meets the target is still found, with the same down payment (±1 %)."""
    case = CASES["P3_min_allocating"]
    old = [r for r in _golden()["P3_min_allocating"] if r["is_acceptable"]]
    new = {tuple(tuple(x) for x in r["sig"]): r for r in _run(case)}
    assert old
    new_names = {tuple(sorted(n for n, _ in sig)) for sig in new}
    for o in old:
        names = tuple(sorted(n for n, _ in o["sig"]))
        assert names in new_names, f"old accepted portfolio {names} missing from the new results"
        same = new.get(tuple(tuple(x) for x in o["sig"]))
        if same is not None:
            assert same["apport_total"] == pytest.approx(o["apport_total"], rel=0.01)
        assert all(r["cf_monthly_stab"] >= case["cf_cible"] - 1e-6 for r in new.values())
