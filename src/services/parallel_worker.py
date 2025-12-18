"""
Parallel worker functions for strategy evaluation.

These are module-level functions designed to be pickle-safe for ProcessPoolExecutor.
They avoid class method references which cannot be pickled.
"""
from typing import Any


def evaluate_combo_worker(args: tuple) -> dict[str, Any] | None:
    """
    Worker function to evaluate a single combination.

    Designed to be called by ProcessPoolExecutor. All dependencies are
    recreated in the worker process to avoid pickle issues.

    Args:
        args: Tuple of (combo, budget, target_cf, tolerance, horizon, scorer_weights, qualite_weight, mode_cf)

    Returns:
        Strategy dict or None if invalid
    """
    (
        combo,
        budget,
        target_cf,
        tolerance,
        horizon,
        scorer_weights,
        qualite_weight,
        mode_cf
    ) = args

    # Import here to avoid circular imports and ensure fresh instances
    from src.core.financial import generate_amortization_schedule
    from src.core.glossary import calculate_cashflow_metrics
    from src.core.scoring import calculate_qualitative_score
    from src.core.simulation import SimulationEngine
    from src.services.allocator import PortfolioAllocator

    try:
        # Create allocator with the right mode
        allocator = PortfolioAllocator(mode_cf=mode_cf)

        # 1. Allocation
        ok, details, cf_final, apport_used = allocator.allocate(
            list(combo), budget, target_cf, tolerance
        )

        if not ok:
            return None

        # 2. Generate schedules
        schedules = []
        for p in details:
            principal = float(p.get("credit_final", p.get("capital_restant", 0)))
            sch = generate_amortization_schedule(
                principal=principal,
                annual_rate_pct=float(p.get("taux_pret", 0.0)),
                duration_months=int(p.get("duree_pret", 20)) * 12,
                annual_insurance_pct=float(p.get("assurance_ann_pct", 0.0))
            )
            schedules.append(sch)

        # 3. Simulation
        strategy = {"details": details, "apport_total": apport_used}
        simulator = SimulationEngine()
        df_sim, bilan = simulator.simulate(strategy, horizon, schedules)

        # 4. CF metrics
        cf_metrics = calculate_cashflow_metrics(df_sim, target_cf, tolerance, mode_cf=mode_cf)

        # 5. Finance score
        w = scorer_weights or {
            "irr": 0.25, "enrich_net": 0.30, "dscr": 0.15,
            "cf_proximity": 0.20, "cap_eff": 0.10
        }

        tri = bilan.get("tri_annuel", 0.0)
        s_tri = max(0.0, min(1.0, tri / 20.0))

        enrich = bilan.get("enrichissement_net", 0.0)
        safe_apport = max(1.0, apport_used)
        roe = enrich / safe_apport
        s_enrich = max(0.0, min(1.0, roe / 2.0))

        dscr = float(bilan.get("dscr_y1", 0.0) or 0.0)
        s_dscr = max(0.0, min(1.0, dscr / 1.3))

        gap = cf_metrics.get("gap", 0.0)
        safe_tol = tolerance if tolerance > 1.0 else 100.0
        s_cf = max(0.0, 1.0 - (gap / (safe_tol * 2.0)))

        finance_score = (
            w.get("irr", 0.25) * s_tri +
            w.get("enrich_net", 0.30) * s_enrich +
            w.get("dscr", 0.15) * s_dscr +
            w.get("cf_proximity", 0.20) * s_cf +
            w.get("cap_eff", 0.10) * s_enrich
        )

        # 6. Quality score
        qual_score = calculate_qualitative_score({"details": details})

        # 7. Combined score
        q_w = qualite_weight if qualite_weight is not None else 0.5
        if q_w >= 1.0:
            combined = qual_score / 100.0
        elif q_w <= 0.0:
            combined = finance_score
        else:
            combined = (1.0 - q_w) * finance_score + q_w * (qual_score / 100.0)

        # Build result
        result = {
            "details": details,
            "apport_total": apport_used,
            "cash_flow_final": cf_final,
            "allocation_ok": ok,
            "liquidation_nette": bilan.get("liquidation_nette", 0.0),
            "enrich_net": bilan.get("enrichissement_net", 0.0),
            "tri_annuel": bilan.get("tri_annuel", 0.0),
            "tri_global": bilan.get("tri_annuel", 0.0),
            "dscr_y1": bilan.get("dscr_y1", 0.0),
            "cf_monthly_y1": cf_metrics.get("cf_year_1_monthly", 0.0),
            "cf_monthly_avg": cf_metrics.get("cf_avg_5y_monthly", 0.0),
            "is_acceptable": cf_metrics.get("is_acceptable", False),
            "fitness": max(0.01, combined * 100),
            "qual_score": qual_score,
        }

        return result

    except Exception:
        return None


def evaluate_batch_worker(args: tuple) -> list[dict[str, Any]]:
    """
    Worker function to evaluate a batch of combinations.

    This reduces IPC overhead by processing multiple combos per worker.

    Args:
        args: Tuple of (combo_batch, budget, target_cf, tolerance, horizon, scorer_weights, qualite_weight, mode_cf)

    Returns:
        List of valid strategy dicts
    """
    (
        combo_batch,
        budget,
        target_cf,
        tolerance,
        horizon,
        scorer_weights,
        qualite_weight,
        mode_cf
    ) = args

    results = []
    for combo in combo_batch:
        single_args = (combo, budget, target_cf, tolerance, horizon, scorer_weights, qualite_weight, mode_cf)
        result = evaluate_combo_worker(single_args)
        if result is not None:
            results.append(result)

    return results
