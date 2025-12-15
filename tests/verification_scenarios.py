"""Scenario Verification Runner.

Runs the strategy engine against defined user personas to verify output consistency
and sanity.
"""

import json
import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logging import configure_logging, get_logger
from src.models.archetype import ArchetypeV2
from src.services.brick_factory import FinancingConfig, OperatingConfig, create_investment_bricks
from src.services.exporter import ResultExporter
from src.services.strategy_finder import StrategyFinder

# Configure logging to console
configure_logging(json_output=False)
log = get_logger("scenario_runner")


def load_archetypes():
    with open("data/archetypes_recale_2025_v2.json") as f:
        return [ArchetypeV2(**a).model_dump() for a in json.load(f)]


SCENARIOS = {
    "CONSERVATIVE": {
        "apport": 50000.0,
        "cash_flow_target": -200.0,
        "finance": FinancingConfig(
            credit_rates={15: 3.2},  # Only short term
            apport_min_pct=20.0,     # High down payment
        ),
        "checks": {
            "min_dscr": 1.1,
            "max_leverage": 0.85,
        }
    },
    "AGGRESSIVE": {
        "apport": 15000.0,
        "cash_flow_target": 0.0,
        "finance": FinancingConfig(
            credit_rates={25: 3.8},  # Long term only
            apport_min_pct=10.0,     # Low down payment
        ),
        "checks": {
            "min_yield": 4.0,
        }
    },
    "BALANCED": {
        "apport": 30000.0,
        "cash_flow_target": -50.0,
        "finance": FinancingConfig(
            credit_rates={20: 3.5, 25: 3.7},
            apport_min_pct=15.0,
        ),
        "checks": {}
    }
}


def run_scenarios():
    log.info("scenario_run_started")
    archetypes = load_archetypes()
    exporter = ResultExporter("results/scenarios")

    op_config = OperatingConfig() # Default

    results = {}

    for name, cfg in SCENARIOS.items():
        log.info(f"running_scenario_{name}")

        # 1. Create Bricks
        bricks = create_investment_bricks(
            archetypes,
            cfg["finance"],
            op_config
        )

        # 2. Find Strategies
        finder = StrategyFinder(
            bricks=bricks,
            apport_disponible=cfg["apport"],
            cash_flow_cible=cfg["cash_flow_target"]
        )

        strategies = finder.find_strategies()

        # 3. Validation Check
        sanity_issues = []
        checks = cfg.get("checks", {})

        for s in strategies:
            # Check DSCR
            if "min_dscr" in checks and float(s.get("dscr_y1", 0)) < checks["min_dscr"]:
                sanity_issues.append(f"Low DSCR: {s['dscr_y1']}")

            # Check Yield
            if "min_yield" in checks and float(s.get("renta_brute", 0)) < checks["min_yield"]:
                sanity_issues.append(f"Low Yield: {s['renta_brute']}")

        if sanity_issues:
            log.warning(f"sanity_check_warnings_{name}", count=len(sanity_issues), sample=sanity_issues[:3])
        else:
            log.info(f"sanity_check_passed_{name}")

        # 4. Save
        exporter.save_results(strategies, prefix=f"scenario_{name}", metadata=cfg.get("checks"))
        results[name] = len(strategies)

    log.info("all_scenarios_completed", results=results)


if __name__ == "__main__":
    run_scenarios()
