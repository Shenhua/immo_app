"""AI Debug Context Exporter.

This module provides tools to capture the entire state of the application
(configurations, inputs, intermediate results, and final outputs) into a 
structured JSON file. This file is designed to be consumed by an AI agent 
to instantly understand the context, reproduce issues, and spot logic errors.
"""

import json
import abc
from datetime import datetime
from typing import Any, Dict, List
import numpy as np
import pandas as pd

class AIContextEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy and Pandas types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super().default(obj)

def collect_debug_context(
    session_state: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
    strategies: List[Dict[str, Any]] = None,
    last_simulation: pd.DataFrame = None,
    log_file_path: str = "logs/app.log"
) -> Dict[str, Any]:
    """Collect full application context."""
    
    timestamp = datetime.now().isoformat()
    
    context = {
        "meta": {
            "timestamp": timestamp,
            "version": "v27.7 (Hybrid-Traceable)",
            "purpose": "AI Agent Debugging & Logic Verification"
        },
        "user_inputs": {},
        "derived_configs": {},
        "data_snapshot": {},
        "results": {},
        "logs_tail": []
    }

    # 1. User Inputs (Raw Params)
    if params:
        # Sanitize sensitive data if any (none really in this app)
        context["user_inputs"] = params

    # 2. Derived Configs (Logic Check)
    # Re-construct configs to see if translation from UI -> Domain was correct
    try:
        from src.ui.app_controller import build_financing_config, build_operating_config
        if params:
             # We try to rebuild them to show the AI what the domain layer 'sees'
             # This helps spot issued in the build_config functions themselves
             fin = build_financing_config(params.get("credit_params", {}))
             op = build_operating_config(
                 params.get("gestion", 0), 
                 params.get("vacance", 0), 
                 params.get("cfe", 0)
             )
             context["derived_configs"]["financing_config"] = fin.__dict__
             context["derived_configs"]["operating_config"] = op.__dict__
    except Exception as e:
        context["derived_configs"]["error"] = str(e)

    # 3. Data Snapshot (Archetypes)
    if params and "raw_archetypes" in params:
        raw = params["raw_archetypes"]
        # Summarize to avoid massive file if too many
        context["data_snapshot"]["archetypes_count"] = len(raw)
        context["data_snapshot"]["sample_archetype"] = raw[0] if raw else None
        # Statistics
        prices = [b.get("prix_achat", 0) for b in raw]
        rents = [b.get("loyer_CC", 0) for b in raw]
        context["data_snapshot"]["stats"] = {
            "min_price": min(prices) if prices else 0,
            "max_price": max(prices) if prices else 0,
            "avg_rent": sum(rents)/len(rents) if rents else 0
        }

    # 4. Results (Strategies)
    if strategies:
        context["results"]["strategies_found_count"] = len(strategies)
        # Deep dump of top 3 strategies
        context["results"]["top_strategies"] = strategies[:3]
        
        # Logic Verification Helper: Check totals
        # Add a "AI Verification" block where we manually sum things up for the AI to check against
        if strategies:
            top = strategies[0]
            calc_check = {
                "sum_prices": sum(d.get("prix_achat_bien", 0) for d in top.get("details", [])),
                "sum_rents": sum(d.get("loyer_mensuel_initial", 0) for d in top.get("details", [])),
                "reported_cf": top.get("cash_flow_final", 0)
            }
            context["results"]["top_strategy_checksum"] = calc_check

    # 5. Simulation Details
    if last_simulation is not None and not last_simulation.empty:
        context["results"]["selected_simulation"] = {
            "year_1": last_simulation.iloc[0].to_dict(),
            "year_5": last_simulation.iloc[4].to_dict() if len(last_simulation) > 4 else None,
            "year_final": last_simulation.iloc[-1].to_dict(),
            "full_dataframe_dump": last_simulation.to_dict(orient="list") # Efficient col-wise
        }

    # 6. Logs (Last 50 lines)
    try:
        if log_file_path:
            import os
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as f:
                    lines = f.readlines()
                    context["logs_tail"] = [l.strip() for l in lines[-100:]] # Increased to 100
                    
                    # 7. AI Traceability Extraction (Phase 22)
                    # Parse critical events from logs to structurally expose Solver Mode
                    solver_trace = {}
                    for line in reversed(context["logs_tail"]):
                        if "hybrid_solver_selected" in line:
                            # Extract JSON-like content if structlog or just quick parse
                            # Assuming line format: ... event=hybrid_solver_selected mode=EXHAUSTIVE combos=123 ...
                            try:
                                # Simple partial extraction for robustness
                                parts = line.split()
                                for p in parts:
                                    if "mode=" in p: solver_trace["mode"] = p.split("=")[1].replace('"', '')
                                    if "combos=" in p: solver_trace["combos"] = str(p.split("=")[1])
                                    if "threshold=" in p: solver_trace["threshold"] = str(p.split("=")[1])
                            except:
                                solver_trace["raw_event"] = line
                            break
                            
                    # Extract Tier Distribution if available
                    for line in reversed(context["logs_tail"]):
                         if "strategy_tiers" in line:
                             try:
                                 parts = line.split()
                                 tiers = {}
                                 for p in parts:
                                     if "tier_" in p:
                                         k, v = p.split("=")
                                         tiers[k] = int(v)
                                 solver_trace["tier_distribution"] = tiers
                             except:
                                 pass
                             break
                             
                    if solver_trace:
                        context["results"]["solver_trace"] = solver_trace

    except Exception as e:
        context["logs_tail"] = [f"Could not read logs: {e}"]

    return context

def save_debug_context(context: Dict[str, Any], filepath: str = "debug_context.json") -> str:
    """Save context to disk."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(context, f, cls=AIContextEncoder, indent=2, ensure_ascii=False)
    return filepath
