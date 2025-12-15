"""Strategy finder service.

Modular strategy finding service extracted from strategy_finder.py.
Provides combination generation, scoring, and ranking.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, List, Optional, Tuple, Callable

from src.core.logging import get_logger

log = get_logger(__name__)


# Base financial scoring weights
BASE_WEIGHTS = {
    "enrich_net": 0.30,
    "irr": 0.25,
    "cf_proximity": 0.20,
    "dscr": 0.15,
    "cap_eff": 0.10,
}


@dataclass
class EvaluationParams:
    """Parameters for strategy evaluation."""
    
    duree_simulation_ans: int = 25
    hypotheses_marche: Dict[str, float] = field(default_factory=lambda: {
        "appreciation_bien_pct": 2.0,
        "revalo_loyer_pct": 1.5,
        "inflation_charges_pct": 2.0,
    })
    regime_fiscal: str = "lmnp"
    tmi_pct: float = 30.0
    frais_vente_pct: float = 6.0
    cfe_par_bien_ann: float = 150.0
    apply_ira: bool = True
    ira_cap_pct: float = 3.0
    finance_weights_override: Optional[Dict[str, float]] = None
    finance_preset_name: str = "Équilibré (défaut)"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationParams":
        return cls(
            duree_simulation_ans=d.get("duree_simulation_ans", 25),
            hypotheses_marche=d.get("hypotheses_marche", cls.__dataclass_fields__["hypotheses_marche"].default_factory()),
            regime_fiscal=d.get("regime_fiscal", "lmnp"),
            tmi_pct=d.get("tmi_pct", 30.0),
            frais_vente_pct=d.get("frais_vente_pct", 6.0),
            cfe_par_bien_ann=d.get("cfe_par_bien_ann", 150.0),
            apply_ira=d.get("apply_ira", True),
            ira_cap_pct=d.get("ira_cap_pct", 3.0),
            finance_weights_override=d.get("finance_weights_override"),
            finance_preset_name=d.get("finance_preset_name", "Équilibré (défaut)"),
        )


class StrategyScorer:
    """Scores strategies based on financial and qualitative metrics."""
    
    def __init__(self, qualite_weight: float = 0.25, weights: Optional[Dict[str, float]] = None):
        self.qualite_weight = max(0.0, min(1.0, qualite_weight))
        self.weights = self._normalize_weights(weights or BASE_WEIGHTS)
    
    @staticmethod
    def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        if not isinstance(w, dict):
            return BASE_WEIGHTS.copy()
        s = sum(max(0.0, float(v)) for v in w.values() if v is not None)
        if s <= 0:
            return BASE_WEIGHTS.copy()
        return {k: max(0.0, float(v)) / s for k, v in w.items()}
    
    @staticmethod
    def minmax_normalize(values: List[float], lo: Optional[float] = None, hi: Optional[float] = None) -> List[float]:
        """Min-max normalize a list of values to 0-1 range."""
        finite_vals = [v for v in values if isfinite(v)]
        if not finite_vals:
            return [0.0] * len(values)
        
        if lo is None:
            lo = min(finite_vals)
        if hi is None:
            hi = max(finite_vals)
        
        if hi <= lo:
            return [0.5] * len(values)
        
        return [
            0.0 if not isfinite(v) else max(0.0, min(1.0, (v - lo) / (hi - lo)))
            for v in values
        ]
    
    def compute_finance_score(self, strategy: Dict[str, Any]) -> float:
        """Compute weighted finance score for a single strategy."""
        w = self.weights
        return (
            w.get("enrich_net", 0.0) * strategy.get("enrich_norm", 0.0) +
            w.get("irr", 0.0) * strategy.get("tri_norm", 0.0) +
            w.get("cap_eff", 0.0) * strategy.get("cap_eff_norm", 0.0) +
            w.get("dscr", 0.0) * strategy.get("dscr_norm", 0.0) +
            w.get("cf_proximity", 0.0) * strategy.get("cf_proximity", 0.0)
        )
    
    def compute_balanced_score(self, strategy: Dict[str, Any]) -> float:
        """Combine finance and quality scores."""
        fin = strategy.get("finance_score", 0.0)
        qual = strategy.get("qual_score", 50.0) / 100.0
        return (1.0 - self.qualite_weight) * fin + self.qualite_weight * qual
    
    def score_strategies(self, strategies: List[Dict[str, Any]], cash_flow_cible: float) -> None:
        """Add normalized scores to all strategies (mutates in place)."""
        if not strategies:
            return
        
        # Extract raw metrics
        enrich = [s.get("enrich_net", 0.0) for s in strategies]
        irr = [s.get("tri_annuel", 0.0) for s in strategies]
        cap_eff = [s.get("cap_eff", 0.0) for s in strategies]
        dscr = [1.5 if s.get("dscr_y1") is None else max(0.0, float(s.get("dscr_y1", 0.0))) for s in strategies]
        cf_dist = [abs(s.get("cash_flow_final", 0.0) - cash_flow_cible) for s in strategies]
        cf_prox = [max(0.0, 1.0 - (d / 300.0)) for d in cf_dist]
        
        # Normalize
        enrich_n = self.minmax_normalize(enrich)
        irr_n = self.minmax_normalize(irr)
        cap_n = self.minmax_normalize([min(v, 6.0) for v in cap_eff], 0.0, 6.0)
        dscr_n = self.minmax_normalize([min(v, 1.5) for v in dscr], 0.0, 1.5)
        
        # Apply to strategies
        for s, en, ir, ca, ds, cf in zip(strategies, enrich_n, irr_n, cap_n, dscr_n, cf_prox):
            s["enrich_norm"] = en
            s["tri_norm"] = ir
            s["cap_eff_norm"] = ca
            s["dscr_norm"] = ds
            s["cf_proximity"] = cf
            s["finance_score"] = self.compute_finance_score(s)
            s["balanced_score"] = self.compute_balanced_score(s)


class CombinationGenerator:
    """Generates valid property combinations within budget."""
    
    def __init__(self, max_properties: int = 3):
        self.max_properties = max_properties
    
    def generate(
        self,
        bricks: List[Dict[str, Any]],
        apport_disponible: float,
    ) -> List[Tuple[Dict[str, Any], ...]]:
        """Generate all valid combinations of bricks.
        
        Filters:
        - No duplicate properties (same nom_bien)
        - Total apport_min <= apport_disponible
        """
        combos = []
        for k in range(1, self.max_properties + 1):
            for combo in itertools.combinations(bricks, k):
                # Check unique properties
                noms = {c.get("nom_bien") for c in combo if c.get("nom_bien")}
                if len(noms) != len(combo):
                    continue
                
                # Check budget
                apport_min = sum(c.get("apport_min", 0.0) for c in combo)
                if apport_min > apport_disponible:
                    continue
                
                combos.append(combo)
        
        return combos


class StrategyFinder:
    """Service for finding optimal investment strategies.
    
    Orchestrates combination generation, allocation, simulation, and scoring.
    """
    
    def __init__(
        self,
        bricks: List[Dict[str, Any]],
        apport_disponible: float,
        cash_flow_cible: float,
        tolerance: float = 100.0,
        qualite_weight: float = 0.25,
        mode_cf: str = "target",
    ):
        self.bricks = bricks
        self.apport_disponible = apport_disponible
        self.cash_flow_cible = cash_flow_cible
        self.tolerance = tolerance
        self.qualite_weight = qualite_weight
        self.mode_cf = mode_cf
        
        self.combo_generator = CombinationGenerator()
        self.scorer = StrategyScorer(qualite_weight=qualite_weight)
    
    def find_strategies(
        self,
        eval_params: Optional[Dict[str, Any]] = None,
        horizon_years: int = 25,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find top strategies matching criteria.
        
        Orchestrates the full search process:
        1. Generate combinations
        2. Allocate capital
        3. Simulate financial performance
        4. Score and rank
        """
        # Dependencies
        from src.services.allocator import PortfolioAllocator
        from src.core.simulation import SimulationEngine, MarketHypotheses, TaxParams, IRACalculator
        from src.core.financial import generate_amortization_schedule
        
        # 1. Setup
        ep = EvaluationParams.from_dict(eval_params or {})
        
        # Update scorer weights from eval_params
        if ep.finance_weights_override:
            self.scorer.weights = self.scorer._normalize_weights(ep.finance_weights_override)
        
        market = MarketHypotheses(
            appreciation_bien_pct=ep.hypotheses_marche["appreciation_bien_pct"],
            revalo_loyer_pct=ep.hypotheses_marche["revalo_loyer_pct"],
            inflation_charges_pct=ep.hypotheses_marche["inflation_charges_pct"],
        )
        tax = TaxParams(
            tmi_pct=ep.tmi_pct,
            regime_fiscal=ep.regime_fiscal,
        )
        ira = IRACalculator(
            apply_ira=ep.apply_ira,
            ira_cap_pct=ep.ira_cap_pct,
        )
        engine = SimulationEngine(
            market=market, 
            tax=tax, 
            ira=ira,
            cfe_par_bien_ann=ep.cfe_par_bien_ann,
            frais_vente_pct=ep.frais_vente_pct,
        )
        
        allocator = PortfolioAllocator(self.mode_cf)
        
        log.info("strategy_search_started", 
                 brick_count=len(self.bricks), 
                 apport=self.apport_disponible)
                 
        # 2. Generate combinations
        combos = self.combo_generator.generate(self.bricks, self.apport_disponible)
        log.debug("combinations_generated", count=len(combos))
        
        strategies = []
        
        # 3. Process candidates
        for combo in combos:
            bricks_copy = [dict(b) for b in combo]
            
            ok, details, cf_final, apport_used = allocator.allocate(
                bricks_copy, 
                self.apport_disponible, 
                self.cash_flow_cible, 
                self.tolerance
            )
            
            if ok:
                strat = {
                    "details": details,
                    "apport_total": apport_used,
                    "patrimoine_acquis": sum(b.get("cout_total", 0.0) for b in details),
                    "cash_flow_final": cf_final,
                }
                
                # Basic Metrics
                total_rent = sum(b.get("loyer_mensuel_initial", 0.0) for b in strat["details"]) * 12
                total_cost = sum(b.get("cout_total", 0.0) for b in strat["details"])
                strat["renta_brute"] = (total_rent / total_cost * 100.0) if total_cost > 0 else 0.0
                
                # Simulation
                try:
                    schedules = [
                        generate_amortization_schedule(
                            float(p["credit_final"]), 
                            float(p["taux_pret"]), 
                            int(p["duree_pret"]) * 12, 
                            float(p["assurance_ann_pct"])
                        ) for p in strat["details"]
                    ]
                    
                    df_sim, bilan = engine.simulate(strat, horizon_years, schedules)
                    
                    # Metrics
                    strat["tri_annuel"] = float(bilan.get("tri_annuel", 0.0))
                    strat["liquidation_nette"] = float(bilan.get("liquidation_nette", 0.0))
                    
                    # DSCR Y1
                    if not df_sim.empty:
                        row = df_sim.iloc[0]
                        ds = row["Capital Remboursé"] + row["Intérêts & Assurance"]
                        # Charges Deductibles (neg) includes Interest. Add back to get OpEx (neg).
                        opex = row["Charges Déductibles"] + row["Intérêts & Assurance"] 
                        noi = row["Loyers Bruts"] + opex
                        strat["dscr_y1"] = (noi / ds) if ds > 1e-9 else 0.0
                    else:
                        strat["dscr_y1"] = 0.0

                    # Qualitative
                    strat["qual_score"] = self._calculate_qualitative_score(strat)
                    strat["cf_distance"] = abs(strat["cash_flow_final"] - self.cash_flow_cible)
                    
                    ap = float(strat.get("apport_total", 1.0)) or 1.0
                    strat["cap_eff"] = (strat["liquidation_nette"] - ap) / ap
                    strat["enrich_net"] = strat["liquidation_nette"] - ap
                    
                    strategies.append(strat)

                except Exception as e:
                    log.warning("strategy_simulation_failed", error=str(e))
                    continue
                
        # 4. Score & Rank
        self.scorer.score_strategies(strategies, self.cash_flow_cible)
        
        # Dedupe
        strategies = self.dedupe_strategies(strategies)
        
        top_strategies = self.rank_strategies(strategies, ep.finance_preset_name, top_n)
        log.info("strategy_search_completed", 
                 total_evaluated=len(strategies), 
                 top_kept=len(top_strategies))
                 
        return top_strategies

    def _calculate_qualitative_score(self, strategy: Dict[str, Any]) -> float:
        """Calculate qualitative score (internal helper)."""
        # Import only as needed to avoid top-level circular dependency if any
        # But ideally we should move scoring logic here fully.
        # For now, using simple weighted average of bricks
        
        total_price = sum(b.get("prix_achat_bien", 0) for b in strategy.get("details", []))
        if total_price <= 0:
            return 50.0
            
        score_sum = 0.0
        for b in strategy.get("details", []):
            qs = b.get("qual_score_bien", 50.0)
            price = b.get("prix_achat_bien", 0)
            score_sum += qs * price
            
        return score_sum / total_price
    
    def dedupe_strategies(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove near-duplicate strategies.
        
        Signature = sorted tuples of (nom_bien, durée, apport rounded to 100€).
        """
        seen = set()
        out = []
        for s in strategies:
            details = s.get("details", [])
            sig = tuple(sorted(
                (
                    d.get("nom_bien"),
                    int(d.get("duree_pret", 0)),
                    int(round(float(d.get("apport_final_bien", 0.0)) / 100.0) * 100),
                )
                for d in details
            ))
            if sig not in seen:
                seen.add(sig)
                out.append(s)
        return out
    
    def rank_strategies(
        self,
        strategies: List[Dict[str, Any]],
        preset_name: str = "Équilibré (défaut)",
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Sort strategies by preset priority.
        
        The ranking respects the user's profile selection:
        - Primary: Use the weighted finance_score (incorporates user weights)
        - Secondary: Preset-specific tiebreaker metric
        - Tertiary: CF proximity (closer to target is better)
        """
        name = preset_name.lower()
        
        def sort_key(x: Dict[str, Any]) -> tuple:
            # CF proximity as final tiebreaker (higher is better)
            cf_prox = x.get("cf_proximity", 0)
            finance = x.get("finance_score", 0)
            balanced = x.get("balanced_score", 0)
            
            # Match French preset names
            if "sécurité" in name or "dscr" in name:
                # Safety profile: prioritize DSCR, then finance_score
                return (x.get("dscr_norm", 0), finance, cf_prox)
            
            if "rendement" in name or "irr" in name:
                # IRR profile: prioritize TRI, then finance_score
                return (x.get("tri_norm", 0), finance, cf_prox)
            
            if "cash" in name:
                # Cash-flow profile: prioritize CF proximity, then DSCR for safety
                return (cf_prox, x.get("dscr_norm", 0), finance)
            
            if "patrimoine" in name:
                # Wealth-building profile: prioritize enrichment, then IRR
                return (x.get("enrich_norm", 0), x.get("tri_norm", 0), balanced)
            
            # Default "Équilibré": use balanced_score which combines finance + quality
            return (balanced, finance, cf_prox)
        
        sorted_strategies = sorted(strategies, key=sort_key, reverse=True)
        return sorted_strategies[:top_n]

