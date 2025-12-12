"""Strategy finder service.

Modular strategy finding service extracted from strategy_finder.py.
Provides combination generation, scoring, and ranking.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, List, Optional, Tuple, Callable


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
    ) -> List[Dict[str, Any]]:
        """Find top strategies matching criteria.
        
        Delegates to legacy implementation for now, but uses modular scoring.
        """
        from strategy_finder import trouver_top_strategies
        
        return trouver_top_strategies(
            apport_disponible=self.apport_disponible,
            cash_flow_cible=self.cash_flow_cible,
            tolerance=self.tolerance,
            briques=self.bricks,
            mode_cf=self.mode_cf,
            qualite_weight=self.qualite_weight,
            eval_params=eval_params,
            horizon_years=horizon_years,
        )
    
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
    ) -> List[Dict[str, Any]]:
        """Sort strategies by preset priority."""
        name = preset_name.lower()
        
        def sort_key(x: Dict[str, Any]) -> tuple:
            if "dscr" in name:
                return (x.get("dscr_norm", 0), x.get("finance_score", 0), -abs(x.get("cf_distance", 0)))
            if "irr" in name or "rendement" in name:
                return (x.get("tri_norm", 0), x.get("finance_score", 0), -abs(x.get("cf_distance", 0)))
            if "cash" in name:
                return (x.get("cf_proximity", 0), x.get("finance_score", 0), x.get("dscr_norm", 0))
            if "patrimoine" in name:
                return (x.get("balanced_score", 0), x.get("enrich_norm", 0), x.get("tri_norm", 0))
            # Default: balanced
            return (x.get("balanced_score", 0), x.get("dscr_norm", 0), x.get("tri_norm", 0))
        
        return sorted(strategies, key=sort_key, reverse=True)
