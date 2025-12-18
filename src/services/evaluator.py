"""
Strategy Evaluator - Shared evaluation logic for strategy optimization.

This module consolidates the evaluation pipeline used by both GeneticOptimizer
and ExhaustiveOptimizer, reducing code duplication and ensuring consistency.
"""
from typing import Any, Dict, List, Tuple
import structlog

from src.core.financial import generate_amortization_schedule
from src.core.glossary import calculate_cashflow_metrics, calculate_enrichment_metrics
from src.core.scoring import calculate_qualitative_score
from src.core.simulation import SimulationEngine
from src.services.allocator import PortfolioAllocator

log = structlog.get_logger(__name__)


class StrategyEvaluator:
    """
    Evaluates investment strategies by running simulation and scoring.
    
    Consolidates shared logic previously duplicated in GeneticOptimizer 
    and ExhaustiveOptimizer.
    """
    
    # Default scoring weights (from scoring_constants)
    DEFAULT_WEIGHTS = {
        "irr": 0.25,
        "enrich_net": 0.30,
        "dscr": 0.15,
        "cf_proximity": 0.20,
        "cap_eff": 0.10
    }
    
    def __init__(
        self,
        simulator: SimulationEngine,
        allocator: PortfolioAllocator = None,
        scorer: Any = None
    ):
        """
        Initialize evaluator with dependencies.
        
        Args:
            simulator: Simulation engine for long-term projections
            allocator: Portfolio allocator (for mode_cf)
            scorer: StrategyScorer instance (for weights)
        """
        self.simulator = simulator
        self.allocator = allocator
        self.scorer = scorer
    
    @property
    def weights(self) -> Dict[str, float]:
        """Get scoring weights from scorer or use defaults."""
        if self.scorer and hasattr(self.scorer, 'weights'):
            return self.scorer.weights
        return self.DEFAULT_WEIGHTS
    
    @property
    def quality_weight(self) -> float:
        """Get quality vs finance weighting."""
        if self.scorer and hasattr(self.scorer, 'qualite_weight'):
            return self.scorer.qualite_weight
        return 0.5
    
    @property
    def mode_cf(self) -> str:
        """Get cash flow mode from allocator."""
        if self.allocator and hasattr(self.allocator, 'mode_cf'):
            return self.allocator.mode_cf
        return "target"
    
    def generate_schedules(self, details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate amortization schedules for all properties.
        
        Args:
            details: List of property details (from allocation)
            
        Returns:
            List of amortization schedule dicts
        """
        schedules = []
        for p in details:
            # Use credit_final (already calculated by allocator)
            principal = float(p.get("credit_final", p.get("capital_restant", 0)))
            
            sch = generate_amortization_schedule(
                principal=principal,
                annual_rate_pct=float(p.get("taux_pret", 0.0)),
                duration_months=int(p.get("duree_pret", 20)) * 12,
                annual_insurance_pct=float(p.get("assurance_ann_pct", 0.0))
            )
            schedules.append(sch)
        
        return schedules
    
    def run_simulation(
        self,
        strategy: Dict[str, Any],
        horizon: int,
        schedules: List[Dict[str, Any]]
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run simulation for a strategy.
        
        Args:
            strategy: Strategy dict with 'details' and 'apport_total'
            horizon: Simulation horizon in years
            schedules: Amortization schedules for each property
            
        Returns:
            Tuple of (df_sim, bilan)
        """
        return self.simulator.simulate(strategy, horizon, schedules)
    
    def calculate_cf_metrics(
        self,
        df_sim: Any,
        target_cf: float,
        tolerance: float
    ) -> Dict[str, Any]:
        """
        Calculate cash flow metrics from simulation results.
        
        Args:
            df_sim: Simulation DataFrame
            target_cf: Target monthly cash flow
            tolerance: Tolerance for CF target
            
        Returns:
            Dict with CF metrics (is_acceptable, gap, cf_year_1_monthly, etc.)
        """
        return calculate_cashflow_metrics(df_sim, target_cf, tolerance, mode_cf=self.mode_cf)
    
    def calculate_enrichment_metrics(
        self,
        bilan: Dict[str, Any],
        apport_total: float,
        horizon: int
    ) -> Dict[str, Any]:
        """
        Calculate enrichment metrics from simulation bilan.
        
        Args:
            bilan: Simulation bilan dict
            apport_total: Total capital invested
            horizon: Simulation horizon in years
            
        Returns:
            Dict with enrichment metrics
        """
        return calculate_enrichment_metrics(bilan, apport_total, horizon)
    
    def calculate_finance_score(
        self,
        bilan: Dict[str, Any],
        cf_metrics: Dict[str, Any],
        apport_total: float,
        tolerance: float
    ) -> float:
        """
        Calculate finance score (0.0 to 1.0) using absolute scales.
        
        Uses constant targets to avoid normalization instability:
        - TRI target: 20% is excellent
        - Enrichment target: 2x ROE is good
        - DSCR target: 1.3 is safe
        - CF target: 0 gap is perfect
        
        Args:
            bilan: Simulation bilan dict
            cf_metrics: Cash flow metrics dict
            apport_total: Total capital invested
            tolerance: CF tolerance for proximity scoring
            
        Returns:
            Finance score between 0.0 and 1.0
        """
        w = self.weights
        
        # 1. TRI (Internal Rate of Return) - target 20%
        tri = bilan.get("tri_annuel", 0.0)
        s_tri = max(0.0, min(1.0, tri / 20.0))
        
        # 2. Enrichment (ROE) - target 2x
        enrich = bilan.get("enrichissement_net", 0.0)
        safe_apport = max(1.0, apport_total)
        roe = enrich / safe_apport
        s_enrich = max(0.0, min(1.0, roe / 2.0))
        
        # 3. DSCR - target 1.3
        dscr = float(bilan.get("dscr_y1", 0.0) or 0.0)
        s_dscr = max(0.0, min(1.0, dscr / 1.3))
        
        # 4. Cash Flow Proximity - 0 gap is perfect
        gap = cf_metrics.get("gap", 0.0)
        safe_tol = tolerance if tolerance > 1.0 else 100.0
        s_cf = max(0.0, 1.0 - (gap / (safe_tol * 2.0)))
        
        # Weighted average
        score = (
            w.get("irr", 0.25) * s_tri +
            w.get("enrich_net", 0.30) * s_enrich +
            w.get("dscr", 0.15) * s_dscr +
            w.get("cf_proximity", 0.20) * s_cf +
            w.get("cap_eff", 0.10) * s_enrich  # cap_eff uses enrichment as proxy
        )
        
        return score
    
    def calculate_quality_score(self, details: List[Dict[str, Any]]) -> float:
        """
        Calculate qualitative score for strategy.
        
        Args:
            details: List of property details
            
        Returns:
            Quality score between 0.0 and 100.0
        """
        return calculate_qualitative_score({"details": details})
    
    def calculate_combined_score(
        self,
        finance_score: float,
        quality_score: float
    ) -> float:
        """
        Calculate combined/balanced score.
        
        Blends finance and quality scores based on quality_weight:
        - q_w = 1.0 → 100% quality
        - q_w = 0.0 → 100% finance
        - q_w = 0.5 → 50/50 blend
        
        Args:
            finance_score: Finance score (0.0 to 1.0)
            quality_score: Quality score (0.0 to 100.0)
            
        Returns:
            Combined score (0.0 to 1.0)
        """
        q_w = self.quality_weight
        qual_normalized = quality_score / 100.0
        
        if q_w >= 1.0:
            return qual_normalized
        elif q_w <= 0.0:
            return finance_score
        else:
            return (1.0 - q_w) * finance_score + q_w * qual_normalized
    
    def evaluate(
        self,
        strategy: Dict[str, Any],
        target_cf: float,
        tolerance: float,
        horizon: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Full evaluation pipeline for a strategy.
        
        Runs simulation, calculates all metrics, and scores the strategy.
        
        Args:
            strategy: Strategy dict with 'details' and 'apport_total'
            target_cf: Target monthly cash flow
            tolerance: CF tolerance
            horizon: Simulation horizon in years
            
        Returns:
            Tuple of (is_valid, metrics_dict) where metrics_dict contains:
            - liquidation_nette, enrich_net, tri_annuel, tri_global, dscr_y1
            - cf_monthly_y1, cf_monthly_avg, is_acceptable
            - finance_score, qual_score, fitness
        """
        details = strategy.get("details", [])
        apport_total = strategy.get("apport_total", 0.0)
        
        if not details:
            return False, {"error": "No details in strategy"}
        
        try:
            # Generate schedules
            schedules = self.generate_schedules(details)
            
            # Run simulation
            df_sim, bilan = self.run_simulation(strategy, horizon, schedules)
            
            # Calculate metrics
            cf_metrics = self.calculate_cf_metrics(df_sim, target_cf, tolerance)
            enrich_metrics = self.calculate_enrichment_metrics(bilan, apport_total, horizon)
            
            # Build result dict
            result = {
                # Bilan fields
                "liquidation_nette": bilan.get("liquidation_nette", 0.0),
                "enrich_net": bilan.get("enrichissement_net", 0.0),
                "tri_annuel": bilan.get("tri_annuel", 0.0),
                "tri_global": bilan.get("tri_annuel", 0.0),
                "dscr_y1": bilan.get("dscr_y1", 0.0),
                
                # CF metrics
                "cf_monthly_y1": cf_metrics.get("cf_year_1_monthly", 0.0),
                "cf_monthly_avg": cf_metrics.get("cf_avg_5y_monthly", 0.0),
                "is_acceptable": cf_metrics.get("is_acceptable", False),
                "gap": cf_metrics.get("gap", 0.0),
            }
            
            # Add enrichment metrics
            result.update(enrich_metrics)
            
            # Calculate scores
            finance_score = self.calculate_finance_score(bilan, cf_metrics, apport_total, tolerance)
            qual_score = self.calculate_quality_score(details)
            combined_score = self.calculate_combined_score(finance_score, qual_score)
            
            result["finance_score"] = finance_score
            result["qual_score"] = qual_score
            result["fitness"] = max(0.01, combined_score * 100)
            
            is_valid = cf_metrics.get("is_acceptable", False)
            
            return is_valid, result
            
        except Exception as e:
            log.warning("evaluation_failed", error=str(e))
            return False, {"error": str(e)}
