"""Scoring constants - single source of truth for weight keys.

This module provides standardized constants for the scoring system,
ensuring consistency across GeneticOptimizer, ExhaustiveOptimizer,
and StrategyScorer.
"""

from typing import TypedDict


class WeightConfig(TypedDict):
    """Type definition for scoring weights."""
    enrich_net: float
    irr: float
    cf_proximity: float
    dscr: float
    cap_eff: float


# Standard weight keys used across all scorers
WEIGHT_KEYS = {
    "irr": "irr",
    "enrichment": "enrich_net",
    "dscr": "dscr",
    "cf_proximity": "cf_proximity",
    "capital_efficiency": "cap_eff",
}

# Default balanced weights (sum to 1.0)
# These are used when no custom weights are provided
DEFAULT_WEIGHTS: WeightConfig = {
    "enrich_net": 0.30,      # Net enrichment (wealth building)
    "irr": 0.25,             # Internal Rate of Return (yield)
    "cf_proximity": 0.20,    # Cash flow target precision
    "dscr": 0.15,            # Debt Service Coverage Ratio (safety)
    "cap_eff": 0.10,         # Capital efficiency
}

# Scoring scale reference - used for absolute scoring mode
# These define what constitutes a "perfect" score (1.0) for each metric
SCORING_SCALES = {
    "tri_max": 20.0,              # 20% IRR = 1.0 score (expert recommendation)
    "roe_target": 2.0,            # 2x ROE (doubling equity) = 1.0 score
    "dscr_safe": 1.3,             # DSCR 1.3 = 1.0 score (bank-safe threshold)
    "cf_tolerance_mult": 2.0,     # CF penalty scale multiplier
}

# Preset configurations for different investor profiles
PRESET_WEIGHTS = {
    "Équilibré": DEFAULT_WEIGHTS,
    
    "Sécurité (DSCR)": {
        "dscr": 0.40,
        "cf_proximity": 0.25,
        "enrich_net": 0.20,
        "irr": 0.10,
        "cap_eff": 0.05,
    },
    
    "Cash-flow d'abord": {
        "cf_proximity": 0.40,
        "dscr": 0.25,
        "irr": 0.15,
        "enrich_net": 0.15,
        "cap_eff": 0.05,
    },
    
    "Rendement / IRR": {
        "irr": 0.40,
        "enrich_net": 0.25,
        "cf_proximity": 0.15,
        "dscr": 0.15,
        "cap_eff": 0.05,
    },
    
    "Patrimoine LT": {
        "enrich_net": 0.40,
        "irr": 0.25,
        "cap_eff": 0.15,
        "dscr": 0.15,
        "cf_proximity": 0.05,
    },
}


def validate_weights(weights: dict[str, float]) -> bool:
    """Validate that weights have all required keys and sum to ~1.0.
    
    Args:
        weights: Weight dictionary to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = set(DEFAULT_WEIGHTS.keys())
    provided_keys = set(weights.keys())
    
    missing = required_keys - provided_keys
    if missing:
        raise ValueError(f"Missing weight keys: {missing}")
    
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {total:.2f}")
    
    return True


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1.0.
    
    Args:
        weights: Raw weights (any positive values)
        
    Returns:
        Normalized weights summing to 1.0
    """
    total = sum(weights.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    
    return {k: v / total for k, v in weights.items()}
