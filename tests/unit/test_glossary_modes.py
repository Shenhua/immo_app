
import pandas as pd
from src.core.glossary import calculate_cashflow_metrics

def test_glossary_target_mode():
    # Setup: Target 100, Tolerance 10
    # CF 150 -> Gap 50 (too high) -> Rejected
    df = pd.DataFrame([{"Cash-Flow Net d'Impôt": 150 * 12}])
    
    metrics = calculate_cashflow_metrics(df, target_cf=100.0, tolerance=10.0, mode_cf="target")
    
    # In target mode, 150 is far from 100. Gap = 50. Rejected.
    assert metrics["gap"] == 50.0
    assert metrics["is_acceptable"] is False

def test_glossary_min_mode():
    # Setup: Min 100, Tolerance 10
    # CF 150 -> Gap 0 (good) -> Accepted
    df = pd.DataFrame([{"Cash-Flow Net d'Impôt": 150 * 12}])
    
    metrics = calculate_cashflow_metrics(df, target_cf=100.0, tolerance=10.0, mode_cf="min")
    
    # In min mode, 150 > 100. Gap = 0. Accepted.
    assert metrics["gap"] == 0.0
    assert metrics["is_acceptable"] is True

def test_glossary_min_mode_below_target():
    # Setup: Min 100, Tolerance 10
    # CF 95 -> Gap 5 (within tolerance) -> Accepted
    df = pd.DataFrame([{"Cash-Flow Net d'Impôt": 95 * 12}])
    
    metrics = calculate_cashflow_metrics(df, target_cf=100.0, tolerance=10.0, mode_cf="min")
    
    assert metrics["gap"] == 5.0
    assert metrics["is_acceptable"] is True

def test_glossary_min_mode_too_low():
    # Setup: Min 100, Tolerance 10
    # CF 80 -> Gap 20 (too low) -> Rejected
    df = pd.DataFrame([{"Cash-Flow Net d'Impôt": 80 * 12}])
    
    metrics = calculate_cashflow_metrics(df, target_cf=100.0, tolerance=10.0, mode_cf="min")
    
    assert metrics["gap"] == 20.0
    assert metrics["is_acceptable"] is False
