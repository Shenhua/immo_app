
import pytest
from src.application.services.simulation import TaxParams

def test_lmnp_amortization_cannot_create_deficit():
    """
    Test that amortization CANNOT create a deficit in LMNP.
    
    Scenario:
    - Rents: 1000
    - Deductible Charges: 200
    - Amortization: 1000
    
    Calculation:
    - Net Operating Income (NOI) = 1000 - 200 = 800
    - Max Amortization Usage = 800
    - Remaining Amortization = 1000 - 800 = 200 (Stored as ARD - Amortissements Réputés Différés)
    - Taxable Income = 0
    - Deficit Reportable = 0 (Because Charges 200 < Rents 1000)
    
    Current Bug:
    - Result = 1000 - 200 - 1000 = -200
    - The code currently sets deficit_reportable to -200.
    """
    tax = TaxParams(tmi_pct=30.0, regime_fiscal="lmnp")
    
    loyers = 1000.0
    charges = 200.0
    amort = 1000.0
    deficit_anterior = 0.0
    
    impot, new_deficit, new_ard, base = tax.calculate_tax(loyers, charges, amort, deficit_anterior)
    
    # Assert base is 0 (this should pass even now)
    assert base == 0.0, "Taxable base should be 0"
    
    # This assertion is expected to FAIL currently
    assert sum(new_deficit) == 0.0, f"Amortization should not create reportable deficit. Got {new_deficit}, expected sum 0.0"
