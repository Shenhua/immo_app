from src.domain.tax import LmnpReel, TaxState, YearContext


def test_lmnp_amortization_cannot_create_deficit():
    """Amortization cannot create a deficit in LMNP réel (CGI 39 C II, TAX_SPEC 4.2).

    Rent 1000, charges 200, amortization 1000: R = 800, 800 deducted, 200 carried as ARD,
    taxable base 0 and no BIC deficit.
    """
    state = TaxState()
    ctx = YearContext(
        year=1,
        property_id=0,
        rent=1000.0,
        operating_charges=200.0,
        interest=0.0,
        loan_insurance=0.0,
        amort_building=1000.0,
    )
    res = LmnpReel(tmi_pct=30.0).compute_year(1, [ctx], state)

    assert res.taxable_base == 0.0
    assert state.deficits.total == 0.0, "Amortization should not create a reportable deficit"
    assert state.prop(0).ard == 200.0
    assert res.detail["amortization_deducted"] == 800.0
