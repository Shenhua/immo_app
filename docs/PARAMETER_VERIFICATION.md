# App Immo - Parameter Verification Guide

This document outlines how to verify that all UI parameters correctly influence calculations and that results are accurate.

---

## 1. Parameter Connection Tests

For each parameter, verify it has a **visible effect** on output.

### Method: A/B Testing
1. Run analysis with default values → note results
2. Change ONE parameter significantly
3. Re-run → verify results changed appropriately

| Parameter | Test Change | Expected Effect |
|-----------|------------|-----------------|
| Apport total | 50k → 150k | More strategies available, different combinations |
| CF cible | -100 → +200 | Different strategies selected |
| Tolérance | 50 → 500 | More strategies pass filter |
| Ciblage Précis toggle | On → Off | Min CF mode vs target CF mode |
| Priorité slider | 0% → 100% | Rankings change (finance vs quality priority) |
| Horizon | 15 → 25 ans | IRR and liquidation values change |
| Taux crédit | 3.6% → 4.5% | Monthly payments increase, CF decreases |
| Frais notaire | 7.5% → 10% | Total costs increase |
| Assurance | 0.35% → 0.5% | Monthly payments increase |
| CFE | 500 → 1500 | Annual charges increase |
| Gestion % | 5% → 10% | Operating expenses increase |
| Vacance % | 3% → 8% | Provision increases |
| Frais revente | 6% → 8% | Liquidation value decreases |
| TMI | 30% → 45% | Tax burden increases, CF decreases |
| Régime | LMNP → SCI IS | Tax calculation method changes |
| Appréciation bien | 2% → 4% | Final property value increases |
| Revalo loyer | 1.5% → 3% | Future rents higher |
| Inflation charges | 2% → 4% | Future expenses higher |
| Profil de Tri | Équilibré → DSCR | Strategy ranking order changes |

---

## 2. Value Verification Tests

### 2.1 Loan Calculations
```
Test: Verify PMT matches Excel/financial calculator

Input: 100,000€ loan, 3.6% rate, 25 years
Expected PMT: ~505€/month (principal + interest)
Check: Strategy detail shows matching value
```

### 2.2 Insurance
```
Test: Insurance = Principal × Insurance% / 12

Input: 100,000€ principal, 0.36% annual insurance
Expected: 100,000 × 0.0036 / 12 = 30€/month
```

### 2.3 Total Monthly Payment
```
PMT Total = PMT (P&I) + Insurance
~505 + 30 = ~535€/month
```

### 2.4 Cash Flow
```
CF = Rent - Expenses - Debt Service - CFE/12

Example:
Rent: 800€
Expenses: 150€ (charges + gestion + provision)
Debt: 535€
CFE: 500/12 = 42€
CF = 800 - 150 - 535 - 42 = 73€
```

### 2.5 IRR/TRI
```
Manual verification:
- Export simulation data to Excel
- Use XIRR function on cash flows
- Compare to displayed TRI
```

---

## 3. Logging-Based Verification

Add temporary logging to verify intermediate values:

```python
# In strategy_finder.py, add after scoring:
for s in strategies[:3]:
    log.info("strategy_debug",
        qual_score=s.get("qual_score"),
        finance_score=s.get("finance_score"),
        balanced_score=s.get("balanced_score"),
    )
```

Check that:
- `qual_score` varies between strategies (not all 50.0)
- `finance_score` varies based on weights
- `balanced_score` responds to qualite_weight

---

## 4. Edge Case Tests

| Test | Input | Expected |
|------|-------|----------|
| Zero apport | 0€ | Strategies use minimum apport |
| Negative CF target | -500€ | Works, shows deficit strategies |
| Max horizon | 30 years | Simulation completes |
| Min horizon | 10 years | Simulation completes |
| No properties selected | Empty filter | "No strategies found" message |
| Single property | 1 ville, 1 type | Limited strategies |

---

## 5. Cross-Validation Checklist

### Before Each Release
- [ ] Change each parameter → verify output changes
- [ ] Compare PMT to financial calculator
- [ ] Compare IRR to Excel XIRR
- [ ] Verify qual_score varies across properties
- [ ] Verify preset changes affect ranking order
- [ ] Test with extreme values (min/max sliders)
- [ ] Export results → verify data matches UI

### Monthly
- [ ] Update test fixtures with current market rates
- [ ] Verify tax calculations against official rates
- [ ] Check capital gains abatement formula is current

---

## 6. Automated Verification (Future)

Consider adding:
```python
# tests/test_parameter_effects.py

def test_apport_affects_strategy_count():
    """More apport should enable more strategies."""
    low = run_analysis(apport=30000)
    high = run_analysis(apport=150000)
    assert len(high) >= len(low)

def test_qualite_weight_changes_ranking():
    """Different weights should produce different rankings."""
    finance_first = run_analysis(qualite_weight=0.0)
    quality_first = run_analysis(qualite_weight=1.0)
    assert finance_first[0] != quality_first[0]

def test_qual_scores_vary():
    """Properties should have different quality scores."""
    bricks = create_investment_bricks(archetypes, config)
    scores = [b["qual_score_bien"] for b in bricks]
    assert len(set(scores)) > 1  # Not all identical
```
