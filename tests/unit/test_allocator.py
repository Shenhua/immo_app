import pytest

from src.application.services.allocator import PortfolioAllocator
from src.domain.models.brick import InvestmentBrick


class MockBrick(dict):
    pass


@pytest.fixture
def allocator():
    return PortfolioAllocator()


@pytest.fixture
def sample_bricks():
    # K-factor roughly: (pmt / capital) ... wait, k is DeltaCF / DeltaApport.
    # Loan logic: Pmt changes when Apport changes.
    # If I add 1000€ apport, I borrow 1000€ less.
    # PMT decreases by (1000 * rate/12) roughly.
    # CF increases by that amount.
    # So k is small (approx monthly rate).
    # wait, existing allocator uses k_factor(taux, duree, assurance).
    # For 3% 20y:
    # Monthly payment for 100k is ~554.
    # Monthly payment for 99k is ~548.
    # Delta PMT = 6€.
    # Delta Apport = 1000€.
    # k = 0.006 (approx).
    # Inverse k = 166.

    return [
        InvestmentBrick(
            nom="B1",
            prix_achat_bien=110000.0,
            apport_min=10000.0,
            capital_emprunte=100000.0,
            taux_annuel_pct=3.0,
            duree_credit_mois=240,
            assurance_annuelle_pct=0.36,
            loyer_mensuel_initial=600.0,
            depenses_mensuelles_hors_credit_initial=100.0,
        )
    ]


def test_allocator_step_logic_coarse(allocator, sample_bricks):
    """Test that large gaps trigger coarse steps."""
    # Target +500 CF. Current -80. Gap 580.
    # Tolerance 10. Gap is 58x tolerance.
    # Should use coarse step.

    # We can't easily spy on internal variables without mocking logic,
    # but we can check if it converges reasonably fast or hits the target.

    ok, details, cf_final, apport_used = allocator.allocate(
        sample_bricks,
        apport_disponible=100000.0,
        target_cf=100.0,  # Target +100
        tolerance=10.0,
    )

    # We expect it to find a solution
    # Initial CF -80. Need +180 delta.
    # k approx 0.0055.
    # Delta apport needed approx 180 / 0.0055 = 32,727

    assert ok is True
    assert abs(cf_final - 100.0) <= 10.0
    assert 30000 < apport_used < 45000


def test_allocator_precise_landing(allocator, sample_bricks):
    """Test that allocator lands precisely within tolerance when close."""
    # Target -70 CF. Current -80. Gap 10.
    # Tolerance 1.
    # Needs very small step.

    ok, details, cf_final, apport_used = allocator.allocate(
        sample_bricks,
        apport_disponible=50000.0,
        target_cf=-75.0,  # Just 5€ boost
        tolerance=1.0,
    )

    # Delta CF 5€. k ~ 0.0055.
    # Apport needed ~ 900€.
    # If step logic was "apport/1000" (50€ step), we might jump 0.25€ CF per step?
    # Wait, apport/1000 of 50k is 50€. 50€ * 0.0055 = 0.27€ CF change.
    # That is smaller than tolerance 1.0, so old logic would work here too.
    # Let's try aggressive budget.

    assert ok is True
    assert abs(cf_final - (-75.0)) <= 1.0


def test_allocator_impossible_target(allocator, sample_bricks):
    """Test strict fail behavior."""
    ok, _, _, _ = allocator.allocate(
        sample_bricks,
        apport_disponible=1000.0,  # Not enough money
        target_cf=1000.0,  # Impossible CF
        tolerance=10.0,
    )
    assert ok is False


# --- T31: analytic allocation (no step truncation) -------------------------------------------------------


@pytest.mark.parametrize("target", [-60.0, -1.0, 0.0, 150.0, 333.33])
def test_min_mode_lands_on_the_target_not_one_step_short(sample_bricks, target):
    """Regression: the stepped allocator stopped at e.g. -1,501 for a -1,500 target and reported ok=False."""
    alloc = PortfolioAllocator(mode_cf="min")
    ok, details, cf, apport = alloc.allocate(sample_bricks, 200000.0, target, 100.0)
    assert ok is True
    assert target <= cf < target + 0.05
    assert apport == pytest.approx(sum(d["apport_final_bien"] for d in details))


def test_min_mode_uses_the_smallest_down_payment(sample_bricks):
    alloc = PortfolioAllocator(mode_cf="min")
    prepared = [alloc.prepare(b) for b in sample_bricks]
    cf0 = alloc.plan(prepared, 200000.0, -10_000.0).cf
    plan = alloc.plan(prepared, 200000.0, cf0 + 50.0)
    # 50 EUR/month of cash flow costs exactly 50 / k euros of down payment
    assert plan.extra[0] == pytest.approx((50.0 + alloc.MIN_MODE_EPS) / prepared[0].k)


def test_combination_over_budget_is_never_ok(sample_bricks):
    alloc = PortfolioAllocator(mode_cf="min")
    ok, _, _, apport = alloc.allocate(sample_bricks, 5000.0, -10_000.0, 100.0)  # apport_min is 10,000
    assert ok is False
    assert apport == 10000.0


def test_highest_k_brick_is_filled_first(sample_bricks):
    short = sample_bricks[0].model_copy(update={"nom": "short", "duree_credit_mois": 120})
    long_ = sample_bricks[0].model_copy(update={"nom": "long", "duree_credit_mois": 300})
    alloc = PortfolioAllocator(mode_cf="min")
    prepared = [alloc.prepare(long_), alloc.prepare(short)]
    cf0 = alloc.plan(prepared, 300000.0, -1e9).cf
    plan = alloc.plan(prepared, 300000.0, cf0 + 100.0)
    assert plan.extra[0] == 0.0 and plan.extra[1] > 0.0  # the 10-year loan has the higher k


def test_full_cash_when_the_whole_loan_is_needed(sample_bricks):
    alloc = PortfolioAllocator(mode_cf="min")
    prepared = [alloc.prepare(b) for b in sample_bricks]
    top = alloc.max_cf(prepared, 500000.0)
    plan = alloc.plan(prepared, 500000.0, top - 0.5)
    assert plan.ok and plan.extra[0] == pytest.approx(prepared[0].capital)
    assert alloc.details(prepared, plan.extra)[0]["credit_final"] == 0.0
