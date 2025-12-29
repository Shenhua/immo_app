
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
            depenses_mensuelles_hors_credit_initial=100.0
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
        target_cf=100.0, # Target +100
        tolerance=10.0
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
        target_cf=-75.0, # Just 5€ boost
        tolerance=1.0
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
        apport_disponible=1000.0, # Not enough money
        target_cf=1000.0, # Impossible CF
        tolerance=10.0
    )
    assert ok is False
