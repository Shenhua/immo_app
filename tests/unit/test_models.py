"""Unit tests for src.models Pydantic models."""

import pytest
from pydantic import ValidationError

from src.domain.models.archetype import ArchetypeV2
from src.domain.models.brick import InvestmentBrick
from src.domain.models.strategy import PortfolioStrategy, SimulationParams


class TestArchetypeV2:
    """Tests for ArchetypeV2 Pydantic model."""

    def test_valid_archetype(self, sample_archetype_data):
        """Valid data should create model successfully."""
        arch = ArchetypeV2(**sample_archetype_data)
        assert arch.nom == "Studio Paris 11"
        assert arch.ville == "Paris"
        assert arch.surface == 25.0

    def test_computed_properties(self, sample_archetype_data):
        """Computed properties should work correctly."""
        arch = ArchetypeV2(**sample_archetype_data)
        assert arch.prix_achat_total == 250000.0  # 25 * 10000
        assert arch.loyer_mensuel == 875.0  # 25 * 35
        assert arch.charges_annuelles == 1250.0  # 25 * 50

    def test_dpe_validation(self, sample_archetype_data):
        """DPE should be normalized to uppercase."""
        sample_archetype_data["dpe_initial"] = "c"
        arch = ArchetypeV2(**sample_archetype_data)
        assert arch.dpe_initial == "C"

    def test_invalid_dpe_defaults(self, sample_archetype_data):
        """Invalid DPE should default to D."""
        sample_archetype_data["dpe_initial"] = "Z"
        arch = ArchetypeV2(**sample_archetype_data)
        assert arch.dpe_initial == "D"

    def test_missing_required_field(self):
        """Missing required field should raise error."""
        with pytest.raises(ValidationError):
            ArchetypeV2(nom="Test", ville="Paris")  # Missing surface, prix_m2, etc.

    def test_negative_surface_rejected(self, sample_archetype_data):
        """Negative surface should be rejected."""
        sample_archetype_data["surface"] = -10
        with pytest.raises(ValidationError):
            ArchetypeV2(**sample_archetype_data)


class TestInvestmentBrick:
    """Tests for InvestmentBrick model."""

    def test_computed_cout_total(self):
        """cout_total should sum all costs."""
        brick = InvestmentBrick(
            nom="Test",
            ville="Paris",
            mode_loyer="meuble",
            surface=30,
            prix_achat_bien=200000,
            frais_notaire=15000,
            travaux=10000,
            renovation_energetique=5000,
            mobilier=3000,
        )
        assert brick.cout_total == 233000

    def test_computed_pmt_total(self):
        """pmt_total should sum P&I and insurance."""
        brick = InvestmentBrick(
            nom="Test",
            ville="Paris",
            mode_loyer="meuble",
            surface=30,
            prix_achat_bien=200000,
            pmt_principal_interet=800,
            pmt_assurance=50,
        )
        assert brick.pmt_total == 850

    def test_computed_cash_flow(self):
        """cash_flow should be rent - expenses - payment."""
        brick = InvestmentBrick(
            nom="Test",
            ville="Paris",
            mode_loyer="meuble",
            surface=30,
            prix_achat_bien=200000,
            loyer_mensuel_initial=1000,
            depenses_mensuelles_hors_credit_initial=200,
            pmt_principal_interet=700,
            pmt_assurance=50,
        )
        # 1000 - 200 - (700+50) = 50
        assert brick.cash_flow_mensuel_initial == 50


class TestPortfolioStrategy:
    """Tests for PortfolioStrategy model."""

    def test_nombre_biens(self, sample_strategy_data):
        """Should correctly count bricks in details."""
        strategy = PortfolioStrategy(**sample_strategy_data)
        assert strategy.nombre_biens == 2


class TestSimulationParams:
    """Tests for SimulationParams model."""

    def test_defaults(self):
        """Should have sensible defaults."""
        params = SimulationParams()
        assert params.horizon_years == 25
        assert params.tmi_pct == 30.0
        assert params.regime_fiscal == "lmnp"

    def test_horizon_bounds(self):
        """Horizon should be bounded 1-50."""
        with pytest.raises(ValidationError):
            SimulationParams(horizon_years=0)
        with pytest.raises(ValidationError):
            SimulationParams(horizon_years=100)
