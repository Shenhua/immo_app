"""Unit tests for src.core.simulation module."""

import pytest
from src.core.simulation import (
    MarketHypotheses,
    TaxParams,
    IRACalculator,
    YearResult,
    SimulationEngine,
)


class TestMarketHypotheses:
    """Tests for MarketHypotheses dataclass."""
    
    def test_defaults(self):
        """Should have sensible defaults."""
        m = MarketHypotheses()
        assert m.appreciation_bien_pct == 2.0
        assert m.revalo_loyer_pct == 1.5
        assert m.inflation_charges_pct == 2.0
    
    def test_from_dict(self):
        """Should create from dictionary."""
        d = {"appreciation_bien_pct": 3.0, "revalo_loyer_pct": 2.0}
        m = MarketHypotheses.from_dict(d)
        assert m.appreciation_bien_pct == 3.0
        assert m.revalo_loyer_pct == 2.0
        assert m.inflation_charges_pct == 2.0  # default


class TestTaxParams:
    """Tests for TaxParams."""
    
    def test_lmnp_tax_calculation(self):
        """LMNP real should use amortization deductions."""
        tax = TaxParams(tmi_pct=30.0, regime_fiscal="lmnp")
        
        impot, deficit, base = tax.calculate_tax(
            loyers_bruts=10000,
            charges_deductibles=3000,
            amortissements=5000,
        )
        
        # 10000 - 3000 - 5000 = 2000 base
        assert base == 2000.0
        assert impot > 0
    
    def test_microbic_tax_calculation(self):
        """Micro-BIC should use 50% abatement."""
        tax = TaxParams(regime_fiscal="microbic", micro_bic_abatt_pct=50.0)
        
        impot, deficit, base = tax.calculate_tax(
            loyers_bruts=10000,
            charges_deductibles=3000,  # ignored in micro-bic
            amortissements=5000,  # ignored in micro-bic
        )
        
        # 10000 * (1 - 0.50) = 5000 base
        assert base == 5000.0


class TestIRACalculator:
    """Tests for IRA calculator."""
    
    def test_no_ira_when_disabled(self):
        """Should return 0 when disabled."""
        ira = IRACalculator(apply_ira=False)
        result = ira.calculate([], [], 25)
        assert result == 0.0
    
    def test_no_ira_when_loan_finished(self):
        """Should return 0 when loan duration <= horizon."""
        ira = IRACalculator(apply_ira=True)
        projets = [{"duree_pret": 20}]
        schedules = [{"nmois": 240, "balances": [0] * 240}]
        
        result = ira.calculate(projets, schedules, 25)  # 25 > 20
        assert result == 0.0


class TestYearResult:
    """Tests for YearResult dataclass."""
    
    def test_to_dict(self):
        """Should convert to dictionary with French keys."""
        r = YearResult(
            year=1, valeur_biens=250000, dette=200000, patrimoine_net=50000,
            resultat_fiscal=5000, impot_du=2000, cash_flow_net=1000,
            capital_rembourse=8000, interets_assurance=3000,
            loyers_bruts=12000, charges_deductibles=4000, amortissements=3000
        )
        d = r.to_dict()
        
        assert d["Année"] == 1
        assert d["Valeur Biens"] == 250000
        assert d["Dette"] == 200000
        assert d["Cash-Flow Net d'Impôt"] == 1000


class TestSimulationEngine:
    """Tests for SimulationEngine."""
    
    def test_empty_strategy_returns_empty(self):
        """Empty strategy should return empty results."""
        engine = SimulationEngine(
            market=MarketHypotheses(),
            tax=TaxParams(),
            ira=IRACalculator(),
        )
        
        df, bilan = engine.simulate({}, 25, [])
        assert df.empty
        assert bilan == {}
    
    def test_simulation_returns_dataframe_and_bilan(self):
        """Full simulation should return DataFrame and bilan dict."""
        engine = SimulationEngine(
            market=MarketHypotheses(),
            tax=TaxParams(tmi_pct=30.0),
            ira=IRACalculator(apply_ira=False),
        )
        
        strategy = {
            "apport_total": 50000,
            "details": [{
                "prix_achat_bien": 200000,
                "frais_notaire": 15000,
                "budget_travaux": 5000,
                "renovation_energetique": 0,
                "mobilier": 3000,
                "credit_final": 175000,
                "taux_pret": 3.5,
                "duree_pret": 20,
                "assurance_ann_pct": 0.36,
                "loyer_mensuel_initial": 800,
                "charges_const_mth0": 80,
                "tf_const_mth0": 40,
                "frais_gestion_pct": 5.0,
                "provision_pct": 5.0,
            }],
        }
        
        # Create mock schedule
        schedule = {
            "nmois": 240,
            "interets": [500] * 240,
            "principals": [300] * 240,
            "balances": [175000 - i * 300 for i in range(240)],
            "pmt_assur": 50,
            "pmt_total": 1100,
        }
        
        df, bilan = engine.simulate(strategy, 10, [schedule])
        
        assert len(df) == 10
        assert "tri_annuel" in bilan
        assert "liquidation_nette" in bilan
