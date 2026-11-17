import pytest

from src.application.services.simulation import (
    IRACalculator,
    MarketHypotheses,
    SimulationEngine,
    SimulationError,
    TaxParams,
    YearResult,
)
from src.domain.tax import RegimeId


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

    def test_maps_to_one_regime_per_category(self):
        """regime_fiscal is the furnished regime, regime_foncier the unfurnished one (TAX_SPEC 8)."""
        cfg = TaxParams(tmi_pct=30.0, regime_fiscal="microbic", regime_foncier="micro").to_config()
        assert cfg.furnished_regime is RegimeId.MICRO_BIC
        assert cfg.unfurnished_regime is RegimeId.MICRO_FONCIER
        assert not cfg.sci_is

    def test_legacy_sci_is_value_sets_flag(self):
        tax = TaxParams(regime_fiscal="sci_is")
        assert tax.sci_is and tax.to_config().sci_is

    def test_unknown_regime_rejected(self):
        with pytest.raises(ValueError):
            TaxParams(regime_fiscal="pinel").to_config()
        with pytest.raises(ValueError):
            TaxParams(regime_foncier="lmnp").to_config()


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
            year=1,
            valeur_biens=250000,
            dette=200000,
            patrimoine_net=50000,
            resultat_fiscal=5000,
            impot_du=2000,
            cash_flow_net=1000,
            capital_rembourse=8000,
            interets_assurance=3000,
            loyers_bruts=12000,
            charges_deductibles=4000,
            amortissements=3000,
        )
        d = r.to_dict()

        assert d["Année"] == 1
        assert d["Valeur Biens"] == 250000
        assert d["Dette"] == 200000
        assert d["Cash-Flow Net d'Impôt"] == 1000


class TestSimulationEngine:
    """Tests for SimulationEngine."""

    def test_empty_strategy_throws_error(self):
        """Empty strategy should raise SimulationError."""
        engine = SimulationEngine(
            market=MarketHypotheses(),
            tax=TaxParams(),
            ira=IRACalculator(),
        )

        with pytest.raises(SimulationError):
            engine.simulate({}, 25, [])

    def test_simulation_returns_dataframe_and_bilan(self):
        """Full simulation should return DataFrame and bilan dict."""
        engine = SimulationEngine(
            market=MarketHypotheses(),
            tax=TaxParams(tmi_pct=30.0),
            ira=IRACalculator(apply_ira=False),
        )

        strategy = {
            "apport_total": 50000,
            "details": [
                {
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
                }
            ],
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


# --- T24: CFE, wealth, failed IRR -------------------------------------------------------------
def _t24_strategy(meuble, rent=1000.0):
    prop = {
        "prix_achat_bien": 200000.0,
        "frais_notaire": 0.0,
        "budget_travaux": 0.0,
        "renovation_energetique": 0.0,
        "mobilier": 0.0,
        "credit_final": 100000.0,
        "taux_pret": 3.0,
        "duree_pret": 20,
        "assurance_ann_pct": 0.0,
        "loyer_mensuel_initial": rent,
        "charges_const_mth0": 0.0,
        "tf_const_mth0": 0.0,
        "frais_gestion_pct": 0.0,
        "provision_pct": 0.0,
        "meuble": meuble,
    }
    return {"apport_total": 100000.0, "details": [prop]}


def _t24_run(strategy, years=4, charges_idx=0.0):
    from src.application.services.simulation import simulate_long_term_strategy

    return simulate_long_term_strategy(
        strategy,
        years,
        {"appreciation_bien_pct": 2.0, "inflation_charges_pct": charges_idx},
        {"tmi_pct": 0.0, "regime_fiscal": "microbic", "cfe_par_bien_ann": 300.0, "apply_ira": False},
    )


def test_cfe_furnished_only_year1_exempt_and_indexed():
    df, _ = _t24_run(_t24_strategy(True), charges_idx=2.0)
    df_nu, _ = _t24_run(_t24_strategy(False), charges_idx=2.0)
    diffs = (df["Charges d'exploitation"] - df_nu["Charges d'exploitation"]).abs().tolist()
    assert diffs[0] == pytest.approx(0.0)
    assert diffs[1] == pytest.approx(300.0 * 1.02)
    assert diffs[2] == pytest.approx(300.0 * 1.02**2)


def test_cfe_exempt_below_5000_receipts():
    df, _ = _t24_run(_t24_strategy(True, rent=300.0))  # 3,600 EUR/yr, rent growth keeps it below 5,000
    df_nu, _ = _t24_run(_t24_strategy(False, rent=300.0))
    # Compare charges, not after-tax CF: since T23 the unfurnished twin is taxed under revenus fonciers.
    assert df["Charges d'exploitation"].tolist() == pytest.approx(df_nu["Charges d'exploitation"].tolist())


def test_wealth_includes_cash_flows():
    """Different cash flows give different wealth = proceeds + sum CF - apport."""
    df_a, bil_a = _t24_run(_t24_strategy(False, rent=1000.0))
    df_b, bil_b = _t24_run(_t24_strategy(False, rent=900.0))
    for df, bil in ((df_a, bil_a), (df_b, bil_b)):
        expected = bil["liquidation_nette"] + df["Cash-Flow Net d'Impôt"].sum() - 100000.0
        assert bil["enrichissement_net"] == pytest.approx(expected)
    assert bil_a["enrichissement_net"] != pytest.approx(bil_b["enrichissement_net"])


def test_nan_irr_is_none_and_never_displays_as_zero(monkeypatch):
    import numpy as np

    from src.application.services import simulation as sim
    from src.ui.helpers import format_pct

    monkeypatch.setattr(sim.npf, "irr", lambda flux: np.nan)
    _, bil = _t24_run(_t24_strategy(False))
    assert bil["tri_annuel"] is None
    assert bil["tri_warning"]
    assert format_pct(bil["tri_annuel"]) == "n/a"
    assert format_pct(float("nan")) == "n/a"


def test_evaluator_scores_failed_irr_as_zero_component():
    from unittest.mock import MagicMock

    from src.application.services.evaluator import StrategyEvaluator

    scorer = MagicMock()
    scorer.weights = {"irr": 1.0, "enrich_net": 0.0, "dscr": 0.0, "cf_proximity": 0.0, "cap_eff": 0.0}
    ev = StrategyEvaluator.__new__(StrategyEvaluator)
    ev.scorer = scorer
    score = ev.calculate_finance_score(
        {"tri_annuel": None, "enrichissement_net": 0.0, "dscr_stab": 1.0}, {"gap": 0.0}, 100000.0, 100.0, 10
    )
    assert score == 0.0
