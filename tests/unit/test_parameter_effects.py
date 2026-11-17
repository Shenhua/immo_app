"""Parameter effect tests.

Verifies that changing UI parameters actually affects the output.
Based on docs/PARAMETER_VERIFICATION.md.
"""

from typing import Any

import pytest

from src.application.services.brick_factory import (
    FinancingConfig,
    OperatingConfig,
    create_investment_bricks,
)
from src.application.services.strategy_finder import StrategyFinder


# Test fixtures
@pytest.fixture
def sample_archetypes() -> list[dict[str, Any]]:
    """Minimal archetypes for testing."""
    return [
        {
            "nom": "Test Apt 1",
            "ville": "Paris",
            "surface": 30,
            "prix_m2": 5000,
            "loyer_m2": 20,
            "charges_m2_an": 30,
            "taxe_fonciere_m2_an": 15,
            "dpe_initial": "C",
            "tension_locative_score_norm": 0.8,
            "transport_score": 0.9,
            "liquidite_score": 0.7,
            "soumis_encadrement": False,
            "budget_travaux": 5000,
            "valeur_mobilier": 3000,
        },
        {
            "nom": "Test Apt 2",
            "ville": "Lyon",
            "surface": 40,
            "prix_m2": 4000,
            "loyer_m2": 15,
            "charges_m2_an": 25,
            "taxe_fonciere_m2_an": 12,
            "dpe_initial": "D",
            "tension_locative_score_norm": 0.6,
            "transport_score": 0.7,
            "liquidite_score": 0.5,
            "soumis_encadrement": False,
            "budget_travaux": 8000,
            "valeur_mobilier": 4000,
        },
        {
            "nom": "Test Apt 3",
            "ville": "Marseille",
            "surface": 50,
            "prix_m2": 3000,
            "loyer_m2": 12,
            "charges_m2_an": 20,
            "taxe_fonciere_m2_an": 10,
            "dpe_initial": "B",
            "tension_locative_score_norm": 0.9,
            "transport_score": 0.5,
            "liquidite_score": 0.8,
            "soumis_encadrement": False,
            "budget_travaux": 3000,
            "valeur_mobilier": 2000,
        },
    ]


@pytest.fixture
def default_finance_config() -> FinancingConfig:
    return FinancingConfig(
        credit_rates={20: 3.4, 25: 3.6},
        frais_notaire_pct=7.5,
        apport_min_pct=0.0,
        assurance_ann_pct=0.36,
        frais_pret_pct=1.0,
    )


@pytest.fixture
def default_operating_config() -> OperatingConfig:
    return OperatingConfig(
        frais_gestion_pct=5.0,
        provision_pct=3.0,
        cfe_par_bien_ann=500.0,
    )


class TestQualScoreVariation:
    """Verify qualitative scores vary between properties."""

    def test_qual_scores_vary_across_bricks(
        self,
        sample_archetypes,
        default_finance_config,
        default_operating_config,
    ):
        """Properties should have different quality scores."""
        bricks = create_investment_bricks(
            sample_archetypes,
            default_finance_config,
            default_operating_config,
        )

        scores = [b.qual_score_bien for b in bricks]
        unique_scores = set(scores)

        # Should have more than one unique score
        assert len(unique_scores) > 1, f"All qual_scores are identical: {scores}"

    def test_qual_scores_in_valid_range(
        self,
        sample_archetypes,
        default_finance_config,
        default_operating_config,
    ):
        """Quality scores should be 0-100."""
        bricks = create_investment_bricks(
            sample_archetypes,
            default_finance_config,
            default_operating_config,
        )

        for brick in bricks:
            score = brick.qual_score_bien
            assert 0 <= score <= 100, f"Score {score} out of range for {brick.nom}"


class TestParameterEffects:
    """Verify parameters have measurable effects."""

    def test_interest_rate_affects_pmt(
        self,
        sample_archetypes,
        default_operating_config,
    ):
        """Higher interest rate should increase monthly payment."""
        low_rate = FinancingConfig(
            credit_rates={25: 3.0},
            frais_notaire_pct=7.5,
        )
        high_rate = FinancingConfig(
            credit_rates={25: 5.0},
            frais_notaire_pct=7.5,
        )

        bricks_low = create_investment_bricks(sample_archetypes, low_rate, default_operating_config)
        bricks_high = create_investment_bricks(sample_archetypes, high_rate, default_operating_config)

        # Compare same property
        pmt_low = bricks_low[0].pmt_total
        pmt_high = bricks_high[0].pmt_total

        assert pmt_high > pmt_low, f"Higher rate should increase PMT: {pmt_low} vs {pmt_high}"

    def test_gestion_pct_affects_expenses(
        self,
        sample_archetypes,
        default_finance_config,
    ):
        """Higher management fee should increase expenses."""
        low_gestion = OperatingConfig(frais_gestion_pct=3.0)
        high_gestion = OperatingConfig(frais_gestion_pct=10.0)

        bricks_low = create_investment_bricks(sample_archetypes, default_finance_config, low_gestion)
        bricks_high = create_investment_bricks(sample_archetypes, default_finance_config, high_gestion)

        exp_low = bricks_low[0].depenses_mensuelles_hors_credit_initial
        exp_high = bricks_high[0].depenses_mensuelles_hors_credit_initial

        assert exp_high > exp_low, f"Higher gestion should increase expenses: {exp_low} vs {exp_high}"


class TestPresetRanking:
    """Verify different presets produce different rankings."""

    def test_preset_changes_affect_ranking(self):
        """Different presets should potentially reorder strategies."""
        finder = StrategyFinder([], 100000, -100)

        strategies = [
            {
                "balanced_score": 0.7,
                "dscr_norm": 0.9,  # High safety
                "tri_norm": 0.4,  # Low IRR
                "cf_proximity": 0.5,
                "finance_score": 0.6,
                "enrich_norm": 0.5,
            },
            {
                "balanced_score": 0.8,
                "dscr_norm": 0.4,  # Low safety
                "tri_norm": 0.9,  # High IRR
                "cf_proximity": 0.5,
                "finance_score": 0.7,
                "enrich_norm": 0.7,
            },
        ]

        # Safety preset should prefer high DSCR
        ranked_safety = finder.rank_strategies(strategies, "Sécurité (DSCR)")
        assert ranked_safety[0]["dscr_norm"] == 0.9

        # IRR preset should prefer high TRI
        ranked_irr = finder.rank_strategies(strategies, "Rendement / IRR")
        assert ranked_irr[0]["tri_norm"] == 0.9


# --------------------------------------------------------------------------- #
# T26a: realistic assumptions
# --------------------------------------------------------------------------- #
def _sim_strategy(**over: Any) -> dict[str, Any]:
    prop = {
        "prix_achat_bien": 100_000.0,
        "frais_notaire": 8_000.0,
        "budget_travaux": 0.0,
        "renovation_energetique": 0.0,
        "mobilier": 0.0,
        "ratio_terrain_pct": 15.0,
        "surface": 30.0,
        "credit_final": 90_000.0,
        "taux_pret": 3.5,
        "duree_pret": 20,
        "assurance_ann_pct": 0.3,
        "loyer_mensuel_initial": 600.0,
        "charges_const_mth0": 50.0,
        "tf_const_mth0": 50.0,
        "frais_gestion_pct": 7.0,
        "provision_pct": 0.0,
        "meuble": False,
    }
    prop.update(over)
    return {"apport_total": 20_000.0, "details": [prop]}


def _run(regime_foncier: str = "reel", market: dict[str, float] | None = None, **over: Any):
    from src.application.services.simulation import simulate_long_term_strategy

    mk = {"appreciation_bien_pct": 2.0, "revalo_loyer_pct": 0.0, "inflation_charges_pct": 0.0}
    mk.update(market or {})
    return simulate_long_term_strategy(
        _sim_strategy(**over),
        10,
        mk,
        {"regime_foncier": regime_foncier, "tmi_pct": 30.0, "apply_ira": False},
    )


class TestRealisticAssumptions:
    def test_maintenance_lowers_cash_flow_and_tax_base(self):
        df0, b0 = _run(maintenance_m2_an=0.0)
        df1, b1 = _run(maintenance_m2_an=10.0)
        assert df1.iloc[0]["Cash-Flow Net d'Impôt"] < df0.iloc[0]["Cash-Flow Net d'Impôt"]
        # Deductible under foncier reel: taxable base falls.
        assert df1.iloc[0]["Résultat Fiscal"] < df0.iloc[0]["Résultat Fiscal"]
        assert b1["enrichissement_net"] < b0["enrichissement_net"]
        drop = df0.iloc[0]["Cash-Flow Net d'Impôt"] - df1.iloc[0]["Cash-Flow Net d'Impôt"]
        assert 0 < drop <= 300.0  # 10 EUR x 30 m2

    def test_gli_lowers_cash_flow(self):
        df0, _ = _run(gli_pct=0.0)
        df1, _ = _run(gli_pct=3.0)
        gap = df0.iloc[0]["Cash-Flow Net d'Impôt"] - df1.iloc[0]["Cash-Flow Net d'Impôt"]
        assert 0 < gap <= 7200 * 0.03

    def test_works_months_remove_year_1_rent_only(self):
        df0, _ = _run(budget_travaux=10_000.0, mois_travaux=0)
        df3, _ = _run(budget_travaux=10_000.0, mois_travaux=3)
        assert df3.iloc[0]["Loyers Bruts"] == pytest.approx(df0.iloc[0]["Loyers Bruts"] * 9 / 12)
        assert df3.iloc[1]["Loyers Bruts"] == pytest.approx(df0.iloc[1]["Loyers Bruts"])
        assert df3.iloc[0]["Cash-Flow Net d'Impôt"] < df0.iloc[0]["Cash-Flow Net d'Impôt"]

    def test_tf_growth_raises_later_charges(self):
        df_lo, _ = _run(market={"tf_growth_pct": 0.0})
        df_hi, _ = _run(market={"tf_growth_pct": 3.0})
        assert df_hi.iloc[0]["Cash-Flow Net d'Impôt"] == pytest.approx(df_lo.iloc[0]["Cash-Flow Net d'Impôt"])
        assert df_hi.iloc[9]["Cash-Flow Net d'Impôt"] < df_lo.iloc[9]["Cash-Flow Net d'Impôt"]

    def test_vacancy_reduces_receipts_and_micro_base(self):
        """Vacancy lowers the gross receipts, so the micro-foncier base (70 % of receipts) drops with it."""
        kw = {"regime_foncier": "micro", "charges_const_mth0": 0.0, "tf_const_mth0": 0.0}
        df0, _ = _run(provision_pct=0.0, **kw)
        df10, _ = _run(provision_pct=10.0, **kw)
        assert df10.iloc[0]["Loyers Bruts"] == pytest.approx(df0.iloc[0]["Loyers Bruts"] * 0.9)
        assert df10.iloc[0]["Résultat Fiscal"] == pytest.approx(df0.iloc[0]["Résultat Fiscal"] * 0.9)

    def test_new_defaults(self):
        from src.core.defaults import OPERATING, REALISM

        assert OPERATING.gestion_pct == 7.0
        assert (REALISM.maintenance_m2_an, REALISM.gli_pct, REALISM.mois_travaux, REALISM.tf_growth_pct) == (
            10.0,
            3.0,
            3,
            3.0,
        )

    def test_vacancy_slider_is_a_floor(self, sample_archetypes):
        arch = [{**sample_archetypes[0], "vacance_pct": 4.0}]
        fin = FinancingConfig(credit_rates={20: 3.5})

        def effective(slider: float) -> float:
            return create_investment_bricks(arch, fin, OperatingConfig(provision_pct=slider))[0].provision_pct

        assert effective(2.0) == 4.0  # slider below the archetype: archetype wins
        assert effective(8.0) == 8.0  # slider above: slider wins

    def test_brick_factory_parameters_flow(self, sample_archetypes):
        fin = FinancingConfig(credit_rates={20: 3.5})
        base = create_investment_bricks(sample_archetypes, fin, OperatingConfig(maintenance_m2_an=0, gli_pct=0))[0]
        more = create_investment_bricks(sample_archetypes, fin, OperatingConfig(maintenance_m2_an=20, gli_pct=5))[0]
        assert more.depenses_mensuelles_hors_credit_initial > base.depenses_mensuelles_hors_credit_initial
        assert more.maintenance_m2_an == 20 and more.gli_pct == 5
        assert more.mois_travaux == 3  # archetype 1 has works
        fin_nw = FinancingConfig(credit_rates={20: 3.5}, inclure_travaux=False, inclure_reno_ener=False)
        assert create_investment_bricks(sample_archetypes, fin_nw, OperatingConfig())[0].mois_travaux == 0

    def test_run_context_carries_new_params(self):
        from src.application.run_context import RunContext

        ctx = RunContext(maintenance_m2_an=15.0, gli_pct=4.0, mois_travaux=2, tf_growth_pct=5.0)
        assert ctx.market_hypotheses["tf_growth_pct"] == 5.0
        assert RunContext.from_dict(ctx.to_dict()) == ctx
