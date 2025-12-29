
import pytest

from src.application.services.brick_factory import FinancingConfig, OperatingConfig, apply_rent_caps, create_investment_bricks


@pytest.fixture
def sample_archetypes():
    return [
        {
            "nom": "Appart A",
            "ville": "Paris",
            "mode_loyer": "nu",
            "surface": 20.0,
            "prix_m2": 10000.0,
            "loyer_m2": 35.0,
            "loyer_m2_max": 30.0,
            "soumis_encadrement": True,
            "charges_m2_an": 20.0,
            "taxe_fonciere_m2_an": 10.0,
        },
        {
            "nom": "Appart B",
            "ville": "Lyon",
            "mode_loyer": "nu",
            "surface": 30.0,
            "prix_m2": 5000.0,
            "loyer_m2": 20.0,
            "loyer_m2_max": 25.0, # Cap higher than rent, should not change
            "soumis_encadrement": True,
        },
        {
            "nom": "Appart C",
            "ville": "Bordeaux",
            "mode_loyer": "nu",
            "surface": 25.0,
            "prix_m2": 6000.0,
            "loyer_m2": 20.0,
            "soumis_encadrement": False, # No cap
        }
    ]

def test_apply_rent_caps(sample_archetypes):
    """Verify correct application of rent control caps."""
    # Test strict application
    capped = apply_rent_caps(sample_archetypes, apply_cap=True)

    # Paris (Appart A): Should be capped at 30
    assert capped[0]["loyer_m2"] == 30.0

    # Lyon (Appart B): Rent 20 < Cap 25, should remain 20
    assert capped[1]["loyer_m2"] == 20.0

    # Bordeaux (Appart C): Not subject, should remain 20
    assert capped[2]["loyer_m2"] == 20.0

def test_apply_rent_caps_disabled(sample_archetypes):
    """Verify caps are ignored when flag is False."""
    uncapped = apply_rent_caps(sample_archetypes, apply_cap=False)
    assert uncapped[0]["loyer_m2"] == 35.0

def test_create_investment_bricks():
    """Verify investment bricks are created with correct variants."""
    input_data = [{
        "nom": "TestBrick",
        "ville": "TestCity",
        "mode_loyer": "nu",
        "surface": 10.0,
        "prix_m2": 2000.0,
        "loyer_m2": 10.0,
        # Required extra fields that might default safely but good to be explicit
        "charges_m2_an": 0, "taxe_fonciere_m2_an": 0,
        "budget_travaux": 0, "renovation_energetique_cout": 0, "valeur_mobilier": 0
    }]

    fin_config = FinancingConfig(
        credit_rates={20: 3.5, 25: 3.8}, # Two variants expected
        frais_notaire_pct=8.0,
        apport_min_pct=10.0,
        assurance_ann_pct=0.36,
        frais_pret_pct=1.0
    )
    op_config = OperatingConfig()

    bricks = create_investment_bricks(input_data, fin_config, op_config)

    # Should produce 2 bricks (one for 20y, one for 25y)
    assert len(bricks) == 2

    # Check durations
    # durre_credit_mois is in months in InvestmentBrick
    durations = sorted([b.duree_credit_mois // 12 for b in bricks])
    assert durations == [20, 25]

    # Check basic cost calc (Surface 10 * Prix 2000 = 20000)
    # Notaire 8% = 1600. Total Project ~ 21600 (+ fees)
    b20 = bricks[0]
    assert b20.prix_achat_bien == 20000.0
    assert b20.frais_notaire == 1600.0
