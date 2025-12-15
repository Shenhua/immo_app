"""Pytest fixtures for app_immo tests."""

import os
import sys

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_archetype_data():
    """Sample archetype data for testing."""
    return {
        "nom": "Studio Paris 11",
        "ville": "Paris",
        "mode_loyer": "meuble_classique",
        "surface": 25.0,
        "prix_m2": 10000.0,
        "loyer_m2": 35.0,
        "loyer_m2_max": 40.0,
        "charges_m2_an": 50.0,
        "taxe_fonciere_m2_an": 30.0,
        "meuble": True,
        "soumis_encadrement": True,
        "dpe_initial": "C",
        "budget_travaux": 5000.0,
        "valeur_mobilier": 3000.0,
        "indice_tension": 0.8,
        "transport_score": 0.9,
        "liquidite_score": 0.7,
    }


@pytest.fixture
def sample_strategy_data():
    """Sample strategy data for testing."""
    return {
        "details": [
            {
                "nom": "Studio Paris",
                "cout_total": 275000,
                "loyer_m2": 35.0,
                "loyer_m2_max": 40.0,
                "dpe_initial": "C",
                "indice_tension": 0.8,
                "transport_score": 0.9,
                "liquidite_score": 0.7,
                "qual_score_bien": 75.0,
            },
            {
                "nom": "T2 Lyon",
                "cout_total": 180000,
                "loyer_m2": 15.0,
                "dpe_initial": "D",
                "indice_tension": 0.6,
                "transport_score": 0.7,
                "liquidite_score": 0.6,
                "qual_score_bien": 60.0,
            },
        ],
        "apport_total": 80000,
        "patrimoine_acquis": 455000,
        "cash_flow_final": 150,
        "balanced_score": 72.5,
    }
