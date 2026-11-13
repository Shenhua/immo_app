"""Pytest fixtures for app_immo tests."""

import os
import shutil
import sys
import tempfile

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_session_log_dir: str | None = None


def pytest_configure(config):
    """Point logging at a temp dir before collection.

    Modules call get_logger() at import time, which configures the file handler
    before any test (and any autouse fixture) runs. Setting APP_LOG_DIR here keeps
    that first configuration out of the repo.
    """
    global _session_log_dir
    _session_log_dir = tempfile.mkdtemp(prefix="app_immo_test_logs_")
    os.environ["APP_LOG_DIR"] = _session_log_dir


def pytest_unconfigure(config):
    if _session_log_dir:
        shutil.rmtree(_session_log_dir, ignore_errors=True)
    os.environ.pop("APP_LOG_DIR", None)


@pytest.fixture(autouse=True)
def _isolate_log_dir(tmp_path, monkeypatch):
    """Send each test's log output to its own tmp_path, never to the repo's logs/."""
    monkeypatch.setenv("APP_LOG_DIR", str(tmp_path / "logs"))


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
        "tension_locative_score_norm": 0.8,
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
                "tension_locative_score_norm": 0.8,
                "transport_score": 0.9,
                "liquidite_score": 0.7,
                "qual_score_bien": 75.0,
            },
            {
                "nom": "T2 Lyon",
                "cout_total": 180000,
                "loyer_m2": 15.0,
                "dpe_initial": "D",
                "tension_locative_score_norm": 0.6,
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
