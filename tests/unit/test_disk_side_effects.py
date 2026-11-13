"""T06: disk side effects of a default run are opt-in (APP_AUTOSAVE / APP_RESULTS_DIR)."""

import importlib
import os

import pytest

from src.ui import app_controller


def test_autosave_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without APP_AUTOSAVE=1 the module flag is off and nothing is read from env at call time."""
    monkeypatch.delenv("APP_AUTOSAVE", raising=False)
    reloaded = importlib.reload(app_controller)
    assert reloaded.AUTOSAVE_ENABLED is False


def test_env_vars_read_once_at_import(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """APP_AUTOSAVE=1 and APP_RESULTS_DIR are honoured when the module is imported."""
    monkeypatch.setenv("APP_AUTOSAVE", "1")
    monkeypatch.setenv("APP_RESULTS_DIR", str(tmp_path))
    try:
        reloaded = importlib.reload(app_controller)
        assert reloaded.AUTOSAVE_ENABLED is True
        assert reloaded.RESULTS_DIR == str(tmp_path)
    finally:
        monkeypatch.delenv("APP_AUTOSAVE", raising=False)
        monkeypatch.delenv("APP_RESULTS_DIR", raising=False)
        importlib.reload(app_controller)


def test_autosave_results_writes_nothing_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A default run must not create the results directory or any JSON file."""
    monkeypatch.setattr(app_controller, "AUTOSAVE_ENABLED", False)
    monkeypatch.setattr(app_controller, "RESULTS_DIR", str(tmp_path / "results"))

    app_controller.autosave_results([{"dummy": 1}], {"horizon": 25})

    assert not os.path.exists(tmp_path / "results")
    assert list(tmp_path.iterdir()) == []


def test_autosave_results_writes_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """With the flag on, results go to the configured directory."""
    out_dir = tmp_path / "out"
    monkeypatch.setattr(app_controller, "AUTOSAVE_ENABLED", True)
    monkeypatch.setattr(app_controller, "RESULTS_DIR", str(out_dir))

    app_controller.autosave_results([{"dummy": 1}], {"horizon": 25})

    files = list(out_dir.glob("simulation_*.json"))
    assert len(files) == 1
