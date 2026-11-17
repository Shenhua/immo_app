"""T06: disk side effects of a default run are opt-in (APP_AUTOSAVE / APP_RESULTS_DIR)."""

import importlib
import importlib.util
import os

import pytest

from src.core import defaults as _defaults_module
from src.ui import app_controller


def _fresh_defaults():
    """Execute src/core/defaults.py again as a throw-away module (env is read at import).

    Reloading the real module would replace PresetId and break isinstance checks elsewhere.
    """
    spec = importlib.util.spec_from_file_location("_defaults_fresh", _defaults_module.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_autosave_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without APP_AUTOSAVE=1 the module flag is off and nothing is read from env at call time."""
    monkeypatch.delenv("APP_AUTOSAVE", raising=False)
    assert _fresh_defaults().AUTOSAVE_ENABLED is False
    reloaded = importlib.reload(app_controller)
    assert reloaded.AUTOSAVE_ENABLED is False


def test_env_vars_read_once_at_import(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """APP_AUTOSAVE=1 and APP_RESULTS_DIR are honoured when the module is imported."""
    monkeypatch.setenv("APP_AUTOSAVE", "1")
    monkeypatch.setenv("APP_RESULTS_DIR", str(tmp_path))
    fresh = _fresh_defaults()
    assert fresh.AUTOSAVE_ENABLED is True
    assert fresh.RESULTS_DIR == str(tmp_path)
    # app_controller mirrors the defaults module values at import; patch them as a reload would.
    monkeypatch.setattr(_defaults_module, "AUTOSAVE_ENABLED", fresh.AUTOSAVE_ENABLED)
    monkeypatch.setattr(_defaults_module, "RESULTS_DIR", fresh.RESULTS_DIR)
    try:
        reloaded = importlib.reload(app_controller)
        assert reloaded.AUTOSAVE_ENABLED is True
        assert reloaded.RESULTS_DIR == str(tmp_path)
    finally:
        monkeypatch.undo()  # restore defaults values and env before re-importing the controller
        importlib.reload(app_controller)


def test_autosave_results_writes_nothing_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A default run must not create the results directory or any JSON file."""
    monkeypatch.setattr(app_controller, "AUTOSAVE_ENABLED", False)
    monkeypatch.setattr(app_controller, "RESULTS_DIR", str(tmp_path / "results"))

    app_controller.autosave_results([{"dummy": 1}], {"horizon": 25})

    assert not os.path.exists(tmp_path / "results")
    assert list(tmp_path.iterdir()) == []


def test_autosave_results_writes_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """With the flag on, results go to the configured directory."""
    out_dir = tmp_path / "out"
    monkeypatch.setattr(app_controller, "AUTOSAVE_ENABLED", True)
    monkeypatch.setattr(app_controller, "RESULTS_DIR", str(out_dir))

    app_controller.autosave_results([{"dummy": 1}], {"horizon": 25})

    files = list(out_dir.glob("simulation_*.json"))
    assert len(files) == 1
