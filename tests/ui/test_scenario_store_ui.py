"""T67a: saved scenarios in the UI survive a refresh (a new session), per profile."""

from __future__ import annotations

from src.application.scenario_store import default_store
from src.ui.components.profile import PROFILE_KEY

from .conftest import new_app, set_inputs


def _save(at, name: str):
    at.text_input(key="sc_name").set_value(name)
    at.run()
    at.button(key="sc_save").click()
    at.run()
    return at


def test_saved_scenario_survives_a_new_session():
    at = set_inputs(new_app(), sb_apport=77_000)
    _save(at, "Mon plan")
    assert not at.exception

    fresh = new_app()  # a refresh: new session, same database
    options = fresh.selectbox(key="sc_choice").options
    assert any("Mon plan" in o for o in options)

    # Loading it applies the stored context to the sidebar.
    set_inputs(fresh, sb_apport=1_000)
    fresh.button(key="sc_load").click()
    fresh.run()
    assert fresh.number_input(key="sb_apport").value == 77_000


def test_rename_and_delete():
    at = _save(new_app(), "Avant")
    at.text_input(key="sc_rename_input").set_value("Après")
    at.run()
    at.button(key="sc_rename").click()
    at.run()
    assert any("Après" in o for o in at.selectbox(key="sc_choice").options)
    at.button(key="sc_delete").click()
    at.run()
    assert not at.exception
    assert [s for s in at.selectbox if s.key == "sc_choice"] == []


def test_profiles_are_isolated():
    at = _save(new_app(), "Privé")
    default_store().ensure_user("marie", "Marie")
    at.run()
    profile = at.selectbox(key=PROFILE_KEY)
    assert profile.options == ["local", "Marie"]
    profile.select_index(1)
    at.run()
    assert at.session_state[PROFILE_KEY] == "marie"
    assert [s for s in at.selectbox if s.key == "sc_choice"] == []  # Marie sees nothing
    at.selectbox(key=PROFILE_KEY).select_index(0)
    at.run()
    assert any("Privé" in o for o in at.selectbox(key="sc_choice").options)


def test_last_context_restored_but_url_wins():
    set_inputs(new_app(), sb_apport=91_000)
    assert new_app().number_input(key="sb_apport").value == 91_000  # restored, no ?p=

    from src.application.run_context import RunContext
    from src.application.scenario_io import encode_context

    linked = new_app({"p": encode_context(RunContext(apport=55_000))})
    assert linked.number_input(key="sb_apport").value == 55_000  # ?p= wins


def test_unavailable_store_does_not_break_the_app(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    from src.core import defaults

    monkeypatch.setattr(defaults, "DATA_DIR", str(blocker / "sub"))  # cannot be created
    at = new_app()
    assert not at.exception
