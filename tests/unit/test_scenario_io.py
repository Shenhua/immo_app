"""T65: scenario encode / decode, JSON files, and the app loading a shared link."""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from src.application.run_context import RunContext
from src.application.scenario_io import (
    ScenarioError,
    build_query_params,
    decode_context,
    encode_context,
    scenario_from_json,
    scenario_to_json,
)
from src.ui.components.sidebar_state import session_values

CUSTOM = RunContext(
    apport=120_000,
    cf_cible=-250.0,
    mode_cf="target",
    tolerance=40,
    taux_credits={10: 3.1, 20: 3.6},
    regime_fiscal="microbic",
    ratio_terrain_pct=22.0,
    nb_owners=2,
    tmi_pct=41.0,
    revenu_net_mensuel=5200.0,
)


def _payload(data: Any) -> str:
    raw = json.dumps(data).encode()
    return base64.urlsafe_b64encode(zlib.compress(raw)).rstrip(b"=").decode()


def test_round_trip() -> None:
    assert decode_context(encode_context(CUSTOM)) == CUSTOM


def test_defaults_give_empty_payload_and_decode_to_defaults() -> None:
    assert encode_context(RunContext()) == ""
    assert decode_context("") == RunContext()
    assert build_query_params(RunContext()) == {}


def test_archetypes_version_is_not_shared() -> None:
    assert encode_context(RunContext(archetypes_version="abc")) == ""


def test_payload_is_compact_and_url_safe() -> None:
    payload = encode_context(CUSTOM)
    assert len(payload) < 300
    assert all(c.isalnum() or c in "-_" for c in payload)
    assert build_query_params(CUSTOM, 3)["strategy"] == "3"


@pytest.mark.parametrize("bad", ["!!!not base64!!!", "abcd", _payload({"v": 1})[:-6], "A" * 50])
def test_corrupted_payload_raises_clean_error(bad: str) -> None:
    with pytest.raises(ScenarioError):
        decode_context(bad)


def test_unsupported_version_and_invalid_value_are_clean_errors() -> None:
    with pytest.raises(ScenarioError, match="Version"):
        decode_context(_payload({"v": 99, "apport": 1}))
    with pytest.raises(ScenarioError, match="mode_cf"):
        decode_context(_payload({"v": 1, "mode_cf": "bogus"}))
    with pytest.raises(ScenarioError):
        decode_context(_payload([1, 2]))


def test_unknown_field_is_ignored() -> None:
    ctx = decode_context(_payload({"v": 1, "apport": 99_000, "future_field": {"x": 1}}))
    assert ctx == RunContext(apport=99_000)


def test_json_file_round_trip() -> None:
    text = scenario_to_json("Mon plan", CUSTOM, ["a+b"])
    name, ctx, keys = scenario_from_json(text)
    assert (name, ctx, keys) == ("Mon plan", CUSTOM, ["a+b"])
    assert json.loads(text)["saved_at"]


def test_json_file_errors_are_friendly() -> None:
    for bad in (
        "not json",
        "[]",
        '{"version": 1}',
        '{"version": 7, "context": {}}',
        '{"version": 1, "context": {"apport": "lots"}}',
    ):
        with pytest.raises(ScenarioError):
            scenario_from_json(bad)
    _, ctx, _ = scenario_from_json('{"version": 1, "context": {"apport": 5, "zzz": 1}}')
    assert ctx.apport == 5


def test_session_values_mirror_the_context() -> None:
    values = session_values(CUSTOM)
    assert values["sb_apport"] == 120_000
    assert values["sb_precise"] is True
    assert values["sb_durees"] == [10, 20]
    assert values["taux_duree_10"] == 3.1
    assert values["sb_override_terrain"] is True and values["sb_ratio_terrain"] == 22.0
    assert values["tax_regime_meuble"] == "Micro-BIC"
    # an absurd context is clamped to what the widgets accept
    wild = session_values(RunContext(apport=-5, cf_cible=99999.0, horizon_ans=99))
    assert wild["sb_apport"] == 0 and wild["sb_cf_cible"] == 2000.0 and wild["horizon_ans"] == 30


def test_app_applies_shared_link_to_sidebar() -> None:
    at = AppTest.from_file("../../app.py", default_timeout=180)
    at.query_params["p"] = encode_context(CUSTOM)
    at.run()
    assert not at.exception
    apport = next(n for n in at.sidebar.number_input if n.label.startswith("Apport"))
    assert apport.value == 120_000
    assert at.session_state["sb_tmi"] == 41
    assert decode_context(at.query_params["p"]).apport == 120_000  # the address bar stays a valid link


def test_app_warns_on_corrupted_link_and_keeps_defaults() -> None:
    at = AppTest.from_file("../../app.py", default_timeout=180)
    at.query_params["p"] = "garbage"
    at.run()
    assert not at.exception
    assert any("lien" in w.value.lower() for w in at.sidebar.warning)


def test_default_sidebar_gives_empty_link() -> None:
    at = AppTest.from_file("../../app.py", default_timeout=180)
    at.run()
    assert not at.exception
    assert "p" not in at.query_params


def test_suggestion_patch_sets_only_that_sidebar_value() -> None:
    from src.ui.components.sidebar_state import PENDING_KEY

    at = AppTest.from_file("../../app.py", default_timeout=180)
    at.run()
    at.session_state[PENDING_KEY] = {"patch": {"cf_cible": -400.0, "apport": 95_000.0}}
    at.run()
    assert not at.exception
    cf = next(n for n in at.sidebar.number_input if n.label.startswith("Cash-Flow"))
    apport = next(n for n in at.sidebar.number_input if n.label.startswith("Apport"))
    assert (cf.value, apport.value) == (-400.0, 95_000)
    assert at.session_state["sb_tmi"] == 30  # untouched


def test_applying_a_scenario_with_custom_weights_and_rentier_mode() -> None:
    from src.ui.components.sidebar_state import PENDING_KEY

    ctx = RunContext(
        use_full_capital=True,
        finance_weights={"enrich_net": 0.4, "irr": 0.3, "cap_eff": 0.1, "dscr": 0.1, "cf_proximity": 0.1},
        revenu_net_mensuel=7000.0,
    )
    at = AppTest.from_file("../../app.py", default_timeout=180)
    at.run()
    at.session_state[PENDING_KEY] = {"ctx": ctx.model_dump(mode="json")}
    at.run()
    assert not at.exception
    assert at.session_state["sb_mode"] == "Rentier (Max Cash-Flow)"
    assert at.session_state["sb_custom_on"] is True
    assert at.session_state["sb_w_enrich"] == 40
    assert at.session_state["max_properties"] == 3  # not reset by the mode-change handler
