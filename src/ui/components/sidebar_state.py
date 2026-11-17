"""Sidebar <-> RunContext mapping (T65).

Every sidebar widget has a stable ``key=``; its value lives in ``st.session_state``. This module knows which key
holds which ``RunContext`` field, so a context can be applied to the sidebar (``apply_context_to_session``), a
single field patched (``request_patch``, used by the diagnostics "Appliquer cette suggestion" buttons) and the
defaults seeded (``ensure_defaults``). Defaults are those of ``RunContext()``, i.e. ``src/core/defaults.py``.

A widget value cannot be written after the widget was created in the same run, so applying is two-step: callers
``request_*`` (a callback or any point of the script) then ``consume_pending`` runs at the very top of the next
run, before the sidebar is drawn.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from src.application.run_context import RunContext
from src.core.defaults import DEFAULT_PRESET, DPE, FINANCING, PRESETS, SEARCH, TAX

# --- Labels shared with the widgets -------------------------------------------------------------------------
FINANCIAL_PRESETS: dict[str, dict[str, float]] = {pid.label: w.as_dict() for pid, w in PRESETS.items()}
DEFAULT_PRESET_LABEL = DEFAULT_PRESET.label
MODE_CLASSIQUE = "Classique (Levier)"
MODE_RENTIER = "Rentier (Max Cash-Flow)"
MODE_EMPIRE = "Empire (Max Volume)"
STRATEGY_MODES = [MODE_CLASSIQUE, MODE_RENTIER, MODE_EMPIRE]
DPE_LABELS: dict[str, str] = {
    "Rénover avant l'interdiction": "renover",
    "Ne pas rénover (loyer nul dès l'interdiction)": "ne_pas_renover",
}
REGIMES_MEUBLE: dict[str, str] = {"LMNP réel": "lmnp", "Micro-BIC": "microbic"}
REGIMES_NU: dict[str, str] = {"Réel (déficit foncier)": "reel", "Micro-foncier": "micro"}
WEIGHT_KEYS = {  # finance weight -> slider key
    "enrich_net": "sb_w_enrich",
    "irr": "sb_w_irr",
    "cap_eff": "sb_w_cap_eff",
    "dscr": "sb_w_dscr",
    "cf_proximity": "sb_w_cf_prox",
}

PENDING_KEY = "_pending_sidebar"
URL_APPLIED_KEY = "_url_params_applied"
SAVED_SCENARIOS_KEY = "saved_scenarios"


def _label(options: dict[str, str], value: Any, default: str) -> str:
    for label, code in options.items():
        if code == value:
            return label
    for label, code in options.items():
        if code == default:
            return label
    return next(iter(options))


def _clamp(lo: float | None, hi: float | None, cast: Callable[[Any], Any] = float) -> Callable[[Any], Any]:
    def conv(v: Any) -> Any:
        x = float(v)
        if lo is not None:
            x = max(lo, x)
        if hi is not None:
            x = min(hi, x)
        return cast(x)

    return conv


def _step(step: int, lo: int, hi: int) -> Callable[[Any], int]:
    return lambda v: int(min(hi, max(lo, round(float(v) / step) * step)))


def _bool(v: Any) -> bool:
    return bool(v)


# (session key, RunContext field, converter field value -> widget value). Bounds mirror the widgets' own.
_SPECS: list[tuple[str, str, Callable[[Any], Any]]] = [
    ("sb_apport", "apport", _clamp(0, None, int)),
    ("sb_cf_cible", "cf_cible", _clamp(-2000.0, 2000.0)),
    ("sb_precise", "mode_cf", lambda m: m == "target"),
    ("sb_tolerance", "tolerance", _clamp(0, None, int)),
    ("sb_priorite", "qualite_weight", lambda w: _step(5, 0, 100)(float(w) * 100.0)),
    ("horizon_ans", "horizon_ans", _clamp(10, 30, int)),
    ("max_properties", "max_properties", _clamp(SEARCH.max_properties_min, SEARCH.max_properties_max, int)),
    ("sb_revenu", "revenu_net_mensuel", _clamp(0.0, None)),
    ("sb_charges", "charges_mensuelles", _clamp(0.0, None)),
    ("sb_filtrer_hcsf", "filtrer_hcsf", _bool),
    ("sb_frais_notaire", "frais_notaire_pct", _clamp(0.0, 12.0)),
    ("sb_assurance", "assurance_ann_pct", _clamp(0.0, 1.0)),
    ("sb_frais_pret", "frais_pret_pct", _clamp(0.0, 3.0)),
    ("sb_inc_travaux", "inclure_travaux", _bool),
    ("sb_inc_reno", "inclure_reno_ener", _bool),
    ("sb_inc_mobilier", "inclure_mobilier", _bool),
    ("sb_fin_mobilier", "financer_mobilier", _bool),
    ("sb_dpe_policy", "dpe_policy", lambda p: _label(DPE_LABELS, p, DPE.policy)),
    ("sb_dpe_cout", "dpe_cout_m2_par_classe", _clamp(0.0, 1000.0)),
    ("sb_override_terrain", "ratio_terrain_pct", lambda r: r is not None),
    ("sb_ratio_terrain", "ratio_terrain_pct", lambda r: _clamp(0.0, 60.0)(TAX.ratio_terrain_pct if r is None else r)),
    ("sb_nb_owners", "nb_owners", lambda n: 2 if int(n) == 2 else 1),
    ("apply_ira", "apply_ira", _bool),
    ("ira_cap_pct", "ira_cap_pct", _clamp(0.0, 10.0)),
    ("sb_tmi", "tmi_pct", _clamp(0, 45, int)),
    ("tax_sci_is", "sci_is", _bool),
    ("tax_regime_meuble", "regime_fiscal", lambda r: _label(REGIMES_MEUBLE, r, TAX.regime_fiscal)),
    ("tax_regime_nu", "regime_foncier", lambda r: _label(REGIMES_NU, r, TAX.regime_foncier)),
    ("sb_cfe", "cfe_par_bien_ann", _clamp(0, 2000, int)),
    ("sb_gestion", "gestion_pct", _clamp(0.0, 15.0)),
    ("sb_vacance", "vacance_pct", _clamp(0.0, 10.0)),
    ("sb_gli", "gli_pct", _clamp(0.0, 10.0)),
    ("sb_maintenance", "maintenance_m2_an", _clamp(0.0, 100.0)),
    ("sb_mois_travaux", "mois_travaux", _clamp(0, 12, int)),
    ("sb_tf_growth", "tf_growth_pct", _clamp(-2.0, 10.0)),
    ("sb_frais_vente", "frais_vente_pct", _clamp(0.0, 10.0)),
    ("sb_top_n", "top_n", _step(10, 10, 500)),
    ("sb_apply_cap", "apply_cap", _bool),
    ("appreciation_bien_pct", "appreciation_bien_pct", _clamp(-2.0, 8.0)),
    ("revalo_loyer_pct", "revalo_loyer_pct", _clamp(-2.0, 5.0)),
    ("inflation_charges_pct", "inflation_charges_pct", _clamp(-2.0, 5.0)),
]


def _mode_for(ctx: RunContext) -> str:
    if ctx.use_full_capital:
        return MODE_RENTIER
    return MODE_EMPIRE if ctx.max_properties >= SEARCH.max_properties_empire else MODE_CLASSIQUE


def _finance_values(ctx: RunContext) -> dict[str, Any]:
    preset = ctx.finance_preset_name if ctx.finance_preset_name in FINANCIAL_PRESETS else DEFAULT_PRESET_LABEL
    base = FINANCIAL_PRESETS[preset]
    weights = {k: float(ctx.finance_weights.get(k, base[k])) for k in base}
    custom = any(abs(weights[k] - base[k]) > 0.005 for k in base)
    shown = weights if custom else base
    out: dict[str, Any] = {
        "finance_preset": preset,
        "finance_custom": dict(shown),
        "sb_custom_on": custom,
    }
    for name, key in WEIGHT_KEYS.items():
        out[key] = int(min(100, max(0, round(shown[name] * 100))))
    return out


def session_values(ctx: RunContext) -> dict[str, Any]:
    """Widget values (by session key) that make the sidebar show ``ctx``. Pure: usable without streamlit."""
    values: dict[str, Any] = {}
    for key, field, conv in _SPECS:
        values[key] = conv(getattr(ctx, field))
    mode = _mode_for(ctx)
    values["sb_mode"] = mode
    values["last_strategy_mode"] = mode  # stops the mode-change handler from resetting preset / max properties
    years = sorted(y for y in ctx.taux_credits if FINANCING.duree_min_ans <= int(y) <= FINANCING.duree_max_ans)
    years = years[: FINANCING.max_durations] or list(FINANCING.durees_defaut)
    values["sb_durees"] = [int(y) for y in years]
    for y in years:
        values[f"taux_duree_{y}"] = _clamp(0.0, None)(ctx.taux_credits.get(y, FINANCING.default_rate(int(y))))
    values.update(_finance_values(ctx))
    return values


def ensure_defaults() -> None:
    """Seed every sidebar key that is missing with the ``RunContext()`` default."""
    skip = set(WEIGHT_KEYS.values())  # the sliders are hidden unless "Personnaliser" is on: seeded when shown
    for key, value in session_values(RunContext()).items():
        if key not in skip and st.session_state.get(key) is None:
            st.session_state[key] = value


def apply_context_to_session(ctx: RunContext) -> None:
    """Write ``ctx`` to the sidebar widget keys. Must run before the sidebar is drawn."""
    for key, value in session_values(ctx).items():
        st.session_state[key] = value


def request_apply(ctx: RunContext) -> None:
    """Ask for ``ctx`` to be applied at the start of the next run (callable from callbacks)."""
    st.session_state[PENDING_KEY] = {"ctx": ctx.model_dump(mode="json")}


def request_patch(field: str, value: Any) -> None:
    """Ask for one ``RunContext`` field to be set in the sidebar at the start of the next run."""
    pending = st.session_state.get(PENDING_KEY) or {}
    pending.setdefault("patch", {})[field] = value
    st.session_state[PENDING_KEY] = pending


def consume_pending() -> None:
    """Apply what ``request_apply`` / ``request_patch`` stored. Call before the sidebar is rendered."""
    pending = st.session_state.pop(PENDING_KEY, None)
    if not pending:
        return
    if "ctx" in pending:
        apply_context_to_session(RunContext.model_validate(pending["ctx"]))
    patch = pending.get("patch")
    if patch:
        try:
            patched = RunContext().model_copy(update=patch)
        except Exception:  # pragma: no cover - patch comes from our own suggestions
            return
        fields = set(patch)
        for key, field, conv in _SPECS:
            if field in fields:
                st.session_state[key] = conv(getattr(patched, field))
        if "mode_cf" in fields:
            st.session_state["sb_precise"] = patched.mode_cf == "target"
