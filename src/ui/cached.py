"""Streamlit cache wrappers (T42). The domain and application layers stay free of streamlit.

``st.cache_data`` pickles each return value and hands every caller a fresh copy, so callers may
mutate what they get without corrupting the cache.

The search itself is NOT cached here: it runs only on the "Lancer l'analyse" click (it takes a progress
callback and may use a process pool) and its results live in the session.
"""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st

from src.application.run_context import RunContext, hash_archetypes
from src.application.services.brick_factory import (
    apply_rent_caps,
    build_financing_config,
    build_operating_config,
    create_investment_bricks,
)
from src.ui.components.filters import filter_archetypes


def file_signature(path: str) -> tuple[str, float, int]:
    """Cache key part for a data file: path, mtime and size (changes when the file is edited)."""
    st_ = os.stat(path)
    return (path, st_.st_mtime, st_.st_size)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_archetypes_from_file(path: str, mtime: float, size: int) -> list[dict[str, Any]]:
    """Archetypes of a data file. ``mtime`` and ``size`` only serve as cache key."""
    from src.ui.app_controller import load_archetypes_from_disk

    return load_archetypes_from_disk(path)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_filter_and_caps(
    _archetypes: list[dict[str, Any]],
    archetypes_hash: str,
    villes: tuple[str, ...],
    types: tuple[str, ...],
    apply_cap: bool,
) -> list[dict[str, Any]]:
    """Filter by city and type, then apply rent caps. ``_archetypes`` is not hashed; ``archetypes_hash`` is."""
    filtered = filter_archetypes(_archetypes, list(villes), list(types))
    return apply_rent_caps(filtered, apply_cap)


def filter_and_caps(
    archetypes: list[dict[str, Any]], villes: list[str], types: list[str], apply_cap: bool
) -> list[dict[str, Any]]:
    return cached_filter_and_caps(archetypes, hash_archetypes(archetypes), tuple(villes), tuple(types), apply_cap)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_bricks(_archetypes: list[dict[str, Any]], archetypes_hash: str, brick_params_json: str) -> list[Any]:
    p = json.loads(brick_params_json)
    fin_config = build_financing_config(p["credit"])
    op_config = build_operating_config(
        p["gestion_pct"], p["vacance_pct"], p["cfe_par_bien_ann"],
        maintenance_m2_an=p["maintenance_m2_an"], gli_pct=p["gli_pct"], mois_travaux=p["mois_travaux"],
    )
    return create_investment_bricks(_archetypes, fin_config, op_config)


def cached_bricks(archetypes: list[dict[str, Any]], ctx: RunContext) -> list[Any]:
    """Bricks for these archetypes and the financing / operating fields of ``ctx`` (a fresh copy each call).

    The key holds only the ctx fields the bricks depend on: apport, horizon, etc. do not change them.
    """
    params = {
        "credit": ctx.to_credit_params(),
        "gestion_pct": ctx.gestion_pct,
        "vacance_pct": ctx.vacance_pct,
        "cfe_par_bien_ann": ctx.cfe_par_bien_ann,
        "maintenance_m2_an": ctx.maintenance_m2_an,
        "gli_pct": ctx.gli_pct,
        "mois_travaux": ctx.mois_travaux,
    }
    return _cached_bricks(archetypes, hash_archetypes(archetypes), json.dumps(params, sort_keys=True, default=str))
