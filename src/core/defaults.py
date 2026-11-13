"""Single source of truth for default parameters (T10).

Every module and every UI widget takes its default values from here. Do not
repeat these numbers elsewhere: ``tests/unit/test_defaults.py`` checks that the
sidebar and the services stay in line with this module.

Environment overrides are read once, at import time, in the "Environment"
section at the bottom of this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


# --------------------------------------------------------------------------- #
# Market, financing, operating costs, tax
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarketDefaults:
    """Yearly market hypotheses (percent)."""

    appreciation_bien_pct: float = 2.0
    revalo_loyer_pct: float = 1.5
    inflation_charges_pct: float = 2.0


@dataclass(frozen=True)
class FinancingDefaults:
    """Loan and acquisition cost defaults."""

    taux_15_pct: float = 3.5
    taux_20_pct: float = 3.6
    taux_25_pct: float = 3.8
    frais_notaire_pct: float = 8.0
    assurance_ann_pct: float = 0.35
    frais_pret_pct: float = 1.0
    apport_min_pct: float = 10.0
    inclure_travaux: bool = True
    inclure_reno_ener: bool = True
    inclure_mobilier: bool = True
    financer_mobilier: bool = True
    apply_ira: bool = True
    ira_cap_pct: float = 3.0

    @property
    def taux_credits(self) -> dict[int, float]:
        """Rates by loan duration in years (a fresh dict on every access)."""
        return {15: self.taux_15_pct, 20: self.taux_20_pct, 25: self.taux_25_pct}


@dataclass(frozen=True)
class OperatingDefaults:
    """Operating cost defaults."""

    cfe_par_bien_ann: float = 300.0  # per furnished property; 0 in year 1 and when receipts <= 5,000 (TAX_SPEC 5)
    gestion_pct: float = 5.0
    vacance_pct: float = 3.0
    provision_pct: float = 5.0


@dataclass(frozen=True)
class TaxDefaults:
    """Tax and exit defaults."""

    regime_fiscal: str = "lmnp"
    tmi_pct: float = 30.0
    micro_bic_abatt_pct: float = 50.0
    frais_vente_pct: float = 6.0
    # Land share of the purchase price (not amortizable). TAX_SPEC 4.1: 15 % by default, never 0.
    ratio_terrain_pct: float = 15.0
    # Works are their own amortization component (TAX_SPEC 4.1, 15 years, range 10-15).
    amort_works_years: int = 15
    # Number of co-owners selling each property (1 or 2). Splits the gain for the surtax threshold only
    # (CGI 1609 nonies G, TAX_SPEC 3.5 / 11b).
    nb_owners: int = 1


@dataclass(frozen=True)
class SearchDefaults:
    """Objective / search defaults."""

    apport: int = 50000
    cf_cible: float = 0.0
    tolerance: int = 100
    horizon_ans: int = 25
    qualite_weight: float = 0.25
    priorite_pct: int = 25  # qualite_weight as shown on the sidebar slider (0-100)
    max_properties_classique: int = 3
    max_properties_empire: int = 5
    top_n: int = 50


MARKET = MarketDefaults()
FINANCING = FinancingDefaults()
OPERATING = OperatingDefaults()
TAX = TaxDefaults()
SEARCH = SearchDefaults()


# --------------------------------------------------------------------------- #
# Scoring presets
# --------------------------------------------------------------------------- #
class PresetId(str, Enum):
    """Identifier of a financial scoring profile."""

    BALANCED = "balanced"
    CASHFLOW = "cashflow"
    IRR = "irr"
    SECURITY = "security"
    WEALTH = "wealth"

    @property
    def label(self) -> str:
        return PRESET_LABELS[self]

    @classmethod
    def parse(cls, value: PresetId | str | None) -> PresetId:
        """Resolve an id, an enum name or a French UI label. Unknown values mean BALANCED."""
        if isinstance(value, cls):
            return value
        if value:
            low = str(value).strip().lower()
            for pid in cls:
                if low in (pid.value, pid.name.lower(), pid.label.lower()):
                    return pid
            # Short labels such as "Équilibré" (without the "(défaut)" suffix).
            for pid in cls:
                if pid.label.lower().startswith(low):
                    return pid
        return cls.BALANCED


PRESET_LABELS: dict[PresetId, str] = {
    PresetId.BALANCED: "Équilibré (défaut)",
    PresetId.CASHFLOW: "Cash-flow d'abord",
    PresetId.IRR: "Rendement / IRR",
    PresetId.SECURITY: "Sécurité (DSCR)",
    PresetId.WEALTH: "Patrimoine LT",
}
DEFAULT_PRESET = PresetId.BALANCED


@dataclass(frozen=True)
class ScoringWeights:
    """Weights of the financial score components. They sum to 1."""

    enrich_net: float
    irr: float
    cap_eff: float
    dscr: float
    cf_proximity: float

    def as_dict(self) -> dict[str, float]:
        return {
            "enrich_net": self.enrich_net,
            "irr": self.irr,
            "cap_eff": self.cap_eff,
            "dscr": self.dscr,
            "cf_proximity": self.cf_proximity,
        }


PRESETS: dict[PresetId, ScoringWeights] = {
    PresetId.BALANCED: ScoringWeights(enrich_net=0.30, irr=0.25, cap_eff=0.20, dscr=0.15, cf_proximity=0.10),
    PresetId.CASHFLOW: ScoringWeights(enrich_net=0.10, irr=0.10, cap_eff=0.10, dscr=0.20, cf_proximity=0.50),
    PresetId.IRR: ScoringWeights(enrich_net=0.15, irr=0.50, cap_eff=0.30, dscr=0.05, cf_proximity=0.00),
    PresetId.SECURITY: ScoringWeights(enrich_net=0.05, irr=0.15, cap_eff=0.10, dscr=0.50, cf_proximity=0.20),
    PresetId.WEALTH: ScoringWeights(enrich_net=0.60, irr=0.15, cap_eff=0.20, dscr=0.03, cf_proximity=0.02),
}

BASE_WEIGHTS: dict[str, float] = PRESETS[DEFAULT_PRESET].as_dict()


def preset_weights(preset: PresetId | str | None) -> dict[str, float]:
    """Return a fresh weights dict for a preset (id or label)."""
    return PRESETS[PresetId.parse(preset)].as_dict()


# --------------------------------------------------------------------------- #
# Environment (read once, at import)
# --------------------------------------------------------------------------- #
# Disk side effects are opt-in.
AUTOSAVE_ENABLED: bool = os.environ.get("APP_AUTOSAVE", "0") == "1"
RESULTS_DIR: str = os.environ.get("APP_RESULTS_DIR", "results")
# Fraction of the available apport the allocator may exceed (see PortfolioAllocator).
MAX_EXTRA_APPORT_PCT: float = float(os.environ.get("MAX_EXTRA_APPORT_PCT", "0.95"))
