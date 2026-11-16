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
    # T43: loan durations the user may pick (whole years) and the most that can be searched at once.
    duree_min_ans: int = 10
    duree_max_ans: int = 25
    max_durations: int = 5
    durees_defaut: tuple[int, ...] = (15, 20, 25)

    @property
    def taux_credits(self) -> dict[int, float]:
        """Rates by loan duration in years (a fresh dict on every access)."""
        return {15: self.taux_15_pct, 20: self.taux_20_pct, 25: self.taux_25_pct}

    def default_rate(self, years: int) -> float:
        """Suggested rate (%) for a loan duration: the 15/20/25-year rates, extended linearly in between and outside."""
        pts = sorted(self.taux_credits.items())
        if years <= pts[0][0]:
            (x0, y0), (x1, y1) = pts[0], pts[1]
        elif years >= pts[-1][0]:
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
        else:
            x0, y0 = max(p for p in pts if p[0] <= years)
            x1, y1 = min(p for p in pts if p[0] >= years)
            if x0 == x1:
                return y0
        return round(y0 + (y1 - y0) * (years - x0) / (x1 - x0), 2)


@dataclass(frozen=True)
class OperatingDefaults:
    """Operating cost defaults."""

    cfe_par_bien_ann: float = 300.0  # per furnished property; 0 in year 1 and when receipts <= 5,000 (TAX_SPEC 5)
    gestion_pct: float = 7.0  # T26a: realistic agency fee (was 5)
    vacance_pct: float = 3.0
    provision_pct: float = 5.0


@dataclass(frozen=True)
class TaxDefaults:
    """Tax and exit defaults."""

    # Regime per income category for the whole foyer (TAX_SPEC 8): furnished properties use
    # ``regime_fiscal`` ("lmnp" = LMNP reel | "microbic"), unfurnished ones use ``regime_foncier``
    # ("reel" | "micro"). ``sci_is`` puts every property in one SCI at IS and overrides both.
    regime_fiscal: str = "lmnp"
    regime_foncier: str = "reel"
    sci_is: bool = False
    # SCI IS: profits are capitalised, PFU once on the liquidation boni at exit (TAX_SPEC 4.6, 11b).
    sci_distribute_at_exit: bool = True
    tmi_pct: float = 30.0
    micro_bic_abatt_pct: float = 50.0
    frais_vente_pct: float = 6.0
    # Land share of the purchase price (not amortizable). TAX_SPEC 4.1: 15 % by default, never 0.
    ratio_terrain_pct: float = 15.0
    # Works are their own amortization component (TAX_SPEC 4.1, 15 years, range 10-15).
    amort_works_years: int = 15
    # Building (30 y) and furniture (10 y) amortization durations (TAX_SPEC 4.1).
    amort_immo_years: int = 30
    amort_mobilier_years: int = 10
    # Number of co-owners selling each property (1 or 2). Splits the gain for the surtax threshold only
    # (CGI 1609 nonies G, TAX_SPEC 3.5 / 11b).
    nb_owners: int = 1


@dataclass(frozen=True)
class SearchDefaults:
    """Objective / search defaults."""

    # T43: tuned so that a first run returns >= 5 strategies (see tests/unit/test_validation.py).
    # With 50 000 EUR and a 0 EUR/month floor, only 1 strategy survived: the CF floor was the binding
    # constraint (a leveraged portfolio is rarely cash-flow positive in year 2), then the down payment.
    apport: int = 80000
    cf_cible: float = -100.0
    tolerance: int = 100
    horizon_ans: int = 25
    qualite_weight: float = 0.25
    priorite_pct: int = 25  # qualite_weight as shown on the sidebar slider (0-100)
    max_properties_classique: int = 3
    max_properties_empire: int = 5
    max_properties_min: int = 1
    max_properties_max: int = 5
    top_n: int = 50


@dataclass(frozen=True)
class TaxRates:
    """2026 tax rates and thresholds, frozen for the whole horizon (TAX_SPEC 1.2, 2). Sources in TAX_SPEC 12."""

    # Prélèvements sociaux per income category (LFSS 2026, loi 2025-1403 art. 12; TAX_SPEC 2, 11.1).
    ps_bic_pct: float = 18.6  # LMNP / micro-BIC income (CSS L136-6, CSG 10.6 % from 2025 income)
    ps_foncier_pct: float = 17.2  # revenus fonciers (CSG kept at 9.2 %, L136-8 IV)
    ir_plus_value_pct: float = 19.0  # IR on real-estate plus-values of individuals (CGI 200 B)
    ps_plus_value_pct: float = 17.2  # real-estate plus-values of individuals (L136-7 I 2°)
    # PFU on dividends / boni de liquidation: 12.8 % IR (CGI 200 A) + 18.6 % PS (L136-7, from 2026).
    pfu_ir_pct: float = 12.8
    pfu_ps_pct: float = 18.6
    # Micro-BIC (CGI 50-0; loi Le Meur 2024-1039; service-public F32744).
    micro_bic_ceiling: float = 83_600.0  # long-term or classified tourist letting, 2026 income [TAX_SPEC 11.7]
    micro_bic_ceiling_non_classe: float = 15_000.0  # non-classified tourist letting (30 % allowance)
    micro_bic_min_abattement: float = 305.0
    # Micro-foncier (CGI 32; service-public F1991).
    micro_foncier_abattement_pct: float = 30.0
    micro_foncier_ceiling: float = 15_000.0
    # Déficit foncier on global income (CGI 156 I 3°; F1991).
    deficit_foncier_global_cap: float = 10_700.0
    deficit_foncier_energy_extra_cap: float = 10_700.0  # energy works paid up to 31/12/2027 [TAX_SPEC 11.6]
    deficit_foncier_energy_last_year: int = 2027
    deficit_recapture_years: int = 3  # letting must go on until 31/12 of N + 3 (CGI 156 I 3°)
    # BIC non-pro (CGI 156 I 1° ter) and foncier deficits (CGI 156 I 3°) carry forward 10 years.
    deficit_carry_years: int = 10
    # IS (CGI 219 I b; PME conditions assumed) [TAX_SPEC 11.12].
    is_reduced_pct: float = 15.0
    is_reduced_limit: float = 42_500.0
    is_normal_pct: float = 25.0
    # LMP status warning (CGI 155 IV): furnished receipts above this amount.
    lmp_receipts_threshold: float = 23_000.0
    # Simulation year 1 is calendar year 2026 (TAX_SPEC 4.5).
    first_calendar_year: int = 2026


MARKET = MarketDefaults()
FINANCING = FinancingDefaults()
OPERATING = OperatingDefaults()
TAX = TaxDefaults()
TAX_RATES = TaxRates()
SEARCH = SearchDefaults()


# --------------------------------------------------------------------------- #
# T26a: realistic assumptions (separate block)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RealismDefaults:
    """Costs a naive model forgets: capex provision, unpaid rent insurance, works vacancy, TF growth."""

    maintenance_m2_an: float = 10.0  # EUR/m2/year, provision for repairs and capex (cash outflow, deductible in reel)
    gli_pct: float = 3.0  # % of collected rent: unpaid-rent insurance (GLI) / unpaid rent
    mois_travaux: int = 3  # months without rent in year 1 when the property has works
    tf_growth_pct: float = 3.0  # %/year taxe fonciere growth (separate from charges inflation)


REALISM = RealismDefaults()


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
# Scoring benchmarks (T25)
# --------------------------------------------------------------------------- #
# Capital efficiency (net enrichment / total down payment) scored on a fixed 0..5x scale.
CAP_EFF_BENCHMARK_MAX: float = 5.0


# --------------------------------------------------------------------------- #
# Environment (read once, at import)
# --------------------------------------------------------------------------- #
# Disk side effects are opt-in.
AUTOSAVE_ENABLED: bool = os.environ.get("APP_AUTOSAVE", "0") == "1"
RESULTS_DIR: str = os.environ.get("APP_RESULTS_DIR", "results")
# Fraction of the available apport the allocator may exceed (see PortfolioAllocator).
MAX_EXTRA_APPORT_PCT: float = float(os.environ.get("MAX_EXTRA_APPORT_PCT", "0.95"))
