"""Foyer-level regime choice and aggregation (TAX_SPEC 8).

The law applies **one regime per income category for the whole foyer**, not per property:
- furnished properties (``meuble=True``): micro-BIC or LMNP réel;
- unfurnished properties: micro-foncier or foncier réel;
- the global "SCI IS" option puts every property in one SCI at IS and overrides both.

Micro ceilings are checked on the category's total receipts. If the receipts exceed the ceiling
in any year of the horizon, the category falls back to its régime réel for the whole horizon,
with a warning (TAX_SPEC 4.3, 4.4: one regime for the whole horizon; the one-year tolerance
after a first overshoot is not modelled). Deficits and the 10,700 EUR cap apply at foyer level;
plus-values are always per property (TAX_SPEC 1.10).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.defaults import TAX_RATES, TaxRates
from src.domain.tax.base import (
    ExitContext,
    ExitTax,
    PropertyExit,
    RegimeId,
    TaxRegime,
    TaxResult,
    TaxState,
    YearContext,
    merge_details,
)
from src.domain.tax.individuals import FoncierReel, LmnpReel, MicroBic, MicroFoncier
from src.domain.tax.plus_value import sum_breakdowns
from src.domain.tax.sci_is import SciIs

FURNISHED, UNFURNISHED, SCI = "furnished", "unfurnished", "sci"

# Legacy / UI spellings accepted for each category.
_FURNISHED_ALIASES = {
    "lmnp": RegimeId.LMNP_REEL, "lmnp_reel": RegimeId.LMNP_REEL, "reel": RegimeId.LMNP_REEL,
    "microbic": RegimeId.MICRO_BIC, "micro_bic": RegimeId.MICRO_BIC, "micro": RegimeId.MICRO_BIC,
}
_UNFURNISHED_ALIASES = {
    "reel": RegimeId.FONCIER_REEL, "foncier_reel": RegimeId.FONCIER_REEL,
    "micro": RegimeId.MICRO_FONCIER, "micro_foncier": RegimeId.MICRO_FONCIER,
}


def parse_furnished_regime(value: str) -> RegimeId:
    try:
        return _FURNISHED_ALIASES[str(value).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown furnished regime {value!r} (expected lmnp or microbic)") from None


def parse_unfurnished_regime(value: str) -> RegimeId:
    try:
        return _UNFURNISHED_ALIASES[str(value).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown unfurnished regime {value!r} (expected reel or micro)") from None


@dataclass(frozen=True)
class TaxConfig:
    """User's tax choices (one regime per category, TAX_SPEC 8)."""

    tmi_pct: float = 30.0
    furnished_regime: RegimeId = RegimeId.LMNP_REEL
    unfurnished_regime: RegimeId = RegimeId.FONCIER_REEL
    sci_is: bool = False
    micro_bic_abattement_pct: float = 50.0
    amortize_furniture: bool = True
    sci_distribute_at_exit: bool = True
    nb_owners: int = 1
    rates: TaxRates = TAX_RATES


@dataclass
class TaxHousehold:
    """Tax of the whole foyer: one regime and one carryforward state per income category."""

    config: TaxConfig
    furnished: list[bool]  # per property, in simulation order
    regimes: dict[str, TaxRegime] = field(default_factory=dict)
    states: dict[str, TaxState] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        config: TaxConfig,
        furnished: list[bool],
        max_receipts: dict[str, float] | None = None,
    ) -> TaxHousehold:
        """Pick the regime of each category.

        ``max_receipts`` maps ``"furnished"`` / ``"unfurnished"`` to the highest yearly total of
        the category's receipts over the horizon (used for the micro ceilings and the LMP flag).
        """
        hh = cls(config=config, furnished=list(furnished))
        receipts = max_receipts or {}
        c, r = config, config.rates
        if c.sci_is:
            hh.regimes[SCI] = SciIs(distribute_at_exit=c.sci_distribute_at_exit, rates=r)
        else:
            if any(furnished):
                hh.regimes[FURNISHED] = hh._furnished_regime(receipts.get(FURNISHED, 0.0))
            if not all(furnished):
                hh.regimes[UNFURNISHED] = hh._unfurnished_regime(receipts.get(UNFURNISHED, 0.0))
        hh.states = {cat: TaxState() for cat in hh.regimes}
        return hh

    def _furnished_regime(self, receipts: float) -> TaxRegime:
        c, r = self.config, self.config.rates
        if receipts > r.lmp_receipts_threshold:
            # CGI 155 IV: flag only (TAX_SPEC 4.2, 9.2).
            self.warnings.append(
                f"Recettes meublées jusqu'à {receipts:,.0f} € > {r.lmp_receipts_threshold:,.0f} € : statut LMP "
                "possible (cotisations sociales et plus-value professionnelle non modélisées)."
            )
        if c.furnished_regime == RegimeId.MICRO_BIC:
            micro = MicroBic(tmi_pct=c.tmi_pct, abattement_pct=float(c.micro_bic_abattement_pct), rates=r)
            if receipts <= micro.ceiling:
                return micro
            self.warnings.append(
                f"Micro-BIC impossible : recettes meublées jusqu'à {receipts:,.0f} € > plafond "
                f"{micro.ceiling:,.0f} € (CGI 50-0). LMNP réel appliqué."
            )
        return LmnpReel(tmi_pct=c.tmi_pct, amortize_furniture=c.amortize_furniture, rates=r)

    def _unfurnished_regime(self, receipts: float) -> TaxRegime:
        c, r = self.config, self.config.rates
        if c.unfurnished_regime == RegimeId.MICRO_FONCIER:
            micro = MicroFoncier(tmi_pct=c.tmi_pct, rates=r)
            if receipts <= micro.ceiling:
                return micro
            self.warnings.append(
                f"Micro-foncier impossible : loyers nus jusqu'à {receipts:,.0f} € > plafond "
                f"{micro.ceiling:,.0f} € (CGI 32). Régime réel appliqué."
            )
        return FoncierReel(tmi_pct=c.tmi_pct, rates=r)

    # ------------------------------------------------------------------ #
    def category_of(self, property_id: int) -> str:
        if SCI in self.regimes:
            return SCI
        return FURNISHED if self.furnished[property_id] else UNFURNISHED

    def regime_of(self, property_id: int) -> RegimeId:
        return self.regimes[self.category_of(property_id)].regime_id

    @property
    def regime_ids(self) -> dict[str, str]:
        return {cat: reg.regime_id.value for cat, reg in self.regimes.items()}

    def amortizes(self, property_id: int) -> bool:
        """True if the property's regime amortizes (LMNP réel and SCI IS only, TAX_SPEC 4.1)."""
        return self.regime_of(property_id) in (RegimeId.LMNP_REEL, RegimeId.SCI_IS)

    def state(self, category: str) -> TaxState:
        return self.states[category]

    def compute_year(self, year: int, ctxs: list[YearContext]) -> TaxResult:
        """Tax of the foyer for one year. Categories are independent and summed."""
        results: list[TaxResult] = []
        for cat, regime in self.regimes.items():
            sub = [c for c in ctxs if self.category_of(c.property_id) == cat]
            results.append(regime.compute_year(year, sub, self.states[cat]))
        return TaxResult(
            taxable_base=sum(r.taxable_base for r in results),
            ir=sum(r.ir for r in results),
            ps=sum(r.ps for r in results),
            is_=sum(r.is_ for r in results),
            amortization_booked=sum(r.amortization_booked for r in results),
            detail=merge_details(*(r.detail for r in results)),
        )

    def compute_exit(
        self,
        years_held: int,
        properties: list[PropertyExit],
        *,
        debt_repaid: float = 0.0,
        cum_cash_flow: float = 0.0,
        apport: float = 0.0,
    ) -> ExitTax:
        """Exit tax of the foyer. Foyer-level amounts only matter for the SCI's boni de liquidation."""
        exits: list[ExitTax] = []
        for cat, regime in self.regimes.items():
            sub = tuple(p for p in properties if self.category_of(p.property_id) == cat)
            ctx = ExitContext(
                years_held=years_held,
                properties=sub,
                nb_owners=self.config.nb_owners,
                tmi_pct=self.config.tmi_pct,
                debt_repaid=debt_repaid,
                cum_cash_flow=cum_cash_flow,
                apport=apport,
            )
            exits.append(regime.compute_exit(ctx, self.states[cat]))
        per_property = tuple(sorted((it for e in exits for it in e.per_property), key=lambda it: it[0]))
        recapture = sum(e.deficit_recapture for e in exits)
        if SCI in self.regimes:
            details = dict(exits[0].details)
        else:
            details = sum_breakdowns([b for _, _, b in per_property]).to_dict()
            details["deficit_recapture"] = recapture
            details["total_tax"] = details["total_tax"] + recapture
        return ExitTax(
            ir=sum(e.ir for e in exits),
            ps=sum(e.ps for e in exits),
            surtax=sum(e.surtax for e in exits),
            is_=sum(e.is_ for e in exits),
            pfu=sum(e.pfu for e in exits),
            deficit_recapture=recapture,
            per_property=per_property,
            details=details,
        )
