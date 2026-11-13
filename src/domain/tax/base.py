"""Tax regime engine: shared types (TAX_SPEC 7).

A regime computes the tax of **one income category of the foyer** (furnished, unfurnished, or the
SCI) for one year, from the list of that category's properties. A single-property run is a list of
one. Carryforwards live in an explicit :class:`TaxState`, one per category, owned by the caller
(:class:`src.domain.tax.household.TaxHousehold`).

Pure domain code: no I/O, no UI import. Amounts in euros, rates in percent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from src.domain.tax.plus_value import PlusValueBreakdown


class RegimeId(str, Enum):
    """Tax regimes of TAX_SPEC 4. Values match the golden cases' ``regime`` field."""

    LMNP_REEL = "lmnp_reel"
    MICRO_BIC = "micro_bic"
    MICRO_FONCIER = "micro_foncier"
    FONCIER_REEL = "foncier_reel"
    SCI_IS = "sci_is"


# --------------------------------------------------------------------------- #
# Carryforward state
# --------------------------------------------------------------------------- #
@dataclass
class DeficitPool:
    """Deficits carried forward FIFO, each expiring ``life_years`` after the year it was created.

    ``life_years=None`` means no time limit (IS losses, CGI 209 I).
    A deficit created in year N can be used up to year N + life_years included.
    """

    life_years: int | None = 10
    items: list[list[float]] = field(default_factory=list)  # [year_created, amount]

    def add(self, year: int, amount: float) -> None:
        if amount > 0:
            self.items.append([year, amount])

    def expire(self, year: int) -> float:
        """Drop the deficits that can no longer be used in ``year``. Returns the amount lost."""
        if self.life_years is None:
            return 0.0
        life = self.life_years
        lost = sum(a for y0, a in self.items if year > y0 + life)
        self.items = [it for it in self.items if year <= it[0] + life]
        return lost

    def use(self, income: float) -> float:
        """Offset ``income`` with the oldest deficits first. Returns the amount used."""
        if income <= 0:
            return 0.0
        used = 0.0
        for it in self.items:
            take = min(it[1], income - used)
            it[1] -= take
            used += take
            if used >= income:
                break
        self.items = [it for it in self.items if it[1] > 1e-9]
        return used

    @property
    def total(self) -> float:
        return float(sum(a for _, a in self.items))


@dataclass
class PropertyTaxState:
    """Per-property amortization state (LMNP réel and SCI IS)."""

    # Deferred amortization (ARD, CGI 39 C II), no time limit. Building + works vs furniture pools.
    ard_imm: float = 0.0
    ard_mob: float = 0.0
    # Amortization actually deducted (LMNP réel) or booked (SCI IS), building + works vs furniture.
    cum_amort_imm: float = 0.0
    cum_amort_mob: float = 0.0

    @property
    def ard(self) -> float:
        return self.ard_imm + self.ard_mob

    @property
    def cum_amort(self) -> float:
        return self.cum_amort_imm + self.cum_amort_mob


@dataclass
class TaxState:
    """Carryforward state of one income category of the foyer (TAX_SPEC 7)."""

    properties: dict[int, PropertyTaxState] = field(default_factory=dict)
    deficits: DeficitPool = field(default_factory=DeficitPool)  # BIC non-pro or foncier, 10 years
    is_losses: DeficitPool = field(default_factory=lambda: DeficitPool(life_years=None))
    global_imputed_by_year: dict[int, float] = field(default_factory=dict)  # déficit foncier on global income
    last_taxable_base: float = 0.0  # SCI IS: base of the last year, for the exit (TAX_SPEC 4.6)

    def prop(self, property_id: int) -> PropertyTaxState:
        return self.properties.setdefault(property_id, PropertyTaxState())


# --------------------------------------------------------------------------- #
# Inputs and outputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class YearContext:
    """One property, one year. All amounts for the 12 months of the year."""

    year: int
    property_id: int
    rent: float  # receipts of the year (micro ceilings and micro allowances apply to it)
    operating_charges: float  # deductible operating charges: co-ownership, TF, management, provisions, CFE
    interest: float
    loan_insurance: float
    works_paid: float = 0.0  # works paid this year (deductible under foncier réel)
    energy_works_paid: float = 0.0  # part of ``works_paid`` that is energy renovation (TAX_SPEC 4.5)
    amort_building: float = 0.0  # this year's straight-line amortization, 0 once fully amortized
    amort_works: float = 0.0
    amort_furniture: float = 0.0

    @property
    def result_before_amortization(self) -> float:
        return self.rent - self.operating_charges - self.interest - self.loan_insurance


@dataclass(frozen=True)
class TaxResult:
    """Tax of one category for one year. ``ir`` may be negative (déficit foncier on global income)."""

    taxable_base: float = 0.0
    ir: float = 0.0
    ps: float = 0.0
    is_: float = 0.0
    amortization_booked: float = 0.0  # amortization of the year shown in the yearly table
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.ir + self.ps + self.is_


@dataclass(frozen=True)
class PropertyExit:
    """One property sold at the end of the horizon."""

    property_id: int
    name: str
    price: float
    notary_fees: float
    works: float  # budget_travaux + renovation_energetique
    furniture: float
    sale_price: float
    sale_fees: float
    residence_services: bool = False


@dataclass(frozen=True)
class ExitContext:
    """Exit of one category. Foyer-level amounts are only used by SCI IS (boni de liquidation)."""

    years_held: int
    properties: tuple[PropertyExit, ...]
    nb_owners: int = 1
    tmi_pct: float = 30.0
    debt_repaid: float = 0.0  # CRD + IRA of the category's properties
    cum_cash_flow: float = 0.0  # sum of after-tax yearly cash flows of the category
    apport: float = 0.0  # down payment of the category's properties


@dataclass(frozen=True)
class ExitTax:
    """Exit tax of one category. ``total`` is deducted from the sale proceeds."""

    ir: float = 0.0
    ps: float = 0.0
    surtax: float = 0.0
    is_: float = 0.0
    pfu: float = 0.0
    deficit_recapture: float = 0.0
    per_property: tuple[tuple[int, str, PlusValueBreakdown], ...] = ()  # (id, name, PV), individuals only
    details: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.ir + self.ps + self.surtax + self.is_ + self.pfu + self.deficit_recapture


class TaxRegime(Protocol):
    """One tax regime, applied to the properties of one income category of the foyer."""

    regime_id: RegimeId

    def compute_year(self, year: int, ctxs: list[YearContext], state: TaxState) -> TaxResult: ...

    def compute_exit(self, ctx: ExitContext, state: TaxState) -> ExitTax: ...


def merge_details(*dicts: dict[str, Any]) -> dict[str, float]:
    """Sum detail dicts key by key."""
    out: dict[str, float] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + float(v)
    return out
