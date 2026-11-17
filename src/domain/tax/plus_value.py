"""Plus-value of individuals on the sale of a rental property (TAX_SPEC section 3).

Pure functions, no I/O. All amounts in euros, percentages in percent.
Applies to LMNP reel, micro-BIC, micro-foncier and foncier reel (not to SCI at IS).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.core.defaults import TAX_RATES

# Rates come from TAX_RATES: ir_plus_value_pct (CGI 200 B) and ps_plus_value_pct
# (CSS L136-7 I 2, CSG kept at 9.2 %, LFSS 2026 art. 12).
# BOI-RFPI-PVI-20-10-20-20: flat rates on the acquisition price.
FORFAIT_FRAIS_ACQ_PCT = 7.5  # acquisition costs (CGI 150 VB II 1)
FORFAIT_TRAVAUX_PCT = 15.0  # works, sale more than 5 years after purchase (CGI 150 VB II 4)
SURTAX_THRESHOLD = 50_000.0  # CGI 1609 nonies G


def abattement_ir_pct(years_held: int) -> float:
    """IR allowance (CGI 150 VC I): 6 %/yr for years 6-21 (96 % at 21), 100 % from year 22."""
    if years_held <= 5:
        return 0.0
    if years_held >= 22:
        return 100.0
    return 6.0 * (years_held - 5)


def abattement_ps_pct(years_held: int) -> float:
    """PS allowance (CGI 150 VC I): 1.65 %/yr years 6-21, 28 % at 22, +9 %/yr years 23-30 (100 % at 30)."""
    if years_held <= 5:
        return 0.0
    if years_held <= 21:
        return 1.65 * (years_held - 5)
    if years_held >= 30:
        return 100.0
    return 28.0 + 9.0 * (years_held - 22)


def surtax(base_ir: float) -> float:
    """Surtax on high plus-values (CGI 1609 nonies G; BOI-RFPI-TPVIE-20), for one seller and one sale.

    ``base_ir`` is the IR-taxable gain after the IR allowance.
    """
    pv = base_ir
    if pv <= SURTAX_THRESHOLD:
        return 0.0
    if pv <= 60_000:
        return 0.02 * pv - (60_000 - pv) / 20
    if pv <= 100_000:
        return 0.02 * pv
    if pv <= 110_000:
        return 0.03 * pv - (110_000 - pv) / 10
    if pv <= 150_000:
        return 0.03 * pv
    if pv <= 160_000:
        return 0.04 * pv - (160_000 - pv) * 15 / 100
    if pv <= 200_000:
        return 0.04 * pv
    if pv <= 210_000:
        return 0.05 * pv - (210_000 - pv) * 20 / 100
    if pv <= 250_000:
        return 0.05 * pv
    if pv <= 260_000:
        return 0.06 * pv - (260_000 - pv) * 25 / 100
    return 0.06 * pv


@dataclass(frozen=True)
class PlusValueBreakdown:
    """Plus-value computation for one property (or the sum over several)."""

    sale_price: float = 0.0
    sale_fees: float = 0.0
    net_sale_price: float = 0.0
    acquisition_fees_retained: float = 0.0
    works_majoration: float = 0.0
    amortization_addback: float = 0.0
    corrected_acquisition_price: float = 0.0
    plus_value_brute: float = 0.0
    abattement_ir_pct: float = 0.0
    abattement_ps_pct: float = 0.0
    base_ir: float = 0.0
    base_ps: float = 0.0
    ir_pv: float = 0.0
    ps_pv: float = 0.0
    surtax: float = 0.0

    @property
    def total_tax(self) -> float:
        return self.ir_pv + self.ps_pv + self.surtax

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["total_tax"] = self.total_tax
        return d


def compute_plus_value(
    *,
    price: float,
    notary_fees: float,
    works: float,
    sale_price: float,
    sale_fees: float,
    years_held: int,
    amortization_deducted_imm: float = 0.0,
    amortization_addback_applies: bool = False,
    works_already_deducted: bool = False,
    nb_owners: int = 1,
) -> PlusValueBreakdown:
    """Plus-value of one property (TAX_SPEC section 3), floored at 0, no netting across properties.

    ``amortization_deducted_imm`` is the building/works amortization actually deducted (LMNP reel);
    it is added back (CGI 150 VB III, loi 2025-127 art. 84, sales from 15/02/2025) only if
    ``amortization_addback_applies`` (LMNP reel and not a residence-services).
    ``works_already_deducted`` (foncier reel) excludes real works from the majoration.
    The surtax threshold is applied per owner: the gain is split by ``nb_owners``.
    """
    owners = max(1, int(nb_owners))
    prix_cession = sale_price - sale_fees  # CGI 150 VA: seller's costs reduce the price
    acq_fees = max(notary_fees, FORFAIT_FRAIS_ACQ_PCT / 100.0 * price)
    eligible_works = 0.0 if works_already_deducted else works
    works_maj = max(eligible_works, FORFAIT_TRAVAUX_PCT / 100.0 * price) if years_held > 5 else eligible_works
    addback = amortization_deducted_imm if amortization_addback_applies else 0.0
    prix_acq = price + acq_fees + works_maj - addback
    pv = max(0.0, prix_cession - prix_acq)

    a_ir, a_ps = abattement_ir_pct(years_held), abattement_ps_pct(years_held)
    base_ir = pv * (1.0 - a_ir / 100.0)
    base_ps = pv * (1.0 - a_ps / 100.0)
    return PlusValueBreakdown(
        sale_price=sale_price,
        sale_fees=sale_fees,
        net_sale_price=prix_cession,
        acquisition_fees_retained=acq_fees,
        works_majoration=works_maj,
        amortization_addback=addback,
        corrected_acquisition_price=prix_acq,
        plus_value_brute=pv,
        abattement_ir_pct=a_ir,
        abattement_ps_pct=a_ps,
        base_ir=base_ir,
        base_ps=base_ps,
        ir_pv=base_ir * TAX_RATES.ir_plus_value_pct / 100.0,
        ps_pv=base_ps * TAX_RATES.ps_plus_value_pct / 100.0,
        surtax=owners * surtax(base_ir / owners),
    )


def sum_breakdowns(items: list[PlusValueBreakdown]) -> PlusValueBreakdown:
    """Total over several properties (each already floored at 0). Allowances are gain-weighted averages."""
    if not items:
        return PlusValueBreakdown()
    total_pv = sum(i.plus_value_brute for i in items)

    def wavg(attr: str) -> float:
        if total_pv <= 0:
            return 0.0
        return float(sum(getattr(i, attr) * i.plus_value_brute for i in items)) / total_pv

    fields = (
        "sale_price",
        "sale_fees",
        "net_sale_price",
        "acquisition_fees_retained",
        "works_majoration",
        "amortization_addback",
        "corrected_acquisition_price",
        "plus_value_brute",
        "base_ir",
        "base_ps",
        "ir_pv",
        "ps_pv",
        "surtax",
    )
    kwargs = {f: sum(getattr(i, f) for i in items) for f in fields}
    return PlusValueBreakdown(
        **kwargs,
        abattement_ir_pct=wavg("abattement_ir_pct"),
        abattement_ps_pct=wavg("abattement_ps_pct"),
    )
