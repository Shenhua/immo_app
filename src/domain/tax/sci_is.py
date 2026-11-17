"""SCI at IS (TAX_SPEC 4.6; CGI 209 I, 219 I b).

Every property sits in one company. The yearly result is taxed at IS (15 % up to 42,500 EUR, then
25 %), with amortization of building + works + furniture that may create a loss (no 39 C II limit).
Losses carry forward with no time limit. No IR and no PS on the yearly result.

Exit: professional gain on the net book value, taxed with the year-H result. Profits are
capitalised and distributed once, at liquidation: PFU 31.4 % on the boni (TAX_SPEC 9.1, 11b).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.defaults import TAX_RATES, TaxRates
from src.domain.tax.base import ExitContext, ExitTax, RegimeId, TaxResult, TaxState, YearContext


@dataclass
class SciIs:
    distribute_at_exit: bool = True
    rates: TaxRates = TAX_RATES
    regime_id: RegimeId = RegimeId.SCI_IS

    def is_tax(self, base: float) -> float:
        """IS on a taxable base (CGI 219 I b; PME conditions assumed, TAX_SPEC 11.12)."""
        base = max(0.0, base)
        r = self.rates
        return r.is_reduced_pct / 100.0 * min(base, r.is_reduced_limit) + r.is_normal_pct / 100.0 * max(
            0.0, base - r.is_reduced_limit
        )

    def compute_year(self, year: int, ctxs: list[YearContext], state: TaxState) -> TaxResult:
        res = 0.0
        booked = 0.0
        for c in ctxs:
            p = state.prop(c.property_id)
            p.cum_amort_imm += c.amort_building + c.amort_works
            p.cum_amort_mob += c.amort_furniture
            amort = c.amort_building + c.amort_works + c.amort_furniture
            booked += amort
            res += c.result_before_amortization - amort
        if res < 0:
            state.is_losses.add(year, -res)
            base = 0.0
        else:
            base = res - state.is_losses.use(res)
        state.last_taxable_base = base
        return TaxResult(
            taxable_base=base,
            is_=self.is_tax(base),
            amortization_booked=booked,
            detail={
                "accounting_result": res,
                "amortization_booked": booked,
                "is_loss_stock_end": state.is_losses.total,
            },
        )

    def compute_exit(self, ctx: ExitContext, state: TaxState) -> ExitTax:
        cost = sum(p.price + p.notary_fees + p.works + p.furniture for p in ctx.properties)
        cum_amort = sum(state.prop(p.property_id).cum_amort for p in ctx.properties)
        vnc = cost - cum_amort
        sale_price = sum(p.sale_price for p in ctx.properties)
        sale_fees = sum(p.sale_fees for p in ctx.properties)
        pv = sale_price - sale_fees - vnc  # professional gain: no allowance, no surtax
        # The gain falls in the year-H exercise: remaining losses offset it; a capital loss reduces
        # the year-H base (floored at 0, the company is liquidated).
        pv_taxable = pv - state.is_losses.use(pv) if pv > 0 else pv
        base_h = state.last_taxable_base
        is_exit = self.is_tax(base_h + pv_taxable) - self.is_tax(base_h)
        # Boni de liquidation = what the associates get back above their contribution (TAX_SPEC 4.6).
        proceeds_before_pfu = sale_price - sale_fees - ctx.debt_repaid - is_exit
        boni = proceeds_before_pfu + ctx.cum_cash_flow - ctx.apport
        pfu_rate = (self.rates.pfu_ir_pct + self.rates.pfu_ps_pct) / 100.0
        pfu = max(0.0, boni) * pfu_rate if self.distribute_at_exit else 0.0
        return ExitTax(
            is_=is_exit,
            pfu=pfu,
            details={
                "sale_price": sale_price,
                "sale_fees": sale_fees,
                "net_book_value": vnc,
                "cumulative_amortization": cum_amort,
                "plus_value_pro": pv,
                "plus_value_taxable_after_losses": pv_taxable,
                "is_on_exit": is_exit,
                "boni_de_liquidation": boni,
                "pfu_on_distribution": pfu,
                "total_tax": is_exit + pfu,
            },
        )
