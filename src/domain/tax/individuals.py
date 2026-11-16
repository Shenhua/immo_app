"""Regimes of individuals: LMNP réel, micro-BIC, micro-foncier, foncier réel (TAX_SPEC 4.2-4.5).

IR = base x TMI, PS = base x the category's PS rate (TAX_SPEC 1.3, 2). The exit is the plus-value of
individuals, per property, floored at 0, no netting (TAX_SPEC 3, 1.10).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.defaults import TAX_RATES, TaxRates
from src.domain.tax.base import (
    ExitContext,
    ExitTax,
    RegimeId,
    TaxResult,
    TaxState,
    YearContext,
)
from src.domain.tax.plus_value import PlusValueBreakdown, compute_plus_value, sum_breakdowns


def _plus_value_exit(
    ctx: ExitContext,
    *,
    addback_by_property: dict[int, float] | None = None,
    works_already_deducted: bool = False,
    deficit_recapture: float = 0.0,
) -> ExitTax:
    """Plus-value of individuals for every property of the category (TAX_SPEC 3)."""
    addbacks = addback_by_property or {}
    items: list[tuple[int, str, PlusValueBreakdown]] = []
    for p in ctx.properties:
        addback = addbacks.get(p.property_id, 0.0)
        items.append(
            (
                p.property_id,
                p.name,
                compute_plus_value(
                    price=p.price,
                    notary_fees=p.notary_fees,
                    works=p.works,
                    sale_price=p.sale_price,
                    sale_fees=p.sale_fees,
                    years_held=ctx.years_held,
                    amortization_deducted_imm=addback,
                    # CGI 150 VB III (loi 2025-127 art. 84): LMNP réel only; residences-services exempt.
                    amortization_addback_applies=addback > 0 and not p.residence_services,
                    works_already_deducted=works_already_deducted,
                    nb_owners=ctx.nb_owners,
                ),
            )
        )
    total = sum_breakdowns([b for _, _, b in items])
    details = total.to_dict()
    details["deficit_recapture"] = deficit_recapture
    return ExitTax(
        ir=total.ir_pv,
        ps=total.ps_pv,
        surtax=total.surtax,
        deficit_recapture=deficit_recapture,
        per_property=tuple(items),
        details=details,
    )


# --------------------------------------------------------------------------- #
# Furnished letting (BIC non professionnel)
# --------------------------------------------------------------------------- #
@dataclass
class LmnpReel:
    """LMNP au régime réel (TAX_SPEC 4.2; CGI 39 C, 156 I 1° ter).

    Per property: amortization is limited to the property's own result before amortization
    (CGI 39 C II); the excess is deferred (ARD, no time limit) [TAX_SPEC 11.10].
    Foyer: the properties' profits and losses are netted within the year; a net loss is a BIC
    non-pro deficit carried forward 10 years against BIC non-pro income only. Amortization is
    limited first, then prior deficits are used [TAX_SPEC 11.2].
    """

    tmi_pct: float
    ps_pct: float = TAX_RATES.ps_bic_pct
    amortize_furniture: bool = True
    rates: TaxRates = TAX_RATES
    regime_id: RegimeId = RegimeId.LMNP_REEL

    def compute_year(self, year: int, ctxs: list[YearContext], state: TaxState) -> TaxResult:
        state.deficits.life_years = self.rates.deficit_carry_years
        state.deficits.expire(year)
        r_total = cur_total = ded_total = 0.0
        benefits = losses = 0.0
        for c in ctxs:
            ps_ = state.prop(c.property_id)
            r = c.result_before_amortization
            cur_imm = c.amort_building + c.amort_works
            cur_mob = c.amort_furniture if self.amortize_furniture else 0.0
            pool_imm, pool_mob = cur_imm + ps_.ard_imm, cur_mob + ps_.ard_mob
            pool = pool_imm + pool_mob
            if r <= 0:
                losses += -r
                deducted = 0.0
            else:
                deducted = min(r, pool)  # CGI 39 C II: amortization cannot create or increase a deficit
                benefits += r - deducted
            # Deducted amortization split pro rata of the two pools (model convention, TAX_SPEC 11.3).
            share_imm = pool_imm / pool if pool > 0 else 0.0
            ded_imm = deducted * share_imm
            ps_.ard_imm, ps_.ard_mob = pool_imm - ded_imm, pool_mob - (deducted - ded_imm)
            ps_.cum_amort_imm += ded_imm
            ps_.cum_amort_mob += deducted - ded_imm
            r_total += r
            cur_total += cur_imm + cur_mob
            ded_total += deducted

        net = benefits - losses
        if net < 0:
            state.deficits.add(year, -net)
            base = 0.0
        else:
            base = net - state.deficits.use(net)  # prior BIC deficits, oldest first
        return TaxResult(
            taxable_base=base,
            ir=base * self.tmi_pct / 100.0,
            ps=base * self.ps_pct / 100.0,
            amortization_booked=cur_total,
            detail={
                "result_before_amortization": r_total,
                "amortization_current_year": cur_total,
                "amortization_deducted": ded_total,
                "ard_stock_end": sum(state.prop(c.property_id).ard for c in ctxs),
                "bic_deficit_stock_end": state.deficits.total,
            },
        )

    def compute_exit(self, ctx: ExitContext, state: TaxState) -> ExitTax:
        # Add-back of the building + works amortization actually deducted (CGI 150 VB III, TAX_SPEC 3.4).
        addbacks = {p.property_id: state.prop(p.property_id).cum_amort_imm for p in ctx.properties}
        return _plus_value_exit(ctx, addback_by_property=addbacks)


@dataclass
class MicroBic:
    """Micro-BIC (TAX_SPEC 4.3; CGI 50-0, loi Le Meur).

    Allowance on the foyer's total furnished receipts: 50 % (long-term or classified tourist
    letting) or 30 % (non-classified tourist letting), minimum 305 EUR. No charges, no
    amortization, no deficit. Eligibility (ceilings) is checked by the household.
    """

    tmi_pct: float
    abattement_pct: float = 50.0
    ps_pct: float = TAX_RATES.ps_bic_pct
    rates: TaxRates = TAX_RATES
    regime_id: RegimeId = RegimeId.MICRO_BIC

    def __post_init__(self) -> None:
        # The old 71 % rate is gone for 2026 income (TAX_SPEC 4.3).
        if self.abattement_pct not in (30.0, 50.0):
            raise ValueError(f"micro-BIC allowance must be 50 or 30 %, got {self.abattement_pct}")

    @property
    def ceiling(self) -> float:
        if self.abattement_pct == 30.0:
            return self.rates.micro_bic_ceiling_non_classe
        return self.rates.micro_bic_ceiling

    def compute_year(self, year: int, ctxs: list[YearContext], state: TaxState) -> TaxResult:
        rent = sum(c.rent for c in ctxs)
        if rent <= 0:
            base = 0.0
        else:
            abatt = max(self.rates.micro_bic_min_abattement, rent * self.abattement_pct / 100.0)
            base = max(0.0, rent - abatt)
        return TaxResult(
            taxable_base=base,
            ir=base * self.tmi_pct / 100.0,
            ps=base * self.ps_pct / 100.0,
        )

    def compute_exit(self, ctx: ExitContext, state: TaxState) -> ExitTax:
        return _plus_value_exit(ctx)  # no amortization deducted, so no add-back


# --------------------------------------------------------------------------- #
# Unfurnished letting (revenus fonciers)
# --------------------------------------------------------------------------- #
@dataclass
class MicroFoncier:
    """Micro-foncier (TAX_SPEC 4.4; CGI 32): 30 % allowance on gross foncier income, no deductions."""

    tmi_pct: float
    ps_pct: float = TAX_RATES.ps_foncier_pct
    rates: TaxRates = TAX_RATES
    regime_id: RegimeId = RegimeId.MICRO_FONCIER

    @property
    def ceiling(self) -> float:
        return self.rates.micro_foncier_ceiling

    def compute_year(self, year: int, ctxs: list[YearContext], state: TaxState) -> TaxResult:
        rent = sum(c.rent for c in ctxs)
        base = max(0.0, rent * (1.0 - self.rates.micro_foncier_abattement_pct / 100.0))
        return TaxResult(
            taxable_base=base,
            ir=base * self.tmi_pct / 100.0,
            ps=base * self.ps_pct / 100.0,
        )

    def compute_exit(self, ctx: ExitContext, state: TaxState) -> ExitTax:
        return _plus_value_exit(ctx)  # works were not deducted: real works are eligible


@dataclass
class FoncierReel:
    """Revenus fonciers au régime réel, with déficit foncier (TAX_SPEC 4.5; CGI 156 I 3°).

    Computed on the foyer's unfurnished properties together. Works are deducted the year they are
    paid. A net deficit offsets the global income up to 10,700 EUR (plus up to 10,700 EUR for
    energy works paid by 2027), excluding its interest part; the rest is carried forward 10 years
    against revenus fonciers only. The imputation saves IR at the TMI, never PS.
    """

    tmi_pct: float
    ps_pct: float = TAX_RATES.ps_foncier_pct
    rates: TaxRates = TAX_RATES
    regime_id: RegimeId = RegimeId.FONCIER_REEL

    def global_cap(self, year: int, energy_works_paid: float) -> float:
        """Cap on the global-income imputation for simulation ``year`` (TAX_SPEC 4.5)."""
        cap = self.rates.deficit_foncier_global_cap
        calendar_year = self.rates.first_calendar_year + year - 1
        if calendar_year <= self.rates.deficit_foncier_energy_last_year and energy_works_paid > 0:
            # TODO(TAX_SPEC §11.6): the extra cap covers the energy-works share only; the model assumes
            # renovation_energetique lifts the DPE from E/F/G to A-D.
            cap += min(self.rates.deficit_foncier_energy_extra_cap, energy_works_paid)
        return cap

    def compute_year(self, year: int, ctxs: list[YearContext], state: TaxState) -> TaxResult:
        state.deficits.life_years = self.rates.deficit_carry_years
        state.deficits.expire(year)
        rent = sum(c.rent for c in ctxs)
        loan_cost = sum(c.interest + c.loan_insurance for c in ctxs)  # form 2044 line 250
        charges = sum(c.operating_charges + c.works_paid for c in ctxs)
        energy = sum(c.energy_works_paid for c in ctxs)
        net = rent - loan_cost - charges
        imputed = 0.0
        if net < 0:
            deficit = -net
            non_interest = min(deficit, charges)  # rents absorb the interest first
            imputed = min(non_interest, self.global_cap(year, energy))
            state.deficits.add(year, deficit - imputed)
            base = 0.0
        else:
            base = net - state.deficits.use(net)
        state.global_imputed_by_year[year] = imputed
        return TaxResult(
            taxable_base=base,
            ir=(base - imputed) * self.tmi_pct / 100.0,
            ps=base * self.ps_pct / 100.0,
            detail={
                "net_foncier_result": net,
                "deficit_imputed_on_global_income": imputed,
                "foncier_deficit_stock_end": state.deficits.total,
            },
        )

    def deficit_recapture(self, years_held: int, state: TaxState) -> float:
        """IR clawed back when the letting stops before 31/12 of N + 3 (CGI 156 I 3°, TAX_SPEC 4.5)."""
        horizon = self.rates.deficit_recapture_years
        clawed = sum(v for n, v in state.global_imputed_by_year.items() if n + horizon > years_held)
        return clawed * self.tmi_pct / 100.0

    def compute_exit(self, ctx: ExitContext, state: TaxState) -> ExitTax:
        # Deducted works are excluded from the majoration; only the 15 % flat rate applies (TAX_SPEC 3.2).
        return _plus_value_exit(
            ctx,
            works_already_deducted=True,
            deficit_recapture=self.deficit_recapture(ctx.years_held, state),
        )


__all__ = ["FoncierReel", "LmnpReel", "MicroBic", "MicroFoncier"]
