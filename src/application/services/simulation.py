"""Long-term strategy simulation.

Modular simulation engine extracted from financial_calculations.py.
Provides year-by-year projections and final liquidation calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy_financial as npf
import pandas as pd

from src.core.defaults import FINANCING, MARKET, OPERATING, TAX
from src.core.exceptions import SimulationError
from src.domain.tax import (
    PropertyExit,
    RegimeId,
    TaxConfig,
    TaxHousehold,
    YearContext,
    parse_furnished_regime,
    parse_unfurnished_regime,
)


@dataclass
class MarketHypotheses:
    """Market growth assumptions for simulation."""

    appreciation_bien_pct: float = MARKET.appreciation_bien_pct
    revalo_loyer_pct: float = MARKET.revalo_loyer_pct
    inflation_charges_pct: float = MARKET.inflation_charges_pct

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MarketHypotheses:
        return cls(
            appreciation_bien_pct=d.get("appreciation_bien_pct", MARKET.appreciation_bien_pct),
            revalo_loyer_pct=d.get("revalo_loyer_pct", MARKET.revalo_loyer_pct),
            inflation_charges_pct=d.get("inflation_charges_pct", MARKET.inflation_charges_pct),
        )


@dataclass
class TaxParams:
    """User's tax choices: one regime per income category of the foyer (TAX_SPEC 8).

    ``regime_fiscal`` is the furnished regime ("lmnp" = LMNP réel, or "microbic"); the legacy value
    "sci_is" is read as ``sci_is=True``. ``regime_foncier`` is the unfurnished regime ("reel" or
    "micro"). ``sci_is`` puts every property in one SCI at IS and overrides both.
    """

    tmi_pct: float = TAX.tmi_pct
    regime_fiscal: str = TAX.regime_fiscal
    amort_mobilier: bool = True
    micro_bic_abatt_pct: float = TAX.micro_bic_abatt_pct  # 50 (long-term / classé) or 30 (non classé)
    regime_foncier: str = TAX.regime_foncier
    sci_is: bool = TAX.sci_is
    sci_distribute_at_exit: bool = TAX.sci_distribute_at_exit

    def __post_init__(self) -> None:
        if str(self.regime_fiscal).strip().lower() == RegimeId.SCI_IS.value:
            self.sci_is = True
            self.regime_fiscal = TAX.regime_fiscal

    def to_config(self, nb_owners: int = TAX.nb_owners) -> TaxConfig:
        return TaxConfig(
            tmi_pct=self.tmi_pct,
            furnished_regime=parse_furnished_regime(self.regime_fiscal),
            unfurnished_regime=parse_unfurnished_regime(self.regime_foncier),
            sci_is=self.sci_is,
            micro_bic_abattement_pct=float(self.micro_bic_abatt_pct),
            amortize_furniture=self.amort_mobilier,
            sci_distribute_at_exit=self.sci_distribute_at_exit,
            nb_owners=nb_owners,
        )


@dataclass
class IRACalculator:
    """Early repayment indemnity calculator."""

    apply_ira: bool = True
    ira_cap_pct: float = 3.0

    def calculate(
        self,
        projets: list[dict[str, Any]],
        schedules: list[dict[str, Any]],
        horizon_years: int,
    ) -> float:
        """Calculate total IRA across all properties."""
        if not self.apply_ira:
            return 0.0

        ira_total = 0.0
        for i, p in enumerate(projets):
            duree_pret = int(p.get("duree_pret", 0))
            if horizon_years < duree_pret:
                sch = schedules[i]
                nmois = int(sch.get("nmois", 0))
                mH = min(horizon_years * 12, nmois)

                crd_H = float(sch["balances"][mH - 1]) if mH > 0 else float(p.get("credit_final", 0.0))
                monthly_rate = (float(p.get("taux_pret", 0.0)) / 100.0) / 12.0

                cap_pct_fee = (self.ira_cap_pct / 100.0) * crd_H
                six_months_interest = 6.0 * monthly_rate * crd_H

                ira_total += max(0.0, min(cap_pct_fee, six_months_interest))

        return ira_total


@dataclass
class YearResult:
    """Single year simulation result."""

    year: int
    valeur_biens: float
    dette: float
    patrimoine_net: float
    resultat_fiscal: float
    impot_du: float
    cash_flow_net: float
    capital_rembourse: float
    interets_assurance: float
    loyers_bruts: float
    charges_deductibles: float
    amortissements: float
    # Tax split (TAX_SPEC 7): IR (negative in a déficit-foncier year), PS, IS.
    ir: float = 0.0
    ps: float = 0.0
    is_: float = 0.0
    # Per-regime detail of the year (ARD and deficit stocks, imputations...), keys as in TAX_SPEC 6.
    tax_detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "Année": self.year,
            "Valeur Biens": self.valeur_biens,
            "Dette": self.dette,
            "Patrimoine Net": self.patrimoine_net,
            "Résultat Fiscal": self.resultat_fiscal,
            "Impôt Dû": self.impot_du,
            "Cash-Flow Net d'Impôt": self.cash_flow_net,
            "Capital Remboursé": self.capital_rembourse,
            "Intérêts & Assurance": self.interets_assurance,
            "Loyers Bruts": self.loyers_bruts,
            "Charges Déductibles": -self.charges_deductibles,
            "Amortissements": -self.amortissements,
            "IR": self.ir,
            "PS": self.ps,
            "IS": self.is_,
        }


class SimulationEngine:
    """Long-term strategy simulation engine.

    Runs year-by-year projections for a property portfolio. The tax of each year and the exit tax
    are delegated to the foyer's tax regimes (``src.domain.tax``, TAX_SPEC 7-8).
    """

    def __init__(
        self,
        market: MarketHypotheses,
        tax: TaxParams,
        ira: IRACalculator,
        cfe_par_bien_ann: float = OPERATING.cfe_par_bien_ann,
        frais_vente_pct: float = TAX.frais_vente_pct,
        amort_immo_years: int = 30,
        amort_mobilier_years: int = 10,
        ratio_terrain_pct: float = TAX.ratio_terrain_pct,
        amort_works_years: int = TAX.amort_works_years,
        nb_owners: int = TAX.nb_owners,
    ):
        self.market = market
        self.tax = tax
        self.ira = ira
        self.cfe_par_bien_ann = cfe_par_bien_ann
        self.frais_vente_pct = frais_vente_pct
        self.ratio_terrain_pct = max(0.0, min(100.0, ratio_terrain_pct))
        if nb_owners not in (1, 2):
            raise ValueError("nb_owners must be 1 or 2")
        self.nb_owners = nb_owners

        if amort_immo_years <= 0:
            raise ValueError("amort_immo_years must be > 0 to avoid division by zero")
        if amort_mobilier_years <= 0:
            raise ValueError("amort_mobilier_years must be > 0 to avoid division by zero")

        if amort_works_years <= 0:
            raise ValueError("amort_works_years must be > 0 to avoid division by zero")

        self.amort_immo_years = amort_immo_years
        self.amort_works_years = amort_works_years
        self.amort_mobilier_years = amort_mobilier_years

    def build_household(self, projets: list[dict[str, Any]], horizon_years: int) -> TaxHousehold:
        """Regime of each income category, with the micro ceilings checked on the horizon's receipts."""
        furnished = [_is_furnished(p) for p in projets]
        g = 1.0 + self.market.revalo_loyer_pct / 100.0
        max_receipts: dict[str, float] = {}
        for cat, flag in (("furnished", True), ("unfurnished", False)):
            base = sum(p["loyer_mensuel_initial"] * 12 for p, f in zip(projets, furnished, strict=True) if f == flag)
            # Receipts move monotonically with the years, so the maximum is in year 1 or in year H.
            max_receipts[cat] = max(base, base * g ** max(0, horizon_years - 1))
        return TaxHousehold.build(self.tax.to_config(self.nb_owners), furnished, max_receipts)

    def simulate(
        self,
        strategy: dict[str, Any],
        horizons_years: int,
        schedules: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run full simulation.

        Args:
            strategy: Strategy dict with 'details' and 'apport_total'
            horizons_years: Number of years to simulate
            schedules: Pre-computed amortization schedules for each property

        Returns:
            Tuple of (yearly DataFrame, summary bilan dict)
        """
        if not strategy or not strategy.get("details"):
            raise SimulationError("Cannot simulate an empty strategy")

        projets = [p.copy() for p in strategy["details"]]

        # Validate schedule count matches property count
        if len(schedules) != len(projets):
            raise ValueError(
                f"Schedule count ({len(schedules)}) != property count ({len(projets)}). "
                "Each property requires a pre-computed amortization schedule."
            )

        household = self.build_household(projets, horizons_years)
        valeur_biens = sum(p["prix_achat_bien"] for p in projets)
        flux = [-float(strategy["apport_total"])]
        cum_cash_flow = 0.0
        results: list[YearResult] = []

        for year in range(1, horizons_years + 1):
            result = self._simulate_year(year, projets, schedules, valeur_biens, household)
            results.append(result)
            valeur_biens = result.valeur_biens
            flux.append(result.cash_flow_net)
            cum_cash_flow += result.cash_flow_net

        bilan = self._calculate_liquidation(
            results, projets, schedules, horizons_years, flux, household, cum_cash_flow=cum_cash_flow,
        )

        df = pd.DataFrame([r.to_dict() for r in results])
        return df, bilan

    def _cfe_for_year(self, p: dict[str, Any], year: int, receipts: float) -> float:
        """CFE of one property for one year (TAX_SPEC 5).

        Furnished letting only; nothing in year 1 (year of creation, CGI 1478 II); nothing when the year's
        receipts are <= 5,000 EUR (CGI 1647 D, AN question 7364); indexed like fixed charges (TAX_SPEC 1.8).
        """
        if not _is_furnished(p) or year <= 1 or receipts <= CFE_EXEMPTION_RECEIPTS:
            return 0.0
        return self.cfe_par_bien_ann * (1.0 + self.market.inflation_charges_pct / 100.0) ** (year - 1)

    def _simulate_year(
        self,
        year: int,
        projets: list[dict[str, Any]],
        schedules: list[dict[str, Any]],
        valeur_biens: float,
        household: TaxHousehold,
    ) -> YearResult:
        """Simulate a single year: cash flows per property, then the foyer's tax."""
        # Market value at the end of year y = price x (1 + g)^y (TAX_SPEC 1.7)
        valeur_biens *= (1.0 + self.market.appreciation_bien_pct / 100.0)

        cf_ann, dette_fin = 0.0, 0.0
        capital_rembourse_ann, interets_assurance_ann = 0.0, 0.0
        loyers_bruts_pf, charges_deductibles_pf = 0.0, 0.0
        year_ctxs: list[YearContext] = []

        for i, p in enumerate(projets):
            # Update rents and charges
            if year == 1:
                p["loyer_mth_courant"] = p["loyer_mensuel_initial"]
                p["charges_fix_mth_courant"] = p["charges_const_mth0"] + p["tf_const_mth0"]
            else:
                p["loyer_mth_courant"] *= (1.0 + self.market.revalo_loyer_pct / 100.0)
                p["charges_fix_mth_courant"] *= (1.0 + self.market.inflation_charges_pct / 100.0)

            loyer_ann = p["loyer_mth_courant"] * 12
            gest_ann = loyer_ann * (p["frais_gestion_pct"] / 100.0)
            prov_ann = loyer_ann * (p["provision_pct"] / 100.0)
            charges_fixes_ann = p["charges_fix_mth_courant"] * 12
            cfe_ann = self._cfe_for_year(p, year, loyer_ann)

            # Loan schedule
            sch = schedules[i]
            start, end = (year - 1) * 12, min(year * 12, sch["nmois"])
            months = max(0, end - start)

            interets = sum(sch["interets"][start:end]) if months > 0 else 0.0
            capital_rembourse_ann += sum(sch["principals"][start:end]) if months > 0 else 0.0
            assurance = months * sch["pmt_assur"]
            debt_service = months * sch["pmt_total"]

            solde_fin = (
                sch["balances"][end - 1] if months > 0
                else sch["balances"][-1] if sch["nmois"] > 0 and year * 12 > sch["nmois"]
                else p["credit_final"] if sch["nmois"] == 0 else 0.0
            )

            operating = charges_fixes_ann + gest_ann + prov_ann + cfe_ann
            loyers_bruts_pf += loyer_ann
            charges_deductibles_pf += operating + interets + assurance

            # Amortization by component, straight-line, each stopping after its own duration
            # (TAX_SPEC 4.1; CGI 39 C; BOI-BIC-AMT-20-40-10-20). Land is never amortized.
            # Notary fees are capitalized with the purchase price and split pro rata land/building.
            # Only LMNP réel and SCI IS amortize; unfurnished regimes never do (TAX_SPEC 4.1, 10 item 7).
            amortizes = household.amortizes(i)
            ratio_terrain = p.get("ratio_terrain_pct")
            ratio_terrain = self.ratio_terrain_pct if ratio_terrain is None else max(0.0, min(100.0, float(ratio_terrain)))
            base_batiment = (p["prix_achat_bien"] + p.get("frais_notaire", 0.0)) * (1.0 - ratio_terrain / 100.0)
            energy_works = float(p.get("renovation_energetique", 0.0) or 0.0)
            base_travaux = p["budget_travaux"] + energy_works
            year_ctxs.append(
                YearContext(
                    year=year,
                    property_id=i,
                    rent=loyer_ann,
                    operating_charges=operating,
                    interest=interets,
                    loan_insurance=assurance,
                    # Works are paid at t0, i.e. in year 1 (TAX_SPEC 1.9); deductible under foncier réel (4.5).
                    works_paid=base_travaux if year == 1 else 0.0,
                    energy_works_paid=energy_works if year == 1 else 0.0,
                    amort_building=(
                        base_batiment / self.amort_immo_years if amortizes and year <= self.amort_immo_years else 0.0
                    ),
                    amort_works=(
                        base_travaux / self.amort_works_years if amortizes and year <= self.amort_works_years else 0.0
                    ),
                    amort_furniture=(
                        p["mobilier"] / self.amort_mobilier_years
                        if amortizes and year <= self.amort_mobilier_years else 0.0
                    ),
                )
            )

            cf_ann += loyer_ann - operating - debt_service
            interets_assurance_ann += interets + assurance
            dette_fin += solde_fin

        tax = household.compute_year(year, year_ctxs)
        cf_ann -= tax.total

        return YearResult(
            year=year,
            valeur_biens=valeur_biens,
            dette=max(0.0, dette_fin),
            patrimoine_net=valeur_biens - dette_fin,
            resultat_fiscal=tax.taxable_base,
            impot_du=tax.total,
            cash_flow_net=cf_ann,
            capital_rembourse=capital_rembourse_ann,
            interets_assurance=interets_assurance_ann,
            loyers_bruts=loyers_bruts_pf,
            charges_deductibles=charges_deductibles_pf,
            amortissements=tax.amortization_booked,
            ir=tax.ir,
            ps=tax.ps,
            is_=tax.is_,
            tax_detail=dict(tax.detail),
        )

    def _calculate_liquidation(
        self,
        results: list[YearResult],
        projets: list[dict[str, Any]],
        schedules: list[dict[str, Any]],
        horizon: int,
        flux: list[float],
        household: TaxHousehold,
        cum_cash_flow: float = 0.0,
    ) -> dict[str, Any]:
        """Calculate final liquidation and IRR."""
        if not results:
            return {}

        final = results[-1]
        valeur_biens = final.valeur_biens
        dette_finale = final.dette

        frais_vente = valeur_biens * (self.frais_vente_pct / 100.0)
        ira_total = self.ira.calculate(projets, schedules, horizon)
        apport_initial = -flux[0] if flux else 0.0

        # Exit tax per regime: plus-value of individuals per property, floored at 0, no netting
        # (TAX_SPEC 3, 1.10), plus déficit-foncier recapture; or IS on exit + PFU on the boni (SCI IS, 4.6).
        g = 1.0 + self.market.appreciation_bien_pct / 100.0
        exits = []
        for i, p in enumerate(projets):
            sale_price = p["prix_achat_bien"] * g**horizon
            exits.append(
                PropertyExit(
                    property_id=i,
                    name=str(p.get("nom_bien", p.get("nom", f"bien_{i + 1}"))),
                    price=p["prix_achat_bien"],
                    notary_fees=p.get("frais_notaire", 0.0),
                    works=p["budget_travaux"] + p.get("renovation_energetique", 0.0),
                    furniture=p.get("mobilier", 0.0),
                    sale_price=sale_price,
                    sale_fees=sale_price * self.frais_vente_pct / 100.0,
                    residence_services=bool(p.get("residence_services", False)),
                )
            )
        exit_tax = household.compute_exit(
            horizon, exits, debt_repaid=dette_finale + ira_total, cum_cash_flow=cum_cash_flow, apport=apport_initial,
        )
        impot_pv = exit_tax.total

        liquidation_nette = valeur_biens - dette_finale - frais_vente - impot_pv - ira_total

        # IRR
        if len(flux) >= 2:
            flux[-1] += liquidation_nette
        else:
            flux.append(liquidation_nette)

        # Failed IRR is None plus a warning, never 0 % (TAX_SPEC 5, 10 item 23).
        tri: float | None
        try:
            tri = float(npf.irr(flux)) * 100.0
            tri_warning = None
            if not isfinite(tri):
                tri = None
                tri_warning = "IRR is non-finite (NaN or Inf)"
        except Exception:
            tri = None
            tri_warning = "IRR calculation failed (unstable cashflow pattern)"

        # Wealth = net exit proceeds + sum of after-tax yearly cash flows - down payment (TAX_SPEC 5)
        enrichissement_net = liquidation_nette + cum_cash_flow - apport_initial

        # Calculate DSCR from Year 1 data
        from src.core.glossary import calculate_dscr_metric
        if results:
            df_y1 = pd.DataFrame([results[0].to_dict()])
            dscr_y1 = calculate_dscr_metric(df_y1)
        else:
            dscr_y1 = 0.0

        return {
            "tri_annuel": tri,
            "tri_warning": tri_warning,
            "liquidation_nette": liquidation_nette,
            "enrichissement_net": enrichissement_net,
            "ira_total": ira_total,
            "dscr_y1": dscr_y1,
            "impot_pv": impot_pv,
            "plus_value": exit_tax.details,
            "plus_value_par_bien": [{"nom": name, **b.to_dict()} for _, name, b in exit_tax.per_property],
            "exit_tax": {
                "ir": exit_tax.ir,
                "ps": exit_tax.ps,
                "surtax": exit_tax.surtax,
                "is": exit_tax.is_,
                "pfu": exit_tax.pfu,
                "deficit_recapture": exit_tax.deficit_recapture,
                "total": exit_tax.total,
            },
            # Regime applied per category, and why a choice was overridden (micro ceilings, LMP flag).
            "regimes": household.regime_ids,
            "tax_warnings": list(household.warnings),
            # Per-year tax detail: IR/PS/IS split plus the regimes' carryforward stocks (TAX_SPEC 6-7).
            "fiscal_annuel": [
                {
                    "year": r.year, "taxable_base": r.resultat_fiscal, "ir": r.ir, "ps": r.ps, "is": r.is_,
                    "tax": r.impot_du, **r.tax_detail,
                }
                for r in results
            ],
        }


# CGI 1647 D: no minimum CFE contribution when receipts are <= 5,000 EUR (AN question 7364)
CFE_EXEMPTION_RECEIPTS = 5000.0


def _is_furnished(p: dict[str, Any]) -> bool:
    """Furnished flag of a property: `meuble`, else inferred from `mode_loyer` (`nu_*` = unfurnished).

    Unknown (neither field present) counts as furnished, the conservative choice for cash flow.
    """
    if p.get("meuble") is not None:
        return bool(p["meuble"])
    mode = p.get("mode_loyer")
    if mode:
        return not str(mode).startswith("nu")
    return True


def _ratio_terrain_or_default(value: float | None) -> float:
    return TAX.ratio_terrain_pct if value is None else float(value)


# Convenience function for backward compatibility
def simulate_long_term_strategy(
    strategy: dict[str, Any],
    duration_years: int = 25,
    market_hypotheses: dict[str, float] | None = None,
    tax_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate strategy performance over time.

    This is a convenience wrapper around SimulationEngine.
    For full control, use SimulationEngine directly.
    """
    # Import modular generator
    from src.domain.calculator.financial import (
        generate_amortization_schedule,
    )
    if not strategy or not strategy.get("details"):
        raise SimulationError("Cannot simulate an empty strategy")

    market = MarketHypotheses.from_dict(market_hypotheses or {})

    tax_dict = tax_params or {}
    tax = TaxParams(
        tmi_pct=tax_dict.get("tmi_pct", TAX.tmi_pct),
        regime_fiscal=tax_dict.get("regime_fiscal", TAX.regime_fiscal),
        micro_bic_abatt_pct=tax_dict.get("micro_bic_abatt_pct", TAX.micro_bic_abatt_pct),
        regime_foncier=tax_dict.get("regime_foncier", TAX.regime_foncier),
        sci_is=bool(tax_dict.get("sci_is", TAX.sci_is)),
        sci_distribute_at_exit=bool(tax_dict.get("sci_distribute_at_exit", TAX.sci_distribute_at_exit)),
    )

    ira = IRACalculator(
        apply_ira=tax_dict.get("apply_ira", FINANCING.apply_ira),
        ira_cap_pct=tax_dict.get("ira_cap_pct", FINANCING.ira_cap_pct),
    )

    engine = SimulationEngine(
        market=market,
        tax=tax,
        ira=ira,
        cfe_par_bien_ann=tax_dict.get("cfe_par_bien_ann", OPERATING.cfe_par_bien_ann),
        frais_vente_pct=tax_dict.get("frais_vente_pct", TAX.frais_vente_pct),
        ratio_terrain_pct=_ratio_terrain_or_default(tax_dict.get("ratio_terrain_pct")),
        nb_owners=int(tax_dict.get("nb_owners", TAX.nb_owners)),
    )

    # Generate schedules using modular function
    schedules = [
        generate_amortization_schedule(
            float(p["credit_final"]),
            float(p["taux_pret"]),
            int(p["duree_pret"]) * 12,  # Convert years to months
            float(p["assurance_ann_pct"])
        )
        for p in strategy["details"]
    ]

    return engine.simulate(strategy, duration_years, schedules)

