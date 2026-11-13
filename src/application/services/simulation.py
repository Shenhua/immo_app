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
from src.domain.tax.plus_value import PlusValueBreakdown, compute_plus_value, sum_breakdowns

# Tax constants
TAUX_PRELEVEMENTS_SOCIAUX = 17.2


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
    """Tax calculation parameters."""

    tmi_pct: float = TAX.tmi_pct
    regime_fiscal: str = TAX.regime_fiscal  # lmnp or microbic
    amort_mobilier: bool = True
    micro_bic_abatt_pct: float = TAX.micro_bic_abatt_pct  # Default 50%, can be 71% for classé


    def calculate_tax(
        self,
        loyers_bruts: float,
        charges_deductibles: float,
        amortissements: float,
        deficit_reportable: float = 0.0,
        amort_differes_reportable: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """Calculate income tax for the year.

        Returns:
            Tuple of (tax_due, new_deficit_reportable, new_amort_differes, taxable_base)
        """
        if self.regime_fiscal == "microbic":
            base_imposable = max(0.0, loyers_bruts * (1.0 - self.micro_bic_abatt_pct / 100.0))
            new_deficit = 0.0
            new_ard = 0.0
        else:
            # LMNP réel logic
            noi = loyers_bruts - charges_deductibles

            # 1. Deficit Foncier (Charges > Rents)
            if noi < 0:
                # Deficit created/increased. Amortization is largely deferred.
                new_deficit = deficit_reportable + noi
                new_ard = amort_differes_reportable + amortissements
                base_imposable = 0.0
            else:
                # Positive NOI - absorb deficit first
                # deficit_reportable is negative
                res_apres_deficit = noi + deficit_reportable

                if res_apres_deficit < 0:
                    # Still in deficit
                    new_deficit = res_apres_deficit
                    new_ard = amort_differes_reportable + amortissements
                    base_imposable = 0.0
                else:
                    # Deficit cleared, now attack amortization
                    new_deficit = 0.0

                    # Deduct stock ARD + current amort
                    # (In LMNP, you can use ARD to reduce profit to 0, but not create deficit)
                    # We treat (ard + amort) as the total pool of deductive power
                    total_amort_pool = amort_differes_reportable + amortissements

                    res_final = res_apres_deficit - total_amort_pool

                    if res_final < 0:
                        # Profit reduced to 0, remaining amort becomes new ARD
                        base_imposable = 0.0
                        new_ard = -res_final
                    else:
                        # Profit remaining after full amort deduction
                        base_imposable = res_final
                        new_ard = 0.0

        impot = base_imposable * (self.tmi_pct / 100.0 + TAUX_PRELEVEMENTS_SOCIAUX / 100.0)
        return impot, new_deficit, new_ard, base_imposable


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
    # Building + works amortization booked this year, per property (the rest of ``amortissements`` is furniture).
    amort_imm_par_bien: list[float] = field(default_factory=list)

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
        }


class SimulationEngine:
    """Long-term strategy simulation engine.

    Runs year-by-year projections for a property portfolio.
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

        valeur_biens = sum(p["prix_achat_bien"] for p in projets)
        # Amortization actually deducted, building/works pool vs furniture pool (TAX_SPEC 3.4, 11.3).
        ard_imm = ard_mob = 0.0
        cum_ded_imm = 0.0
        cum_booked_imm = [0.0] * len(projets)
        flux = [-float(strategy["apport_total"])]

        results: list[YearResult] = []
        deficit_reportable = 0.0
        ard_reportable = 0.0
        total_impot = 0.0

        for year in range(1, horizons_years + 1):
            result, next_deficit, next_ard = self._simulate_year(
                year, projets, schedules, valeur_biens, deficit_reportable, ard_reportable
            )
            results.append(result)

            cur_imm = sum(result.amort_imm_par_bien)
            cur_mob = max(0.0, result.amortissements - cur_imm)
            pool_imm, pool_mob = cur_imm + ard_imm, cur_mob + ard_mob
            pool = pool_imm + pool_mob
            deducted = max(0.0, min(pool, pool - next_ard))
            share_imm = pool_imm / pool if pool > 0 else 0.0
            ded_imm = deducted * share_imm
            ard_imm, ard_mob = pool_imm - ded_imm, pool_mob - (deducted - ded_imm)
            cum_ded_imm += ded_imm
            for i, a in enumerate(result.amort_imm_par_bien):
                cum_booked_imm[i] += a

            total_impot += result.impot_du
            deficit_reportable = next_deficit
            ard_reportable = next_ard
            valeur_biens = result.valeur_biens
            flux.append(result.cash_flow_net)

        # Final liquidation
        bilan = self._calculate_liquidation(
            results, projets, schedules, horizons_years, flux,
            cum_ded_imm=cum_ded_imm, cum_booked_imm=cum_booked_imm,
        )

        df = pd.DataFrame([r.to_dict() for r in results])
        return df, bilan

    def _simulate_year(
        self,
        year: int,
        projets: list[dict[str, Any]],
        schedules: list[dict[str, Any]],
        valeur_biens: float,
        deficit_reportable: float,
        ard_reportable: float = 0.0,
    ) -> tuple[YearResult, float, float]:
        """Simulate a single year.

        Returns:
            Tuple of (YearResult, next_year_deficit_reportable, next_year_ard)
        """
        # Market value at the end of year y = price x (1 + g)^y (TAX_SPEC 1.7)
        valeur_biens *= (1.0 + self.market.appreciation_bien_pct / 100.0)

        cf_ann, dette_fin = 0.0, 0.0
        capital_rembourse_ann, interets_assurance_ann = 0.0, 0.0
        loyers_bruts_pf, charges_deductibles_pf, amortissements_pf = 0.0, 0.0, 0.0
        amort_imm_par_bien: list[float] = []

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

            loyers_bruts_pf += loyer_ann
            charges_deductibles_pf += (
                charges_fixes_ann + gest_ann + prov_ann +
                interets + assurance + self.cfe_par_bien_ann
            )

            # Amortization by component, straight-line, each stopping after its own duration
            # (TAX_SPEC 4.1; CGI 39 C; BOI-BIC-AMT-20-40-10-20). Land is never amortized.
            # Notary fees are capitalized with the purchase price and split pro rata land/building.
            ratio_terrain = p.get("ratio_terrain_pct")
            ratio_terrain = self.ratio_terrain_pct if ratio_terrain is None else max(0.0, min(100.0, float(ratio_terrain)))
            base_batiment = (p["prix_achat_bien"] + p.get("frais_notaire", 0.0)) * (1.0 - ratio_terrain / 100.0)
            base_travaux = p["budget_travaux"] + p.get("renovation_energetique", 0.0)
            amort_immo = base_batiment / self.amort_immo_years if year <= self.amort_immo_years else 0.0
            amort_trav = base_travaux / self.amort_works_years if year <= self.amort_works_years else 0.0
            amort_mob = p["mobilier"] / self.amort_mobilier_years if year <= self.amort_mobilier_years else 0.0
            amort_imm_par_bien.append(amort_immo + amort_trav if self.tax.regime_fiscal == "lmnp" else 0.0)
            if self.tax.regime_fiscal == "lmnp":
                amortissements_pf += amort_immo + amort_trav + (amort_mob if self.tax.amort_mobilier else 0.0)

            cf_ann += loyer_ann - (charges_fixes_ann + gest_ann + prov_ann) - debt_service - self.cfe_par_bien_ann
            interets_assurance_ann += interets + assurance
            dette_fin += solde_fin

        # Tax calculation
        impot, new_deficit, new_ard, base_imposable = self.tax.calculate_tax(
            loyers_bruts_pf, charges_deductibles_pf, amortissements_pf,
            deficit_reportable, ard_reportable
        )
        cf_ann -= impot

        return YearResult(
            year=year,
            valeur_biens=valeur_biens,
            dette=max(0.0, dette_fin),
            patrimoine_net=valeur_biens - dette_fin,
            resultat_fiscal=base_imposable,
            impot_du=impot,
            cash_flow_net=cf_ann,
            capital_rembourse=capital_rembourse_ann,
            interets_assurance=interets_assurance_ann,
            loyers_bruts=loyers_bruts_pf,
            charges_deductibles=charges_deductibles_pf,
            amortissements=amortissements_pf,
            amort_imm_par_bien=amort_imm_par_bien,
        ), new_deficit, new_ard

    def _calculate_liquidation(
        self,
        results: list[YearResult],
        projets: list[dict[str, Any]],
        schedules: list[dict[str, Any]],
        horizon: int,
        flux: list[float],
        cum_ded_imm: float = 0.0,
        cum_booked_imm: list[float] | None = None,
    ) -> dict[str, Any]:
        """Calculate final liquidation and IRR."""
        if not results:
            return {}

        final = results[-1]
        valeur_biens = final.valeur_biens
        dette_finale = final.dette

        frais_vente = valeur_biens * (self.frais_vente_pct / 100.0)

        # Plus-value of individuals, per property, floored at 0, no netting (TAX_SPEC 3, 1.10).
        # The deducted amortization is shared between properties pro rata of the amortization each booked.
        booked = cum_booked_imm or [0.0] * len(projets)
        total_booked = sum(booked)
        g = 1.0 + self.market.appreciation_bien_pct / 100.0
        is_lmnp = self.tax.regime_fiscal == "lmnp"
        par_bien: list[PlusValueBreakdown] = []
        for i, p in enumerate(projets):
            sale_price = p["prix_achat_bien"] * g**horizon
            share = booked[i] / total_booked if total_booked > 0 else 0.0
            par_bien.append(
                compute_plus_value(
                    price=p["prix_achat_bien"],
                    notary_fees=p.get("frais_notaire", 0.0),
                    works=p["budget_travaux"] + p.get("renovation_energetique", 0.0),
                    sale_price=sale_price,
                    sale_fees=sale_price * self.frais_vente_pct / 100.0,
                    years_held=horizon,
                    amortization_deducted_imm=cum_ded_imm * share,
                    # CGI 150 VB III: LMNP reel only; residences-services are exempt (TAX_SPEC 3.4)
                    amortization_addback_applies=is_lmnp and not bool(p.get("residence_services", False)),
                    nb_owners=self.nb_owners,
                )
            )
        pv_total = sum_breakdowns(par_bien)
        impot_pv = pv_total.total_tax

        # IRA
        ira_total = self.ira.calculate(projets, schedules, horizon)

        liquidation_nette = valeur_biens - dette_finale - frais_vente - impot_pv - ira_total

        # IRR
        if len(flux) >= 2:
            flux[-1] += liquidation_nette
        else:
            flux.append(liquidation_nette)

        try:
            tri = float(npf.irr(flux)) * 100.0
            tri_warning = None
        except Exception:
            tri = 0.0
            tri_warning = "IRR calculation failed (unstable cashflow pattern)"

        # Handle non-finite IRR
        if not isfinite(tri):
            tri = 0.0
            tri_warning = "IRR is non-finite (NaN or Inf)"

        # Initial investment (flux[0] is negative apport)
        apport_initial = -flux[0] if flux else 0.0
        enrichissement_net = liquidation_nette - apport_initial

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
            "plus_value": pv_total.to_dict(),
            "plus_value_par_bien": [
                {"nom": p.get("nom_bien", p.get("nom", f"bien_{i + 1}")), **b.to_dict()}
                for i, (p, b) in enumerate(zip(projets, par_bien, strict=True))
            ],
        }


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

