"""Portfolio allocation service.

Logic for distributing capital across properties to meet cash flow targets.
Converted from legacy `allouer_apport_pour_cf`.
"""

from __future__ import annotations

import os
from typing import Any

from src.core.financial import calculate_total_monthly_payment, k_factor


class PortfolioAllocator:
    """Service for allocating capital across properties."""

    def __init__(self, mode_cf: str = "target"):
        self.mode_cf = mode_cf
        self.max_extra_apport_pct = float(os.getenv("MAX_EXTRA_APPORT_PCT", "0.75"))

    def allocate(
        self,
        bricks: list[dict[str, Any]],
        apport_disponible: float,
        target_cf: float,
        tolerance: float = 100.0,
        use_full_capital: bool = False,
    ) -> tuple[bool, list[dict[str, Any]], float, float]:
        """Allocate capital to maximize cash flow proximity or usage.

        Args:
            bricks: List of investment bricks (properties)
            apport_disponible: Total capital available
            target_cf: Target monthly cash flow
            tolerance: Tolerance for CF target
            use_full_capital: If True, try to use all available apport (unless precise target)

        Returns:
            Tuple of (success, details_list, final_cf, total_apport_used)
        """
        # Initial setup: minimal apport
        biens = []
        for b in bricks:
            # Setup initial state with minimum apport
            biens.append({
                **b,
                "capital_restant": b["capital_emprunte"],  # Initial loan amount
                "pmt_total": b["pmt_total"],  # Monthly payment at min apport
                "apport_add_bien": 0.0,
            })

        # Calculate initial CF
        cf0 = sum(
            b["loyer_mensuel_initial"] -
            b["depenses_mensuelles_hors_credit_initial"] -
            b["pmt_total"]
            for b in biens
        )

        apport_total_min = sum(b.get("apport_min", 0.0) for b in bricks)
        apport_suppl_max = max(0.0, apport_disponible - apport_total_min)

        # check precise mode
        is_precise = self.mode_cf == "target"

        # Check if already good
        # Only return early if we DON'T want to burn full capital OR if we are in strict precise mode
        if self._accept_cf(cf0, target_cf, tolerance):
             if not use_full_capital or is_precise or apport_suppl_max < 1.0:
                details = [
                    {
                        **b,
                        "apport_final_bien": b["apport_min"],
                        "credit_final": b["capital_restant"]
                    }
                    for b in biens
                ]
                return True, details, cf0, apport_total_min

        # Iterative greedy allocation
        apport_rest = apport_suppl_max

        # Sort by efficiency (k-factor)
        ordre = sorted(
            biens,
            key=lambda x: k_factor(
                x["taux_pret"], x["duree_pret"], x["assurance_ann_pct"]
            ),
            reverse=True
        )

        manque_cf = self._calc_need(cf0, target_cf, tolerance)
        
        # If forcing full capital, behave as if we have an infinite need for CF
        if use_full_capital and not is_precise:
            manque_cf = 1e9 

        for b in ordre:
            if apport_rest <= 1e-9:
                break
            
            # Stop if we hit target (unless forcing capital)
            if manque_cf <= 1e-9 and not (use_full_capital and not is_precise):
                break

            k = k_factor(b["taux_pret"], b["duree_pret"], b["assurance_ann_pct"])
            if k <= 0:
                continue

            apport_necessaire = manque_cf / k
            delta = min(apport_rest, apport_necessaire, b["capital_restant"])

            # Cap extra apport to avoid 0 loan
            cap_extra = self.max_extra_apport_pct * float(b.get("capital_emprunte", 0.0))
            deja = float(b.get("apport_add_bien", 0.0))
            reste_cap = max(0.0, cap_extra - deja)

            if delta > reste_cap:
                delta = reste_cap

            # Update state
            b["capital_restant"] = max(1e-2, b["capital_restant"] - delta)

            # Recompute payment
            _, _, p_tot = calculate_total_monthly_payment(
                b["capital_restant"],
                b["taux_pret"],
                b["duree_pret"] * 12,
                b["assurance_ann_pct"]
            )
            b["pmt_total"] = p_tot
            b["apport_add_bien"] = b.get("apport_add_bien", 0.0) + delta

            apport_rest -= delta

            # Recompute global CF
            cf0 = sum(
                x["loyer_mensuel_initial"] -
                x["depenses_mensuelles_hors_credit_initial"] -
                x["pmt_total"]
                for x in biens
            )
            manque_cf = self._calc_need(cf0, target_cf, tolerance)

        success = self._accept_cf(cf0, target_cf, tolerance)

        details = [
            {
                **b,
                "apport_final_bien": b["apport_min"] + b.get("apport_add_bien", 0.0),
                "credit_final": b["capital_restant"]
            }
            for b in biens
        ]

        total_apport_final = sum(d["apport_final_bien"] for d in details)
        return success, details, cf0, total_apport_final

    def _accept_cf(self, cf_observe: float, cf_cible: float, tol: float) -> bool:
        if self.mode_cf == "min" or self.mode_cf == "≥":
             return cf_observe >= (cf_cible - tol)
        return abs(cf_observe - cf_cible) <= tol

    def _calc_need(self, cf_observe: float, cf_cible: float, tol: float) -> float:
        if self.mode_cf == "min" or self.mode_cf == "≥":
            return max(0.0, cf_cible - cf_observe)
        else:
            return cf_cible - cf_observe if abs(cf_cible - cf_observe) > tol else 0.0
