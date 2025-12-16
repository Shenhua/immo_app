"""Portfolio allocation service.

Logic for distributing capital across properties to meet cash flow targets.
Converted from legacy `allouer_apport_pour_cf`.
"""

from __future__ import annotations

import os
from typing import Any

from src.core.financial import calculate_total_monthly_payment, k_factor
from src.core.logging import get_logger

log = get_logger(__name__)


class PortfolioAllocator:
    """Service for allocating capital across properties."""

    def __init__(self, mode_cf: str = "target"):
        self.mode_cf = mode_cf
        self.max_extra_apport_pct = float(os.getenv("MAX_EXTRA_APPORT_PCT", "0.95"))
        log.info("allocator_initialized", mode=self.mode_cf, max_extra_apport_ratio=self.max_extra_apport_pct)

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
                log.debug(
                    "allocation_early_exit",
                    cf0=cf0,
                    target=target_cf,
                    reason="Target met & no aggressive deployment"
                )
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
        
        log.debug(
            "starting_allocation",
            cf_start=cf0,
            target=target_cf,
            apport_rest=apport_rest,
            use_full_capital=use_full_capital
        )

        # Sort by efficiency (k-factor)
        ordre = sorted(
            biens,
            key=lambda x: k_factor(
                x["taux_pret"], x["duree_pret"], x["assurance_ann_pct"]
            ),
            reverse=True
        )

        # Convergence loop (Multi-pass for Coarse -> Refine strategy)
        # Pass 1 may use coarse steps and undershoot (due to truncation).
        # Pass 2+ will refine with smaller steps to hit target.
        for pass_idx in range(5):
            # Check success at start of pass
            # Phase 18.2 Fix: Do NOT break if we want to use full capital (Empire mode)
            if self._accept_cf(cf0, target_cf, tolerance):
                if not (use_full_capital and not is_precise):
                    break
                
            # Recalculate need
            manque_cf = self._calc_need(cf0, target_cf, tolerance)
             # If forcing full capital, behave as if we have an infinite need for CF
            if use_full_capital and not is_precise:
                manque_cf = 1e9

            # Stop if no need (and not forced) 
            if manque_cf <= 1e-9 and not (use_full_capital and not is_precise):
                break
            
            # Allow visiting properties again to fill the gap
            changes_made = False
            
            for b in ordre:
                if apport_rest <= 1e-9:
                    break
                
                # Re-check local need inside loop (since others might have filled it)
                manque_cf = self._calc_need(cf0, target_cf, tolerance)
                if use_full_capital and not is_precise: manque_cf = 1e9
                
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

                # Adaptive step size (Expert Recommendation Part 4)
                # Avoid jumping over the target window
                # Approx k ~ 150-250 for typical loans. Using 200 as proxy if not calcable.
                k_proxy = k if k > 0 else 200.0
                
                if is_precise:
                    # Dynamic Refinement (Coarse -> Fine):
                    # If gap is huge, stride big. If close, tiptoe.
                    # This implements "Coarse then Refine" inside the loop efficiently.
                    current_gap = abs(manque_cf)
                    fine_step = max(1.0, (0.5 * tolerance) / k_proxy)
                    
                    if current_gap > 5 * tolerance:
                        # Coarse mode: go fast (10x fine step or generic coarse)
                        step_size = max(fine_step * 10.0, apport_disponible / 1000.0)
                    else:
                        # Refine mode: go precise
                        step_size = fine_step
                else:
                    # In min mode (>= target), coarser steps are efficient
                    # But kept reasonable to not overshoot massive amounts
                    step_size = max(apport_disponible / 2000, 10.0) 

                # Round delta
                if step_size > 0:
                    # Use int() to floor/truncate. 
                    # Rounding up (nearest) risks overshooting the target in an additive-only process.
                    # It's safer to under-step and let the next iteration finish the job.
                    delta = int(delta / step_size) * step_size

                # Safety cap against overspending
                if delta > apport_rest:
                    delta = apport_rest

                if delta < 0.01:
                    continue

                # Update state
                b["capital_restant"] = max(1.0, b["capital_restant"] - delta)

                # Recompute payment
                _, _, p_tot = calculate_total_monthly_payment(
                    b["capital_restant"],
                    b["taux_pret"],
                    b["duree_pret"] * 12,
                    b["assurance_ann_pct"]
                )
                b["pmt_total"] = p_tot
                b["apport_add_bien"] = b.get("apport_add_bien", 0.0) + delta
                changes_made = True

                apport_rest -= delta

                # Recompute global CF
                cf0 = sum(
                    x["loyer_mensuel_initial"] -
                    x["depenses_mensuelles_hors_credit_initial"] -
                    x["pmt_total"]
                    for x in biens
                )
                # log.debug("step_allocation", delta=delta,  new_cf=cf0, property=b.get("nom_bien"))
            
            if not changes_made:
                # If we iterated all properties and made no changes, we are stuck. Stop.
                break

        success = self._accept_cf(cf0, target_cf, tolerance)
        
        log.debug("allocation_finished", final_cf=cf0, success=success, remaining_apport=apport_rest)

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
             return bool(cf_observe >= (cf_cible - tol))
        return bool(abs(cf_observe - cf_cible) <= tol)

    def _calc_need(self, cf_observe: float, cf_cible: float, tol: float) -> float:
        if self.mode_cf == "min" or self.mode_cf == "≥":
            return max(0.0, cf_cible - cf_observe)
        else:
            return cf_cible - cf_observe if abs(cf_cible - cf_observe) > tol else 0.0
