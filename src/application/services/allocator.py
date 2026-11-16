"""Portfolio allocation service.

Logic for distributing capital across properties to meet cash flow targets.
Converted from legacy `allouer_apport_pour_cf`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.core.defaults import MARKET, MAX_EXTRA_APPORT_PCT, REALISM
from src.core.glossary import cf_accepts
from src.core.logging import get_logger
from src.domain.calculator.financial import calculate_total_monthly_payment, k_factor
from src.domain.models.brick import InvestmentBrick

log = get_logger(__name__)


def _get_val(obj: Any, key: str, default: Any = 0.0) -> Any:
    """Safe access for both InvestmentBrick and dict with legacy key mapping."""
    mapping = {
        "nom_bien": "nom",
        "taux_pret": "taux_annuel_pct",
        "assurance_ann_pct": "assurance_annuelle_pct",
        "duree_pret": "duree_credit_mois",
        "cout_total_bien": "cout_total",
    }
    actual_key = mapping.get(key, key)

    # 1. Try dict-style access (for speed and model_dump result)
    if isinstance(obj, dict):
        val = obj.get(actual_key, obj.get(key))
        if val is not None:
            if key == "duree_pret" and actual_key == "duree_credit_mois" and val > 100:
                return val // 12
            return val
        return default

    # 2. Try attribute access (for InvestmentBrick objects)
    val = getattr(obj, actual_key, getattr(obj, key, None))
    if val is not None:
        if key == "duree_pret" and actual_key == "duree_credit_mois" and isinstance(val, int):
            return val // 12
        return val

    # 3. Try subscripting if supported (fallback)
    try:
        val = obj[actual_key]
        if key == "duree_pret" and actual_key == "duree_credit_mois":
             return val // 12
        return val
    except (KeyError, TypeError):
        try:
            return obj[key]
        except (KeyError, TypeError):
            return default


@dataclass(frozen=True, slots=True)
class PreparedBrick:
    """Per-brick data the allocation needs, computed once per search (T31).

    ``k`` is the monthly debt service (P&I + insurance) per euro borrowed: the payment is linear in the loan,
    so one more euro of down payment raises the monthly cash flow by exactly ``k``. ``noi_stab`` is the
    stabilised-year monthly cash flow before debt service and before income tax (same formula as
    ``PortfolioAllocator.stabilised_monthly_cf``). ``data`` is the brick's ``model_dump()``, taken once.
    """

    nom: str
    duree_mois: int
    apport_min: float
    capital: float
    k: float
    noi_stab: float
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Allocation:
    """Result of ``PortfolioAllocator.plan``: extra down payment per brick, in the order given."""

    ok: bool  # the pre-tax estimate meets the target (see glossary.CF_ACCEPTANCE_BASIS)
    extra: tuple[float, ...]
    cf: float  # stabilised monthly CF estimate, before income tax
    apport_total: float
    within_budget: bool


class PortfolioAllocator:
    """Service for allocating capital across properties.

    The allocation is analytic (T31): the stabilised cash flow is linear in each loan amount, so the extra down
    payment needed to close a cash-flow gap is ``gap / k``. Bricks are filled by decreasing ``k`` (most cash flow
    per euro first), each up to ``max_extra_apport_pct`` of its loan, or paid fully in cash when the remaining
    gap needs the whole loan. No step rounding, so the result lands on the target instead of one step short.
    """

    # Aim this far above the target in "min" mode so float rounding never puts the result a cent below it.
    MIN_MODE_EPS = 0.01

    def __init__(self, mode_cf: str = "target", market: dict[str, float] | None = None):
        self.mode_cf = mode_cf
        # Growth rates of the stabilised year (glossary.STABILISED_YEAR = year 2): the estimate below must
        # use the same ones as the simulation, or the allocator accepts what the evaluator rejects.
        m = market or {}
        self.revalo_loyer_pct = float(m.get("revalo_loyer_pct", MARKET.revalo_loyer_pct))
        self.inflation_charges_pct = float(m.get("inflation_charges_pct", MARKET.inflation_charges_pct))
        self.tf_growth_pct = float(m.get("tf_growth_pct", REALISM.tf_growth_pct))
        self.max_extra_apport_pct = MAX_EXTRA_APPORT_PCT
        log.info("allocator_initialized", mode=self.mode_cf, max_extra_apport_ratio=self.max_extra_apport_pct)

    # ------------------------------------------------------------------ preparation
    def prepare(self, brick: InvestmentBrick | dict[str, Any]) -> PreparedBrick:
        """Flatten a brick once: loan constant, stabilised pre-tax NOI and the dumped dict."""
        data = brick.model_dump() if hasattr(brick, "model_dump") else dict(brick)
        capital = float(data.get("capital_emprunte", 0.0) or 0.0)
        years = int(_get_val(brick, "duree_pret"))
        rate = float(_get_val(brick, "taux_pret"))
        ins = float(_get_val(brick, "assurance_ann_pct"))
        k = k_factor(rate, years, ins) if years > 0 else 0.0
        noi = self.stabilised_monthly_cf([{**data, "pmt_total": 0.0}])
        return PreparedBrick(
            nom=str(_get_val(brick, "nom_bien", "")),
            duree_mois=years * 12,
            apport_min=float(_get_val(brick, "apport_min")),
            capital=capital,
            k=k,
            noi_stab=noi,
            data=data,
        )

    # ------------------------------------------------------------------ allocation
    def plan(
        self,
        bricks: Sequence[PreparedBrick],
        apport_disponible: float,
        target_cf: float,
        tolerance: float = 100.0,
        use_full_capital: bool = False,
    ) -> Allocation:
        """Smallest extra down payment that brings the pre-tax stabilised CF to the target.

        "min": aim at ``target + MIN_MODE_EPS``. "target": aim at the centre of the window when the CF is below
        it (a CF above the window cannot be lowered: the down payment is already the minimum).
        ``use_full_capital`` (min mode only) spends the whole budget instead.
        """
        n = len(bricks)
        apport_min = sum(b.apport_min for b in bricks)
        cf0 = sum(b.noi_stab - b.k * b.capital for b in bricks)
        zeros = (0.0,) * n
        if apport_min > apport_disponible + 1e-6:
            return Allocation(False, zeros, cf0, apport_min, within_budget=False)

        if use_full_capital and self.mode_cf != "target":
            need = float("inf")
        elif self.mode_cf == "target":
            need = target_cf - cf0 if cf0 < target_cf - tolerance else 0.0
        else:
            need = target_cf + self.MIN_MODE_EPS - cf0 if cf0 < target_cf else 0.0

        extra = self._fill(bricks, need, apport_disponible - apport_min) if need > 0 else list(zeros)
        cf = cf0 + sum(b.k * x for b, x in zip(bricks, extra, strict=True))
        return Allocation(self._accept_cf(cf, target_cf, tolerance), tuple(extra), cf, apport_min + sum(extra), True)

    def max_cf(self, bricks: Sequence[PreparedBrick], apport_disponible: float) -> float:
        """Highest pre-tax stabilised CF the budget can buy for these bricks."""
        apport_min = sum(b.apport_min for b in bricks)
        cf0 = sum(b.noi_stab - b.k * b.capital for b in bricks)
        extra = self._fill(bricks, float("inf"), max(0.0, apport_disponible - apport_min))
        return cf0 + sum(b.k * x for b, x in zip(bricks, extra, strict=True))

    def _fill(self, bricks: Sequence[PreparedBrick], need: float, budget: float) -> list[float]:
        """Greedy fill of a CF gap ``need`` with at most ``budget`` euros, highest ``k`` first.

        Pass 1: each brick takes ``need / k``, capped at ``max_extra_apport_pct`` of its loan (no near-zero
        residual loans), except that a brick whose whole loan is needed is paid in cash. Pass 2: if a gap is
        left, partially paid loans are paid off in full while the budget allows.
        Ties on ``k`` keep the input order, so the result is deterministic.
        """
        extra = [0.0] * len(bricks)
        order = sorted(range(len(bricks)), key=lambda i: -bricks[i].k)
        left, money = need, budget
        for i in order:
            if left <= 1e-9 or money <= 1e-9:
                break
            b = bricks[i]
            if b.k <= 0 or b.capital <= 0:
                continue
            want = left / b.k
            if want >= b.capital and money >= b.capital:
                x = b.capital
            else:
                x = min(want, self.max_extra_apport_pct * b.capital, money)
            extra[i] = x
            left -= b.k * x
            money -= x
        if left > 1e-9:
            for i in order:
                b = bricks[i]
                rest = b.capital - extra[i]
                if left <= 1e-9 or b.k <= 0 or rest <= 0 or money < rest:
                    continue
                extra[i] = b.capital
                left -= b.k * rest
                money -= rest
        return extra

    def details(self, bricks: Sequence[PreparedBrick], extra: Sequence[float]) -> list[dict[str, Any]]:
        """Per-property dicts for the simulation (same keys as the legacy allocator output)."""
        out = []
        for b, x in zip(bricks, extra, strict=True):
            credit = max(0.0, b.capital - x)
            _, _, pmt = calculate_total_monthly_payment(
                credit, float(b.data.get("taux_annuel_pct", 0.0)), b.duree_mois,
                float(b.data.get("assurance_annuelle_pct", 0.0)),
            )
            out.append({
                **b.data,
                "capital_restant": credit,
                "pmt_total": pmt,
                "apport_add_bien": x,
                "apport_final_bien": b.apport_min + x,
                "credit_final": credit,
            })
        return out

    def allocate(
        self,
        bricks: list[InvestmentBrick],
        apport_disponible: float,
        target_cf: float,
        tolerance: float = 100.0,
        use_full_capital: bool = False,
    ) -> tuple[bool, list[dict[str, Any]], float, float]:
        """Allocate capital to meet the cash-flow target (pre-tax estimate). Returns (ok, details, cf, apport).

        Convenience wrapper over ``prepare`` + ``plan`` + ``details``. Searches prepare their bricks once instead.
        A combination whose minimum down payments exceed the budget is never ``ok``.
        """
        if not bricks:
            return True, [], 0.0, 0.0
        prepared = [self.prepare(b) for b in bricks]
        alloc = self.plan(prepared, apport_disponible, target_cf, tolerance, use_full_capital)
        log.debug("allocation_finished", final_cf=alloc.cf, success=alloc.ok, apport=alloc.apport_total)
        return alloc.ok, self.details(prepared, alloc.extra), alloc.cf, alloc.apport_total

    def stabilised_monthly_cf(self, biens: list[dict[str, Any]]) -> float:
        """Monthly cash-flow estimate for the stabilised year (year 2), before income tax.

        Year 2 = first year with every month rented (no works vacancy), with indexation applied once:
        rent x (1 + revalo), charges and maintenance x (1 + inflation), taxe foncière x (1 + TF growth).
        Rent-linked costs (management, vacancy provision, GLI) follow the rent. CFE, which the brick only
        exposes inside its total operating costs, is the residual and is indexed like charges.
        """
        g_rent = 1.0 + self.revalo_loyer_pct / 100.0
        g_chg = 1.0 + self.inflation_charges_pct / 100.0
        g_tf = 1.0 + self.tf_growth_pct / 100.0
        total = 0.0
        for b in biens:
            rent = float(_get_val(b, "loyer_mensuel_initial"))
            charges = float(_get_val(b, "charges_const_mth0"))
            tf = float(_get_val(b, "tf_const_mth0"))
            prov = float(_get_val(b, "provision_pct")) / 100.0
            rent_linked = rent * (float(_get_val(b, "frais_gestion_pct")) / 100.0 + prov
                                  + (1.0 - prov) * float(_get_val(b, "gli_pct")) / 100.0)
            maint = float(_get_val(b, "maintenance_m2_an")) * float(_get_val(b, "surface")) / 12.0
            depenses0 = float(_get_val(b, "depenses_mensuelles_hors_credit_initial"))
            cfe = max(0.0, depenses0 - charges - tf - rent_linked - maint)
            opex = rent_linked * g_rent + charges * g_chg + tf * g_tf + (maint + cfe) * g_chg
            total += rent * g_rent - opex - float(_get_val(b, "pmt_total"))
        return total

    def _accept_cf(self, cf_observe: float, cf_cible: float, tol: float) -> bool:
        return cf_accepts(cf_observe, cf_cible, tol, self.mode_cf)

    def _calc_need(self, cf_observe: float, cf_cible: float, tol: float) -> float:
        if self.mode_cf == "min" or self.mode_cf == "≥":
            return max(0.0, cf_cible - cf_observe)
        else:
            return cf_cible - cf_observe if abs(cf_cible - cf_observe) > tol else 0.0
