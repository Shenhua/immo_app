"""Golden reference calculator for docs/TAX_SPEC.md (task T20).

Standalone on purpose: standard library only, no import from ``src/``. The app's
tax code is what the golden cases are meant to test, so it must not be used to
produce them.

Usage (from the repo root):
    python tests/golden/reference_calc.py           # print cases.yaml to stdout
    python tests/golden/reference_calc.py --write   # (re)write tests/golden/cases.yaml
    python tests/golden/reference_calc.py --check   # exit 1 if cases.yaml is stale

Every rule below refers to a section of docs/TAX_SPEC.md (``§x.y``). Conventions
(TAX_SPEC §1): acquisition on 1 January of year 1, sale on 31 December of year H,
H full years of holding; IR at the marginal rate (TMI) on the whole base; rates
and thresholds of 2026 frozen for the whole horizon.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Rates and thresholds (TAX_SPEC §2). Sources are cited in the spec.
# ---------------------------------------------------------------------------
PS_BIC = 0.186  # LMNP / micro-BIC income: CSG 10.6% from 2025 income (LFSS 2026 art. 12)
PS_FONCIER = 0.172  # revenus fonciers stay at CSG 9.2% (LFSS 2026 art. 12, new L136-8 IV)
PS_PV_IMMO = 0.172  # real-estate gains of individuals stay at CSG 9.2%
IR_PV_RATE = 0.19  # CGI 200 B
PFU_IR = 0.128  # CGI 200 A
PS_RCM = 0.186  # dividends / boni de liquidation (L136-7, from 1 Jan 2026)
DEFICIT_FONCIER_GLOBAL_CAP = 10_700.0  # CGI 156 I 3°
DEFICIT_CARRY_YEARS = 10  # BIC non-pro (CGI 156 I 1° ter) and fonciers (CGI 156 I 3°)
MICRO_BIC_MIN_ABATTEMENT = 305.0  # CGI 50-0
MICRO_FONCIER_ABATTEMENT = 0.30  # CGI 32
IS_REDUCED_RATE = 0.15  # CGI 219 I b
IS_REDUCED_LIMIT = 42_500.0
IS_NORMAL_RATE = 0.25  # CGI 219 I
FORFAIT_ACQ_FEES = 0.075  # CGI 150 VB II 4° / BOI-RFPI-PVI-20-10-20-20
FORFAIT_WORKS = 0.15  # CGI 150 VB II 4°, only when held > 5 years
CFE_EXEMPT_RECEIPTS = 5_000.0  # CGI 1647 D I (no minimum contribution below this)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class CaseInput:
    id: str
    description: str
    regime: str  # lmnp_reel | micro_bic | micro_foncier | foncier_reel | sci_is
    price: float
    notary_fees: float
    works: float
    furniture: float
    land_ratio_pct: float
    loan_amount: float
    loan_rate_pct: float
    loan_years: int
    loan_insurance_pct: float  # % of initial principal per year
    rent_monthly: float  # rent charged when let (before vacancy), excluding recoverable charges
    copro_charges_annual: float  # non-recoverable co-ownership charges + PNO insurance
    property_tax_annual: float  # taxe foncière
    vacancy_pct: float
    management_pct: float  # % of collected rent
    accounting_fees_annual: float
    cfe_annual: float  # CFE amount when due (furnished only, TAX_SPEC §2.8)
    rent_indexation_pct: float
    charges_indexation_pct: float  # applies to copro, TF, accounting, CFE
    appreciation_pct: float
    tmi_pct: float
    horizon_years: int
    sale_fees_pct: float  # borne by the seller, deducted from the sale price
    furnished: bool
    amort_building_years: int = 30
    amort_works_years: int = 15
    amort_furniture_years: int = 10
    apply_ira: bool = True
    residence_services: bool = False
    micro_bic_abattement_pct: float = 50.0
    sci_distribute_at_exit: bool = True
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loan (monthly French annuity, insurance on initial principal)
# ---------------------------------------------------------------------------
def loan_years_table(principal: float, rate_pct: float, years: int, ins_pct: float, horizon: int) -> list[dict]:
    n = years * 12
    r = rate_pct / 100 / 12
    pmt = principal * r / (1 - (1 + r) ** -n) if r > 0 else principal / n
    ins_m = principal * ins_pct / 100 / 12
    bal = principal
    rows = []
    for y in range(1, horizon + 1):
        interest = principal_paid = ins = 0.0
        for m in range((y - 1) * 12, y * 12):
            if m >= n:
                break
            i = bal * r
            interest += i
            principal_paid += pmt - i
            ins += ins_m
            bal = max(0.0, bal - (pmt - i))
        rows.append(
            {
                "interest": interest,
                "principal": principal_paid,
                "insurance": ins,
                "debt_service": interest + principal_paid + ins,
                "crd_end": bal,
            }
        )
    return rows


def ira(inp: CaseInput, crd: float) -> float:
    """Early-repayment penalty (Code de la consommation R313-25): min(6 months interest, 3% CRD)."""
    if not inp.apply_ira or inp.horizon_years >= inp.loan_years or crd <= 0:
        return 0.0
    return min(0.03 * crd, 6 * inp.loan_rate_pct / 100 / 12 * crd)


# ---------------------------------------------------------------------------
# Exit rules for individuals (TAX_SPEC §3)
# ---------------------------------------------------------------------------
def abattement_ir(h: int) -> float:
    if h <= 5:
        return 0.0
    if h >= 22:
        return 1.0
    return 0.06 * (h - 5)


def abattement_ps(h: int) -> float:
    if h <= 5:
        return 0.0
    if h >= 30:
        return 1.0
    a = 0.0165 * (min(h, 21) - 5)
    if h >= 22:
        a += 0.016
    if h >= 23:
        a += 0.09 * (h - 22)
    return min(1.0, a)


def surtax(pv: float) -> float:
    """CGI 1609 nonies G, BOI-RFPI-TPVIE-20 (table with smoothing)."""
    if pv <= 50_000:
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


def is_tax(base: float) -> float:
    base = max(0.0, base)
    return IS_REDUCED_RATE * min(base, IS_REDUCED_LIMIT) + IS_NORMAL_RATE * max(0.0, base - IS_REDUCED_LIMIT)


# ---------------------------------------------------------------------------
# Deficit pool with 10-year expiry (FIFO)
# ---------------------------------------------------------------------------
class DeficitPool:
    def __init__(self, life: int | None):
        self.life = life
        self.items: list[list[float]] = []  # [year_created, amount]

    def add(self, year: int, amount: float) -> None:
        if amount > 0:
            self.items.append([year, amount])

    def expire(self, year: int) -> float:
        """Drop deficits that can no longer be used in ``year``; return amount lost."""
        if self.life is None:
            return 0.0
        lost = sum(a for y0, a in self.items if year > y0 + self.life)
        self.items = [it for it in self.items if year <= it[0] + self.life]
        return lost

    def use(self, income: float) -> float:
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
        return sum(a for _, a in self.items)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def irr(flows: list[float]) -> float | None:
    def npv(rate: float) -> float:
        return sum(f / (1 + rate) ** t for t, f in enumerate(flows))

    lo, hi = -0.99, 1.0
    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def simulate(inp: CaseInput) -> dict:
    H = inp.horizon_years
    tmi = inp.tmi_pct / 100
    land = inp.land_ratio_pct / 100
    loan = loan_years_table(inp.loan_amount, inp.loan_rate_pct, inp.loan_years, inp.loan_insurance_pct, H)
    apport = inp.price + inp.notary_fees + inp.works + inp.furniture - inp.loan_amount

    # Amortizable bases (TAX_SPEC §4.1): notary fees capitalized pro rata (land share not amortizable).
    base_building = (inp.price + inp.notary_fees) * (1 - land)
    amort_b = base_building / inp.amort_building_years
    amort_w = inp.works / inp.amort_works_years if inp.works else 0.0
    amort_f = inp.furniture / inp.amort_furniture_years if inp.furniture else 0.0

    bic_deficits = DeficitPool(DEFICIT_CARRY_YEARS)
    foncier_deficits = DeficitPool(DEFICIT_CARRY_YEARS)
    is_losses = DeficitPool(None)  # CGI 209 I: indefinite carryforward
    ard_imm = ard_mob = 0.0
    cum_ded_imm = cum_ded_mob = 0.0  # amortization actually deducted (LMNP) / booked (IS)
    global_imputed_by_year: dict[int, float] = {}

    years = []
    for y in range(1, H + 1):
        g_r = (1 + inp.rent_indexation_pct / 100) ** (y - 1)
        g_c = (1 + inp.charges_indexation_pct / 100) ** (y - 1)
        rent = inp.rent_monthly * 12 * g_r * (1 - inp.vacancy_pct / 100)
        mgmt = rent * inp.management_pct / 100
        copro = inp.copro_charges_annual * g_c
        tf = inp.property_tax_annual * g_c
        acct = inp.accounting_fees_annual * g_c
        cfe = inp.cfe_annual * g_c if (inp.furnished and y >= 2 and rent > CFE_EXEMPT_RECEIPTS) else 0.0
        ln = loan[y - 1]
        interest, ins = ln["interest"], ln["insurance"]
        cash_pre_tax = rent - mgmt - copro - tf - acct - cfe - ln["debt_service"]

        row: dict = {
            "rent": rent,
            "operating_charges": mgmt + copro + tf + acct + cfe,
            "interest_and_insurance": interest + ins,
            "cash_flow_before_tax": cash_pre_tax,
        }
        ir = ps = 0.0
        base = 0.0

        if inp.regime == "lmnp_reel":
            # TAX_SPEC §4.2. Amortization first (CGI 39 C II), then prior BIC deficits.
            bic_deficits.expire(y)
            R = rent - mgmt - copro - tf - acct - cfe - interest - ins
            cur_imm = (amort_b if y <= inp.amort_building_years else 0.0) + (
                amort_w if y <= inp.amort_works_years else 0.0
            )
            cur_mob = amort_f if y <= inp.amort_furniture_years else 0.0
            pool_imm, pool_mob = cur_imm + ard_imm, cur_mob + ard_mob
            pool = pool_imm + pool_mob
            if R <= 0:
                bic_deficits.add(y, -R)
                ded = 0.0
                benefice = 0.0
            else:
                ded = min(R, pool)
                benefice = R - ded
            share_imm = pool_imm / pool if pool > 0 else 0.0
            ded_imm, ded_mob = ded * share_imm, ded * (1 - share_imm)
            ard_imm, ard_mob = pool_imm - ded_imm, pool_mob - ded_mob
            cum_ded_imm += ded_imm
            cum_ded_mob += ded_mob
            used = bic_deficits.use(benefice) if benefice > 0 else 0.0
            base = benefice - used
            ir, ps = base * tmi, base * PS_BIC
            row.update(
                {
                    "result_before_amortization": R,
                    "amortization_current_year": cur_imm + cur_mob,
                    "amortization_deducted": ded,
                    "ard_stock_end": ard_imm + ard_mob,
                    "bic_deficit_stock_end": bic_deficits.total,
                }
            )

        elif inp.regime == "micro_bic":
            # TAX_SPEC §4.3
            abatt = max(MICRO_BIC_MIN_ABATTEMENT, rent * inp.micro_bic_abattement_pct / 100)
            base = max(0.0, rent - abatt)
            ir, ps = base * tmi, base * PS_BIC

        elif inp.regime == "micro_foncier":
            # TAX_SPEC §4.4
            base = rent * (1 - MICRO_FONCIER_ABATTEMENT)
            ir, ps = base * tmi, base * PS_FONCIER

        elif inp.regime == "foncier_reel":
            # TAX_SPEC §4.5 (CGI 156 I 3°). Works deductible in year 1 (paid in year 1).
            foncier_deficits.expire(y)
            works_ded = inp.works if y == 1 else 0.0
            loan_cost = interest + ins
            C = mgmt + copro + tf + acct + works_ded
            net = rent - loan_cost - C
            imputed = 0.0
            if net < 0:
                D = -net
                G = min(D, C)  # rents absorb interest first
                imputed = min(G, DEFICIT_FONCIER_GLOBAL_CAP)
                foncier_deficits.add(y, D - imputed)
                base = 0.0
            else:
                base = net - foncier_deficits.use(net)
            global_imputed_by_year[y] = imputed
            ir = base * tmi - imputed * tmi
            ps = base * PS_FONCIER
            row.update(
                {
                    "net_foncier_result": net,
                    "deficit_imputed_on_global_income": imputed,
                    "foncier_deficit_stock_end": foncier_deficits.total,
                }
            )

        elif inp.regime == "sci_is":
            # TAX_SPEC §4.6
            cur = (amort_b if y <= inp.amort_building_years else 0.0) + (amort_w if y <= inp.amort_works_years else 0.0)
            cur_mob = amort_f if y <= inp.amort_furniture_years else 0.0
            cum_ded_imm += cur
            cum_ded_mob += cur_mob
            res = rent - mgmt - copro - tf - acct - cfe - interest - ins - cur - cur_mob
            if res < 0:
                is_losses.add(y, -res)
                base = 0.0
            else:
                base = res - is_losses.use(res)
            ir = is_tax(base)  # reported as "is" below
            row.update(
                {
                    "accounting_result": res,
                    "amortization_booked": cur + cur_mob,
                    "is_loss_stock_end": is_losses.total,
                }
            )
        else:
            raise ValueError(inp.regime)

        tax = ir + ps
        row.update(
            {
                "taxable_base": base,
                ("is" if inp.regime == "sci_is" else "ir"): ir,
                "ps": ps,
                "tax": tax,
                "cash_flow_after_tax": cash_pre_tax - tax,
            }
        )
        if inp.regime == "sci_is":
            row.pop("ps")
        years.append(row)

    # ----------------------------------------------------------------- exit
    sale_price = inp.price * (1 + inp.appreciation_pct / 100) ** H
    sale_fees = sale_price * inp.sale_fees_pct / 100
    crd = loan[H - 1]["crd_end"]
    ira_amt = ira(inp, crd)
    ex: dict = {"sale_price": sale_price, "sale_fees": sale_fees, "crd": crd, "ira": ira_amt}

    if inp.regime == "sci_is":
        vnc = inp.price + inp.notary_fees + inp.works + inp.furniture - cum_ded_imm - cum_ded_mob
        pv = sale_price - sale_fees - vnc
        base_H = years[-1]["taxable_base"]
        # Gain and the last year's result fall in the same exercise; remaining losses offset the gain.
        # A capital loss reduces the last year's base (floored at 0; the company is liquidated).
        pv_taxable = pv - is_losses.use(pv) if pv > 0 else pv
        is_exit = is_tax(base_H + pv_taxable) - is_tax(base_H)
        proceeds_pre = sale_price - sale_fees - crd - ira_amt - is_exit
        cum_cf = sum(r["cash_flow_after_tax"] for r in years)
        boni = proceeds_pre + cum_cf - apport
        pfu = max(0.0, boni) * (PFU_IR + PS_RCM) if inp.sci_distribute_at_exit else 0.0
        proceeds = proceeds_pre - pfu
        ex.update(
            {
                "net_book_value": vnc,
                "cumulative_amortization": cum_ded_imm + cum_ded_mob,
                "plus_value_pro": pv,
                "plus_value_taxable_after_losses": pv_taxable,
                "is_on_exit": is_exit,
                "boni_de_liquidation": boni,
                "pfu_on_distribution": pfu,
                "net_exit_proceeds": proceeds,
            }
        )
    else:
        acq_fees = max(inp.notary_fees, FORFAIT_ACQ_FEES * inp.price)
        # Real works eligible for the majoration (TAX_SPEC §3.2): not if deducted (foncier réel).
        eligible_works = 0.0 if inp.regime == "foncier_reel" else inp.works
        works_maj = max(eligible_works, FORFAIT_WORKS * inp.price) if H > 5 else eligible_works
        addback = cum_ded_imm if (inp.regime == "lmnp_reel" and not inp.residence_services) else 0.0
        prix_acq = inp.price + acq_fees + works_maj - addback
        prix_cession = sale_price - sale_fees
        pv = max(0.0, prix_cession - prix_acq)
        a_ir, a_ps = abattement_ir(H), abattement_ps(H)
        base_ir, base_ps = pv * (1 - a_ir), pv * (1 - a_ps)
        ir_pv, ps_pv, st = base_ir * IR_PV_RATE, base_ps * PS_PV_IMMO, surtax(base_ir)
        # Déficit foncier recapture if the letting stops before 31/12 of year N+3 (TAX_SPEC §4.5).
        recapture = 0.0
        if inp.regime == "foncier_reel":
            recapture = sum(v for yy, v in global_imputed_by_year.items() if yy + 3 > H) * tmi
        proceeds = prix_cession - crd - ira_amt - ir_pv - ps_pv - st - recapture
        ex.update(
            {
                "acquisition_fees_retained": acq_fees,
                "works_majoration": works_maj,
                "amortization_addback": addback,
                "corrected_acquisition_price": prix_acq,
                "net_sale_price": prix_cession,
                "plus_value_brute": pv,
                "abattement_ir_pct": a_ir * 100,
                "abattement_ps_pct": a_ps * 100,
                "base_ir": base_ir,
                "base_ps": base_ps,
                "ir_pv": ir_pv,
                "ps_pv": ps_pv,
                "surtax": st,
                "deficit_recapture": recapture,
                "net_exit_proceeds": proceeds,
            }
        )

    cum_cf = sum(r["cash_flow_after_tax"] for r in years)
    flows = [-apport] + [r["cash_flow_after_tax"] for r in years]
    flows[-1] += ex["net_exit_proceeds"]
    rate = irr(flows)
    return {
        "apport": apport,
        "amortization_bases": {
            "building_incl_notary": base_building,
            "building_annual": amort_b,
            "works_annual": amort_w,
            "furniture_annual": amort_f,
        },
        "years": years,
        "exit": ex,
        "cumulative_cash_flow_after_tax": cum_cf,
        "wealth": ex["net_exit_proceeds"] + cum_cf - apport,
        "irr_pct": None if rate is None else rate * 100,
        "total_tax_on_income": sum(r["tax"] for r in years),
    }


# ---------------------------------------------------------------------------
# Cases (TAX_SPEC §6). Round inputs; indexations off unless stated.
# ---------------------------------------------------------------------------
LMNP_BASE = {
    "price": 200_000,
    "notary_fees": 16_000,
    "works": 10_000,
    "furniture": 10_000,
    "land_ratio_pct": 15,
    "loan_amount": 150_000,
    "loan_rate_pct": 3.5,
    "loan_years": 20,
    "loan_insurance_pct": 0.30,
    "rent_monthly": 1_150,
    "copro_charges_annual": 1_200,
    "property_tax_annual": 1_000,
    "vacancy_pct": 0,
    "management_pct": 0,
    "accounting_fees_annual": 400,
    "cfe_annual": 300,
    "rent_indexation_pct": 0,
    "charges_indexation_pct": 0,
    "appreciation_pct": 2.0,
    "tmi_pct": 30,
    "sale_fees_pct": 5,
    "furnished": True,
}


def _c(id_: str, description: str, regime: str, horizon: int, **over) -> CaseInput:
    d = dict(LMNP_BASE)
    d.update(over)
    return CaseInput(id=id_, description=description, regime=regime, horizon_years=horizon, **d)


CASES = [
    _c("C1_lmnp_reel_10y", "LMNP réel, held 10 years (sold before loan end: IRA due).", "lmnp_reel", 10),
    _c(
        "C2_lmnp_reel_20y",
        "LMNP réel, held 20 years; furniture (10y) and works (15y) amortization stop.",
        "lmnp_reel",
        20,
    ),
    _c(
        "C3_lmnp_reel_25y_indexed",
        "LMNP réel, held 25 years, with indexations on and 5% vacancy, 6% management; IR on gain fully exempt.",
        "lmnp_reel",
        25,
        vacancy_pct=5,
        management_pct=6,
        rent_indexation_pct=1.5,
        charges_indexation_pct=2.0,
    ),
    _c(
        "C4_micro_bic_15y",
        "Same furnished flat as C1, micro-BIC long-term letting (50%), held 15 years.",
        "micro_bic",
        15,
        accounting_fees_annual=0,
    ),
    _c(
        "C5_micro_foncier_20y",
        "Unfurnished flat, micro-foncier (30%), held 20 years.",
        "micro_foncier",
        20,
        price=150_000,
        notary_fees=12_000,
        works=0,
        furniture=0,
        loan_amount=150_000,
        rent_monthly=700,
        copro_charges_annual=900,
        property_tax_annual=800,
        accounting_fees_annual=0,
        cfe_annual=0,
        furnished=False,
    ),
    _c(
        "C6_foncier_reel_deficit_15y",
        "Unfurnished flat with 30k€ non-energy works in year 1, régime réel: déficit foncier "
        "(10,700€ on global income, rest carried forward 10 years), held 15 years.",
        "foncier_reel",
        15,
        price=150_000,
        notary_fees=12_000,
        works=30_000,
        furniture=0,
        loan_amount=180_000,
        rent_monthly=850,
        management_pct=6,
        copro_charges_annual=900,
        property_tax_annual=800,
        accounting_fees_annual=0,
        cfe_annual=0,
        furnished=False,
    ),
    _c(
        "C7_sci_is_20y",
        "SCI at IS, unfurnished, held 20 years, sale of the property then liquidation with PFU on the boni.",
        "sci_is",
        20,
        works=0,
        furniture=0,
        loan_amount=150_000,
        rent_monthly=1_200,
        accounting_fees_annual=1_000,
        cfe_annual=0,
        furnished=False,
    ),
    _c(
        "C8_lmnp_reel_20y_land30",
        "C2 with a 30% land share: lower building amortization, lower add-back, more tax.",
        "lmnp_reel",
        20,
        land_ratio_pct=30,
    ),
    _c(
        "C9_lmnp_reel_12y_surtax",
        "Larger LMNP réel flat, held 12 years, 3% appreciation: gain above 50k€ after IR allowance (surtax).",
        "lmnp_reel",
        12,
        price=400_000,
        notary_fees=32_000,
        works=20_000,
        furniture=15_000,
        loan_amount=420_000,
        rent_monthly=1_900,
        copro_charges_annual=2_000,
        property_tax_annual=1_800,
        appreciation_pct=3.0,
        tmi_pct=41,
    ),
]

# Hand-check trail notes per case (appended as YAML comments).
TRAIL_EXTRA = {
    "C6_foncier_reel_deficit_15y": [
        "Year 1: D = -(rent - (interest+ins) - (copro+TF+works)); G = min(D, C) = non-interest part;",
        "imputed on global income = min(G, 10,700); remainder carried forward against fonciers for 10 years.",
        "IR saving year 1 = 10,700 x TMI 30% = 3,210 (no PS saving). Exit: works were deducted, so only",
        "the 15% forfait is added to the acquisition price.",
    ],
    "C7_sci_is_20y": [
        "IS: 15% up to 42,500, 25% above (CGI 219). Exit gain = net sale price - net book value",
        "(cost incl. notary - cumulative amortization). IS on exit = IS(base_H + PV) - IS(base_H).",
        "Distribution at exit: PFU 31.4% (12.8% IR + 18.6% PS) on boni = proceeds after IS + sum(CF) - apport.",
    ],
}


# ---------------------------------------------------------------------------
# YAML writer (no PyYAML dependency)
# ---------------------------------------------------------------------------
def _num(x: float | None) -> str:
    if x is None:
        return "null"
    v = round(float(x), 2)
    if v == 0:
        v = 0.0
    return f"{v:.2f}"


def _scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v) if isinstance(v, int) else (str(int(v)) if float(v).is_integer() else str(v))
    return '"' + str(v).replace('"', '\\"') + '"'


INPUT_FIELDS = [
    "price", "notary_fees", "works", "furniture", "land_ratio_pct",
    "loan_amount", "loan_rate_pct", "loan_years", "loan_insurance_pct",
    "rent_monthly", "copro_charges_annual", "property_tax_annual", "vacancy_pct", "management_pct",
    "accounting_fees_annual", "cfe_annual", "rent_indexation_pct", "charges_indexation_pct",
    "appreciation_pct", "tmi_pct", "horizon_years", "sale_fees_pct", "furnished",
    "amort_building_years", "amort_works_years", "amort_furniture_years",
    "apply_ira", "residence_services", "micro_bic_abattement_pct", "sci_distribute_at_exit",
]  # fmt: skip


def report_years(h: int) -> list[int]:
    return sorted({1, 2, min(10, h), h})


def trail(inp: CaseInput, res: dict) -> list[str]:
    ab = res["amortization_bases"]
    ex = res["exit"]
    L = [
        f"apport = {inp.price:,.0f} + {inp.notary_fees:,.0f} + {inp.works:,.0f} + {inp.furniture:,.0f}"
        f" - {inp.loan_amount:,.0f} = {res['apport']:,.2f}",
        f"loan: monthly annuity {inp.loan_rate_pct}% / {inp.loan_years}y, insurance {inp.loan_insurance_pct}%"
        f" x {inp.loan_amount:,.0f} / 12 per month",
    ]
    if inp.regime in ("lmnp_reel", "sci_is"):
        L.append(
            f"amortization: building ({inp.price:,.0f} + {inp.notary_fees:,.0f}) x (1 - {inp.land_ratio_pct}%)"
            f" = {ab['building_incl_notary']:,.2f} / {inp.amort_building_years} = {ab['building_annual']:,.2f}/y;"
            f" works {inp.works:,.0f} / {inp.amort_works_years} = {ab['works_annual']:,.2f}/y;"
            f" furniture {inp.furniture:,.0f} / {inp.amort_furniture_years} = {ab['furniture_annual']:,.2f}/y"
        )
    y1 = res["years"][0]
    L.append(
        f"year 1: rent {y1['rent']:,.2f}; operating charges {y1['operating_charges']:,.2f};"
        f" interest+insurance {y1['interest_and_insurance']:,.2f}"
    )
    if inp.regime == "lmnp_reel":
        L.append(
            f"year 1: R = {y1['result_before_amortization']:,.2f}; amortization {y1['amortization_current_year']:,.2f}"
            f" -> deducted min(R, pool) = {y1['amortization_deducted']:,.2f}; ARD carried {y1['ard_stock_end']:,.2f}"
        )
    if inp.regime == "micro_bic":
        L.append(f"base = rent x (1 - {inp.micro_bic_abattement_pct}%); tax = base x (TMI {inp.tmi_pct}% + PS 18.6%)")
    if inp.regime == "micro_foncier":
        L.append(f"base = rent x 70%; tax = base x (TMI {inp.tmi_pct}% + PS 17.2%)")
    L.append(
        f"exit: sale price {inp.price:,.0f} x (1 + {inp.appreciation_pct}%)^{inp.horizon_years}"
        f" = {ex['sale_price']:,.2f}; sale fees {inp.sale_fees_pct}% = {ex['sale_fees']:,.2f};"
        f" CRD {ex['crd']:,.2f}; IRA {ex['ira']:,.2f}"
    )
    if inp.regime != "sci_is":
        L.append(
            f"corrected acquisition price = {inp.price:,.0f} + fees max(real, 7.5%) {ex['acquisition_fees_retained']:,.2f}"
            f" + works {ex['works_majoration']:,.2f} - amortization add-back {ex['amortization_addback']:,.2f}"
            f" = {ex['corrected_acquisition_price']:,.2f}"
        )
        L.append(
            f"PV brute = {ex['net_sale_price']:,.2f} - {ex['corrected_acquisition_price']:,.2f} = {ex['plus_value_brute']:,.2f};"
            f" allowances IR {ex['abattement_ir_pct']:.2f}% / PS {ex['abattement_ps_pct']:.2f}%"
        )
        L.append(
            f"IR 19% x {ex['base_ir']:,.2f} = {ex['ir_pv']:,.2f}; PS 17.2% x {ex['base_ps']:,.2f} = {ex['ps_pv']:,.2f};"
            f" surtax on {ex['base_ir']:,.2f} = {ex['surtax']:,.2f}"
        )
    else:
        L.append(
            f"VNC = cost - cumulative amortization {ex['cumulative_amortization']:,.2f} = {ex['net_book_value']:,.2f};"
            f" PV = {ex['plus_value_pro']:,.2f}; IS on exit {ex['is_on_exit']:,.2f}; boni {ex['boni_de_liquidation']:,.2f};"
            f" PFU {ex['pfu_on_distribution']:,.2f}"
        )
    L.append(
        f"net exit proceeds = {ex['net_exit_proceeds']:,.2f}; cumulative CF = {res['cumulative_cash_flow_after_tax']:,.2f};"
        f" wealth = proceeds + CF - apport = {res['wealth']:,.2f}"
    )
    L.append("IRR flows: t0 = -apport; t1..tH = after-tax cash flow; tH also + net exit proceeds")
    if inp.id == "C8_lmnp_reel_20y_land30":
        twin = simulate(next(c for c in CASES if c.id == "C2_lmnp_reel_20y"))
        L.append(
            f"land-ratio check vs C2 (land 15%): building amortization {twin['amortization_bases']['building_annual']:,.2f}"
            f" -> {ab['building_annual']:,.2f}/y; total income tax {twin['total_tax_on_income']:,.2f}"
            f" -> {res['total_tax_on_income']:,.2f}; add-back {twin['exit']['amortization_addback']:,.2f}"
            f" -> {ex['amortization_addback']:,.2f}"
        )
    return L + TRAIL_EXTRA.get(inp.id, [])


def render() -> str:
    out = [
        "# Golden reference cases for the tax engine (task T20).",
        "# GENERATED by tests/golden/reference_calc.py -- do not edit by hand; edit the script and run --write.",
        "# Rules: docs/TAX_SPEC.md. Conventions: buy 1 Jan year 1, sell 31 Dec year H; IR at TMI on the whole base;",
        "# 2026 rates frozen. Money in euros, rounded to cents; compare with a 1 EUR tolerance (loan schedule rounding).",
        "# Yearly keys: taxable_base, ir (or is), ps, tax = ir + ps, cash_flow_after_tax.",
        "# Negative ir = IR saving from a déficit foncier imputed on global income.",
        "version: 1",
        "tolerance_eur: 1.0",
        "tolerance_irr_pct: 0.01",
        "cases:",
    ]
    for inp in CASES:
        res = simulate(inp)
        out.append("")
        for line in trail(inp, res):
            out.append(f"  # {line}")
        out.append(f"  - id: {inp.id}")
        out.append(f"    description: {_scalar(inp.description)}")
        out.append(f"    regime: {inp.regime}")
        out.append("    inputs:")
        for k in INPUT_FIELDS:
            out.append(f"      {k}: {_scalar(getattr(inp, k))}")
        out.append("    expected:")
        out.append(f"      apport: {_num(res['apport'])}")
        out.append("      yearly:")
        for y in report_years(inp.horizon_years):
            r = res["years"][y - 1]
            out.append(f"        {y}:")
            for k, v in r.items():
                out.append(f"          {k}: {_num(v)}")
        out.append("      exit:")
        for k, v in res["exit"].items():
            out.append(f"        {k}: {_num(v)}")
        out.append(f"      total_tax_on_income: {_num(res['total_tax_on_income'])}")
        out.append(f"      cumulative_cash_flow_after_tax: {_num(res['cumulative_cash_flow_after_tax'])}")
        out.append(f"      wealth: {_num(res['wealth'])}")
        out.append(f"      irr_pct: {_num(res['irr_pct'])}")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    target = Path(__file__).with_name("cases.yaml")
    text = render()
    if "--write" in argv:
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
        return 0
    if "--check" in argv:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != text:
            print("cases.yaml is stale: run reference_calc.py --write")
            return 1
        print("cases.yaml is up to date")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
