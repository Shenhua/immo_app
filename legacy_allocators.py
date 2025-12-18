# --- allocators.py -----------------------------------------------------------


def _pmt_factor(annual_rate: float, years: int) -> float:
    """Monthly payment per € borrowed for a fixed-rate loan (€/mo per €)."""
    rm = float(annual_rate) / 12.0
    n = int(years) * 12
    if rm <= 0 or n <= 0:  # degenerate guard
        return 0.0
    return rm / (1.0 - (1.0 + rm) ** (-n))

def _marginal_cf_gain_per_euro(
    brique: dict,
    rates_by_term: dict[int, float] | None,
    insurance_annual_pct: float,
    default_rate: float = 0.035,
) -> float:
    """
    k_i = alpha_i + beta_i
    alpha_i: mortgage per € (depends on rate/term)
    beta_i:  insurance per € (annual%/12)
    """
    duree = int(brique.get("duree_pret", 0) or brique.get("duree", 0) or 0)
    if rates_by_term is not None:
        r = float(rates_by_term.get(duree, default_rate))
    else:
        r = float(brique.get("taux_annuel", default_rate))
    alpha = _pmt_factor(r, duree)                  # €/mo per €
    beta  = float(insurance_annual_pct) / 12.0     # €/mo per €
    return max(0.0, alpha + beta)

def allocate_portfolio_cf(
    briques: list[dict],
    budget_eur: float,
    rates_by_term: dict[int, float] | None = None,
    insurance_annual_pct: float = 0.003,  # 0.30%/year
    max_extra_apport_pct: float = 0.30,   # per-bien cap vs initial loan
    ltv_min: float | None = None,      # e.g. 0.55 to avoid de-levering too much
    price_key_fallback: str = "prix_total",
) -> float:
    """
    Greedy continuous-knapsack: maximize portfolio CF under an apport budget.
    Mutates each brique in-place:
      - increases `apport_final_bien`
      - decreases `capital_emprunte` (leaves a tiny remainder to avoid zero)
    Returns remaining budget (if any).
    """
    rest = float(budget_eur)
    if rest <= 1e-9 or not briques:
        return rest

    items = []
    for b in briques:
        k = _marginal_cf_gain_per_euro(b, rates_by_term, insurance_annual_pct)
        cap_init = float(b.get("capital_emprunte_initiale", b.get("capital_emprunte", 0.0)))
        cap_now  = float(b.get("capital_emprunte", 0.0))
        if cap_now <= 1e-6 or k <= 0:
            continue

        # per-bien cap (relative to initial loan)
        per_bien_cap = max_extra_apport_pct * cap_init

        # optional LTV floor
        if ltv_min is not None:
            price = float(b.get("prix_achat", b.get(price_key_fallback, 0.0)))
            if price > 0.0:
                min_cap = float(ltv_min) * price
                per_bien_cap = min(per_bien_cap, max(0.0, cap_now - min_cap))

        # cannot exceed outstanding capital (keep a small remainder)
        per_bien_cap = min(per_bien_cap, max(0.0, cap_now - 1e-2))
        if per_bien_cap > 1e-6:
            items.append({"b": b, "k": k, "cap": per_bien_cap})

    # allocate highest marginal CF first
    items.sort(key=lambda x: x["k"], reverse=True)
    for it in items:
        if rest <= 1e-9:
            break
        b   = it["b"]
        cap = float(it["cap"])
        delta = min(rest, cap)
        if delta <= 0.0:
            continue

        b["apport_final_bien"] = float(b.get("apport_final_bien", b.get("apport_min", 0.0) or 0.0)) + delta
        b["capital_emprunte"]  = max(1e-2, float(b.get("capital_emprunte", 0.0)) - delta)
        rest -= delta

    return rest
# ---------------------------------------------------------------------------
