"""Scoring functions for properties and strategies.

Contains DPE scoring, qualitative scoring, and balanced scoring logic.
"""

from __future__ import annotations

from typing import Any

# Default weights for qualitative scoring
DEFAULT_QUAL_WEIGHTS = {
    "tension": 0.25,       # Market tension (high = low vacancy)
    "transport": 0.15,     # Transport accessibility
    "dpe": 0.15,           # Energy rating
    "encadrement": 0.15,   # Rent cap margin
    "vacance": 0.10,       # Vacancy proxy (uses tension)
    "travaux": 0.10,       # Renovation ratio (inverse)
    "liquidite": 0.10,     # Resale liquidity
}

# DPE rating to score mapping
DPE_SCORES = {
    "A": 1.0,
    "B": 0.9,
    "C": 0.8,
    "D": 0.6,
    "E": 0.3,
    "F": 0.0,
    "G": 0.0,
    "ND": 0.6,
}


def calculate_dpe_score(dpe_letter: str) -> float:
    """Convert DPE rating to normalized score (0-1).

    Args:
        dpe_letter: DPE rating letter (A-G or ND)

    Returns:
        Score from 0.0 (worst) to 1.0 (best)
    """
    letter = (dpe_letter or "D").upper()
    return DPE_SCORES.get(letter, 0.6)


def calculate_property_qualitative_score(
    source: dict[str, Any],
    *,
    loyer_m2: float | None = None,
    loyer_m2_max: float | None = None,
    prix_achat: float = 0.0,
    travaux: float = 0.0,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Calculate qualitative score for a single property.

    Args:
        source: Property data dictionary
        loyer_m2: Rent per m² (override)
        loyer_m2_max: Rent cap per m² (override)
        prix_achat: Purchase price
        travaux: Renovation costs
        weights: Custom weights (defaults to DEFAULT_QUAL_WEIGHTS)

    Returns:
        Tuple of (score 0-100, feature breakdown dict)
    """
    w = weights or DEFAULT_QUAL_WEIGHTS

    # Extract features
    tension = float(source.get("indice_tension", source.get("tension_locative_score_norm", 0.5)))
    transport = float(source.get("transport_score", 0.5))
    dpe = calculate_dpe_score(source.get("dpe_initial", source.get("dpe", "D")))

    # Rent cap margin (higher is better if under cap)
    loyer = loyer_m2 if loyer_m2 is not None else float(source.get("loyer_m2", 0.0))
    cap = loyer_m2_max if loyer_m2_max is not None else source.get("loyer_m2_max")

    if source.get("soumis_encadrement") and cap:
        try:
            cap_val = float(cap)
            enc_marge = max(0.0, min(1.0, (cap_val - loyer) / max(1e-6, cap_val)))
        except (TypeError, ValueError):
            enc_marge = 0.6
    else:
        enc_marge = 0.6  # Neutral if not regulated

    # Vacancy proxy: Use actual vacancy_pct if available, else derive from tension
    # High tension = low vacancy = good (invert for positive scoring)
    vacancy_pct = source.get("vacance_pct", source.get("vacancy_pct"))
    if vacancy_pct is not None:
        # Normalize: 0% vacancy = 1.0 score, 10% vacancy = 0.0 score
        vac_score = max(0.0, min(1.0, 1.0 - float(vacancy_pct) / 10.0))
    else:
        # Fall back to tension as proxy (high tension = low vacancy)
        vac_score = tension

    # Renovation ratio (lower is better)
    ratio_trav = min(1.0, float(travaux or 0.0) / max(1e-6, float(prix_achat or 0.0)))
    trav_pos = 1.0 - ratio_trav

    # Liquidity
    liq = float(source.get("liquidite_score", 0.5))

    # Build feature dict
    features = {
        "tension": tension,  # Market tension (0-1, higher = tighter market)
        "transport": transport,
        "dpe": dpe,
        "encadrement": enc_marge,
        "vacance": vac_score,  # Based on actual vacancy or tension proxy
        "travaux": trav_pos,
        "liquidite": liq,
    }

    # Weighted sum
    score_01 = sum(
        w.get(k, 0.0) * max(0.0, min(1.0, features.get(k, 0.5)))
        for k in w
    )

    return round(score_01 * 100.0, 1), features


def calculate_qualitative_score(strategy: dict[str, Any]) -> float:
    """Calculate aggregate qualitative score for a strategy.

    Weights by property cost (cout_total).

    Args:
        strategy: Strategy dict with 'details' list

    Returns:
        Score 0-100
    """
    details = strategy.get("details") or []
    if not details:
        return 50.0

    total_cost = sum(d.get("cout_total", 0.0) for d in details) or 1.0
    aggregate = 0.0

    for d in details:
        # Use precomputed score if available
        qs = d.get("qual_score_bien")
        if qs is None:
            qs, _ = calculate_property_qualitative_score(
                d,
                loyer_m2=d.get("loyer_m2"),
                loyer_m2_max=d.get("loyer_m2_max"),
                prix_achat=d.get("prix_achat_bien", d.get("cout_total", 0.0)),
                travaux=d.get("travaux", 0.0),
            )
        aggregate += qs * (d.get("cout_total", 0.0) / total_cost)

    return round(aggregate, 1)


def calculate_balanced_score(
    tri: float | None,
    enrichissement_net: float,
    dscr: float,
    qual_score: float,
    *,
    qualite_weight: float = 0.25,
) -> float:
    """Calculate balanced strategy score.

    Combines financial metrics with qualitative score.

    Args:
        tri: IRR as percent (e.g., 8.0 for 8%)
        enrichissement_net: Net wealth creation in €
        dscr: Debt service coverage ratio
        qual_score: Qualitative score 0-100
        qualite_weight: Weight for qualitative component (0-1)

    Returns:
        Score 0-100
    """
    # Financial component weights (sum to 1 - qualite_weight)
    finance_weight = 1.0 - qualite_weight

    # Normalize TRI (0-20% range mapped to 0-100)
    # TRI is already in percent (e.g., 8.0 for 8%)
    tri_score = min(100.0, max(0.0, (tri or 0.0) / 20.0 * 100.0))

    # Normalize enrichissement (0-500k range mapped to 0-100)
    enrich_score = min(100.0, max(0.0, enrichissement_net / 500_000.0 * 100.0))

    # Normalize DSCR (0.8-2.0 range mapped to 0-100)
    dscr_score = min(100.0, max(0.0, (dscr - 0.8) / 1.2 * 100.0))

    # Weighted combination
    finance_score = (tri_score * 0.4 + enrich_score * 0.4 + dscr_score * 0.2)

    return round(finance_score * finance_weight + qual_score * qualite_weight, 1)
