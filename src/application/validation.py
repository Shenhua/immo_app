"""Cross-field validation of the search inputs (T43). Pure functions, no streamlit.

The UI calls :func:`validate_search_inputs` before every search and blocks the run when it returns
errors. Messages are in French because they are shown as is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.core.defaults import FINANCING, SEARCH

MAX_RATE_PCT = 15.0  # sanity ceiling for a loan rate typed in the sidebar


def validate_durations(taux_credits: Mapping[int, float] | None) -> list[str]:
    """Loan durations must be 1 to ``FINANCING.max_durations`` whole years in 10-25, with a sane rate each."""
    errors: list[str] = []
    if not taux_credits:
        return ["Durées de prêt : choisissez au moins une durée."]
    if len(taux_credits) > FINANCING.max_durations:
        errors.append(f"Durées de prêt : {FINANCING.max_durations} durées au maximum.")
    for years, rate in taux_credits.items():
        if not FINANCING.duree_min_ans <= int(years) <= FINANCING.duree_max_ans:
            errors.append(
                f"Durée de prêt {years} ans hors plage ({FINANCING.duree_min_ans}-{FINANCING.duree_max_ans} ans)."
            )
        if not 0.0 < float(rate) <= MAX_RATE_PCT:
            errors.append(f"Taux du prêt {years} ans : {rate} % hors plage (0-{MAX_RATE_PCT:g} %).")
    return errors


def validate_search_inputs(
    *,
    apport: float,
    tolerance: float,
    mode_cf: str = "min",
    max_properties: int = SEARCH.max_properties_classique,
    min_brick_apport: float | None = None,
    budget: float | None = None,
    taux_credits: Mapping[int, float] | None = None,
    n_archetypes: int | None = None,
) -> list[str]:
    """Return the blocking errors of a search request (empty list = valid).

    Args:
        apport: Down payment available (EUR).
        tolerance: Cash-flow tolerance (EUR/month). In "min" mode it has no effect but must still be >= 0.
        mode_cf: "min" or "target" (kept for callers; the tolerance rule is the same in both modes).
        max_properties: Maximum number of properties in a portfolio.
        min_brick_apport: Minimum down payment of the cheapest brick, None when unknown.
        budget: Total budget (EUR), None when the user gave none.
        taux_credits: Loan rates by duration; validated when given.
        n_archetypes: Number of properties left after the filters; validated when given.
    """
    errors: list[str] = []
    if apport < 0:
        errors.append("Apport : doit être positif ou nul.")
    if tolerance < 0:
        errors.append("Tolérance : doit être supérieure ou égale à 0.")
    if not SEARCH.max_properties_min <= int(max_properties) <= SEARCH.max_properties_max:
        errors.append(f"Nombre max de biens : entre {SEARCH.max_properties_min} et {SEARCH.max_properties_max}.")
    if n_archetypes is not None and n_archetypes <= 0:
        errors.append("Aucun bien ne correspond aux filtres (villes / types de bien).")
    if min_brick_apport is not None and apport < min_brick_apport:
        errors.append(
            f"Apport : {apport:,.0f} € ne couvre pas l'apport minimum du bien le moins cher "
            f"({min_brick_apport:,.0f} € : frais de notaire, hors travaux et mobilier financés)."
        )
    if budget is not None and budget < apport:
        errors.append(f"Budget : {budget:,.0f} € est inférieur à l'apport ({apport:,.0f} €).")
    if taux_credits is not None:
        errors.extend(validate_durations(taux_credits))
    return errors


def cheapest_brick_apport(bricks: Iterable[object]) -> float | None:
    """Smallest ``apport_min`` among bricks (None for an empty list)."""
    values = [float(getattr(b, "apport_min", 0.0)) for b in bricks]
    return min(values) if values else None
