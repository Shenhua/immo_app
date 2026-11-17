"""HCSF bankability check (T60): debt ratio and loan duration of a strategy.

Source: HCSF decision D-HCSF-2021-7 of 29 September 2021 (in force since 1 January 2022), taken under art. L.631-2-1
of the Code monétaire et financier: debt ratio (taux d'effort) <= 35 %, insurance included; maturity <= 25 years
(27 for a primary residence with works >= 10 % of the cost; not modelled here: a rental strategy is capped at 25).
https://www.economie.gouv.fr/hcsf/mesures/mesure-relative-loctroi-de-credits-immobiliers
The 20 % flexibility margin granted to banks is out of scope.

The 70 % weighting of the new properties' gross rents in the income is bank practice, not part of the HCSF text.
Pure functions: no streamlit, no engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.defaults import HCSF


@dataclass(frozen=True)
class HcsfResult:
    taux_endettement: float  # percent
    hcsf_ok: bool
    duree_ok: bool
    nouvelles_mensualites: float  # EUR/month, insurance included
    loyers_bruts: float  # EUR/month, gross rents of the new properties
    revenu_requis: float  # household net monthly income needed for a ratio of exactly the maximum


def hcsf_check(
    details: list[dict[str, Any]],
    revenu_net_mensuel: float,
    charges_mensuelles: float = 0.0,
    *,
    taux_max_pct: float = HCSF.taux_max_pct,
    ponderation_loyers: float = HCSF.ponderation_loyers,
    duree_max_ans: int = HCSF.duree_max_ans,
) -> HcsfResult:
    """Debt ratio = (existing debts + new payments incl. insurance) / (net income + weighted gross rents).

    A strategy with no new loan needs no bank: always ok, whatever the existing debts.
    """
    pmt = sum(float(d.get("pmt_total", 0.0) or 0.0) for d in details if float(d.get("credit_final", 0.0) or 0.0) > 0)
    loans = [d for d in details if float(d.get("credit_final", 0.0) or 0.0) > 0]
    rents = sum(float(d.get("loyer_mensuel_initial", 0.0) or 0.0) for d in details)
    debts = float(charges_mensuelles) + pmt
    income = float(revenu_net_mensuel) + ponderation_loyers * rents
    ratio = 100.0 * debts / income if income > 0 else (0.0 if debts <= 0 else float("inf"))
    duree_ok = all(int(d.get("duree_credit_mois", 0) or 0) <= duree_max_ans * 12 for d in loans)
    ok = not loans or (ratio <= taux_max_pct + 1e-9 and duree_ok)
    required = max(0.0, debts / (taux_max_pct / 100.0) - ponderation_loyers * rents)
    return HcsfResult(ratio, ok, duree_ok, pmt, rents, required)
