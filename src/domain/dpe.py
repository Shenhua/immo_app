"""DPE rental ban (T61): which year a property becomes unrentable, and the renovations that avoid it.

Source: loi Climat et Résilience 2021-1104 art. 160 (logement décent, loi 89-462 art. 6); service-public.fr and
ecologie.gouv.fr "Ce que vous devez savoir sur la location des logements mal isolés": class G is unrentable from
1 Jan 2025, F from 1 Jan 2028, E from 1 Jan 2034. Simplification: the ban is applied from that date to the whole
property (in law, the G ban only bites at new or renewed leases).

Pure functions, no streamlit: used by the simulation engine and by the UI badge.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.defaults import DPE

CLASSES = "ABCDEFG"


def _rank(dpe: str) -> int:
    return CLASSES.index(dpe.strip().upper())


def normalize_dpe(dpe: object) -> str | None:
    s = str(dpe or "").strip().upper()
    return s if len(s) == 1 and s in CLASSES else None


def ban_year(dpe: object, ban_years: tuple[tuple[str, int], ...] = DPE.ban_years) -> int | None:
    """First calendar year in which a property of this class cannot be rented (None if never)."""
    c = normalize_dpe(dpe)
    if c is None:
        return None
    years = [y for banned, y in ban_years if _rank(c) >= _rank(banned)]
    return min(years) if years else None


@dataclass(frozen=True)
class RenovationEvent:
    """One renovation: paid in ``pay_year`` (calendar), before the ban of ``ban_year``, lifting ``jumps`` classes."""

    pay_year: int
    ban_year: int
    jumps: int
    from_class: str
    to_class: str


def renovation_plan(
    dpe: object, first_year: int = DPE.first_year, ban_years: tuple[tuple[str, int], ...] = DPE.ban_years,
) -> list[RenovationEvent]:
    """Renovations needed over time: before each ban that hits the current class, go to the minimum allowed class.

    A ban already in force at purchase is handled at purchase (``pay_year`` = ``first_year``, i.e. from the down payment).
    """
    cls = normalize_dpe(dpe)
    events: list[RenovationEvent] = []
    if cls is None:
        return events
    for banned, year in sorted(ban_years, key=lambda t: t[1]):
        if _rank(cls) >= _rank(banned):
            target = CLASSES[_rank(banned) - 1]  # one class better than the banned one
            events.append(RenovationEvent(max(first_year, year - 1), year, _rank(cls) - _rank(target), cls, target))
            cls = target
    return events


def ban_warning(dpe: object, horizon_years: int, first_year: int = DPE.first_year) -> str | None:
    """Badge text if the property is hit by the ban within the horizon (calendar first_year .. first_year+H-1)."""
    c = normalize_dpe(dpe)
    y = ban_year(c)
    if c is None or y is None or y > first_year + horizon_years - 1:
        return None
    return f"DPE {c} : interdit à la location {'depuis' if y <= first_year else 'en'} {y}"


def property_ban_warning(bien: dict, horizon_years: int) -> str | None:
    """Badge for one brick dict; a brick already renovated at purchase ("Reno E→D") counts as class D."""
    renovated = float(bien.get("renovation_energetique", 0.0) or 0.0) > 0
    return ban_warning("D" if renovated else bien.get("dpe_initial"), horizon_years)
