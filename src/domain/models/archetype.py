"""Archetype data models.

An archetype represents a template for a real estate property with
its financial characteristics and market data.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.core.defaults import TAX


def is_furnished(item: Any) -> bool:
    """Single furnished/unfurnished rule for archetypes, bricks and property dicts.

    An explicit ``meuble`` wins; otherwise ``mode_loyer`` starting with "nu" means unfurnished;
    anything else (including no information) counts as furnished. ``item`` is a mapping or an object
    with ``meuble`` / ``mode_loyer`` attributes.
    """
    if isinstance(item, dict):
        meuble, mode = item.get("meuble"), item.get("mode_loyer")
    else:
        meuble, mode = getattr(item, "meuble", None), getattr(item, "mode_loyer", None)
    if meuble is not None:
        return bool(meuble)
    if mode:
        return not str(mode).startswith("nu")
    return True


class ArchetypeV2(BaseModel):
    """Validated archetype model for v2 JSON data format.

    Represents a property template with pricing, rental income,
    and qualitative market indicators.
    """

    # Identity
    nom: str = Field(..., description="Unique identifier/name for the archetype")
    ville: str = Field(..., description="City name")
    mode_loyer: str = Field(..., description="Rental mode (meuble_classique, meuble_boost, nu)")

    # Physical characteristics
    surface: float = Field(..., gt=0, description="Surface area in m²")

    # Pricing
    prix_m2: float = Field(..., ge=0, description="Purchase price per m² in €")
    loyer_m2: float = Field(..., ge=0, description="Monthly rent per m² in €")
    loyer_m2_max: float | None = Field(None, description="Maximum rent cap per m² if regulated")

    # Costs
    charges_m2_an: float = Field(default=0.0, ge=0, description="Annual charges per m²")
    taxe_fonciere_m2_an: float = Field(default=0.0, ge=0, description="Property tax per m² per year")

    # Renovation
    budget_travaux: float = Field(default=0.0, ge=0, description="Renovation budget in €")
    renovation_energetique_cout: float = Field(default=0.0, ge=0, description="Energy renovation cost in €")
    valeur_mobilier: float = Field(default=0.0, ge=0, description="Furniture value in €")

    # Tax: land share of the price, not amortizable (TAX_SPEC 4.1)
    ratio_terrain_pct: float = Field(
        default=TAX.ratio_terrain_pct, ge=0, le=100, description="Land share of the price in % (not amortizable)"
    )

    # Tax: residence-services exemption from the LMNP amortization add-back at sale (TAX_SPEC 3.4)
    residence_services: bool = Field(
        default=False, description="Student/senior/disabled residence with services (no amortization add-back)"
    )

    # Status
    meuble: bool | None = Field(
        default=None, description="Is furnished (explicit value wins; default derived from mode_loyer: nu* = unfurnished)"
    )
    soumis_encadrement: bool = Field(default=False, description="Subject to rent control")
    dpe_initial: str = Field(default="D", description="Initial DPE rating (A-G)")

    # Market indicators (normalized 0-1)
    tension_locative_score_norm: float = Field(default=0.5, ge=0, le=1, description="Market tension score")
    transport_score: float = Field(default=0.5, ge=0, le=1, description="Transport accessibility score")
    liquidite_score: float = Field(default=0.5, ge=0, le=1, description="Resale liquidity score")

    # Timing
    delai_vente_j_median: int = Field(default=60, ge=0, description="Median days to sell")

    # Additional fields from JSON data
    vacance_pct: float = Field(default=2.0, ge=0, le=100, description="Vacancy rate percentage")
    transport_modes: list[str] = Field(default_factory=list, description="Available transport modes")
    mode_loyer_label: str = Field(default="", description="Human-readable rental mode label")
    frais_gestion_pct: float | None = Field(None, ge=0, le=100, description="Management fees % (colocation)")
    facteurs_qualitatifs: dict[str, str] = Field(default_factory=dict, description="Qualitative factors")

    model_config = {
        "extra": "forbid",  # Reject unexpected fields (now that schema matches data)
    }

    @field_validator("dpe_initial")
    @classmethod
    def validate_dpe(cls, v: str) -> str:
        """Normalize DPE rating to uppercase."""
        if v:
            v = v.upper()
        valid_ratings = {"A", "B", "C", "D", "E", "F", "G", "ND"}
        return v if v in valid_ratings else "D"

    @model_validator(mode="after")
    def _default_meuble(self) -> ArchetypeV2:
        """Fill ``meuble`` from ``mode_loyer`` when not given (same rule as ``is_furnished``)."""
        if self.meuble is None:
            self.meuble = is_furnished({"mode_loyer": self.mode_loyer})
        return self

    @property
    def prix_achat_total(self) -> float:
        """Calculate total purchase price."""
        return self.surface * self.prix_m2

    @property
    def loyer_mensuel(self) -> float:
        """Calculate monthly rent."""
        return self.surface * self.loyer_m2

    @property
    def charges_annuelles(self) -> float:
        """Calculate annual charges."""
        return self.surface * self.charges_m2_an


# Alias for backwards compatibility
Archetype = ArchetypeV2
