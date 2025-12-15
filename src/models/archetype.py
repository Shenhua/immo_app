"""Archetype data models.

An archetype represents a template for a real estate property with
its financial characteristics and market data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


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

    # Status
    meuble: bool = Field(default=False, description="Is furnished")
    soumis_encadrement: bool = Field(default=False, description="Subject to rent control")
    dpe_initial: str = Field(default="D", description="Initial DPE rating (A-G)")

    # Market indicators (normalized 0-1)
    tension_locative_score_norm: float = Field(default=0.5, ge=0, le=1, description="Market tension score")
    transport_score: float = Field(default=0.5, ge=0, le=1, description="Transport accessibility score")
    liquidite_score: float = Field(default=0.5, ge=0, le=1, description="Resale liquidity score")
    indice_tension: float = Field(default=0.5, ge=0, le=1, description="Rental demand tension index")

    # Timing
    delai_vente_j_median: int = Field(default=60, ge=0, description="Median days to sell")

    model_config = {
        "extra": "allow",  # Allow additional fields from JSON
    }

    @field_validator("dpe_initial")
    @classmethod
    def validate_dpe(cls, v: str) -> str:
        """Normalize DPE rating to uppercase."""
        if v:
            v = v.upper()
        valid_ratings = {"A", "B", "C", "D", "E", "F", "G", "ND"}
        return v if v in valid_ratings else "D"

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
