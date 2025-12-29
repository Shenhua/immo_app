"""Investment brick data model.

A brick represents a specific property investment opportunity derived
from an archetype, with calculated financial details.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, computed_field


class InvestmentBrick(BaseModel):
    """Investment brick with full financial breakdown.

    Created from an archetype with specific financing parameters applied.
    """

    # Identity (from archetype)
    nom: str = Field(..., description="Property identifier")
    ville: str = Field(default="Paris", description="City")
    mode_loyer: str = Field(default="nu", description="Rental mode")
    surface: float = Field(default=20.0, gt=0, description="Surface in m²")

    # Purchase
    prix_achat_bien: float = Field(..., ge=0, description="Purchase price in €")
    frais_notaire: float = Field(default=0.0, ge=0, description="Notary fees in €")
    travaux: float = Field(default=0.0, ge=0, description="Renovation costs in €")
    renovation_energetique: float = Field(default=0.0, ge=0, description="Energy renovation in €")
    mobilier: float = Field(default=0.0, ge=0, description="Furniture value in €")

    # Financing
    apport_min: float = Field(default=0.0, ge=0, description="Minimum down payment in €")
    capital_emprunte: float = Field(default=0.0, ge=0, description="Loan amount in €")
    duree_credit_mois: int = Field(default=240, gt=0, description="Loan term in months")
    taux_annuel_pct: float = Field(default=3.5, ge=0, description="Annual interest rate %")
    assurance_annuelle_pct: float = Field(default=0.36, ge=0, description="Annual insurance rate %")
    pmt_principal_interet: float = Field(default=0.0, description="Monthly P&I payment in €")
    pmt_assurance: float = Field(default=0.0, description="Monthly insurance in €")

    # Income & Expenses
    loyer_m2: float = Field(default=0.0, ge=0, description="Rent per m² in €")
    loyer_m2_max: float | None = Field(None, description="Rent cap per m²")
    loyer_mensuel_initial: float = Field(default=0.0, ge=0, description="Initial monthly rent in €")
    depenses_mensuelles_hors_credit_initial: float = Field(default=0.0, description="Monthly expenses ex-loan in €")

    # Qualitative
    dpe_initial: str = Field(default="D", description="DPE rating")
    meuble: bool = Field(default=False, description="Is furnished")
    soumis_encadrement: bool = Field(default=False, description="Subject to rent control")
    qual_score_bien: float | None = Field(None, description="Qualitative score 0-100")

    # Market indicators
    indice_tension: float = Field(default=0.5, description="Market tension")
    transport_score: float = Field(default=0.5, description="Transport score")
    liquidite_score: float = Field(default=0.5, description="Liquidity score")

    model_config = {
        "extra": "allow",
    }

    @computed_field
    @property
    def cout_total(self) -> float:
        """Total acquisition cost."""
        return (
            self.prix_achat_bien
            + self.frais_notaire
            + self.travaux
            + self.renovation_energetique
            + self.mobilier
        )

    @computed_field
    @property
    def pmt_total(self) -> float:
        """Total monthly loan payment (P&I + insurance)."""
        return self.pmt_principal_interet + self.pmt_assurance

    @computed_field
    @property
    def cash_flow_mensuel_initial(self) -> float:
        """Initial monthly cash flow before taxes."""
        return (
            self.loyer_mensuel_initial
            - self.depenses_mensuelles_hors_credit_initial
            - self.pmt_total
        )

    def __getitem__(self, key: str) -> Any:
        """Legacy dict-style access for backward compatibility."""
        mapping = {
            "nom_bien": "nom",
            "taux_pret": "taux_annuel_pct",
            "assurance_ann_pct": "assurance_annuelle_pct",
            "cout_total_bien": "cout_total",
            "capital_emprunte": "capital_emprunte",
            "apport_add_bien": "apport_add_bien",
            "duree_pret": "duree_credit_mois",
            "cf_proximity": "cf_proximity", # For sorting tests
        }
        
        # Resolve key
        actual_key = mapping.get(key, key)
        
        # Check attributes first
        if hasattr(self, actual_key):
            val = getattr(self, actual_key)
            # Special case for duration: return years if legacy key 'duree_pret' is used
            if key == "duree_pret" and isinstance(val, int):
                return val // 12
            return val
        
        # Fallback to model_extra
        if self.model_extra and actual_key in self.model_extra:
            return self.model_extra[actual_key]
            
        raise KeyError(f"Key '{key}' (mapped to '{actual_key}') not found in InvestmentBrick")

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for legacy keys."""
        mapping = {
            "nom_bien": "nom",
            "taux_pret": "taux_annuel_pct",
            "assurance_ann_pct": "assurance_annuelle_pct",
            "cout_total_bien": "cout_total",
            "duree_pret": "duree_credit_mois",
        }
        actual_key = mapping.get(key, key)
        return hasattr(self, actual_key) or (self.model_extra and actual_key in self.model_extra)

    def get(self, key: str, default: Any = None) -> Any:
        """Legacy dict-style get for backward compatibility."""
        try:
            return self[key]
        except KeyError:
            return default
