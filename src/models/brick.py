"""Investment brick data model.

A brick represents a specific property investment opportunity derived
from an archetype, with calculated financial details.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, computed_field


class InvestmentBrick(BaseModel):
    """Investment brick with full financial breakdown.
    
    Created from an archetype with specific financing parameters applied.
    """
    
    # Identity (from archetype)
    nom: str = Field(..., description="Property identifier")
    ville: str = Field(..., description="City")
    mode_loyer: str = Field(..., description="Rental mode")
    surface: float = Field(..., gt=0, description="Surface in m²")
    
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
    loyer_m2_max: Optional[float] = Field(None, description="Rent cap per m²")
    loyer_mensuel_initial: float = Field(default=0.0, ge=0, description="Initial monthly rent in €")
    depenses_mensuelles_hors_credit_initial: float = Field(default=0.0, description="Monthly expenses ex-loan in €")
    
    # Qualitative
    dpe_initial: str = Field(default="D", description="DPE rating")
    meuble: bool = Field(default=False, description="Is furnished")
    soumis_encadrement: bool = Field(default=False, description="Subject to rent control")
    qual_score_bien: Optional[float] = Field(None, description="Qualitative score 0-100")
    
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
