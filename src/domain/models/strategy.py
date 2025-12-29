"""Strategy and result data models.

A strategy represents a combination of investment bricks with
aggregated financial metrics and scoring.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, computed_field


class PortfolioStrategy(BaseModel):
    """Investment strategy combining multiple property bricks.

    Contains the portfolio composition and aggregated metrics.
    """

    # Portfolio composition
    details: list[dict[str, Any]] = Field(default_factory=list, description="List of property bricks")

    # Aggregated financials
    apport_total: float = Field(default=0.0, ge=0, description="Total down payment in €")
    patrimoine_acquis: float = Field(default=0.0, ge=0, description="Total acquired assets in €")
    cash_flow_final: float = Field(default=0.0, description="Net monthly cash flow in €")

    # Scoring
    balanced_score: float = Field(default=0.0, description="Balanced score 0-100")
    qual_score: float = Field(default=50.0, description="Qualitative score 0-100")

    # Classification
    taxonomy: str | None = Field(None, description="Strategy type: Optimisé, Patrimonial, Mix")

    model_config = {
        "extra": "allow",
    }

    @computed_field
    @property
    def nombre_biens(self) -> int:
        """Number of properties in the strategy."""
        return len(self.details)


class StrategyResult(BaseModel):
    """Complete strategy result with simulation data.

    Contains the strategy, simulation DataFrame (as dict), and financial summary.
    """

    strategy_data: dict[str, Any] = Field(..., description="Strategy dictionary")
    bilan: dict[str, Any] = Field(default_factory=dict, description="Financial summary")
    kpis: dict[str, Any] = Field(default_factory=dict, description="Key performance indicators")

    # Simulation data (DataFrame serialized)
    simulation_years: list[dict[str, Any]] = Field(default_factory=list, description="Yearly simulation data")

    model_config = {
        "extra": "allow",
    }

    @computed_field
    @property
    def enrichissement_net(self) -> float:
        """Net wealth creation (liquidation - initial investment)."""
        return self.bilan.get("enrichissement_net", 0.0)

    @computed_field
    @property
    def tri_pct(self) -> float | None:
        """Internal rate of return percentage."""
        return self.bilan.get("tri_annuel")


class SimulationParams(BaseModel):
    """Parameters for long-term strategy simulation."""

    # Duration
    horizon_years: int = Field(default=25, ge=1, le=50, description="Simulation horizon in years")

    # Market hypotheses
    inflation_pct: float = Field(default=2.0, description="Annual inflation %")
    hausse_loyers_pct: float = Field(default=1.5, description="Annual rent increase %")
    hausse_prix_immo_pct: float = Field(default=2.0, description="Annual property price increase %")
    hausse_charges_pct: float = Field(default=2.0, description="Annual charges increase %")

    # Tax parameters
    tmi_pct: float = Field(default=30.0, ge=0, le=45, description="Marginal tax rate %")
    regime_fiscal: str = Field(default="lmnp", description="Tax regime: lmnp, micro_bic")
    micro_bic_abatt_pct: float = Field(default=50.0, description="Micro-BIC abatement %")

    # Costs
    frais_vente_pct: float = Field(default=6.0, description="Sale costs %")
    cfe_par_bien_ann: float = Field(default=500.0, description="Annual CFE per property")

    # IRA (early repayment)
    apply_ira: bool = Field(default=True, description="Apply early repayment penalties")
    ira_cap_pct: float = Field(default=3.0, description="IRA cap as % of remaining capital")

    model_config = {
        "extra": "allow",
    }
