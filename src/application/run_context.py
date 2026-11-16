"""RunContext: immutable snapshot of every parameter used by one analysis run (T11).

It is built once, when the user clicks "Lancer l'analyse", and stored next to the
results. The search, the detail simulation, the charts, the result cards and the
sensitivity tab all read their parameters from it, never from the live sidebar.
The sidebar is compared to it only to show the "stale results" banner.

This module must stay free of streamlit / src.ui imports (layering rule).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.defaults import DEFAULT_PRESET, DPE, FINANCING, MARKET, OPERATING, REALISM, SEARCH, TAX, preset_weights

if TYPE_CHECKING:
    from src.application.services.simulation import SimulationEngine


def hash_archetypes(archetypes: list[dict[str, Any]]) -> str:
    """Stable short hash of the archetype list actually searched (after filters and rent caps)."""
    payload = json.dumps(archetypes, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class RunContext(BaseModel):
    """Frozen set of search and simulation parameters for one run."""

    model_config = ConfigDict(frozen=True)

    # Objectives
    apport: float = SEARCH.apport
    cf_cible: float = SEARCH.cf_cible
    tolerance: float = SEARCH.tolerance
    mode_cf: Literal["target", "min"] = "min"
    qualite_weight: float = SEARCH.qualite_weight
    horizon_ans: int = SEARCH.horizon_ans
    max_properties: int = SEARCH.max_properties_classique
    use_full_capital: bool = False
    top_n: int = SEARCH.top_n

    # Financing
    taux_credits: dict[int, float] = Field(default_factory=lambda: FINANCING.taux_credits)
    frais_notaire_pct: float = FINANCING.frais_notaire_pct
    assurance_ann_pct: float = FINANCING.assurance_ann_pct
    frais_pret_pct: float = FINANCING.frais_pret_pct
    inclure_travaux: bool = FINANCING.inclure_travaux
    inclure_reno_ener: bool = FINANCING.inclure_reno_ener
    inclure_mobilier: bool = FINANCING.inclure_mobilier
    financer_mobilier: bool = FINANCING.financer_mobilier
    apply_ira: bool = FINANCING.apply_ira
    ira_cap_pct: float = FINANCING.ira_cap_pct

    # Operating costs and exit
    cfe_par_bien_ann: float = OPERATING.cfe_par_bien_ann
    gestion_pct: float = OPERATING.gestion_pct
    vacance_pct: float = OPERATING.vacance_pct
    frais_vente_pct: float = TAX.frais_vente_pct
    # T26a realistic assumptions. ``vacance_pct`` is a floor on each archetype's own vacancy.
    maintenance_m2_an: float = REALISM.maintenance_m2_an
    gli_pct: float = REALISM.gli_pct
    mois_travaux: int = REALISM.mois_travaux
    tf_growth_pct: float = REALISM.tf_growth_pct

    # DPE rental ban (T61): "renover" = works before the ban year, "ne_pas_renover" = rent 0 from the ban year.
    dpe_policy: Literal["renover", "ne_pas_renover"] = DPE.policy  # type: ignore[assignment]
    dpe_cout_m2_par_classe: float = DPE.cout_m2_par_classe

    # Tax: one regime per income category (TAX_SPEC 8). ``regime_fiscal`` = furnished regime
    # ("lmnp" | "microbic"), ``regime_foncier`` = unfurnished regime ("reel" | "micro"),
    # ``sci_is`` puts every property in one SCI at IS and overrides both.
    regime_fiscal: str = TAX.regime_fiscal
    regime_foncier: str = TAX.regime_foncier
    sci_is: bool = TAX.sci_is
    tmi_pct: float = TAX.tmi_pct
    # Land share override (%). None = use each archetype's own ratio_terrain_pct.
    ratio_terrain_pct: float | None = None
    # Co-owners selling each property (1 or 2): splits the gain for the surtax threshold only.
    nb_owners: Literal[1, 2] = 1  # default: TAX.nb_owners (checked in tests)

    # Market
    appreciation_bien_pct: float = MARKET.appreciation_bien_pct
    revalo_loyer_pct: float = MARKET.revalo_loyer_pct
    inflation_charges_pct: float = MARKET.inflation_charges_pct

    # Scoring
    finance_preset_name: str = DEFAULT_PRESET.label
    finance_weights: dict[str, float] = Field(default_factory=lambda: preset_weights(DEFAULT_PRESET))

    # Data
    apply_cap: bool = True
    archetypes_version: str = ""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_sidebar(cls, params: dict[str, Any], archetypes: list[dict[str, Any]]) -> RunContext:
        """Build a context from the dict the sidebar returns and the archetypes to be searched."""
        credit = params["credit_params"]
        return cls(
            apport=params["apport"],
            cf_cible=params["cf_cible"],
            tolerance=params["tolerance"],
            mode_cf=params["mode_cf"],
            qualite_weight=params["qual_weight"],
            horizon_ans=params["horizon"],
            max_properties=params.get("max_properties", SEARCH.max_properties_classique),
            use_full_capital=params.get("use_full_capital", False),
            top_n=params.get("top_n", SEARCH.top_n),
            taux_credits=credit["taux_credits"],
            frais_notaire_pct=credit["frais_notaire_pct"],
            assurance_ann_pct=credit["assurance_ann_pct"],
            frais_pret_pct=credit["frais_pret_pct"],
            inclure_travaux=credit["inclure_travaux"],
            inclure_reno_ener=credit["inclure_reno_ener"],
            inclure_mobilier=credit["inclure_mobilier"],
            financer_mobilier=credit["financer_mobilier"],
            apply_ira=credit["apply_ira"],
            ira_cap_pct=credit["ira_cap_pct"],
            ratio_terrain_pct=credit.get("ratio_terrain_pct"),
            nb_owners=2 if int(credit.get("nb_owners", TAX.nb_owners)) == 2 else 1,
            cfe_par_bien_ann=params["cfe"],
            gestion_pct=params["gestion"],
            vacance_pct=params["vacance"],
            frais_vente_pct=params["frais_vente"],
            maintenance_m2_an=params.get("maintenance_m2_an", REALISM.maintenance_m2_an),
            gli_pct=params.get("gli_pct", REALISM.gli_pct),
            mois_travaux=int(params.get("mois_travaux", REALISM.mois_travaux)),
            tf_growth_pct=params.get("tf_growth_pct", REALISM.tf_growth_pct),
            dpe_policy=credit.get("dpe_policy", DPE.policy),
            dpe_cout_m2_par_classe=float(credit.get("dpe_cout_m2_par_classe", DPE.cout_m2_par_classe)),
            regime_fiscal=params["regime"],
            regime_foncier=params.get("regime_foncier", TAX.regime_foncier),
            sci_is=bool(params.get("sci_is", TAX.sci_is)),
            tmi_pct=params["tmi"],
            appreciation_bien_pct=params["market_hypo"]["appreciation_bien_pct"],
            revalo_loyer_pct=params["market_hypo"]["revalo_loyer_pct"],
            inflation_charges_pct=params["market_hypo"]["inflation_charges_pct"],
            finance_preset_name=params["finance_preset_name"],
            finance_weights=params["finance_weights"],
            apply_cap=params.get("apply_cap", True),
            archetypes_version=hash_archetypes(archetypes),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunContext:
        """Inverse of :meth:`to_dict` (JSON keys of ``taux_credits`` come back as str and are coerced)."""
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    # ------------------------------------------------------------------ #
    # Views for the existing services
    # ------------------------------------------------------------------ #
    @property
    def market_hypotheses(self) -> dict[str, float]:
        return {
            "appreciation_bien_pct": self.appreciation_bien_pct,
            "revalo_loyer_pct": self.revalo_loyer_pct,
            "inflation_charges_pct": self.inflation_charges_pct,
            "tf_growth_pct": self.tf_growth_pct,
        }

    def build_engine(self) -> SimulationEngine:
        """Simulation engine for this run (the same builder the search uses)."""
        from src.application.services.simulation import build_engine

        return build_engine(self.market_hypotheses, self.to_tax_params())

    def to_credit_params(self) -> dict[str, Any]:
        """Dict expected by ``build_financing_config`` and ``IRACalculator``."""
        return {
            "taux_credits": dict(self.taux_credits),
            "frais_notaire_pct": self.frais_notaire_pct,
            "assurance_ann_pct": self.assurance_ann_pct,
            "frais_pret_pct": self.frais_pret_pct,
            "inclure_travaux": self.inclure_travaux,
            "inclure_reno_ener": self.inclure_reno_ener,
            "inclure_mobilier": self.inclure_mobilier,
            "financer_mobilier": self.financer_mobilier,
            "apply_ira": self.apply_ira,
            "ira_cap_pct": self.ira_cap_pct,
            "ratio_terrain_pct": self.ratio_terrain_pct,
        }

    def to_eval_params(self) -> dict[str, Any]:
        """Dict taken by ``StrategyFinder.find_strategies`` (same keys the old app.py built)."""
        return {
            "tmi_pct": self.tmi_pct,
            "regime_fiscal": self.regime_fiscal,
            "regime_foncier": self.regime_foncier,
            "sci_is": self.sci_is,
            "frais_vente_pct": self.frais_vente_pct,
            "ratio_terrain_pct": self.ratio_terrain_pct,
            "nb_owners": self.nb_owners,
            "apply_ira": self.apply_ira,
            "ira_cap_pct": self.ira_cap_pct,
            "cfe_par_bien_ann": self.cfe_par_bien_ann,
            "hypotheses_marche": self.market_hypotheses,
            "finance_preset_name": self.finance_preset_name,
            "finance_weights_override": dict(self.finance_weights),
            "max_properties": self.max_properties,
            "use_full_capital": self.use_full_capital,
        }

    def to_tax_params(self) -> dict[str, Any]:
        """Dict taken by ``simulate_long_term_strategy(tax_params=...)``."""
        return {
            "tmi_pct": self.tmi_pct,
            "regime_fiscal": self.regime_fiscal,
            "regime_foncier": self.regime_foncier,
            "sci_is": self.sci_is,
            "apply_ira": self.apply_ira,
            "ira_cap_pct": self.ira_cap_pct,
            "cfe_par_bien_ann": self.cfe_par_bien_ann,
            "frais_vente_pct": self.frais_vente_pct,
            "ratio_terrain_pct": self.ratio_terrain_pct,
            "nb_owners": self.nb_owners,
            "dpe_policy": self.dpe_policy,
            "dpe_cout_m2_par_classe": self.dpe_cout_m2_par_classe,
        }
