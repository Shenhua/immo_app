"""Brick creation service.

Generates investable 'bricks' (property + financing + costs) from archetypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from src.domain.calculator.financial import calculate_total_monthly_payment
from src.domain.calculator.scoring import calculate_property_qualitative_score
from src.domain.models.brick import InvestmentBrick


@dataclass
class FinancingConfig:
    """Configuration for financing parameters."""
    credit_rates: dict[int, float]  # duration -> rate
    frais_notaire_pct: float = 7.5
    apport_min_pct: float = 10.0
    assurance_ann_pct: float = 0.36
    frais_pret_pct: float = 1.0
    inclure_travaux: bool = True
    inclure_reno_ener: bool = True
    inclure_mobilier: bool = True
    financer_mobilier: bool = True


@dataclass
class OperatingConfig:
    """Configuration for operating costs."""
    frais_gestion_pct: float = 5.0
    provision_pct: float = 5.0
    cfe_par_bien_ann: float = 150.0


def create_investment_bricks(
    archetypes: list[dict[str, Any]],
    finance: FinancingConfig,
    operating: OperatingConfig,
) -> list[InvestmentBrick]:
    """Create investment bricks from archetypes and configuration.

    A 'brick' is a specific combination of a property archetype and a financing plan.
    It represents an atomic investment unit.

    Args:
        archetypes: List of property archetypes
        finance: Financing configuration
        operating: Operating configuration

    Returns:
        List of expanded investment bricks
    """
    bricks = []

    for archetype in archetypes:
        # 1. Base property costs
        surface = archetype["surface"]
        prix_achat = surface * archetype["prix_m2"]

        # Monthly revenues and costs (Year 0)
        loyer_mth0 = surface * (float(archetype.get("loyer_m2") or 0.0))
        charges_mth0 = (surface * archetype.get("charges_m2_an", 0.0)) / 12.0
        taxe_fonciere_mth0 = (surface * archetype.get("taxe_fonciere_m2_an", 0.0)) / 12.0

        # 2. Project costs
        travaux = archetype.get("budget_travaux", 0.0) if finance.inclure_travaux else 0.0

        # Energy renovation check
        dpe = archetype.get("dpe_initial", "D").upper()
        reno_cout = float(archetype.get("renovation_energetique_cout", 0.0))
        # Apply renovation cost for energy-inefficient properties (E, F, G)
        reno = reno_cout if finance.inclure_reno_ener and dpe in ["E", "F", "G"] else 0.0

        mobilier = archetype.get("valeur_mobilier", 0.0) if finance.inclure_mobilier else 0.0

        frais_notaire = prix_achat * (finance.frais_notaire_pct / 100.0)

        cout_total_projet = prix_achat + frais_notaire + travaux + reno + mobilier

        # 3. Financing Setup
        apport_min = frais_notaire + (prix_achat * (finance.apport_min_pct / 100.0))

        if finance.inclure_mobilier and not finance.financer_mobilier:
            # If furniture included but not financed, add to minimal down payment
            apport_min += mobilier

        credit_base_needed = max(0.0, cout_total_projet - apport_min)
        frais_pret = credit_base_needed * (finance.frais_pret_pct / 100.0)

        capital_emprunte = credit_base_needed + frais_pret

        # 4. Operating costs
        # Allow archetype to override global operating config
        gestion_pct = float(archetype.get("frais_gestion_pct", operating.frais_gestion_pct))
        vacance_pct = float(archetype.get("vacance_pct", operating.provision_pct))

        gest_mth0 = loyer_mth0 * (gestion_pct / 100.0)
        prov_mth0 = loyer_mth0 * (vacance_pct / 100.0)

        # Total operating expenses (excluding debt)
        depenses_mth0 = (
            charges_mth0 +
            taxe_fonciere_mth0 +
            gest_mth0 +
            prov_mth0 +
            (operating.cfe_par_bien_ann / 12.0)
        )

        # 5. Calculate qualitative score for this property
        qual_score, _ = calculate_property_qualitative_score(
            archetype,
            loyer_m2=archetype.get("loyer_m2"),
            loyer_m2_max=archetype.get("loyer_m2_max"),
            prix_achat=prix_achat,
            travaux=travaux,
        )

        # 6. Create variants for each loan duration
        base_data = archetype.copy()
        # Rename 'nom' to 'nom_bien' for internal consistency
        base_data['nom_bien'] = base_data.pop('nom', 'Bien inconnu')

        for duree, taux in finance.credit_rates.items():
            duration_months = int(duree) * 12

            # Calculate Monthly Payment
            p_i, ins, pmt_total = calculate_total_monthly_payment(
                capital_emprunte,
                taux,
                duration_months,
                finance.assurance_ann_pct
            )

            # Create InvestmentBrick instance
            # We use model_validate with a dictionary to handle field mappings
            brick_dict = {
                **base_data,
                # Financials
                "prix_achat_bien": prix_achat,
                "frais_notaire": frais_notaire,
                "budget_travaux": travaux,
                "renovation_energetique": reno,
                "mobilier": mobilier,
                "cout_total": cout_total_projet,
                "apport_min": apport_min,

                # Loan
                "capital_emprunte": capital_emprunte,
                "credit_final": capital_emprunte,  # Alias
                "duree_credit_mois": duration_months,
                "taux_annuel_pct": float(taux),
                "assurance_annuelle_pct": float(finance.assurance_ann_pct),
                "pmt_principal_interet": p_i,
                "pmt_assurance": ins,

                # Legacy fields for backward compatibility (allowed by extra="allow")
                "duree_pret": int(duree),
                "taux_pret": float(taux),
                "assurance_ann_pct": float(finance.assurance_ann_pct),
                "pmt_total": pmt_total,

                # Operating
                "loyer_mensuel_initial": loyer_mth0,
                "charges_const_mth0": charges_mth0,
                "tf_const_mth0": taxe_fonciere_mth0,
                "frais_gestion_pct": gestion_pct,
                "provision_pct": vacance_pct,
                "depenses_mensuelles_hors_credit_initial": depenses_mth0,

                # Qualitative score
                "qual_score_bien": qual_score,
            }

            # Map 'nom_bien' back to 'nom' for the model if needed
            if 'nom_bien' in brick_dict and 'nom' not in brick_dict:
                brick_dict['nom'] = brick_dict['nom_bien']

            bricks.append(InvestmentBrick.model_validate(brick_dict))


    return bricks


def apply_rent_caps(archetypes: list[dict[str, Any]], apply_cap: bool = True) -> list[dict[str, Any]]:
    """Apply regulatory rent caps to archetypes.

    Args:
        archetypes: List of archetype data
        apply_cap: Whether to enforce rent caps

    Returns:
        Processed archetypes with rent caps applied
    """
    if not apply_cap:
        return archetypes

    processed = []
    for item in archetypes:
        a = item.copy()
        if a.get("soumis_encadrement") and a.get("loyer_m2_max") is not None:
            try:
                cap = float(a["loyer_m2_max"])
                current = float(a.get("loyer_m2", 0.0))
                # Apply cap if strictly lower
                if current > cap:
                    a["loyer_m2"] = cap
            except (ValueError, TypeError):
                pass
        processed.append(a)
    return processed


# Required fields for brick validation
REQUIRED_BRICK_FIELDS = [
    ("nom_bien", "Property name"),
    ("prix_achat_bien", "Purchase price"),
    ("loyer_mensuel_initial", "Monthly rent"),
    ("taux_pret", "Loan rate"),
    ("duree_pret", "Loan duration"),
    ("assurance_ann_pct", "Insurance rate"),
    ("cout_total", "Total cost"),
]


class BrickValidationError(ValueError):
    """Raised when a brick is missing required fields."""
    pass


def validate_brick(brick: Union[dict[str, Any], InvestmentBrick], strict: bool = False) -> list[str]:
    """Validate a brick has all required fields for simulation.

    Args:
        brick: Investment brick dictionary
        strict: If True, raise BrickValidationError on first missing field

    Returns:
        List of warning messages for missing/invalid fields

    Raises:
        BrickValidationError: If strict=True and validation fails
    """
    warnings: list[str] = []
    # Handle both dict and InvestmentBrick
    if hasattr(brick, 'model_dump'):
        brick_data = brick.model_dump()
    else:
        brick_data = brick
    brick_name = brick_data.get("nom_bien") or brick_data.get("nom", "Unknown")

    for field, description in REQUIRED_BRICK_FIELDS:
        value = brick_data.get(field)
        if value is None:
            msg = f"Brick '{brick_name}': Missing {description} ({field})"
            warnings.append(msg)
            if strict:
                raise BrickValidationError(msg)
        elif isinstance(value, (int, float)) and value <= 0:
            # Check for zero/negative values on financial fields
            if field in ("taux_pret", "duree_pret", "prix_achat_bien", "cout_total"):
                msg = f"Brick '{brick_name}': Invalid {description} ({field}={value})"
                warnings.append(msg)
                if strict:
                    raise BrickValidationError(msg)

    return warnings


def validate_bricks(bricks: list[Union[dict[str, Any], InvestmentBrick]], strict: bool = False) -> list[str]:
    """Validate all bricks in a list.

    Args:
        bricks: List of investment bricks
        strict: If True, raise on first validation error

    Returns:
        Aggregated list of all warnings
    """
    all_warnings: list[str] = []
    for brick in bricks:
        all_warnings.extend(validate_brick(brick, strict=strict))
    return all_warnings


