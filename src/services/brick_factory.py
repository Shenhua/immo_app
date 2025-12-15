"""Brick creation service.

Generates investable 'bricks' (property + financing + costs) from archetypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.financial import calculate_total_monthly_payment
from src.core.scoring import calculate_property_qualitative_score


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
) -> list[dict[str, Any]]:
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
        gest_mth0 = loyer_mth0 * (operating.frais_gestion_pct / 100.0)
        prov_mth0 = loyer_mth0 * (operating.provision_pct / 100.0)

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
            _, _, pmt_total = calculate_total_monthly_payment(
                capital_emprunte,
                taux,
                duration_months,
                finance.assurance_ann_pct
            )

            brick = {
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
                "duree_pret": int(duree),
                "taux_pret": float(taux),
                "assurance_ann_pct": float(finance.assurance_ann_pct),
                "pmt_total": pmt_total,

                # Operating
                "loyer_mensuel_initial": loyer_mth0,
                "charges_const_mth0": charges_mth0,
                "tf_const_mth0": taxe_fonciere_mth0,
                "frais_gestion_pct": operating.frais_gestion_pct,
                "provision_pct": operating.provision_pct,
                "depenses_mensuelles_hors_credit_initial": depenses_mth0,

                # Qualitative score
                "qual_score_bien": qual_score,
            }

            bricks.append(brick)

    return bricks

