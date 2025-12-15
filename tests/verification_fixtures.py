"""
Excel Comparison Fixture Generator

This script generates CSV files with calculated values that can be
compared against an Excel spreadsheet for validation.

Usage:
    python tests/verification_fixtures.py

Output:
    results/validation_fixture.csv
"""

import csv
import os
from datetime import datetime

from src.core.financial import (
    calculate_monthly_payment,
    calculate_insurance,
    generate_amortization_schedule,
)
from src.core.simulation import (
    SimulationEngine,
    MarketHypotheses,
    TaxParams,
    IRACalculator,
)


def generate_loan_fixtures():
    """Generate loan calculation fixtures for Excel comparison."""
    fixtures = []
    
    test_cases = [
        # (principal, rate%, months, description)
        (100_000, 3.6, 300, "100k @ 3.6% 25yr"),
        (150_000, 3.4, 240, "150k @ 3.4% 20yr"),
        (83_426, 3.6, 300, "83.4k @ 3.6% 25yr"),
        (144_000, 3.6, 300, "144k @ 3.6% 25yr"),
        (200_000, 3.0, 180, "200k @ 3.0% 15yr"),
    ]
    
    for principal, rate, months, desc in test_cases:
        pmt = calculate_monthly_payment(principal, rate, months)
        ins = calculate_insurance(principal, 0.35)
        
        fixtures.append({
            "Description": desc,
            "Principal (€)": principal,
            "Rate (%)": rate,
            "Duration (months)": months,
            "PMT (€/month)": round(pmt, 2),
            "Insurance (€/month)": round(ins, 2),
            "Total Payment (€/month)": round(pmt + ins, 2),
            "Excel Formula PMT": f'=PMT({rate}%/12, {months}, -{principal})',
            "Excel Formula Ins": f'={principal}*0.35%/12',
        })
    
    return fixtures


def generate_simulation_fixture():
    """Generate a full 25-year simulation for Excel comparison."""
    strategy = {
        "apport_total": 10_000,
        "details": [
            {
                "nom_bien": "Test Property",
                "prix_achat_bien": 100_000,
                "budget_travaux": 5_000,
                "mobilier": 3_000,
                "renovation_energetique": 0,
                "frais_notaire": 7_500,
                "credit_final": 100_000,
                "taux_pret": 3.6,
                "duree_pret": 25,
                "assurance_ann_pct": 0.35,
                "loyer_mensuel_initial": 600,
                "charges_const_mth0": 80,
                "tf_const_mth0": 60,
                "frais_gestion_pct": 5.0,
                "provision_pct": 3.0,
            }
        ],
    }
    
    engine = SimulationEngine(
        market=MarketHypotheses(
            appreciation_bien_pct=2.0,
            revalo_loyer_pct=1.5,
            inflation_charges_pct=2.0,
        ),
        tax=TaxParams(tmi_pct=30.0, regime_fiscal="lmnp"),
        ira=IRACalculator(apply_ira=False),
    )
    
    schedules = [
        generate_amortization_schedule(100_000, 3.6, 300, 0.35)
    ]
    
    df, bilan = engine.simulate(strategy, 25, schedules)
    
    return df.to_dict('records'), bilan


def save_fixtures():
    """Save all fixtures to CSV files."""
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Loan fixtures
    loan_fixtures = generate_loan_fixtures()
    loan_file = f"results/loan_validation_{timestamp}.csv"
    
    with open(loan_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=loan_fixtures[0].keys())
        writer.writeheader()
        writer.writerows(loan_fixtures)
    
    print(f"✓ Loan fixtures saved to: {loan_file}")
    
    # Simulation fixtures
    sim_records, bilan = generate_simulation_fixture()
    sim_file = f"results/simulation_validation_{timestamp}.csv"
    
    with open(sim_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sim_records[0].keys())
        writer.writeheader()
        writer.writerows(sim_records)
    
    print(f"✓ Simulation fixtures saved to: {sim_file}")
    
    # Bilan summary
    bilan_file = f"results/bilan_validation_{timestamp}.csv"
    
    with open(bilan_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in bilan.items():
            writer.writerow([key, value])
    
    print(f"✓ Final bilan saved to: {bilan_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("VALIDATION FIXTURE SUMMARY")
    print("="*50)
    print(f"\nLoan Calculations ({len(loan_fixtures)} cases):")
    for fix in loan_fixtures:
        print(f"  {fix['Description']}: PMT = {fix['PMT (€/month)']}€")
    
    print(f"\nSimulation (25 years):")
    last_year = sim_records[-1]
    print(f"  Year 25 Property Value: {last_year['Valeur Biens']:,.0f}€")
    print(f"  Year 25 Net Patrimony: {last_year['Patrimoine Net']:,.0f}€")
    cf_key = "Cash-Flow Net d'Impôt"
    print(f"  Year 25 Cash Flow: {last_year[cf_key]:,.0f}€")
    
    print(f"\nFinal Bilan:")
    print(f"  TRI (IRR): {bilan.get('tri_annuel', 0):.2f}%")
    print(f"  Liquidation Nette: {bilan.get('liquidation_nette', 0):,.0f}€")
    
    return loan_file, sim_file, bilan_file


if __name__ == "__main__":
    save_fixtures()
