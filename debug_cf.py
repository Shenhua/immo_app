#!/usr/bin/env python3
"""Debug script to analyze cash flow at year 25."""
import json
import sys

# Load latest results
with open('results/simulation_20251218_134733.json') as f:
    data = json.load(f)

s = data['strategies'][0]
print("=== Strategy Analysis ===")
print(f"Cash-flow final (Y1 monthly): {s.get('cash_flow_final', 'N/A'):.2f} €")
print(f"Apport total: {s.get('apport_total', 'N/A'):,.0f} €")
print(f"TRI: {s.get('tri_annuel', 'N/A'):.2f}%")

print("\n=== Properties ===")
for i, p in enumerate(s['details'], 1):
    print(f"\nProperty {i}: {p.get('nom_bien', 'N/A')[:40]}...")
    print(f"  Prix achat: {p.get('prix_achat_bien', 0):,.0f} €")
    print(f"  Credit final: {p.get('credit_final', 0):,.0f} €")
    print(f"  Duree pret: {p.get('duree_pret', 0)} ans")
    print(f"  Taux pret: {p.get('taux_pret', 0):.2f}%")
    print(f"  Loyer mensuel initial: {p.get('loyer_mensuel_initial', 0):,.0f} €")
    print(f"  Charges hors credit: {p.get('depenses_mensuelles_hors_credit_initial', 0):,.0f} €")
    print(f"  PMT total mensuel: {p.get('pmt_total', 0):,.2f} €")
    
    # Calculate expected monthly CF at year 1
    loyer = p.get('loyer_mensuel_initial', 0)
    charges = p.get('depenses_mensuelles_hors_credit_initial', 0)
    pmt = p.get('pmt_total', 0)
    cf_mth = loyer - charges - pmt
    print(f"  CF mensuel Y1 (avant impot): {cf_mth:,.2f} €")
    
    # Calculate expected monthly CF after loan payoff (no PMT)
    cf_mth_noloan = loyer - charges
    print(f"  CF mensuel apres pret (avant impot): {cf_mth_noloan:,.2f} €")

# Total CF calculation
total_loyer = sum(p.get('loyer_mensuel_initial', 0) for p in s['details'])
total_charges = sum(p.get('depenses_mensuelles_hors_credit_initial', 0) for p in s['details'])
total_pmt = sum(p.get('pmt_total', 0) for p in s['details'])
n_biens = len(s['details'])
cfe_mensuel = 500 * n_biens / 12  # CFE is annual

print(f"\n=== Totals (monthly) ===")
print(f"Total loyer: {total_loyer:,.0f} €")
print(f"Total charges: {total_charges:,.0f} €")
print(f"Total PMT: {total_pmt:,.2f} €")
print(f"CFE: {cfe_mensuel:,.2f} €")
print(f"CF Y1 (avant impot): {total_loyer - total_charges - total_pmt - cfe_mensuel:,.2f} €")
print(f"CF apres pret (avant impot): {total_loyer - total_charges - cfe_mensuel:,.2f} €")
