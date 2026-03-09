# Architecture Decision Record: LMNP 10-Year Deficit Expiration

## Status
Accepted

## Context
During the simulation of real estate investments under the **LMNP (Location Meublée Non Professionnelle)** tax regime, operating deficits (when deductible charges such as interest, property tax, and insurance exceed rental income) can be carried forward. Under French tax law, these deficits can only offset future LMNP profits for a maximum of **10 consecutive years**.

Previously, the `app_immo` calculation engine in `TaxParams.calculate_tax` managed LMNP deficits as a single accumulated `float`. Because it did not track the vintage (year of origin) of each deficit portion, the engine implicitly allowed deficits to roll forward indefinitely. This architectural gap risked inflating long-term net yields by providing an unwarranted tax shield beyond the legal 10-year limit.

## Decision
We have revised the underlying simulation model to enforce strict chronological tracking and expiration of LMNP deficits:
1. **List-Based Vintage Tracking**: `TaxParams.calculate_tax` now accepts and returns a `list[float]` (representing `deficits_reportables`) instead of a simple scalar `float`.
2. **Chronological Sorting**: The list is bounded to a maximum length of 10. Index `0` represents the deficit created in the immediately preceding year ($T-1$), while index `9` represents the oldest valid deficit from exactly 10 years ago ($T-10$).
3. **Oldest-First Absorption**: When the property portfolio generates a positive Net Operating Income (NOI), the engine iterates from the end of the list (index `9` down to `0`). It absorbs the oldest deficits first, maximizing the taxpayer's advantage legally, before attacking recent deficits.
4. **Natural Expiration**: During the simulation loop in `SimulationEngine._simulate_year`, the list is naturally truncated `[:10]`. If a deficit created in year $N$ has not been absorbed by year $N+10$, it "falls off" the edge of the list and is permanently lost, perfectly mirroring French tax doctrine.
5. **Backward Compatibility**: To support older unit testing formats and potential external module calls, `SimulationEngine._simulate_year` and `TaxParams.calculate_tax` have been retrofitted to cast scalar `float` configurations securely into lists.

## Consequences
- **Positive**:
  - The simulation's financial timeline is now highly accurate over $30$-year horizons.
  - The model prevents the overestimation of Net Cash Flows (Cash-Flow Net d'Impôt) by correctly exposing tax obligations after 10 years, increasing user trust.
- **Negative**:
  - Added slight algorithmic complexity due to array truncation and reverse-iteration within the inner simulation loop. However, since the array is bounded strictly to $n=10$, the runtime cost is entirely negligible ($O(1)$ effectively).

## References
- `RAPPORT_ETONNEMENT_Audit.md`
- `HANDOFF_DataModel_Refactoring.md` (Phase 2 plan)
- French Tax Code specifying the 10-year rolling limit for LMNP deficits.
