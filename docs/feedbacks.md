1. Logic vs. Goal Discrepancies

🔴 OBJECTIVE FLAW: Optimization is scoring the wrong “bilan” keys (TRI/enrichment become 0)
	•	Goal: Rank strategies by meaningful long-term wealth metrics (TRI, enrichment, liquidation).
	•	Reality: The optimizer’s absolute finance score reads bilan["tri_global"] and bilan["enrichissement_net"]  ￼, but the simulation + models consistently expose tri_annuel and liquidation_nette as the canonical outputs  ￼  ￼.
⇒ In many runs, TRI/enrichment will silently collapse to default 0.0, heavily biasing selection.
	•	Severity: Critical
	•	Fix Required: Align “bilan contract” across the codebase (single schema). Either:
	•	change optimizer to use tri_annuel + compute enrichment consistently, or
	•	update simulation to output tri_global/enrichissement_net (but then update models/tests/UI too).

⸻

🔴 OBJECTIVE FLAW: Loan insurance key mismatch in optimizer schedule generation
	•	Goal: Evaluate strategies with correct loan cost (interest + insurance).
	•	Reality: The optimizer uses annual_insurance_pct = p.get("assurance_pret_pct", 0.0)  ￼, while the brick/strategy pipeline uses assurance_ann_pct as the actual field name  ￼.
⇒ Insurance is often treated as 0% during optimization → systematically over-optimistic cashflow and “best” results.
	•	Severity: Critical
	•	Fix Required: Standardize to one key (assurance_ann_pct seems the de-facto), validate it at boundaries (Pydantic model or explicit schema validation).

⸻

🔴 OBJECTIVE FLAW: “Net wealth” property returns liquidation, not net enrichment
	•	Goal: Expose “enrichissement net” = liquidation – initial aport.
	•	Reality: StrategyResult.enrichissement_net returns liquidation_nette directly  ￼ (docstring says “Net wealth creation”).
	•	Severity: Major (misleads users and downstream ranking/UI)
	•	Fix Required: Compute true enrichment (liquidation − apport_total) consistently; rename fields if you want both.

⸻

🔴 OBJECTIVE FLAW: Scoring normalization assumes TRI is a fraction (0.20) but elsewhere TRI is % (8.0)
	•	Goal: A stable 0–100 finance score.
	•	Reality: tri_score = (tri / 0.20 * 100)  ￼ implies TRI is 0.08 not 8.0. But simulation computes TRI in percent (* 100.0)  ￼.
⇒ Any TRI > 0.2% saturates at 100, killing discrimination.
	•	Severity: Critical if this function is used in ranking anywhere user-facing.
	•	Fix Required: Decide unit (percent or fraction) and enforce it everywhere (models + tests). Add unit tests specifically around TRI scaling.

⸻

🟡 SUBJECTIVE IMPROVEMENT: “use_full_capital_override” exists but appears unused in the GA path
	•	Goal: Let the user choose “deploy all capital” vs “minimum to hit CF constraint.”
	•	Reality: find_strategies(... use_full_capital_override=...) is present  ￼, but in the provided dump I don’t see it applied to the GA evaluation/allocation path.
	•	Severity: Major (feature flag silently ignored)
	•	Fix Required: Thread the flag through the evaluation/allocator call (or remove it from UI until actually supported). Add an integration test that proves it changes outcomes.

⸻

🟡 SUBJECTIVE IMPROVEMENT: Missing explicit “Promise” doc (README) → intent is inferred from code
	•	Your own health check expects README.md  ￼. Without it, intent/spec drifts and QA has no source of truth.

⸻

2. “Mental Sandbox” Findings

Workflow A: Strategy search (GA) → allocation → simulation → ranking
	•	Scenario: Insurance field missing/mismatched
	•	Current Behavior: Insurance becomes 0% in optimizer schedule generation  ￼, making “best” strategies unrealistically strong.
	•	Expected Behavior: Fail fast (validation error) or fallback to a documented default, never silently “free insurance”.
	•	Scenario: Simulation returns tri_annuel / liquidation_nette, scorer expects tri_global / enrichissement_net
	•	Current Behavior: Score components default to 0.0, selection becomes distorted  ￼.
	•	Expected Behavior: A single canonical bilan schema; missing keys = hard error in optimizer.

⸻

Workflow B: Simulation engine computes TRI
	•	Scenario: Cashflows produce non-finite IRR (NaN/inf) or unstable sign patterns
	•	Current Behavior: TRI is computed via npf.irr(flux) * 100.0  ￼. If irr returns NaN, downstream ranking/scoring can silently break unless guarded everywhere.
	•	Expected Behavior: Explicit handling: if IRR not finite → set TRI=None + penalty score + surface warning to UI.

⸻

Workflow C: Allocation heuristic (“make CF meet target with extra apport”)
	•	Scenario: Step size causes “no movement” on small deltas
	•	Current Behavior: Extra apport increments are quantized by int(delta / step_size) * step_size  ￼ and capped (10%)  ￼; if delta < step_size, the allocator can stall and return “cannot meet target” even when a small feasible adjustment exists.
	•	Expected Behavior: Adaptive step sizing (smaller steps near feasibility boundary), plus a final “fine-tune” pass.

⸻

3. The “Matrix of Pain” (Test Plan)

Component	Scenario	Input Data	Expected Outcome	Type (Unit/E2E)
Optimizer scoring	Bilan keys mismatch	bilan={"tri_annuel":8.0,"liquidation_nette":150000}	Optimizer reads correct metrics; no silent default to 0	Unit
Optimizer schedules	Insurance key mismatch	brick has assurance_ann_pct=0.35	Insurance included in PMT; if missing → explicit error	Unit
Simulation TRI	Non-finite IRR	flux all positive / single sign	TRI=None + penalty + warning, no crash	Unit
Allocation	Delta smaller than step size	target CF barely missed	Allocator still finds minimal increment (fine-tune)	Unit
Allocation cap	Needs >10% extra apport	high negative CF brick	Strategy rejected (or flagged) deterministically	Unit
End-to-end search	Reproducibility	fixed seed + same inputs	Same top-N strategies returned	E2E
StrategyResult	Enrichment correctness	liquidation=150k, apport=50k	enrichissement_net = 100k (not 150k)	Unit
TRI units	Percent vs fraction	TRI=8.0 vs 0.08	Score behaves consistently; no saturation	Unit
Data contract	Missing required fields	remove duree_pret	Hard validation error, not random KeyError	Integration
Performance	GA memory blowup	store df_sim in every individual  ￼	No per-individual full DF retention, bounded memory	Load/Perf


⸻

4. Recommendations for Refactoring

Untestable / High-Risk code (SRP violations)
	•	High Risk: Optimizer evaluation mixes (1) schedule construction, (2) simulation, (3) constraint checking, (4) scoring, (5) memory storage in one function  ￼. This makes it hard to test each failure mode independently.
	•	High Risk: Multiple “truths” for the same concepts (TRI units, bilan keys, insurance key). This is the #1 reason “best results” can be wrong even if math is fine.

Hardening steps (guard clauses/validation to add now)
	1.	Define a single canonical schema:
	•	BrickFinancing (taux, durée, assurance, apport, credit_final, etc.)
	•	Bilan (tri_annuel, liquidation_nette, enrichissement_net, dscr_y1, …)
	2.	Fail fast on schema mismatch at module boundaries (before GA runs).
	3.	Remove silent defaults like get(..., 0.0) for financially critical fields (insurance, rates).
	4.	Stop storing full simulation frames per individual in GA stats  ￼; store only required KPIs.

“Smart pruning” to reduce millions of possibilities without losing the best

You’re already moving toward heuristics/GA, but you need domain-aware discretization + staged search:
	•	Stage 1 (Coarse): broad steps to map the frontier (fast, approximate).
	•	Stage 2 (Refine around Pareto front): shrink steps only near top candidates.
	•	Always keep diversity: keep top-K per “taxonomy bucket” (CF-focused / patrimonial / mix) before final rank.

Domain-aware step ideas (conceptual, not code):
	•	Loan duration: integer years (1–25), but don’t treat “rate” as continuous—apply your threshold model (15/20/25) while still allowing yearly duration because payment changes each year even if rate bucket stays constant.
	•	Rate/insurance: coarse 0.10% in stage 1, refine to 0.05% in stage 2 (only near winners).
	•	Apport allocation: coarse in 1–2% of property cost steps early, refine to smaller increments only when a strategy is within tolerance of the CF constraint.

⸻

5. Calibration & Reality Check (CRITICAL)
	•	The issues I flagged as 🔴 are not style opinions: they are contract mismatches that will materially change which strategies are selected as “best” (insurance treated as 0%, TRI/enrichment read as 0, wrong enrichment surfaced as “net”).  ￼  ￼  ￼
	•	The 🟡 items are only improvements if they match intended product behavior; but right now “unused override flags” are a product correctness risk because the UI can claim something it doesn’t do.



⸻

1. Logic vs. Goal Discrepancies

1) 🔴 OBJECTIVE FLAW: Optimizer scores metrics the simulator does not produce
	•	Goal: Use “Absolute Scales” scoring on TRI + enrichment to find best strategies.
	•	Reality: GeneticOptimizer._calculate_absolute_finance_score() reads bilan["tri_global"] and bilan["enrichissement_net"].  ￼
But the simulator liquidation step returns a bilan with tri_annuel + liquidation_nette (+ ira_total) — no tri_global, no enrichissement_net.
Result: finance scoring silently collapses toward zeros → GA selection pressure is wrong → “best results” are not actually best.
	•	Severity: Critical
	•	Fix Required: Define one canonical Bilan schema and enforce it everywhere (simulator, scorer, result model, tests). Either:
	•	change optimizer to use tri_annuel + a well-defined “enrichissement” metric computed by simulator, or
	•	update simulator to output tri_global and enrichissement_net (and define what they mean).

⸻

2) 🔴 OBJECTIVE FLAW: Insurance percent key mismatch makes loans unrealistically cheap
	•	Goal: Include loan insurance in amortization schedule and CF.
	•	Reality: Optimizer builds schedules using p.get("assurance_pret_pct", 0.0).  ￼
But bricks/allocator use assurance_ann_pct (seen in tests + allocator expectations).  ￼
So insurance likely becomes 0 in GA simulation → CF and DSCR overstated → ranking is wrong.
	•	Severity: Critical
	•	Fix Required: Standardize the field name (prefer assurance_ann_pct) and validate presence/type before schedule generation.

⸻

3) 🔴 OBJECTIVE FLAW: StrategyResult exposes wrong “enrichissement_net”
	•	Goal: Expose enrichment/net-wealth in the results layer.
	•	Reality: StrategyResult.enrichissement_net returns bilan["liquidation_nette"] (not enrichment).  ￼
This is a semantic lie: UI/export users will read the wrong KPI.
	•	Severity: Critical
	•	Fix Required: Rename fields or compute correct KPI; add tests that compare expected enrichment vs liquidation.

⸻

4) 🔴 OBJECTIVE FLAW: Qualitative scoring uses the wrong value for vacancy
	•	Goal: Include vacancy/tension as a factor in qualitative score.
	•	Reality: In calculate_property_qualitative_score, vacancy is set from travaux ratio: vacance = 1.0 - ratio_travaux.  ￼
That means “more renovation” ⇒ “less vacancy”… which is nonsensical, and it ignores actual tension.
	•	Severity: Major
	•	Fix Required: Use tension-derived vacancy/occupancy (or remove vacancy if you don’t have real vacancy data). Add a regression test proving vacancy varies with tension, not renovation budget.

⸻

5) 🟡 SUBJECTIVE IMPROVEMENT (but high impact): “Acceptable cashflow” checks only Year-1
	•	Goal: Ensure strategies meet CF objective over the horizon.
	•	Reality: calculate_cashflow_metrics() sets is_acceptable from Year-1 monthly CF proximity only.  ￼
A strategy could pass Year-1 then crash later (rate resets, works, tax regime effects, rent cap, etc.) and still be considered “acceptable”.
	•	Severity: Major (logic weakness)
	•	Fix Required: At minimum: check min CF over first N years, or “% of months within tolerance”, or “worst-year CF”.

⸻

6) 🔴 OBJECTIVE FLAW: Tests validate a bilan shape the real simulator doesn’t guarantee
	•	Goal: Tests should catch scoring/metric breakage.
	•	Reality: Optimizer unit tests mock bilan with tri_global and enrichissement_net.  ￼
That makes the current tests blind to the real mismatch described in #1.
	•	Severity: Critical
	•	Fix Required: Replace mocks with a real-ish simulator output contract test (or at least mock the actual simulator schema).

⸻

2. “Mental Sandbox” Findings

Workflow A — GeneticOptimizer evaluation (core search loop)
	•	Scenario: Brick dictionaries missing fields / inconsistent naming
Current Behavior: Silent defaults (insurance = 0, missing bilan keys = 0) distort fitness without failing fast.  ￼  ￼
Expected Behavior: Hard validation: if required keys missing, mark individual invalid with explicit reason (and optionally drop brick).
	•	Scenario: Memory blow during GA run
Current Behavior: Stores entire df_sim inside each Individual: ind.stats["simulation"] = df_sim with a warning comment.  ￼
Expected Behavior: Store only summary metrics (or keep df only for top-K individuals).

⸻

Workflow B — Simulation liquidation / TRI computation
	•	Scenario: IRR undefined (cashflows all same sign, NaN propagation)
Current Behavior: TRI derived from npf.irr(cashflows) (no visible NaN guard in the snippet that builds the bilan).
Expected Behavior: If IRR is NaN/inf: set to 0 (or None) + record error flag; never let NaN infect ranking.

⸻

Workflow C — Allocation loop (cashflow targeting)
	•	Scenario: Insurance key mismatch cascades into wrong k-factor / payment deltas
Current Behavior: System relies on consistent insurance fields; tests use assurance_ann_pct.  ￼
Expected Behavior: Canonical naming + conversion layer at input boundary.

⸻

3. The “Matrix of Pain” (Test Plan)

Component	Scenario	Input Data	Expected Outcome	Type
Optimizer	Missing assurance_ann_pct but present assurance_pret_pct	Brick with only one of the two keys	Fails fast OR normalized to canonical field	Unit
Optimizer+Sim	Simulator bilan lacks tri_global / enrichissement_net	Real simulator output	Optimizer uses correct keys; no silent zeros	Integration
Simulator	IRR NaN	Cashflows all negative	tri_* becomes 0/None + warning flag	Unit
Scoring	Vacancy depends on travaux (bug)	Two identical archetypes, only travaux differ	Vacancy feature should not “improve” from travaux	Unit
StrategyResult	KPI wiring	Bilan has distinct liquidation vs enrichment values	enrichissement_net returns enrichment, not liquidation	Unit
Cashflow Metrics	Year-1 OK, Year-5 bad	df where CF drops after year 2	is_acceptable should fail under new rule	Unit
Allocation	Target CF unreachable under MAX apport	Brick with tiny k-factor + huge target CF	Returns ok=False, deterministic reason	Unit
End-to-End	Consistency across layers	Small dataset 3 bricks	Same top strategy across runs (within tolerance)	E2E
Performance	GA population memory	pop=200, gen=50	No unbounded memory growth; df not stored per individual	Perf
Regression	Schema contract	Snapshot of bilan keys/types	Contract test fails on schema drift	Unit


⸻

4. Recommendations for Refactoring

Untestable / High-Risk Areas (SRP violations)
	•	High risk: GA evaluation function mixes: allocation → schedule building → simulation → constraint checking → scoring → persistence of heavy stats.  ￼
This is hard to test because every test becomes a mini-integration test.

Hardening Steps (do these first)
	1.	Define a strict Bilan contract (Pydantic model or TypedDict) and use it everywhere (simulator output, optimizer scoring, StrategyResult mapping, tests). (#1 is the core break)
	2.	Canonicalize field names at the boundary (assurance_ann_pct, TRI naming, enrichment naming). (#2)
	3.	Remove silent defaults for “important finance fields” (insurance, loan term, rate, bilan keys). Fail fast, or mark invalid with reason.
	4.	Stop storing full df_sim per individual; store only top-K or summaries.  ￼
	5.	Fix qualitative vacancy feature to reflect tension/occupancy, not travaux.  ￼

⸻

5. Calibration & Reality Check (CRITICAL)

Must-fix objective breakages (not style):
	•	🔴 Metric schema mismatch (tri_global/enrichissement_net vs simulator outputs).  ￼
	•	🔴 Insurance naming mismatch causes systematically wrong CF.  ￼  ￼
	•	🔴 StrategyResult KPI wired to wrong value.  ￼
	•	🔴 Vacancy feature computed from travaux ratio.  ￼
	•	🔴 Tests mock the wrong bilan shape, so they won’t catch the above.  ￼

Nice-to-have / design choices (still worth doing, but not “broken”):
	•	🟡 Acceptability based only on Year-1 CF.  ￼
	•	🟡 GA exploration/exploitation tuning (premature convergence risks).

⸻

