#Part 1

1. Project Identity Card
	•	Project Purpose: A Streamlit “super-simulator” that builds and compares multi-property French real-estate investment strategies (LMNP/micro-BIC), trying to hit a target monthly cashflow while optimizing long-term returns and qualitative risk/attractiveness.  ￼
	•	Current Stack: Python 3.10+, Streamlit UI, pandas/numpy/numpy-financial, Plotly, Pydantic(+pydantic-settings), structlog.  ￼
	•	Architecture Overview (current data flow):
	1.	UI (Streamlit) in app.py builds sidebar inputs, binds session state, and calls an AppController to orchestrate work.  ￼
	2.	Data layer loads archetypes from JSON, applies rent caps when “encadrement” is enabled, and turns archetypes into “investment bricks” (property building blocks).  ￼  ￼
	3.	Strategy generation (StrategyFinder) enumerates combinations of bricks (with pruning), then uses an allocator to distribute capital/apport to meet the cashflow objective (within tolerances).  ￼
	4.	Simulation engine runs a year-by-year long-term projection (tax, amortization, liquidation, IRA), producing IRR/TRI, net-enrichment, DSCR-ish indicators, etc.  ￼
	5.	Scoring & ranking computes a finance score + qualitative score → balanced score, then ranks strategies with a cashflow proximity rule and a configurable qualitative weight.
	6.	Results UI renders a table, details of the selected strategy, and exports/autosaves.  ￼

⸻

2. Technical Health Report

Critical Issues (Bugs / Logic errors)
	1.	Concurrency bug that can crash strategy search: as_completed(..., timeout=0.1) is used on an active set of futures; as_completed can raise TimeoutError if not all futures complete inside the timeout, and the loop isn’t shown handling that exception. This is a “random crash under load” class of bug.  ￼
	2.	Threading likely provides little speedup: Strategy search + simulation are CPU-heavy; ThreadPoolExecutor is used (GIL-bound). You’ll likely see poor scaling vs. expectations, and nondeterministic performance.  ￼
	3.	“Relative normalization” scoring instability: the scorer uses min/max normalization across the candidate set (not absolute business ranges), meaning “scores” can change materially when you add/remove archetypes or change pruning limits; ranking becomes hard to validate and explain to users.
	4.	Cashflow objective is operationally underspecified: allocation is tuned to match a cashflow target (and tolerance), but the code path suggests it’s based on a specific computed cashflow metric (often year-1 / averaged monthly cashflow). If the business intent is “never negative cashflow across horizon” or “stress-tested cashflow,” the current logic won’t enforce it. (Needs confirmation against product intent; see section 3.)  ￼

Dependency Risks
	•	Pinned below NumPy 2.x: project pins numpy<2.0.0  ￼ while NumPy 2.3.0 is current.
	•	This may be intentional for stability, but it also means you’re freezing out newer ecosystems and performance fixes; also watch for future incompatibilities with newer pandas. (Pandas current is 2.3.3. )
	•	Streamlit range is broad but old floor: pinned streamlit>=1.28.0,<2.0.0  ￼ while Streamlit 1.52.1 exists.
	•	Positives: the project already uses version ranges, has CI, and pre-commit with Ruff + MyPy + pytest hooks.  ￼  ￼

“Spaghetti Code” Areas (brittle / high coupling)
	•	Controller does too much orchestration + mutation: AppController handles data loading, rent capping, building config objects, triggering search, simulation, and autosave/export. It’s a “god object” risk: changes to business logic ripple into UI orchestration.  ￼  ￼
	•	Domain models degrade into dicts: even though models exist, the strategy pipeline frequently passes dictionaries (dict[str, Any]) for bricks/projects/results, which reduces type safety and makes regressions easy. (This also makes “explainability” features harder.)  ￼  ￼
	•	Duplicate scoring concepts across layers: src/core/scoring.py defines balanced scoring logic and qualitative aggregation, while StrategyFinder also scores/ranks strategies. This split invites divergence (same “concept” computed differently).  ￼

Missing Standards
	•	No single “source of truth” for domain definitions: e.g., what exactly is cashflow, DSCR, net enrichment, liquidation, risk score, etc., and which timeframe they apply to. Today those are implicit across multiple modules.
	•	Type-checking isn’t strict where it matters most: MyPy is present, but disallow_untyped_defs is not yet enabled broadly (core/services still “gradual”).  ￼
	•	Performance/complexity contracts aren’t enforced: there are pruning limits, but no explicit performance tests/benchmarks and no “max combinations evaluated under X seconds” target.  ￼

⸻

3. Goal of the project vs current logic Report

Interpreted project goal (from code)

“Find the best portfolio strategy under capital and financing constraints that hits a target monthly cashflow (or stays close), then rank by a blended score of financial outcomes (TRI/enrichment/DSCR) and qualitative desirability.”
This intent is clearly reflected by: (a) allocation step to match cashflow target + tolerance, (b) finance + qualitative scoring, (c) rank key prioritizing cashflow proximity.  ￼

Critical mismatches / risks vs that goal
	1.	Ranking is “cashflow-first,” but the financial score may contradict user expectations.
The rank key starts with cashflow proximity gating before considering the balanced score. If a user expects “maximize TRI subject to CF>=0,” they’ll get different results than “closest to CF target even if TRI worse.” This needs an explicit product decision + naming in UI.
	2.	Scoring uses normalization that depends on the candidate set (not on finance reality).
TRI/enrichment/DSCR are normalized against min/max observed in the evaluated strategies; add one outlier and everyone else’s score shifts. This is mathematically valid for “relative ranking,” but problematic for:
	•	reproducibility (“same input → same score?”),
	•	explainability (“why 78/100?”),
	•	and calibration (“what does 70 mean?”).
	3.	DSCR definition looks “single-period-ish.”
A DSCR computed from year-1 style NOI and debt service can be very misleading for amortizing loans + inflation/revalo, and may not reflect downside risk (vacancy spikes, capex, rate stress). If the intent is risk screening, you likely want min-DSCR over a horizon (or stressed DSCR).   ￼
	4.	Business logic for encadrement is applied via a UI-time mutation step.
apply_rent_caps rewrites archetypes before building bricks; if other modules assume “raw archetype” vs “capped archetype,” you can get confusing inconsistencies (especially in explanations/export). It should be a first-class transformation with traceability (“capped from X to Y”).  ￼

⸻

4. Master Development Plan

[ ] Phase 0 (Cleanup): Delete dead code and fix dependencies
	•	Create a “Single Source of Truth” domain glossary (cashflow, DSCR, TRI, enrichment, liquidation, IRA, rent cap, vacancy): name, formula, unit, time basis, sign conventions.
	•	Remove/merge duplicate scoring logic: decide whether src/core/scoring.py is authoritative, then refactor StrategyFinder to call it (or vice-versa) so there’s only one implementation.  ￼
	•	Dependency review policy: document why numpy<2 is pinned; add a scheduled “dependency bump” cadence and compatibility checks.  ￼

[ ] Phase 1 (Stabilization): Fix critical bugs and security risks
	•	Fix as_completed(timeout=0.1) crash risk and make parallel evaluation robust (explicit exception handling, deterministic draining, cancellation, progress reporting).  ￼
	•	Determinism: ensure ranking is stable for same inputs (seed any randomness, deterministic sort keys, consistent tie-breakers).
	•	Hard input validation at boundaries: Pydantic validation for any dict-based inputs entering the core simulation/scoring layers.
	•	Operational reliability: exporter/autosave should be safe under Streamlit Cloud / container FS constraints (explicit paths, permissions, graceful failures).

[ ] Phase 2 (Logic vs Goal): Make a deep audit/etonement report
	•	Re-derive formulas with business owner:
	•	What exactly is the cashflow target constraint? (Year-1 monthly? average? minimum across horizon? stress scenario?)
	•	What is the real scoring objective: “closest to CF target” vs “maximize return under CF>=X”?
	•	Replace relative normalization with business-scaled scoring (or provide both modes):
	•	absolute scales (e.g., TRI 0–20%, DSCR 0.8–2.0, enrichment 0–500k) similar to what core scoring already hints at.  ￼
	•	Risk model clarification: vacancy, liquidity score, DPE/reno costs, rent caps—confirm these match the intended French-market assumptions and avoid double-counting.
	•	Performance envelope: define max archetypes/combos and acceptable runtime; then align pruning (max_combinations, max_properties_per_strategy) with those targets.  ￼

[ ] Phase 3 (Standardization): Add logging, tests, and types
	•	Adopt a clearer architecture boundary (Clean Architecture / Hexagonal):
	•	domain/ (entities + value objects + pure functions),
	•	application/ (use-cases: “SearchStrategies”, “SimulateStrategy”, “ExportResults”),
	•	adapters/ (Streamlit UI, file system exporter),
	•	infrastructure/ (caching, persistence).
	•	Typing program: move away from dict pipelines → typed models (Pydantic/dataclasses). Then enable stricter MyPy incrementally for core and services.  ￼
	•	Testing strategy: keep unit tests for math/finance kernels, add “golden scenario” integration tests (fixed archetype set → expected ranking bands), add performance regression tests. (CI is already in place, extend it.)  ￼

⸻

#Part 2

Yeah—when you have “millions of possibilities,” the biggest algorithmic risk is that “best result found” quietly means “best among what we happened to test,” not “best overall.” The goal is to shrink the search without shrinking the set of genuinely competitive candidates.

Common issues in “find the best result” logic

These are the usual ways an algorithm accidentally misses great options:
	•	Local-optimum trap: greedy choices early on make later good combinations unreachable.
	•	Score noise / non-determinism: if evaluations are noisy, you can crown a “winner” that’s just lucky.
	•	Overfitting to the evaluation set: you optimize for your test harness quirks, not real-world quality.
	•	Hard pruning too early: a partial candidate looks mediocre early but becomes excellent once completed.
	•	Single-metric tunnel vision: you optimize one score and unknowingly destroy a second important dimension (latency, cost, UX constraints, etc.).
	•	Bad tie-breaking: equal scores get broken in a biased way (e.g., always picking the first enumerated).

If any of those are true, your “best” is fragile.

⸻

How to reduce the search space smartly (without cutting off great options)

1) Separate hard constraints from optimization

Before “search,” define a feasible set:
	•	Hard constraints: must-haves (compatibility, legal, max CPU, required features).
	•	Soft objectives: what you optimize (quality, cost, time, subjective preference).

This alone often deletes 80–99% of combos safely because you’re not pruning by “seems worse,” you’re pruning by “invalid.”

⸻

2) Use progressive refinement (coarse → fine)

Instead of testing everything at full fidelity:
	1.	Cheap proxy evaluation (fast approximate scoring)
	2.	Keep top X% (or top K)
	3.	Medium evaluation on survivors
	4.	Full evaluation only on finalists

Key trick: make early stages conservative so you don’t throw away late-bloomers. (Keep wider early, narrower late.)

⸻

3) Use beam search (top-K partial solutions)

If your problem is “build a configuration step by step,” beam search is a great middle ground:
	•	At each step, keep the K best partial candidates (K = beam width)
	•	Expand only those
	•	This avoids exploding combinations while still exploring multiple “promising directions”

To avoid missing weird-but-great paths:
	•	add a small “diversity” rule (e.g., keep some candidates that are different, not just the top K clones).

⸻

4) Add branch-and-bound pruning with upper bounds

This is the safest pruning that doesn’t cut off the true best:
	•	For any partial candidate, compute an optimistic upper bound on how good it could possibly become.
	•	If that upper bound is already worse than your current best, prune it.

This is mathematically clean: you’re not saying “it looks bad,” you’re saying “it cannot possibly win.”

⸻

5) Use smart sampling instead of exhaustive enumeration

If the space is huge and “structure” is unclear:
	•	Start with stratified / Latin hypercube sampling (covers the space better than random)
	•	Identify regions that produce strong results
	•	Spend more trials there

This prevents the classic failure mode where you “random sample” and accidentally ignore an entire important region.

⸻

6) Use Bayesian optimization / bandits for expensive evaluations

If each test is expensive:
	•	Build a surrogate model predicting score from parameters
	•	Choose next trials via exploration/exploitation (expected improvement / Thompson sampling)
	•	Re-train as you collect data

This can cut millions of trials down to hundreds or thousands while still finding near-optimal configurations.

⸻

7) Do sensitivity analysis and freeze low-impact dimensions

After a few hundred runs:
	•	Which parameters actually move the score?
	•	Which ones barely matter?

Freeze or coarsen the low-impact ones (e.g., 20 values → 3 buckets). This slashes combinatorics without losing quality.

⸻

A practical recipe that works in real projects
	1.	Define constraints + metrics (including “must not regress” metrics).
	2.	Run broad coverage sampling (enough to see patterns).
	3.	Fit a quick model / do sensitivity analysis.
	4.	Switch to beam search or Bayesian optimization depending on whether it’s “constructive” or “continuous.”
	5.	Use multi-stage evaluation (cheap → expensive).
	6.	Keep a Pareto frontier, not a single winner, until the very end.
	7.	Re-test top candidates multiple times to beat noise.

⸻



#Part 3

Issues in how it finds “best” results

1) Early termination can bias the winners

StrategyFinder stops after collecting a limited number of strategies (STRATEGY_MAX_VIABLE, STRATEGY_MAX_TOTAL).  ￼
That’s not inherently bad, but which strategies you see first matters because combinations are generated in a deterministic order: bricks are sorted by apport_min and then itertools.combinations runs in that order.  ￼
Result: you tend to evaluate lots of “cheap-apport” combos first, and if early termination triggers, you might never evaluate combos that are more expensive on apport_min but far better on IRR/enrichment/quality.

2) Your scoring is relative to the tested population

Finance metrics are min-max normalized over the strategies that were evaluated.  ￼  ￼
If early termination biases the evaluated set, the normalization and final ranking become self-referential: “best among a biased sample,” not “best overall.”

3) Combination generation is “prune-lite”

CombinationGenerator prunes only on:
	•	“brick individually affordable”
	•	“no duplicate nom_bien”
	•	“sum(apport_min) <= apport”
	•	a valid but only budget-related K-level bound (cheapest K unique > budget → stop increasing K)  ￼

There’s no pruning based on CF feasibility, DSCR feasibility, yield, or quality—so millions of combos can survive to the expensive steps.

4) You simulate even when allocation didn’t hit target (by design)

The pipeline keeps strategies even if allocation doesn’t meet the cashflow target (“SOFT FAILURE”), expecting scoring to penalize them later.  ￼
That’s fine, but it increases the candidate count massively unless you add earlier-stage screening.

⸻

How to reduce millions of possibilities without losing best options

A) Fix the search bias first (otherwise pruning is risky)

If you keep early termination, you need to avoid “cheap-first bias”:
	•	Shuffle / stratify combo exploration so you cover the space (by cost buckets, city, yield bands, quality bands) before you stop.
	•	Or replace stop-after-N with a time budget + diversity quota (e.g., always keep the best K per bucket).

This is the single highest-leverage change because it makes all further pruning “safer.”

⸻

B) Add dominance pruning at the brick level (safe and powerful)

Before building combinations, drop “dominated” bricks:
	•	If Brick A has higher apport_min and lower/equal rent, lower/equal qualitative score, and lower/equal yield than Brick B, then A can never be part of an optimal solution (for any reasonable scoring mix).
	•	This often shrinks the brick set by 30–80% without touching the true optimum.

This is much safer than arbitrary thresholds because it’s mathematically grounded.

⸻

C) Switch from “generate all combos” to incremental search with bounds

Right now you materialize all valid combinations then process them.  ￼
Instead, explore combos incrementally and prune partials:
	•	Branch-and-bound: while building a partial combo, compute an optimistic upper bound of how good it could become if you added the best remaining bricks. If that upper bound can’t beat your current top-K, prune the branch.
	•	Pareto frontier pruning for partial combos: for partial combos, keep only non-dominated states across a few cheap dimensions, e.g. (apport_min_total, cf_proxy, qual_proxy, renta_brute). Discard dominated partials early.

These reduce candidates drastically without cutting off winners.

⸻

D) Use multi-stage evaluation (cheap → expensive)

Right now, each combo goes through allocation and then full simulation.  ￼
Instead:
	1.	Stage 1 (very cheap): compute quick proxies (sum of initial rents, rough debt service proxy, gross yield, qual score).
	2.	Stage 2 (medium): run allocator only; keep top M per bucket based on CF proximity + a proxy “return potential”.
	3.	Stage 3 (expensive): run full simulation only on the survivors.

This is how you preserve global optima: you’re not throwing away “weird-but-good” candidates early, you’re deferring expensive evaluation.

⸻

E) Make early termination “quality aware”

Right now it stops after a count.  ￼
Better: stop when the search has “converged,” e.g.:
	•	you haven’t improved the top-K by >X% for N iterations, and
	•	you’ve covered each diversity bucket at least Q times.

That prevents stopping too early in one region of the space.

⸻

Bonus: you’ll also get speed back

Your workload is CPU-heavy and uses ThreadPoolExecutor  ￼; even with perfect pruning, you’ll likely hit diminishing returns due to the GIL. But algorithmic pruning usually beats parallelism anyway.

⸻

If you want, I can point to the exact places to insert “dominance pruning” and “multi-stage evaluation hooks” in the current pipeline (still no new code), but the key takeaway is:

Right now the “best result” can be “best among the first 500 strategies,” and the first 500 are biased by sorting + early termination. Fixing that + adding dominance/bounds gets you huge reduction without losing the true best.



# Part 4

Yes — and your “fractional step increments” idea is exactly the right lever as long as the step size is tied to what actually moves the objective (cash-flow proximity + feasibility), not just “nice round numbers”.

Issues in the current “best result” search (that can hide the true best)
	1.	Hard cap on explored combinations
The settings include max_combinations (default 10,000). That means you can have “millions of possibilities”, but the engine will stop early and never even see many candidates.  ￼
	2.	Cash-flow “precise targeting” can be silently broken by the allocation rounding step
In the allocator, extra downpayment (delta) is rounded to apport_disponible / 1000 (min 1€).  ￼
This is independent of the tolerance you set in the UI (tolerance step=10€)  ￼.
So if you’re in precise mode (± tolerance), your allocation may jump over the feasible window and miss solutions that exist (especially when apport/1000 implies a cash-flow jump larger than the tolerance).
	3.	Greedy allocation + greedy exploration = local optimum risk
The allocation is iterative greedy (sort by efficiency, apply delta, recompute)  ￼. Greedy is fast, but it can miss “complementary” portfolios where a slightly-worse first choice enables a much better overall solution.

⸻

How to reduce the search space without losing top options

The safe approach is coarse-to-fine + goal-aware discretization:
	•	Coarse pass: Use bigger steps to find the best “regions” quickly.
	•	Refinement pass: Re-run only around the top-K candidates with smaller steps (or smaller tolerances / finer delta).
	•	Goal-aware step sizing: Choose steps so that each step causes a meaningful change in the objective (e.g., cash-flow changes by ~½ tolerance). If a step changes CF by 0.5€, it’s noise; if it changes CF by 200€ you’ll skip good solutions.

⸻

Proposed step increments (sound defaults) for all parameters you currently expose

A) Portfolio search controls

These are defined in your sidebar “Configuration Financière” block.  ￼
	•	Apport total disponible (apport)
	•	Current UI step: 5,000€  ￼
	•	Suggested discretization:
	•	Coarse: max(5,000€, round(apport * 2%))
	•	Refine: max(500€, round(apport * 0.5%))
	•	Logic: search space grows quickly with capital granularity; 0.5–2% gives stable coverage without exploding combinations.
	•	Cash-flow cible (cf_cible)
	•	Current UI step: 50€/mois  ￼
	•	Suggested discretization:
	•	Coarse: step = max(50€, round(tolerance))
	•	Refine: step = max(10€, round(tolerance / 2))
	•	Logic: the only useful resolution is relative to tolerance. If tolerance is 30€, a 50€ step is too coarse.
	•	Tolérance (tolerance)
	•	Current UI step: 10€  ￼
	•	Suggested discretization:
	•	Coarse: 25€
	•	Refine: 10€ (current)
	•	Logic: tolerance defines the acceptance band; coarse tolerance widens feasibility early, then tighten.
	•	Horizon (horizon)
	•	Current: integer slider 10→30 (implicit step 1)  ￼
	•	Suggested discretization:
	•	Coarse: 5 years
	•	Refine: 1 year
	•	Logic: long-term metrics usually don’t change meaningfully year-by-year in early exploration.
	•	Qualité vs Finance priority (priorite_pct → qualite_weight)
	•	Current step: 5%  ￼
	•	Suggested discretization:
	•	Keep 5% (it’s already only 21 values)
	•	Optional refine: 2% only in a second pass
	•	Logic: weight changes are second-order; avoid turning this into a dimension explosion.
	•	max_properties / use_full_capital
	•	They’re set by strategy mode and returned in the objectives dict.  ￼
	•	Suggested:
	•	Don’t grid-search these. Treat as scenario variants (e.g., run once with 2/3/4 properties if you need).

⸻

B) Credit parameters (huge impact, but don’t over-discretize)

Credit tab currently defines:
	•	Rates: 15/20/25 years (taux_15/20/25)  ￼
	•	Notary / insurance / loan fees steps  ￼
	•	IRA cap step=0.1  ￼

Suggested discretization:
	•	Interest rates (taux_15/20/25)
	•	Current: number_input with 2 decimals (no explicit step)  ￼
	•	Suggested:
	•	Coarse: 0.10%
	•	Refine: 0.05%
	•	Logic: smaller than 0.05% usually adds combinations without changing ranking meaningfully.
	•	Notary fees (frais_notaire_pct)
	•	Current step: 0.1%  ￼
	•	Suggested:
	•	Coarse: 0.25%
	•	Refine: 0.10% (current)
	•	Logic: notary fees are mostly “macro”; 0.1% is fine only in refinement.
	•	Insurance (assurance_ann_pct)
	•	Current step: 0.01% cap  ￼
	•	Suggested:
	•	Coarse: 0.02%
	•	Refine: 0.01% (current)
	•	Logic: insurance strongly affects payment; keep refinement option, but don’t brute force 0.01 across wide ranges.
	•	Loan fees (frais_pret_pct)
	•	Current step: 0.1%  ￼
	•	Suggested:
	•	Coarse: 0.2%
	•	Refine: 0.1% (current)
	•	IRA cap (ira_cap_pct)
	•	Current step: 0.1  ￼
	•	Suggested:
	•	Keep 0.1 (it’s not a combinatorial driver unless you sweep it)
	•	Cost inclusion toggles (travaux/reno/mobilier/financé)
	•	Keep as categorical scenarios only (don’t brute-force all combos unless you need them).

⸻

C) Market hypotheses (don’t let this dimension explode)

All three use step 0.1 today.  ￼
	•	Appreciation / rent revaluation / inflation charges
	•	Current step: 0.1%  ￼
	•	Suggested:
	•	Coarse: 0.5%
	•	Refine: 0.1% (current)
	•	Logic: these mainly affect long-term outcomes; you don’t want them multiplying your first-pass search space.

⸻

D) Internal allocator step (this one matters for “millions of possibilities” thinking)

Allocator rounds delta like this: step_size = max(apport_disponible / 1000, 1)  ￼

Recommended replacement rule (conceptually):
	•	Make step_size depend on tolerance, especially in precise mode:
	•	Min-mode (CF >= target): step_size ≈ apport/1000 is OK.
	•	Precise mode (± tolerance): step_size should be small enough so one step doesn’t change CF by more than ~½ tolerance.

Practically, you can set:
	•	Coarse: step_size = max(100€, apport/2000)
	•	Precise: step_size = max(10€, min(apport/10000, 0.5 * tolerance / k_max))
where k_max is the best (largest) k_factor among the candidate properties in that strategy (you already compute k-factors).  ￼

That gives you the big search-space reduction (you avoid tiny steps everywhere), while still allowing the allocator to “land” inside the tolerance band when required.

⸻

If you want the single best “smart reduction” rule

Tie every continuous step to the acceptance band.
For anything that ultimately decides acceptance/ranking through cash-flow targeting, a good universal heuristic is:

Choose steps so that a 1-step move changes monthly CF by about ½ tolerance (coarse) or ⅕ tolerance (refine).

That’s the sweet spot: big pruning, low risk of skipping the best.

If you want, I can also point out where to compute k_max cleanly (you already have the ingredients in the allocator loop) — still no new code, just the exact best insertion points.


You’re right to call that out — my earlier suggestion didn’t respect how your code currently models loan duration, and also didn’t account for how badly “1–25 years” would blow up the combinatorics if implemented the naïve way.

What the code does today (so the logic constraints are clear)
	•	The UI only captures 3 credit rates and returns them as a dict keyed by {15, 20, 25}.  ￼
	•	When generating “bricks”, the engine creates variants per duration by iterating for duree, taux in finance.credit_rates.items() and computing payment for each duration.  ￼

So today, “loan duration steps” aren’t a 1..25 slider — they’re literally 3 discrete variants per property.

If you switch to durations 1..25 and keep the same pattern (“create a brick variant for every year”), you multiply the brick count by ~25/3 ≈ 8.3×, and combinations explode accordingly. That’s the opposite of what you want.

⸻

Revised recommendation: “useful precision” rules + corrected step increments

Rule 1 — Don’t discretize things that don’t participate in the combinatorial explosion

Most of your sidebar parameters (market hypotheses, fees) are single run inputs; they don’t create millions of possibilities unless you decide to sweep them. So we should only propose “step increments” for:
	•	parameters you do want to sweep, or
	•	parameters that internally behave like discretization (your loan duration variants and allocator stepping).

Rule 2 — If a parameter is piecewise constant (like your “rates change at 15/20/25”), don’t over-sample inside a flat region

This is exactly your point: if the rate is constant on a band, stepping inside the band adds search cost without adding information.

⸻

A) Loan duration (corrected logic)

If you want “duration 1–25 years” with rates only changing at 15/20/25

That means the rate is piecewise constant and duration mainly changes cashflow via the amortization length.

Best practice to avoid combinatorial explosion:
	•	Do not generate brick variants for every year.
	•	Instead, treat duration as a derived choice during evaluation:
	•	for each property (or each strategy), pick the duration that best meets the objective (min/target cashflow) because payment is monotonic with duration.

Sound step increments:
	•	Coarse exploration: only test durations {15, 20, 25} (what you do today).  ￼
	•	Refinement (only for top candidates): test 1-year increments only near the best duration:
	•	e.g., if the best is 20, test [18..22] by 1 year.
	•	Full 1..25 by 1 year should exist as a final refinement mode, not as a default search dimension.

That honors your “rates change only at thresholds” logic and prevents your brick space from exploding.

⸻

B) Cashflow targeting parameters (precision matters here)

These do interact with feasibility and ranking.
	•	Cash-flow cible currently step 50€/mois in UI.  ￼
Sweeping step (if you sweep it):
	•	Coarse: 100€
	•	Refine: 25–50€
	•	Logic: below ~25€ you’re typically within noise given all other assumptions.
	•	Tolérance currently step 10€.  ￼
Sweeping step:
	•	Coarse: 25€
	•	Refine: 10€
	•	Logic: tolerance is a band, not a scalar objective. Too fine creates fake precision.
	•	Internal allocator “apport delta” step (this is real discretization today)
It uses step_size = max(apport_disponible / 1000, 1) (so ~50€ steps if apport=50k).  ￼
Correct “useful precision” logic:
	•	In precise targeting (± tolerance), the step size must be small enough to not jump over the tolerance window.
	•	In min mode, bigger steps are OK.
Recommendation:
	•	Min mode: keep ~apport/1000
	•	Precise mode: clamp step_size so that a step doesn’t move CF by more than ~½ tolerance (the exact conversion uses your k-factor / payment sensitivity).

This is one of the biggest “precision that matters” fixes.

⸻

C) Capital parameters (don’t over-precise)
	•	Apport total UI step is 5,000€.  ￼
Sweeping step:
	•	Coarse: 10,000€
	•	Refine: 2,500–5,000€
	•	Logic: apport mainly affects feasibility and the allocator; sub-1000€ sweeps aren’t meaningful until final tuning.

⸻

D) Horizon + market hypotheses (over-precision is mostly useless early)
	•	Horizon slider 10–30 years (step 1).  ￼
Sweeping step:
	•	Coarse: 5 years
	•	Refine: 1 year
	•	Logic: only matters for long-term metrics; don’t multiply scenarios early.
	•	Market hypotheses (appreciation / revalo / inflation) currently step 0.1%.  ￼
Sweeping step:
	•	Coarse: 0.5%
	•	Refine: 0.25%
	•	Keep 0.1% only for “final sensitivity analysis”, not for search.

⸻

E) Fees & insurance (very easy to overfit)

These do affect results, but sweeping them finely is rarely worth it.
	•	Frais de notaire slider step 0.1%.  ￼
Sweep: coarse 0.5%, refine 0.25%
	•	Assurance slider step 0.01%.  ￼
Sweep: coarse 0.05%, refine 0.02% (0.01 only for sensitivity checks)
	•	Frais de prêt slider step 0.1%.  ￼
Sweep: coarse 0.2%, refine 0.1%

⸻

The “million possibilities” killer you should avoid

If you add parameters (like duration 1..25) and treat them as another axis to brute-force at brick-generation time, you’ll explode the search.

The pattern that stays efficient is:
	•	Coarse discrete choices early
	•	Refinement only around winners
	•	Use monotonicity (duration→payment) to solve, not enumerate

If you want, I can extract the full list of “setup parameters” you mean (some are in UI, some are in config/constants) and label each one explicitly as:
	•	“don’t sweep”
	•	“sweep coarse”
	•	“refine near winners”
	•	“derive analytically (don’t enumerate)”

…but the biggest correction is already: loan duration should not become 25× more bricks.