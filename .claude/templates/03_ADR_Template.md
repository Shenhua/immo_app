# Architecture Decision Record (ADR): [Short Title of Decision]

**Status:** [Proposed / Accepted / Rejected / Deprecated]
**Date:** [YYYY-MM-DD]
**Author:** [Name]

## 1. Context
*What is the technical force or problem that required a decision? Keep it objective.*
*Example: "Our simulation engine currently uses a brute-force approach to evaluate millions of property combinations. As the number of property variables has increased, the calculation time has exceeded our 15-second timeout limit in production."*

## 2. Decision
*What exactly did we decide to do?*
*Example: "We will implement a Genetic Algorithm using DEAP (Distributed Evolutionary Algorithms in Python) to replace the brute-force exhaustive search when permutations exceed 1,000,000."*

## 3. Alternative Options Considered
*Why didn't we do something else? This stops future developers from saying "Why didn't they just do X?"*
* **Option A:** Rewrite the brute-force logic in C++ or Rust.
  * *Pros:* Extremely fast.
  * *Cons:* Requires maintaining two separate codebases and hiring cross-language experts.
* **Option B:** Move calculations to asynchronous background jobs (Celery/Redis).
  * *Pros:* Solves the timeout issue.
  * *Cons:* Does not reduce the actual compute cost; ruins the real-time UX for the customer.

## 4. Consequences
*What happens next because of this decision? Both good and bad.*
* **Positive:** Processing time drops to ~4 seconds for complex portfolios. UX remains real-time.
* **Negative:** We lose the guarantee of finding the absolute mathematical optimum (Genetic Algorithms find a "good enough" local maximum). Also requires the team to learn evolutionary algorithm tuning.
