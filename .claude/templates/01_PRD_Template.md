# Product Requirements Document (PRD): [Feature / Project Name]

**Document Status:** [Draft / Under Review / Approved]
**Author:** [Name]
**Date:** [YYYY-MM-DD]

## 1. Overview & Goals
### 1.1 Problem Statement
*Describe the exact problem the user is facing that requires this feature or product. Why are we building this?*

### 1.2 Target Audience
*Who is the end user? Be specific (e.g., "Retail investors with no technical background").*

### 1.3 Core Value Proposition
*In one sentence, what is the primary benefit this product delivers?*

### 1.4 Success Metrics (KPIs)
*How will we know this feature is successful? (e.g., "Reduce manual calculation time by 90%", "Achieve a 5-second response time").*

---

## 2. Scope
### 2.1 In Scope
* Bulleted list of exact features that WILL be built in this iteration.

### 2.2 Out of Scope
* Bulleted list of features that WILL NOT be built (crucial for preventing scope creep).

---

## 3. User Journeys & Functional Requirements
*Break down the feature from the user's perspective.*

### Scenario 1: [Name of Journey, e.g., "Running a Simulation"]
1. **Given:** The user has provided a €50,000 budget and selected 'High Yield' property types.
2. **When:** The user clicks 'Optimize Portfolio'.
3. **Then:** The system calculates multiple loan durations and returns the top 10 property combinations ranked by Net Enrichment.
4. **Data Inputs Required:** Budget (Int), Property Type (Enum).
5. **Expected Output:** Ranked List of Portfolios.

### Scenario 2: [...]

---

## 4. Business Logic & Constraints
*List all strict rules the system must follow.*
* **Validation Rule 1:** Loan durations cannot exceed 25 years.
* **Math Formula:** `DSCR = Net Operating Income / Total Debt Service`.
* **Constraint:** The system must never suggest a portfolio where monthly cash flow is negative.

---

## 5. Non-Functional Requirements
* **Performance:** The optimization algorithm must return results within 10 seconds.
* **Compatibility:** Must render properly on mobile Safari and Chrome.
* **Security:** All user tax data must be encrypted at rest.
