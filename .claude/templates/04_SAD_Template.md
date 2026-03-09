# System Architecture Document (As-Built): [System Name]

**Document Status:** Living Document
**Last Updated:** [YYYY-MM-DD]
**Author:** [Name]

## 1. Executive Summary
*What is this system currently doing in production? Provide a high-level summary of the codebase's purpose.*

## 2. Global Architecture & Stack
*List the current technologies in use.*
* **Language/Runtimes:** [e.g., Python 3.11]
* **Frontend:** [e.g., Streamlit]
* **Data Layer:** [e.g., Pandas / Postgres]

## 3. Component Interaction & Workflows
*Describe the macro-level data flow as it exists today. How do the primary modules talk to each other? What is the lifecycle of a core user request through the system?*
1. **User Input:** Flow starts at `app.py`.
2. **Generation:** Hits `service.py`.
3. **Database:** Saves to `db.py`.

## 4. Global Data Dictionary
*A consolidated list of the most critical foundational data models, schemas, and state objects currently active in the codebase.*
* **`UserModel`**: Handles authentication and profile data.
* **`SimulationState`**: Handles active memory for user inputs.

## 5. Detailed Module Breakdown
*Iterate through the core physical chunks/folders of your repository.*

### [Module / Folder Name]
* **Files Included:** `foo.py`, `bar.py`
* **Purpose:** What does this cluster of files achieve?
* **Core Logic:** How it calculates or routes data.

## 6. External Integrations
*What third-party services does this system connect to?*
* **Stripe:** Located in `payments.py` for subscription billing.
* **AWS S3:** Located in `storage.py` for image handling.

## 7. Known Defects & Technical Debt Register
*A crucially important list of known hacks, bad code, or bugs that are currently in production and need refactoring eventually.*
* **High:** `billing.py` triggers an N+1 query loop when fetching invoices.
* **Medium:** `utils.py` suppresses exceptions via `except Exception: pass`, hiding errors.

## 8. Strategic Enhancement Roadmap
*Provide architectural recommendations based on the aggregated technical debt. What is the next major refactor?*
