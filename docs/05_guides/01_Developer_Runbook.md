# Developer Runbook & Local Setup Guide

**Document Status:** Approved 
**Author:** AI Technical Lead
**Date:** 2026-03-06

## 1. Quick Start Guide
If you are a new developer setting up the App Immo engine locally, this section contains the exact terminal commands required to run the simulation engine on your machine.

### 1.1 Environment Setup
```bash
# 1. Ensure you have Python 3.10+ installed
python --version

# 2. Navigate to the project root
cd /path/to/app_immo

# 3. Create a clean virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 4a. Install with pyproject.toml (preferred — includes all extras)
pip install -e ".[dev]"

# 4b. Alternative: install from requirements.txt (pinned, synced with pyproject.toml)
# pip install -r requirements.txt
```

### 1.2 Booting the Application
```bash
# Launch the Streamlit local server
streamlit run app.py
```
*The Streamlit web interface will open automatically in your default browser at `http://localhost:8501`.*

---

## 2. Daily Development Workflows

The codebase is highly regulated using modern Python static tooling to prevent logic bugs in the simulation loop. 

### 2.1 Formatting & Linting
App Immo uses `Ruff` for simultaneous ultra-fast linting and code formatting.
```bash
# Check for linting errors without modifying files
ruff check .

# Automatically fix linting errors (unused imports, spacing)
ruff check . --fix

# Format the entire codebase to the standard Style Guide (120 line length)
ruff format .
```

### 2.2 Type Checking
The Core and Domain layers must be entirely statically typed.
```bash
mypy src/
```

### 2.3 Executing the Test Suite
Refer to `docs/02_design/04_QA_and_Testing_Strategy.md` for methodology.
```bash
# Run the entire test suite (Unit, Integration, E2E)
pytest tests/ -v

# Run only the fast unit tests (Recommended before every atomic commit)
pytest tests/unit/ -q --tb=no

# Run tests with Code Coverage reporting (Requires pytest-cov)
pytest tests/ -v --tb=short --cov=src --cov-report=xml
```

---

## 3. Operational Telemetry & Debugging

When the optimizer crashes or a financial result seems statistically incorrect, utilize the following diagnostic tools.

### 3.1 Environment Variable Toggles
The engine responds to `.env` or temporary terminal environment variables.
```bash
# Enable trace-level JSON logging for debugging the genetic algorithm
LOGLEVEL=DEBUG streamlit run app.py

# Force the allocator to modify capital scaling bounds to test edge cases
MAX_EXTRA_APPORT_PCT=0.99 streamlit run app.py
```

### 3.2 Reading `structlog` Outputs
The application utilizes `structlog`.
*   **In Development:** Logs render natively in the terminal utilizing colored blocks for standard tracebacks.
*   **In Production:** Logs convert mathematically to uncolored single-line `JSON` blocks designed for ingestion by DataDog or ELK stacks. Do not attempt to parse JSON outputs via `grep` manually; utilize `jq`.

### 3.3 Utilizing `debug.py` State Dumps
If a user hits a bug in the Streamlit runtime, the UI state is completely volatile. 
*   **Execution:** A function call to `collect_debug_context()` will forcefully serialize ALL active memory variables (Archetype limits, Session States, Current Matrices).
*   **Output:** It generates a highly dense JSON file on the local disk.
*   **Triage Context:** Developers should pass this exact JSON file into an LLM or diagnostic engine to perfectly recreate the frontend state parameters that caused a backend engine math error.

---

## 4. Subsystem Verification (`health_check.py`)

> ⚠️ **`src/health_check.py` is not yet implemented.** This section documents the intended behavior once the module is created (see QA Strategy §4.1 for the planned gate design).

Once implemented, the intended usage will be:
```bash
python src/health_check.py
```
This will isolate the system from the frontend and assert: (1) all `.py` files parse without syntax errors, (2) no circular imports exist, (3) function sizes are within bounds, (4) `data/archetypes_recale_2025_v2.json` is present and valid.
