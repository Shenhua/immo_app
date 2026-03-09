# 🏠 App Immo - Real Estate Investment Strategy Simulator

> Version 27.6.0 | A Streamlit-based web application that simulates and ranks real estate investment strategies for French investors.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Features

- **Strategy Search**: Find optimal property combinations matching your investment criteria
- **Financial Simulation**: 25-year projections with IRR, DSCR scores, and cash flow analysis
- **Qualitative Scoring**: Rate properties on location, transport, DPE, and market tension
- **Comparison View**: Side-by-side strategy comparison with interactive charts
- **JSON Export**: Save simulation results for further analysis

## Project Structure

```
app_immo/
├── app.py                    # Streamlit entry point
├── src/
│   ├── models/              # Pydantic data models
│   ├── core/                # Financial & scoring engines
│   ├── services/            # Business logic services
│   └── ui/                  # UI components
├── tests/                   # Unit tests
├── data/                    # Archetype JSON data
├── pyproject.toml           # Project configuration
└── requirements.txt         # Dependencies
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check .

# Run type checker
mypy src/
```

## Configuration

Copy `.env.example` to `.env` and adjust settings:

```bash
LOGLEVEL=INFO
```

## License

MIT
