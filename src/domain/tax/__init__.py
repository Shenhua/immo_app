"""Tax regime engine (TAX_SPEC 4, 7, 8). Pure domain code: no streamlit, no src.ui."""

from src.domain.tax.base import (
    DeficitPool,
    ExitContext,
    ExitTax,
    PropertyExit,
    PropertyTaxState,
    RegimeId,
    TaxRegime,
    TaxResult,
    TaxState,
    YearContext,
)
from src.domain.tax.household import (
    TaxConfig,
    TaxHousehold,
    parse_furnished_regime,
    parse_unfurnished_regime,
)
from src.domain.tax.individuals import FoncierReel, LmnpReel, MicroBic, MicroFoncier
from src.domain.tax.sci_is import SciIs

__all__ = [
    "DeficitPool",
    "ExitContext",
    "ExitTax",
    "FoncierReel",
    "LmnpReel",
    "MicroBic",
    "MicroFoncier",
    "PropertyExit",
    "PropertyTaxState",
    "RegimeId",
    "SciIs",
    "TaxConfig",
    "TaxHousehold",
    "TaxRegime",
    "TaxResult",
    "TaxState",
    "YearContext",
    "parse_furnished_regime",
    "parse_unfurnished_regime",
]
