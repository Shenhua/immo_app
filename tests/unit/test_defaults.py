"""T10: one source of truth for defaults (src/core/defaults.py)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.application.run_context import RunContext
from src.application.services import strategy_finder
from src.application.services.allocator import PortfolioAllocator
from src.application.services.brick_factory import FinancingConfig, OperatingConfig
from src.application.services.evaluator import StrategyEvaluator
from src.application.services.simulation import MarketHypotheses, SimulationEngine, TaxParams
from src.application.services.strategy_finder import BASE_WEIGHTS, EvaluationParams, StrategyScorer
from src.core import defaults
from src.core.defaults import FINANCING, MARKET, OPERATING, PRESETS, SEARCH, PresetId
from src.domain.calculator.scoring import calculate_balanced_score
from src.domain.models.strategy import SimulationParams
from src.ui.components import sidebar

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDEBAR_FILES = [REPO_ROOT / "src/ui/components/sidebar.py", REPO_ROOT / "app.py"]


def test_presets_sum_to_one_and_cover_every_id() -> None:
    assert set(PRESETS) == set(PresetId)
    for pid, w in PRESETS.items():
        assert sum(w.as_dict().values()) == pytest.approx(1.0), pid


def test_base_weights_equal_balanced_preset() -> None:
    assert BASE_WEIGHTS == PRESETS[PresetId.BALANCED].as_dict()
    assert StrategyEvaluator.DEFAULT_WEIGHTS == BASE_WEIGHTS
    assert StrategyScorer(mode_cf="min").weights == pytest.approx(BASE_WEIGHTS)


def test_sidebar_presets_come_from_defaults() -> None:
    assert sidebar.FINANCIAL_PRESETS == {pid.label: w.as_dict() for pid, w in PRESETS.items()}


def test_preset_id_parse() -> None:
    assert PresetId.parse("Sécurité (DSCR)") is PresetId.SECURITY
    assert PresetId.parse("Équilibré") is PresetId.BALANCED
    assert PresetId.parse(PresetId.WEALTH) is PresetId.WEALTH
    assert PresetId.parse("irr") is PresetId.IRR
    assert PresetId.parse(None) is PresetId.BALANCED
    assert PresetId.parse("nonsense") is PresetId.BALANCED
    for pid in PresetId:
        assert PresetId.parse(pid.label) is pid


def test_rank_strategies_uses_preset_id() -> None:
    finder = strategy_finder.StrategyFinder(bricks=[], apport_disponible=0.0, cash_flow_cible=0.0)
    strategies = [
        {"tier": 1, "dscr_norm": 0.9, "tri_norm": 0.1, "finance_score": 1, "balanced_score": 1},
        {"tier": 1, "dscr_norm": 0.1, "tri_norm": 0.9, "finance_score": 1, "balanced_score": 1},
    ]
    assert finder.rank_strategies(strategies, PresetId.SECURITY)[0]["dscr_norm"] == 0.9
    assert finder.rank_strategies(strategies, PresetId.IRR)[0]["tri_norm"] == 0.9
    assert finder.rank_strategies(strategies, "Sécurité (DSCR)")[0]["dscr_norm"] == 0.9


def test_drifting_defaults_are_unified() -> None:
    assert OPERATING.cfe_par_bien_ann == 300.0
    assert EvaluationParams.from_dict({}).cfe_par_bien_ann == 300.0
    assert SimulationEngine(MarketHypotheses(), TaxParams(), None).cfe_par_bien_ann == 300.0  # type: ignore[arg-type]
    assert SimulationParams().cfe_par_bien_ann == 300.0
    assert RunContext().cfe_par_bien_ann == 300.0
    assert OperatingConfig().cfe_par_bien_ann == 300.0

    assert MARKET.appreciation_bien_pct == 2.0
    assert MarketHypotheses().appreciation_bien_pct == 2.0
    assert RunContext().appreciation_bien_pct == 2.0

    assert FINANCING.taux_credits == {15: 3.5, 20: 3.6, 25: 3.8}
    assert RunContext().taux_credits == FINANCING.taux_credits
    assert FINANCING.frais_notaire_pct == 8.0
    assert RunContext().frais_notaire_pct == 8.0
    assert FinancingConfig(credit_rates={}).frais_notaire_pct == 8.0

    assert SEARCH.qualite_weight == 0.25
    assert RunContext().qualite_weight == 0.25
    assert StrategyScorer(mode_cf="min").qualite_weight == 0.25
    assert StrategyEvaluator(simulator=None).quality_weight == 0.25  # type: ignore[arg-type]
    assert calculate_balanced_score.__kwdefaults__["qualite_weight"] == 0.25


def test_allocator_reads_env_default_from_defaults() -> None:
    assert PortfolioAllocator().max_extra_apport_pct == defaults.MAX_EXTRA_APPORT_PCT


def test_results_dir_is_redirected_during_tests() -> None:
    """conftest points APP_RESULTS_DIR at a temp dir so tests never write to the repo's results/."""
    assert Path(defaults.RESULTS_DIR).resolve() != (REPO_ROOT / "results").resolve()


# --- Sidebar widget defaults --------------------------------------------------
_WIDGET_VALUE = re.compile(r"\bvalue\s*=\s*([^,\n)]+)")
_NUMERIC_LITERAL = re.compile(r"^[-+]?\d+(\.\d+)?$")


@pytest.mark.parametrize("path", SIDEBAR_FILES, ids=lambda p: p.name)
def test_sidebar_widgets_have_no_numeric_literal_defaults(path: Path) -> None:
    """Widgets take value= from src/core/defaults.py, never from a literal number."""
    offending = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        for match in _WIDGET_VALUE.finditer(line):
            if _NUMERIC_LITERAL.match(match.group(1).strip()):
                offending.append(f"{lineno}: {line.strip()}")
    assert not offending, f"{path.name} has numeric literal value= defaults: {offending}"


@pytest.mark.parametrize("path", SIDEBAR_FILES, ids=lambda p: p.name)
def test_sidebar_has_no_positional_numeric_defaults(path: Path) -> None:
    """Slider / number_input calls pass their default as value=<defaults constant>."""
    text = path.read_text(encoding="utf-8")
    # st.slider("label", lo, hi, <default>, ...) with a bare numeric 4th positional argument.
    positional = re.findall(
        r"st\.(?:slider|number_input)\(\s*\"[^\"]*\",\s*[-\d.]+,\s*[-\d.]+,\s*[-\d.]+\s*[,)]", text
    )
    assert not positional, positional


def test_sidebar_uses_defaults_module() -> None:
    for path in SIDEBAR_FILES:
        text = path.read_text(encoding="utf-8")
        assert "from src.core.defaults import" in text, path.name
