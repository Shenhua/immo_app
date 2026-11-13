"""T12: layering rules. Domain and application code must never depend on the UI or Streamlit."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAYERS = ["src/domain", "src/application"]
FORBIDDEN = re.compile(r"^\s*(?:import|from)\s+(?:streamlit|src\.ui|src\s+import\s+ui)\b")


def _python_files() -> list[Path]:
    files: list[Path] = []
    for layer in LAYERS:
        files.extend(sorted((REPO_ROOT / layer).rglob("*.py")))
    return files


def test_layers_have_python_files() -> None:
    """Guard against the test silently passing because the paths moved."""
    assert _python_files(), "no python files found under src/domain or src/application"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_ui_or_streamlit_import(path: Path) -> None:
    """Every import line (including lazy in-function imports) must avoid src.ui and streamlit."""
    offending = [
        f"{lineno}: {line.strip()}"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FORBIDDEN.match(line)
    ]
    assert not offending, f"{path.relative_to(REPO_ROOT)} imports the UI layer: {offending}"


def test_core_package_does_not_reexport_domain_or_application() -> None:
    """Importing src.core must not pull in domain or application modules."""
    init_src = (REPO_ROOT / "src/core/__init__.py").read_text(encoding="utf-8")
    import_lines = [line for line in init_src.splitlines() if re.match(r"^\s*(import|from)\s", line)]
    offending = [line for line in import_lines if "src.domain" in line or "src.application" in line]
    assert not offending, f"src/core/__init__.py re-exports the inner layers: {offending}"
