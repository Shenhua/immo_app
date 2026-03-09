"""Health check script for App Immo Engine.

This script acts as the automated structural verification gate and performs:
1. Syntax Checking: ast.parse() against all .py files
2. Import Validation: check for circular dependencies or missing imports
3. Function Size Cap: flag functions exceeding 50 lines
4. Configuration Verification: check data/archetypes_recale_2025_v2.json
"""

import ast
import json
import os
import sys
from pathlib import Path


def get_all_py_files(root_dir: Path) -> list[Path]:
    """Return all .py files in the given directory tree, excluding tests if necessary.
    We'll include src/ and app.py for structural checks.
    """
    files = []
    # Add app.py
    app_py = root_dir / "app.py"
    if app_py.exists():
        files.append(app_py)
    # Add all under src
    src_dir = root_dir / "src"
    if src_dir.exists():
        for path in src_dir.rglob("*.py"):
            files.append(path)
    return files


def check_syntax(files: list[Path]) -> bool:
    """Check syntax of all Python files using ast.parse."""
    print("Checking Python syntax...")
    all_passed = True
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            try:
                ast.parse(f.read(), filename=str(file))
            except SyntaxError as e:
                print(f"❌ Syntax error in {file}:\n   {e}")
                all_passed = False
    return all_passed


def check_function_size(files: list[Path], max_lines: int = 50) -> bool:
    """Check that no function exceeds max_lines."""
    print(f"Checking function size limit ({max_lines} lines)...")
    all_passed = True
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=str(file))
        except SyntaxError:
            continue  # Handled by syntax check

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Approximate line count: end_lineno - lineno
                # Note: node.end_lineno is available in Python 3.8+
                if hasattr(node, 'end_lineno') and node.end_lineno is not None:
                    line_count = node.end_lineno - node.lineno
                    if line_count > max_lines:
                        # Exclude some known long methods maybe? The spec says "strictly exceeding"
                        print(f"❌ Function '{node.name}' in {file.name}:{node.lineno} exceeds {max_lines} lines ({line_count} lines)")
                        all_passed = False
    return all_passed


def check_imports() -> bool:
    """Check for circular dependencies by trying to import main modules.
    We'll do a simple import of the UI controller which pulls in most of the architecture.
    """
    print("Checking import validity & dependencies...")
    try:
        import src.ui.app_controller
        return True
    except ImportError as e:
        print(f"❌ Import error detected:\n   {e}")
        return False
    except Exception as e:
        print(f"❌ Exception during import evaluation:\n   {e}")
        return False


def check_configuration(root_dir: Path) -> bool:
    """Verify primary data files are present and valid."""
    print("Checking configuration & data availability...")
    data_path = root_dir / "data" / "archetypes_recale_2025_v2.json"
    if not data_path.exists():
        print(f"❌ Missing required data file: {data_path}")
        return False

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"❌ Data file {data_path} root should be a JSON array (list)")
                return False
        return True
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {data_path}:\n   {e}")
        return False


def main():
    root_dir = Path(__file__).parent.parent
    # Ensure root is in sys.path for the import check
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    files = get_all_py_files(root_dir)
    if not files:
        print("❌ No Python files found to check.")
        sys.exit(1)

    print(f"Starting health check on {len(files)} files...\n")
    
    passed_syntax = check_syntax(files)
    passed_imports = check_imports()
    passed_size = check_function_size(files, max_lines=50)
    passed_config = check_configuration(root_dir)

    print("\n--- Summary ---")
    if passed_syntax and passed_imports and passed_config:
        print("✅ Core health checks passed (Syntax, Imports, Config)!")
        if not passed_size:
            print("⚠️  Warning: Some functions exceed the 50-line recommended limit.")
        sys.exit(0)
    else:
        print("❌ Core health check failed. Please resolve the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
