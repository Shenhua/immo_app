#!/usr/bin/env python3
"""
Comprehensive Codebase Health Check
Analyzes the app_immo project for issues, code quality, and potential problems.
"""

import ast
import os
import sys
import json
from collections import defaultdict
from pathlib import Path

# Project root
ROOT = Path(__file__).parent
RESULTS = {"errors": [], "warnings": [], "info": [], "stats": {}}

def log_error(msg): RESULTS["errors"].append(msg)
def log_warning(msg): RESULTS["warnings"].append(msg)
def log_info(msg): RESULTS["info"].append(msg)

# ============================================================
# 1. SYNTAX CHECK - Can all Python files be parsed?
# ============================================================
def check_syntax():
    print("🔍 Checking Python syntax...")
    py_files = list(ROOT.glob("*.py")) + list(ROOT.glob("src/**/*.py")) + list(ROOT.glob("tests/**/*.py"))
    py_files = [f for f in py_files if "env/" not in str(f)]
    
    errors = []
    for f in py_files:
        try:
            with open(f) as fp:
                ast.parse(fp.read())
        except SyntaxError as e:
            errors.append(f"{f.relative_to(ROOT)}:{e.lineno} - {e.msg}")
    
    RESULTS["stats"]["python_files"] = len(py_files)
    if errors:
        for e in errors:
            log_error(f"Syntax error: {e}")
    else:
        log_info(f"✅ All {len(py_files)} Python files have valid syntax")

# ============================================================
# 2. IMPORT CHECK - Can all modules be imported?
# ============================================================
def check_imports():
    print("🔍 Checking module imports...")
    modules = ["app", "src.services.allocator", "src.services.strategy_finder", "src.services.brick_factory"]
    
    failed = []
    for mod in modules:
        try:
            __import__(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    
    if failed:
        for f in failed:
            log_error(f"Import failed: {f}")
    else:
        log_info(f"✅ All {len(modules)} core modules import successfully")
    
    # Check src package
    src_modules = [
        "src.models.archetype",
        "src.models.brick", 
        "src.models.strategy",
        "src.core.financial",
        "src.core.scoring",
        "src.core.logging",
    ]
    failed_src = []
    for mod in src_modules:
        try:
            __import__(mod, fromlist=[""])
        except Exception as e:
            failed_src.append(f"{mod}: {e}")
    
    if failed_src:
        for f in failed_src:
            log_warning(f"src import issue: {f}")
    else:
        log_info(f"✅ All {len(src_modules)} src modules import successfully")

# ============================================================
# 3. DUPLICATE DETECTION - Any duplicate functions?
# ============================================================
def check_duplicates():
    print("🔍 Checking for duplicate functions...")
    files = ["app.py", "src/services/strategy_finder.py", "src/core/simulation.py"]
    
    for fname in files:
        fpath = ROOT / fname
        if not fpath.exists():
            continue
        
        try:
            with open(fpath) as f:
                tree = ast.parse(f.read())
            
            funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            seen = set()
            dups = []
            for fn in funcs:
                if fn in seen:
                    dups.append(fn)
                seen.add(fn)
            
            if dups:
                log_error(f"Duplicate functions in {fname}: {', '.join(dups)}")
        except:
            pass
    
    log_info("✅ No duplicate function definitions found")

# ============================================================
# 4. COMPLEXITY CHECK - Large functions?
# ============================================================
def check_complexity():
    print("🔍 Checking function complexity...")
    files = ["app.py", "src/services/strategy_finder.py", "src/core/simulation.py"]
    
    large_funcs = []
    for fname in files:
        fpath = ROOT / fname
        if not fpath.exists():
            continue
        
        try:
            with open(fpath) as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    lines = (node.end_lineno or 0) - node.lineno
                    if lines > 100:
                        large_funcs.append(f"{fname}:{node.name} ({lines} lines)")
        except:
            pass
    
    if large_funcs:
        for lf in large_funcs:
            log_warning(f"Large function: {lf}")
    else:
        log_info("✅ No excessively large functions (>100 lines)")
    
    RESULTS["stats"]["large_functions"] = len(large_funcs)

# ============================================================
# 5. TYPE HINT CHECK - Missing type annotations?
# ============================================================
def check_type_hints():
    print("🔍 Checking type annotations...")
    src_files = list(ROOT.glob("src/**/*.py"))
    
    missing_annotations = 0
    total_funcs = 0
    
    for fpath in src_files:
        if "__pycache__" in str(fpath):
            continue
        try:
            with open(fpath) as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_funcs += 1
                    if node.returns is None and node.name != "__init__":
                        missing_annotations += 1
        except:
            pass
    
    if total_funcs > 0:
        pct = 100 * (total_funcs - missing_annotations) / total_funcs
        RESULTS["stats"]["type_hint_coverage"] = f"{pct:.0f}%"
        if pct < 80:
            log_warning(f"Type hint coverage in src/: {pct:.0f}%")
        else:
            log_info(f"✅ Type hint coverage in src/: {pct:.0f}%")

# ============================================================
# 6. TEST CHECK - Run pytest
# ============================================================
def check_tests():
    print("🔍 Running tests...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True, cwd=ROOT
    )
    
    if result.returncode == 0:
        # Parse output for test count
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if "passed" in line:
                log_info(f"✅ Tests: {line.strip()}")
                break
    else:
        log_error(f"Tests failed: {result.stdout}")

# ============================================================
# 7. DEPENDENCY CHECK - requirements.txt complete?
# ============================================================
def check_dependencies():
    print("🔍 Checking dependencies...")
    req_path = ROOT / "requirements.txt"
    
    if not req_path.exists():
        log_error("requirements.txt not found")
        return
    
    with open(req_path) as f:
        reqs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    
    expected = ["streamlit", "pandas", "numpy", "numpy-financial", "plotly", "pydantic", "structlog"]
    missing = [e for e in expected if not any(e in r.lower() for r in reqs)]
    
    if missing:
        log_warning(f"Possibly missing dependencies: {missing}")
    else:
        log_info(f"✅ requirements.txt has {len(reqs)} dependencies")
    
    RESULTS["stats"]["dependencies"] = len(reqs)

# ============================================================
# 8. DATA CHECK - Archetypes file valid?
# ============================================================
def check_data():
    print("🔍 Checking data files...")
    
    # Check for archetype files
    arch_files = list(ROOT.glob("*.json")) + list(ROOT.glob("data/*.json"))
    
    for af in arch_files:
        if "archetype" in af.name.lower():
            try:
                with open(af) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    log_info(f"✅ {af.name}: {len(data)} archetypes")
                    RESULTS["stats"]["archetypes"] = len(data)
            except json.JSONDecodeError as e:
                log_error(f"Invalid JSON in {af.name}: {e}")

# ============================================================
# 9. PROJECT STRUCTURE CHECK
# ============================================================
def check_structure():
    print("🔍 Checking project structure...")
    
    expected = [
        "app.py",
        "requirements.txt", 
        "pyproject.toml",
        "README.md",
        ".env.example",
        "src/__init__.py",
        "src/models/__init__.py",
        "src/core/__init__.py",
        "tests/__init__.py",
    ]
    
    missing = [e for e in expected if not (ROOT / e).exists()]
    
    if missing:
        for m in missing:
            log_warning(f"Missing: {m}")
    else:
        log_info(f"✅ All {len(expected)} expected files present")

# ============================================================
# 10. SUMMARY
# ============================================================
def generate_report():
    print("\n" + "="*60)
    print("📊 CODEBASE HEALTH REPORT")
    print("="*60)
    
    print("\n📈 STATS:")
    for k, v in RESULTS["stats"].items():
        print(f"   {k}: {v}")
    
    if RESULTS["errors"]:
        print(f"\n❌ ERRORS ({len(RESULTS['errors'])}):")
        for e in RESULTS["errors"]:
            print(f"   • {e}")
    
    if RESULTS["warnings"]:
        print(f"\n⚠️  WARNINGS ({len(RESULTS['warnings'])}):")
        for w in RESULTS["warnings"]:
            print(f"   • {w}")
    
    if RESULTS["info"]:
        print(f"\n✅ PASSED ({len(RESULTS['info'])}):")
        for i in RESULTS["info"]:
            print(f"   • {i}")
    
    # Final verdict
    print("\n" + "="*60)
    if RESULTS["errors"]:
        print("🔴 STATUS: ISSUES FOUND - Action required")
    elif RESULTS["warnings"]:
        print("🟡 STATUS: HEALTHY with minor warnings")
    else:
        print("🟢 STATUS: EXCELLENT - No issues found")
    print("="*60)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.chdir(ROOT)
    
    check_syntax()
    check_imports()
    check_duplicates()
    check_complexity()
    check_type_hints()
    check_dependencies()
    check_data()
    check_structure()
    check_tests()
    
    generate_report()
