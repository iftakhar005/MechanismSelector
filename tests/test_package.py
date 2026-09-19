"""The published package must be exactly the code the experiments ran.

Its modules are copies of `src/accounting.py`, `src/mechanisms.py` and
`src/selector.py` with package-relative imports and rewritten docstrings. The
first test compares abstract syntax trees with docstrings removed and relative
imports normalised, so any change to behaviour -- however small -- fails it.
"""

from __future__ import annotations

import ast
import contextlib
import io
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "package") not in sys.path:
    sys.path.insert(1, str(ROOT / "package"))

import mechanism_selector  # noqa: E402

MODULES = ("accounting", "mechanisms", "selector")


def code_only(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            node.level = 0
    return ast.dump(tree)


@pytest.mark.parametrize("module", MODULES)
def test_packaged_code_is_identical_to_the_tested_code(module):
    packaged = Path(mechanism_selector.__file__).parent / f"{module}.py"
    tested = ROOT / "src" / f"{module}.py"
    assert code_only(packaged) == code_only(tested), (
        f"{module}.py in the package differs in code, not just documentation, from src/{module}.py"
    )


def test_public_surface_is_one_class_with_one_method():
    import types

    assert mechanism_selector.__all__ == ["MechanismSelector"]
    public = {n for n in dir(mechanism_selector)
              if not n.startswith("_") and not isinstance(getattr(mechanism_selector, n), types.ModuleType)}
    assert public == {"MechanismSelector"}
    methods = {n for n in dir(mechanism_selector.MechanismSelector)
               if not n.startswith("_") and callable(getattr(mechanism_selector.MechanismSelector, n))}
    assert methods == {"adapt"}


def test_public_class_and_method_are_documented_without_planning_references():
    for obj in (mechanism_selector.MechanismSelector, mechanism_selector.MechanismSelector.adapt):
        doc = obj.__doc__ or ""
        assert len(doc) > 200, f"{obj.__qualname__} needs a real docstring"
        assert not re.search(r"phase \d|spec trap|prompt\.md|experiments/", doc, re.IGNORECASE)


def test_version():
    assert mechanism_selector.__version__ == "0.1.0"


def test_readme_example_runs_and_produces_a_valid_result():
    readme = (ROOT / "package" / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", readme, re.DOTALL).group(1)
    assert len([ln for ln in code.splitlines() if ln.strip()]) <= 11, "the README promises a short example"

    namespace, out = {}, io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "README example", "exec"), namespace)
    result = namespace["result"]

    assert result.action in ("SKIP", "NUDGE", "REBUILD")
    assert result.cost.work_units >= 0
    assert hasattr(result.model, "predict")
    assert out.getvalue().split()[0] == result.action


def test_readme_states_the_negative_findings_before_the_example():
    readme = (ROOT / "package" / "README.md").read_text(encoding="utf-8")
    warning = readme.index("Read this before using it")
    example = readme.index("## Example")
    assert warning < example
    for phrase in ("does not beat simpler policies", "minimum-window guard", "only a fraction",
                   "not an energy-saving tool"):
        assert phrase in readme
