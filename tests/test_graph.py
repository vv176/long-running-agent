"""
Step 2 test: the import graph, against the exact expected edge set of miniledger.

Run: python -m pytest tests/test_graph.py -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foreman import graph  # noqa: E402

REPO = ROOT / "fixtures" / "miniledger"

# Hand-derived from the fixture. If the resolver regresses, this fails.
EXPECTED = {
    ("app/__init__.py", "app/routes/health.py"),        # from .routes import health
    ("app/__init__.py", "app/routes/invoices.py"),      # ...and invoices, same stmt
    ("app/models/invoice.py", "app/utils/money.py"),    # from ..utils.money import Money
    ("app/routes/invoices.py", "app/schemas.py"),
    ("app/routes/invoices.py", "app/services/invoice.py"),   # from app.services import invoice
    ("app/schemas.py", "app/utils/money.py"),
    ("app/services/invoice.py", "app/models/invoice.py"),
    ("app/services/invoice.py", "app/utils/money.py"),
    ("tests/conftest.py", "app/__init__.py"),
    ("tests/test_money.py", "app/utils/money.py"),
    ("wsgi.py", "app/__init__.py"),
}


def test_edges_exactly_match():
    found, unparsable = graph.edges(REPO)
    assert unparsable == []
    assert set(found) == EXPECTED, (
        f"missing: {sorted(EXPECTED - set(found))}\nextra: {sorted(set(found) - EXPECTED)}")


def test_module_names():
    assert graph.module_name("app/utils/money.py") == "app.utils.money"
    assert graph.module_name("app/__init__.py") == "app"
    assert graph.module_name("wsgi.py") == "wsgi"


def test_package_beats_same_named_module():
    idx = graph.build_index(["pkg.py", "pkg/__init__.py", "pkg/mod.py"])
    assert idx["pkg"] == "pkg/__init__.py"


IDX = graph.build_index([str(p.relative_to(REPO)) for p in graph.python_files(REPO)])


@pytest.mark.parametrize("src,importer,expected", [
    # the case a naive resolver gets wrong: submodule imported as a name
    ("from app.services import invoice", "app/routes/invoices.py", "app/services/invoice.py"),
    # ...and the same shape where the name is NOT a module -> falls back to the package
    ("from app.services import create", "app/routes/invoices.py", "app/services/__init__.py"),
    ("import app.utils.money", "wsgi.py", "app/utils/money.py"),
    ("import app.utils.money as m", "wsgi.py", "app/utils/money.py"),
    ("from app.utils.money import Money", "wsgi.py", "app/utils/money.py"),
    # deeper than any real module -> longest existing prefix
    ("from app.utils.money.Money import x", "wsgi.py", "app/utils/money.py"),
    ("from . import health", "app/routes/invoices.py", "app/routes/health.py"),
    # `.money` from app/models/invoice.py means app.models.money, which does NOT
    # exist -> correct Python semantics is to fall back to the package itself.
    ("from .money import Money", "app/models/invoice.py", "app/models/__init__.py"),
    # the sibling that DOES exist, one level up
    ("from ..utils.money import Money", "app/models/invoice.py", "app/utils/money.py"),
    ("from .health import bp", "app/routes/invoices.py", "app/routes/health.py"),
    ("from .. import schemas", "app/routes/invoices.py", "app/schemas.py"),
    ("from app.routes import *", "wsgi.py", "app/routes/__init__.py"),
    # reached inside a function body
    ("def f():\n    from app.schemas import invoice_payload", "wsgi.py", "app/schemas.py"),
    ("if True:\n    import app.utils.money", "wsgi.py", "app/utils/money.py"),
])
def test_resolves(src, importer, expected):
    found = {resolve for ref in graph.extract_imports(src)
             if (resolve := graph.resolve(ref, importer, IDX))}
    assert expected in found, f"got {sorted(found)}"


@pytest.mark.parametrize("src,importer", [
    ("import os", "wsgi.py"),
    ("from flask import Blueprint", "app/routes/health.py"),
    ("from __future__ import annotations", "wsgi.py"),
    ("from decimal import Decimal", "app/utils/money.py"),
    ("from ... import toofar", "app/routes/invoices.py"),   # climbs past root
])
def test_resolves_to_nothing(src, importer):
    assert not {r for ref in graph.extract_imports(src)
                if (r := graph.resolve(ref, importer, IDX))}


def test_unparsable_is_reported_not_swallowed(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "broken.py").write_text("def f(:\n")
    found, unparsable = graph.edges(tmp_path)
    assert found == []
    assert len(unparsable) == 1 and "broken.py" in unparsable[0]


def test_deterministic():
    runs = {tuple(graph.edges(REPO)[0]) for _ in range(5)}
    assert len(runs) == 1
