"""
The import graph. Pure `ast`, no LLM, no network.

Two questions, asked constantly by every later phase:

    deps_of(f)        what does f import?      (outgoing)
    dependents_of(f)  who imports f?           (incoming)

Everything else in Foreman treats those as ground truth, so a silently missing
edge does not raise — it makes the agent plan a caller before the helper it
depends on. Hence the care below.
"""
import ast
from dataclasses import dataclass
from pathlib import Path

IGNORE_DIRS = {"__pycache__", ".venv", "venv", ".git", ".pytest_cache", "node_modules"}


@dataclass(frozen=True)
class ImportRef:
    """One import target as written. `level` = leading dots (0 = absolute)."""
    level: int
    module: str          # "" for `from . import x`
    name: str | None     # the imported name; may itself be a submodule

    def key(self):
        """Sort key. `name` is None for some shapes, so coerce it — comparing
        None to str raises TypeError, and every real file mixes both shapes."""
        return (self.level, self.module, self.name or "")


def python_files(root: Path) -> list[Path]:
    """Every .py file, alphabetically. A flat walk — order does not matter here."""
    return [p for p in sorted(root.rglob("*.py"))
            if not any(part in IGNORE_DIRS for part in p.parts)]


def module_name(rel) -> str:
    """app/utils/money.py -> app.utils.money ;  app/__init__.py -> app"""
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def build_index(rels: list[str]) -> dict[str, str]:
    """dotted module name -> file path. A package beats a same-named module."""
    index: dict[str, str] = {}
    for rel in sorted(rels):
        mod = module_name(rel)
        if not mod:
            continue
        cur = index.get(mod)
        # Claim the name if nobody has it, OR if we are a package __init__ and the
        # current holder is a plain module. Python resolves `pkg` to the PACKAGE when
        # both pkg/__init__.py and pkg.py exist, so we mirror that.
        if cur is None or (Path(rel).name == "__init__.py" and Path(cur).name != "__init__.py"):
            index[mod] = rel
    return index


def extract_imports(source: str) -> list[ImportRef]:
    """
    Every import target in one file. Raises SyntaxError so the caller can report
    the file rather than silently lose its edges.

    `ast.walk` reaches imports inside functions, `if TYPE_CHECKING`, and both arms
    of a try/except ImportError. All of them are real dependencies.
    """
    refs: set[ImportRef] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name:
                    refs.add(ImportRef(0, a.name, None))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            named = [a.name for a in node.names if a.name and a.name != "*"]
            # Each imported name may itself be a submodule: `from app.services
            # import invoice` must yield the edge to app/services/invoice.py, not
            # to app/services/__init__.py. Offering `name` to the resolver is the
            # whole reason this dataclass has three fields.
            for n in named or [None]:
                refs.add(ImportRef(node.level, module, n))
    return sorted(refs, key=ImportRef.key)


def resolve(ref: ImportRef, importer_rel: str, index: dict[str, str]) -> str | None:
    """
    One ImportRef -> the in-repo file it means, or None if external.

    Relative imports anchor on the importer's PARENT directory, which is correct
    for a module and for an __init__.py alike: `from . import x` inside
    app/routes/foo.py and inside app/routes/__init__.py both mean package app.routes.
    """
    importer = Path(importer_rel)
    if ref.level == 0:
        prefix = ref.module.split(".") if ref.module else []
    else:
        package = list(importer.parent.parts)
        climb = ref.level - 1
        if climb > len(package):
            return None                       # climbs past the repo root
        prefix = package[: len(package) - climb] + (ref.module.split(".") if ref.module else [])

    if not prefix and not ref.name:
        return None

    # Two candidates, tried in order:
    #   1. prefix + [name]   -> `from app.services import invoice` means app.services.invoice
    #   2. prefix            -> ...unless `invoice` is a function, then it means app.services
    # `([x] if cond else []) + [y]` is just "maybe-first, then always-second" without
    # an if-statement. Within each candidate, the LONGEST existing prefix wins, so
    # `from app.utils.money import Money` still lands on app/utils/money.py.
    for parts in ([prefix + [ref.name]] if ref.name else []) + [prefix]:
        for cut in range(len(parts), 0, -1):
            hit = index.get(".".join(parts[:cut]))
            if hit is not None:
                return hit
    return None


def edges(root: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """
    The whole graph in one call.

    Returns (edges, unparsable) where an edge is (importer, imported), both
    repo-relative, deduplicated and sorted. Two passes, because an edge can point
    at a file the walk has not reached yet.
    """
    rels = [str(p.relative_to(root)) for p in python_files(root)]
    index = build_index(rels)
    found: set[tuple[str, str]] = set()
    unparsable: list[str] = []

    for rel in rels:
        try:
            refs = extract_imports((root / rel).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            unparsable.append(f"{rel}:{exc.lineno} {exc.msg}")
            continue
        for ref in refs:
            target = resolve(ref, rel, index)
            if target is not None and target != rel:
                found.add((rel, target))
    return sorted(found), unparsable
