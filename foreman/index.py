"""
PHASE 1 — INDEX. Build the map of the repo, once.

Produces three things:

  1. `files` + `edges` in the DB      deterministic, from `ast`
  2. one summary per folder            one weak-model call each
  3. `.foreman/REPO_MAP.md`            the always-in-context orientation block

Idempotent by sha256: a second boot with no file changes makes ZERO model calls.
Change one file and only that file's folder is re-summarised.

The three-tier rule this phase exists to serve:

    always in context   REPO_MAP.md               constant size, ~1-2k tokens
    on demand           read_summary(folder)      ~300 tokens each
    never in context    file contents

Putting every folder summary in the prompt would reproduce, one level up, exactly
the unbounded-context failure this project is about.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, db, graph, llm

_SUMMARY_TOOL = [{
    "type": "function",
    "function": {
        "name": "describe_folder",
        "description": "Describe what this folder does and what each file in it is for.",
        "parameters": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string",
                            "description": "2-3 sentences: what this folder is responsible for"},
                "one_line": {"type": "string",
                             "description": "ONE short line for the repo map, under 90 chars"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "file name only"},
                            "does": {"type": "string", "description": "one short line"},
                        },
                        "required": ["name", "does"],
                    },
                },
            },
            "required": ["purpose", "one_line", "files"],
        },
    },
}]

_REPO_TOOL = [{
    "type": "function",
    "function": {
        "name": "describe_repo",
        "description": "Describe the repository as a whole.",
        "parameters": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "description": "2-4 sentences: what this app is"},
                "stack": {"type": "string",
                          "description": "frameworks and libraries it is built on, comma separated"},
                "entry_points": {"type": "array", "items": {"type": "string"}},
                "conventions": {"type": "array", "items": {"type": "string"},
                                "description": "3-5 observed conventions a contributor must follow"},
            },
            "required": ["purpose", "stack", "entry_points", "conventions"],
        },
    },
}]


# ─────────────────────────── the deterministic half ───────────────────────────

def digest(path: Path) -> tuple[str, int]:
    """(sha256, line count) for one file. The sha is what makes INDEX idempotent:
    same bytes -> same hash -> no model call on the next boot."""
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), raw.count(b"\n") + 1


def folder_of(rel: str) -> str:
    """'app/routes/x.py' -> 'app/routes' ; 'wsgi.py' -> '' (the root)."""
    parent = str(Path(rel).parent)
    return "" if parent == "." else parent


def entry_points(root: Path, rels: list[str]) -> list[str]:
    """
    Which files start the app. Detected, not guessed.

    The model got this wrong when asked (it answered `app/__init__.py`, which
    defines the factory but does not run it). Three deterministic signals:
    a module-level `application`/`app` assigned from a call, an
    `if __name__ == "__main__"` block, or a conventional filename.
    """
    import ast as _ast
    CONVENTIONAL = {"wsgi.py", "asgi.py", "main.py", "manage.py", "app.py"}
    found = []
    for rel in rels:
        if Path(rel).name in CONVENTIONAL:
            found.append(rel); continue
        try:
            tree = _ast.parse((root / rel).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:                          # module level only
            if (isinstance(node, _ast.Assign) and isinstance(node.value, _ast.Call)
                    and any(isinstance(tg, _ast.Name) and tg.id in ("application", "app")
                            for tg in node.targets)):
                found.append(rel); break
            if (isinstance(node, _ast.If) and isinstance(node.test, _ast.Compare)
                    and isinstance(node.test.left, _ast.Name)
                    and node.test.left.id == "__name__"):
                found.append(rel); break
    return sorted(set(found))


def scan(root: Path) -> tuple[dict[str, tuple[str, int]], dict[str, list[str]]]:
    """
    Walk the repo. Returns (file -> (sha, loc), folder -> [file names]).
    Zero model calls.
    """
    files, folders = {}, {}
    for p in graph.python_files(root):
        if config.SIDECAR in p.parts:
            continue
        rel = str(p.relative_to(root))
        files[rel] = digest(p)
        folders.setdefault(folder_of(rel), []).append(Path(rel).name)
    return files, {k: sorted(v) for k, v in folders.items()}


def changed_folders(root: Path, files: dict) -> set[str]:
    """Folders whose contents differ from what the DB last saw. Empty = nothing to do."""
    known = db.get_files(root)
    now = {rel: sha for rel, (sha, _) in files.items()}
    touched = {rel for rel in set(known) | set(now) if known.get(rel) != now.get(rel)}
    return {folder_of(rel) for rel in touched}


# ─────────────────────────── the model half ───────────────────────────

def _folder_prompt(root: Path, folder: str, names: list[str]) -> str:
    """Build the summarisation prompt for one folder.

    Each file contributes three things: its in-repo imports, its importers, and its
    first SUMMARY_HEAD_LINES lines. The head is where imports, docstrings and class
    definitions live, so it identifies a file's job at a fraction of its token cost.
    The edges are included because "who calls this" is the fact a summary most often
    gets wrong from the body alone.
    """
    blocks = []
    for name in names:
        rel = f"{folder}/{name}" if folder else name
        head = "\n".join((root / rel).read_text(encoding="utf-8", errors="replace")
                         .splitlines()[: config.SUMMARY_HEAD_LINES])
        imports = db.deps_of(root, rel)
        used_by = db.dependents_of(root, rel)
        blocks.append(
            f"--- {name} ---\n"
            f"imports (in-repo): {', '.join(imports) or 'none'}\n"
            f"imported by: {', '.join(used_by) or 'nothing'}\n"
            f"first {config.SUMMARY_HEAD_LINES} lines:\n{head}\n")
    where = folder or "the repository root"
    return (f"Describe the folder `{where}` of a Python project.\n\n"
            f"Be concrete and factual. Name the actual responsibilities, not generic phrases "
            f"like 'handles business logic'. If a file is framework-coupled, say which "
            f"framework. If it is pure and framework-agnostic, say so — that fact matters "
            f"to anyone changing this code.\n\n" + "\n".join(blocks))


def summarize_folder(run_id, root: Path, folder: str, names: list[str]) -> dict:
    """One weak-model call. Returns {purpose, one_line, files:{name: does}}."""
    msg, tin, tout, c = llm.call(
        [{"role": "user", "content": _folder_prompt(root, folder, names)}],
        tools=_SUMMARY_TOOL, tool_choice="required", model=config.WEAK_MODEL)
    db.append(run_id, "spend", payload={"cents": c, "tokens_in": tin, "tokens_out": tout})
    args = json.loads(msg.tool_calls[0].function.arguments)
    return {
        "purpose": args["purpose"],
        "one_line": args["one_line"],
        "files": {f["name"]: f["does"] for f in args.get("files", [])},
    }


def summarize_repo(run_id, root: Path, folder_lines: dict[str, str],
                   rels: list[str], entries: list[str]) -> dict:
    """
    One call, after all folders are done. Sees folder one-liners plus the real file
    list — the paths matter, because without them the model invents them (it wrote
    "app/models/schemas.py" for a file that lives at "app/schemas.py").
    """
    layout = "\n".join(f"  {k or '.'}/  — {v}" for k, v in sorted(folder_lines.items()))
    paths = "\n".join(f"  {r}" for r in sorted(rels))
    msg, tin, tout, c = llm.call(
        [{"role": "user", "content":
          f"A Python repository has these folders:\n\n{layout}\n\n"
          f"Its complete file list:\n{paths}\n\n"
          f"Its entry points (already determined, do not change them): "
          f"{', '.join(entries) or 'none found'}\n\n"
          f"Describe the repository as a whole. Be concrete. Conventions will be shown "
          f"to an agent about to modify this code, so make them actionable "
          f"('validation lives in app/schemas.py, not in route handlers').\n"
          f"CITE ONLY PATHS FROM THE LIST ABOVE. Never invent a path."}],
        tools=_REPO_TOOL, tool_choice="required", model=config.WEAK_MODEL)
    db.append(run_id, "spend", payload={"cents": c, "tokens_in": tin, "tokens_out": tout})
    return json.loads(msg.tool_calls[0].function.arguments)


# ─────────────────────────── artefacts on disk ───────────────────────────

def folder_body(folder: str, s: dict) -> str:
    """Render one folder summary dict as the markdown that read_summary() serves.

    Written to disk as well as the DB, so a human can read .foreman/folders/*.md and
    see exactly what the agent sees. Same bytes, one source.
    """
    lines = [f"# {folder or '(repo root)'}", "", s["purpose"], "", "## Files"]
    lines += [f"- `{n}` — {d}" for n, d in sorted(s["files"].items())]
    return "\n".join(lines) + "\n"


def write_repo_map(root: Path, repo: dict, folder_lines: dict[str, str],
                   n_files: int, n_edges: int) -> Path:
    """Write .foreman/REPO_MAP.md and return its path.

    This is TIER 1 of the three-tier context: the only summary that is always in the
    prompt, so it must stay roughly constant in size no matter how big the repo is.
    Hence one line per folder — not per file.
    """
    out = root / config.SIDECAR / "REPO_MAP.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Built line by line on purpose. `textwrap.dedent` on an f-string silently does
    # nothing once a multi-line value is interpolated: the injected lines have no
    # indent, so the common prefix becomes "" and every template line keeps its
    # leading spaces. The first version of this function shipped that bug.
    lines = [
        "# Repository map", "",
        repo["purpose"], "",
        f"**Stack:** {repo['stack']}",
        f"**Entry points:** {', '.join(repo['entry_points']) or 'none found'}",
        f"**Size:** {n_files} python files, {n_edges} in-repo import edges", "",
        "## Layout", "```",
    ]
    lines += [f"{(k or '.') + '/':<24} {v}" for k, v in sorted(folder_lines.items())]
    lines += ["```", "", "## Conventions"]
    lines += [f"- {c}" for c in repo["conventions"]]
    lines += ["", "---", "Generated by Foreman INDEX. Folder detail: `read_summary(folder)`."]
    out.write_text("\n".join(lines) + "\n")
    return out


def write_folder_files(root: Path, bodies: dict[str, str]) -> None:
    """Write each folder summary to .foreman/folders/<slug>.md. Returns nothing.

    'app/routes' becomes 'app__routes.md' — the slug flattens the tree so the sidecar
    stays one directory deep and a summary can never collide with a real source path.
    """
    base = root / config.SIDECAR / "folders"
    base.mkdir(parents=True, exist_ok=True)
    for folder, body in bodies.items():
        slug = folder.replace("/", "__") or "_root"
        (base / f"{slug}.md").write_text(body)


# ─────────────────────────── the phase ───────────────────────────

def build(run_id: str, root: Path, log=print) -> dict:
    """
    Index the repo. Idempotent.

    Returns {files, edges, folders, summarised, skipped} for the console.
    Only folders whose file shas changed are re-summarised.
    """
    db.append(run_id, "phase_entered", payload={"phase": "INDEX"})
    files, folders = scan(root)

    pairs, unparsable = graph.edges(root)
    db.save_edges(root, pairs)
    if unparsable:
        log(f"  ! {len(unparsable)} file(s) failed to parse — their edges are MISSING:")
        for u in unparsable:
            log(f"      {u}")

    dirty = changed_folders(root, files)
    db.save_files(root, [(rel, sha, loc) for rel, (sha, loc) in files.items()])

    have = db.all_summaries(root)
    todo = sorted(f for f in folders if f in dirty or f not in have)
    cached_repo = json.loads(db.get_summary(root, "__repo__") or "{}")
    log(f"  {len(files)} files · {len(pairs)} edges · {len(folders)} folders")

    if not todo and cached_repo:
        # Nothing to SUMMARISE is not the same as nothing to WRITE. The DB cache is
        # keyed by repo and survives across runs, but the artefacts live INSIDE the
        # work copy, which main() deletes and re-copies every run. Returning here
        # without rebuilding them left PLAN reading "(no repo map — run INDEX first)"
        # — a real run planned badly because of exactly this. Rebuild from the cache;
        # still zero model calls.
        bodies = {f: db.get_summary(root, f) or "" for f in folders}
        stored = cached_repo.get("folder_lines") or {}
        one_liners = {f: stored.get(f) or _first_line(bodies[f]) for f in folders}
        write_folder_files(root, bodies)
        path = write_repo_map(root, cached_repo, one_liners, len(files), len(pairs))
        log(f"  index is current — 0 model calls (rebuilt {path.relative_to(root)} "
            f"from cache)")
        return {"files": len(files), "edges": len(pairs), "folders": len(folders),
                "summarised": 0, "skipped": len(folders)}

    bodies, one_liners = {}, {}
    if todo:
        log(f"  summarising {len(todo)} folder(s) (weak model, in parallel): "
            f"{', '.join(f or '.' for f in todo)}")
        # Folder summaries are independent, so run them concurrently. Sequentially this
        # took 175s on a 15-file repo; INDEX is the phase the class watches, and dead
        # air is the one cost we can remove for free.
        with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
            results = list(pool.map(
                lambda f: (f, summarize_folder(run_id, root, f, folders[f])), todo))
        for folder, s in results:
            bodies[folder] = folder_body(folder, s)
            one_liners[folder] = s["one_line"]
            db.save_summary(root, folder, bodies[folder])
            log(f"    {folder or '.'}/  {s['one_line']}")

    # Unchanged folders keep their one-liner, recovered from the stored body's
    # first paragraph, so REPO_MAP stays complete without re-summarising them.
    stored = cached_repo.get("folder_lines") or {}
    for folder in folders:
        bodies.setdefault(folder, db.get_summary(root, folder) or "")
        one_liners.setdefault(folder,
                              stored.get(folder) or _first_line(bodies[folder]))

    entries = entry_points(root, list(files))
    repo = summarize_repo(run_id, root, one_liners, list(files), entries)
    repo["entry_points"] = entries        # detected wins over anything the model said
    # Persist the folder one-liners too. They are TIER 1 and the model wrote them
    # deliberately short; _first_line() can only recover a truncated `purpose`
    # paragraph, which reads badly in the map. Storing them keeps a cache-hit rebuild
    # byte-identical to the original.
    repo["folder_lines"] = one_liners
    db.save_summary(root, "__repo__", json.dumps(repo))
    write_folder_files(root, bodies)
    path = write_repo_map(root, repo, one_liners, len(files), len(pairs))
    log(f"  wrote {path.relative_to(root)}")

    return {"files": len(files), "edges": len(pairs), "folders": len(folders),
            "summarised": len(todo), "skipped": len(folders) - len(todo)}


def _first_line(body: str) -> str:
    """First non-heading line of a summary — the one-liner used in the repo map.

    Skips '#' headings, so it lands on the prose rather than the folder name.
    """
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:90]
    return ""


def repo_map(root: Path) -> str:
    """The always-in-context block. Read from disk so it is identical to the artefact."""
    p = root / config.SIDECAR / "REPO_MAP.md"
    return p.read_text() if p.exists() else "(no repo map — run INDEX first)"
