"""
The tools. Twelve of them, scoped by phase.

Read tools are free and deterministic; only `read_file` and `run_tests` cost
anything real. That ratio is deliberate — navigation should be cheap so the
expensive model spends its budget on judgment, not on orientation.

PHASE SCOPING (Rombaut §4.2.1, Prometheus's per-node tool binding):

    PLAN     8 tools   read + search + write_plan.   CANNOT touch a source file.
    EXECUTE 11 tools   read + search + edit + test.  CANNOT write a plan.

That is a structural guardrail, not a request in a prompt. PLAN physically cannot
edit code, so a planning phase can never quietly become an execution phase.

THE WRITE GUARD. A plan declares its own blast radius (`writes`, `deletes`).
During EXECUTE, a file's unit may only touch paths its own plan declared. So the
agent commits to a scope while planning and is held to it while executing.

ONE WORK TREE. Everything operates on a copy of the repo. A failed attempt is
rolled back by shadow git before the retry, so the retry starts from the same
bytes as the first attempt — the same idempotency a pristine-source tree would
give, using the checkpoint mechanism we already need.
"""
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

from . import config, db

TEST_TIMEOUT = 90
GREP_MAX_HITS = 40


# ─────────────────────────── schemas ───────────────────────────

def _tool(name, desc, props, required):
    """Build one OpenAI function-calling schema. Just boilerplate reduction."""
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


SCHEMAS = {
    "list_tree": _tool(
        "list_tree", "List folders and files under a repo-relative path. Cheap orientation.",
        {"path": {"type": "string", "description": "'' for the repo root"},
         "depth": {"type": "integer", "description": "how many levels down, default 2"}},
        ["path"]),

    "read_summary": _tool(
        "read_summary",
        "Read the pre-built summary of one folder: what it does and what each file is for. "
        "Far cheaper than reading the files. Use this before read_file.",
        {"folder": {"type": "string", "description": "e.g. 'app/routes', or '' for the root"}},
        ["folder"]),

    "deps_of": _tool(
        "deps_of", "OUTGOING edges: which in-repo files this file imports.",
        {"path": {"type": "string"}}, ["path"]),

    "dependents_of": _tool(
        "dependents_of",
        "INCOMING edges: which in-repo files import this one. Use it to find the blast "
        "radius of a change before you commit to it.",
        {"path": {"type": "string"}}, ["path"]),

    "grep": _tool(
        "grep", "Regex search across the repo. Use it to find where a concept lives.",
        {"pattern": {"type": "string"}, "glob": {"type": "string", "description": "default **/*.py"}},
        ["pattern"]),

    "read_file": _tool(
        "read_file", "Read a file, or a line range of it. Costs tokens — prefer read_summary first.",
        {"path": {"type": "string"},
         "start": {"type": "integer", "description": "1-based first line"},
         "end": {"type": "integer"}}, ["path"]),

    "write_plan": _tool(
        "write_plan",
        "Record what must happen to ONE file. Call it once per file you visit. "
        "Calling it again for the same file replaces that file's plan.",
        {"path": {"type": "string", "description": "the file this plan is about"},
         "verdict": {"type": "string", "enum": ["plan", "no_change"],
                     "description": "'no_change' if this file needs nothing — say so and move on"},
         "plan": {"type": "string",
                  "description": "what to change and how, concretely. Empty for no_change."},
         "writes": {"type": "array", "items": {"type": "string"},
                    "description": "every file this work will create or modify. You will be "
                                   "BLOCKED from touching anything not listed here."},
         "deletes": {"type": "array", "items": {"type": "string"},
                     "description": "every file this work will delete"}},
        ["path", "verdict"]),

    "read_plan": _tool(
        "read_plan", "Read the plan already recorded for a file, if any.",
        {"path": {"type": "string"}}, ["path"]),

    "write_file": _tool(
        "write_file", "Create or overwrite a file with complete contents.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),

    "str_replace": _tool(
        "str_replace",
        "Replace an exact string in a file. Preferred over rewriting a whole file: "
        "old_str must appear exactly once.",
        {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}},
        ["path", "old_str", "new_str"]),

    "delete_file": _tool(
        "delete_file", "Delete a file.", {"path": {"type": "string"}}, ["path"]),

    "skip_unit": _tool(
        "skip_unit",
        "Close this unit WITHOUT changing anything, because the plan was wrong and this "
        "file genuinely needs no edit. Use it only after reading the file and confirming "
        "that. Do not use it to avoid difficult work — the whole test suite still has to "
        "pass at the end.",
        {"reason": {"type": "string",
                    "description": "what you checked, and why no change is needed"}},
        ["reason"]),

    "run_tests": _tool(
        "run_tests", "Run pytest on a path (file, folder, or '' for the whole suite).",
        {"target": {"type": "string"}}, ["target"]),
}

PLAN_TOOLS = ["list_tree", "read_summary", "deps_of", "dependents_of",
              "grep", "read_file", "write_plan", "read_plan"]
EXECUTE_TOOLS = ["list_tree", "read_summary", "deps_of", "dependents_of",
                 "grep", "read_file", "read_plan",
                 "write_file", "str_replace", "delete_file", "run_tests", "skip_unit"]


def schemas_for(phase: str) -> list[dict]:
    """The tool schemas this phase is allowed to see. PLAN never sees a write tool,
    which is why PLAN cannot edit code even if the prompt were compromised."""
    names = PLAN_TOOLS if phase == "PLAN" else EXECUTE_TOOLS
    return [SCHEMAS[n] for n in names]


# ─────────────────────────── context ───────────────────────────

class Ctx:
    """
    What the tools need, plus what the loop reads back afterwards.

    `unit` is set during EXECUTE to the file currently being worked on; it is what
    the write guard checks declared paths against. None during PLAN.
    """

    def __init__(self, run_id: str, root: Path, phase: str, unit: str | None = None,
                 allow_writes: list[str] | None = None,
                 allow_deletes: list[str] | None = None):
        """One Ctx per unit (or per phase, for PLAN). `allow_writes` / `allow_deletes`
        come straight from that unit's plan and are what the guards check against."""
        self.run_id, self.root, self.phase = run_id, root, phase
        self.unit = unit
        self.allow_writes = set(allow_writes or [])
        self.allow_deletes = set(allow_deletes or [])
        self.planned: set[str] = set()      # files write_plan was called for this phase
        self.wrote: set[str] = set()
        self.deleted: set[str] = set()
        self.last_tests: bool | None = None
        self.skipped: str | None = None     # set by skip_unit: the plan was wrong


class Refused(Exception):
    """A guard rejected the call. Returned to the model as a string, never raised at it."""


# ─────────────────────────── guards ───────────────────────────

def _resolve(ctx: Ctx, rel: str) -> Path:
    """Repo-relative path -> absolute, refusing anything outside the repo.

    `.resolve()` collapses `..` FIRST, so 'app/../../etc/passwd' becomes a real
    absolute path before we compare. Checking the string before resolving would be
    trivially bypassable.
    """
    p = (ctx.root / rel).resolve()
    root = ctx.root.resolve()
    if not str(p).startswith(str(root)):
        raise Refused(f"'{rel}' resolves outside the repo")
    if config.SIDECAR in Path(rel).parts:
        raise Refused(f"'{rel}' is inside {config.SIDECAR}/ — Foreman's own notes are off limits")
    return p


def _check_write(ctx: Ctx, rel: str) -> None:
    """Raise Refused unless this write is allowed. Three gates, cheapest first:
    right phase, not protected, and declared by this unit's own plan."""
    if ctx.phase != "EXECUTE":
        raise Refused(f"{ctx.phase} cannot modify files. Record intent with write_plan instead.")
    for pattern in config.PROTECTED_GLOBS:
        if fnmatch.fnmatch(rel, pattern):
            raise Refused(f"'{rel}' is protected by PROTECTED_GLOBS and must not change")
    if rel not in ctx.allow_writes:
        raise Refused(
            f"'{rel}' is not in this unit's declared writes {sorted(ctx.allow_writes)}. "
            f"A unit may only touch what its plan declared.")


def _check_delete(ctx: Ctx, rel: str) -> None:
    """Same as _check_write but against the plan's declared `deletes`."""
    if ctx.phase != "EXECUTE":
        raise Refused(f"{ctx.phase} cannot delete files.")
    if rel not in ctx.allow_deletes:
        raise Refused(f"'{rel}' is not in this unit's declared deletes "
                      f"{sorted(ctx.allow_deletes)}.")


# ─────────────────────────── implementations ───────────────────────────

def _list_tree(ctx, path="", depth=2) -> str:
    """An `ls -R` bounded by depth. Indents by nesting level so the shape is obvious."""
    base = _resolve(ctx, path) if path else ctx.root
    if not base.is_dir():
        return f"ERROR: '{path}' is not a directory"
    out = []
    for p in sorted(base.rglob("*")):
        if any(part in config.IGNORE_DIRS for part in p.parts):
            continue
        rel = p.relative_to(base)
        if len(rel.parts) > depth:
            continue
        out.append(f"{'  ' * (len(rel.parts) - 1)}{rel.name}{'/' if p.is_dir() else ''}")
    return "\n".join(out) or "(empty)"


def _read_file(ctx, path, start=None, end=None) -> str:
    """File contents with line numbers, optionally a range.

    Line numbers matter: they let the model refer to 'line 14' and let a plan cite a
    location. Truncated at READ_MAX_CHARS so one huge file cannot blow the context.
    """
    text = _resolve(ctx, path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    a = max(1, start or 1)
    b = min(len(lines), end or len(lines))
    body = "\n".join(f"{i:4} | {lines[i - 1]}" for i in range(a, b + 1))
    if len(body) > config.READ_MAX_CHARS:
        body = body[: config.READ_MAX_CHARS] + "\n... (truncated)"
    return body or "(empty file)"


def _grep(ctx, pattern, glob="**/*.py") -> str:
    """Regex search -> 'path:line: text' hits, capped at GREP_MAX_HITS.

    A bad regex returns an error string rather than raising, so the model can fix it
    on the next turn instead of killing the run.
    """
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: bad regex: {exc}"
    hits = []
    for p in sorted(ctx.root.glob(glob)):
        if any(part in config.IGNORE_DIRS for part in p.parts) or not p.is_file():
            continue
        try:
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace")
                                     .splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(ctx.root)}:{n}: {line.strip()[:140]}")
                    if len(hits) >= GREP_MAX_HITS:
                        return "\n".join(hits) + f"\n... truncated at {GREP_MAX_HITS} hits"
        except OSError:
            continue
    return "\n".join(hits) or "no matches"


def _write_plan(ctx, path, verdict, plan="", writes=None, deletes=None) -> str:
    """Record one file's plan as an event. PLAN's only side effect.

    Rejects verdict='plan' with an empty blast radius: a plan that touches nothing
    cannot be executed, and accepting it would produce a unit that always fails.
    """
    _resolve(ctx, path)                          # must be a real repo path
    if verdict == "plan" and not (writes or deletes):
        return ("ERROR: verdict 'plan' needs at least one entry in writes or deletes — "
                "otherwise the work has no effect and cannot be executed.")
    db.append(ctx.run_id, "plan_written", path,
              {"plan": plan, "verdict": verdict,
               "writes": sorted(set(writes or [])), "deletes": sorted(set(deletes or []))})
    ctx.planned.add(path)
    if verdict == "no_change":
        return f"Recorded: {path} needs no change. Move on."
    return (f"Plan recorded for {path}. Declared writes: {sorted(set(writes or []))}, "
            f"deletes: {sorted(set(deletes or []))}. During execution you will be blocked "
            f"from touching anything else.")


def _read_plan(ctx, path) -> str:
    """The plan recorded for a file, by folding the ledger. Used on revisits."""
    st = db.fold(ctx.run_id)
    f = st.files.get(path)
    if f is None or f.verdict is None:
        return f"No plan recorded for {path} yet."
    return (f"verdict: {f.verdict}\nversion: {f.version}\nwrites: {f.writes}\n"
            f"deletes: {f.deletes}\nplan: {f.plan}")


def _write_file(ctx, path, content) -> str:
    """Create or overwrite a file, then mark its folder's summary stale.

    Staleness is marked, not fixed: re-summarising after every write would cost a
    model call per edit. The batch refresh happens once, later.
    """
    _check_write(ctx, path)
    p = _resolve(ctx, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content)
    ctx.wrote.add(path)
    db.mark_stale(ctx.root, str(Path(path).parent) if str(Path(path).parent) != "." else "")
    return f"{'Overwrote' if existed else 'Created'} {path} ({len(content)} chars)."


def _str_replace(ctx, path, old_str, new_str) -> str:
    """Surgical edit: replace old_str exactly once.

    Refuses 0 matches (the model guessed the text) and refuses 2+ (ambiguous — it
    would silently patch the wrong site). 5 of the 13 agents in the paper converged
    on this exact interface independently, because exact-string matching beats line
    numbers and unified diffs for LLM-generated edits.
    """
    _check_write(ctx, path)
    p = _resolve(ctx, path)
    if not p.exists():
        return f"ERROR: {path} does not exist — use write_file to create it."
    text = p.read_text(encoding="utf-8", errors="replace")
    n = text.count(old_str)
    if n == 0:
        return f"ERROR: old_str not found in {path}. Read the file and copy the exact text."
    if n > 1:
        return f"ERROR: old_str appears {n} times in {path}. Include more surrounding lines."
    p.write_text(text.replace(old_str, new_str, 1))
    ctx.wrote.add(path)
    db.mark_stale(ctx.root, str(Path(path).parent) if str(Path(path).parent) != "." else "")
    return f"Replaced 1 occurrence in {path}."


def _delete_file(ctx, path) -> str:
    """Delete a file or directory tree, if the plan declared it."""
    _check_delete(ctx, path)
    p = _resolve(ctx, path)
    if not p.exists():
        return f"{path} is already gone."
    if p.is_dir():
        import shutil
        shutil.rmtree(p)
    else:
        p.unlink()
    ctx.deleted.add(path)
    return f"Deleted {path}."


def _skip_unit(ctx, reason) -> str:
    """Record that this unit needs no change. EXECUTE only.

    The escape hatch for a wrong plan. Without it, a unit whose plan declared writes
    but which genuinely needs no edit fails accept() ("ended without changing
    anything") three times and lands in the DLQ — punished for being right.
    """
    if ctx.phase != "EXECUTE":
        raise Refused("skip_unit is only available during EXECUTE")
    if ctx.wrote or ctx.deleted:
        raise Refused("you have already changed files in this unit; skip_unit is for "
                      "units that need NO change at all")
    ctx.skipped = str(reason or "").strip() or "no reason given"
    return ("Recorded: this unit needs no change. It will be closed as skipped, not "
            "failed. Stop making tool calls now.")


def run_pytest(root: Path, target: str = "") -> tuple[bool, str]:
    """Run pytest in `root`. Returns (passed, last 3000 chars of output).

    A subprocess, not an in-process pytest call: the repo under repair imports
    modules we have already imported, and re-importing changed code in-process gives
    stale results. `cwd=root` also puts the repo on sys.path, which is how `import
    app` resolves.
    """
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
    if target:
        cmd.append(target)
    try:
        r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True,
                           timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {TEST_TIMEOUT}s"
    return r.returncode == 0, (r.stdout + r.stderr)[-3000:]


def _run_tests(ctx, target="") -> str:
    """run_pytest, plus recording the result on ctx so the loop can report it."""
    passed, out = run_pytest(ctx.root, target)
    ctx.last_tests = passed
    return ("PASSED\n" if passed else "FAILED\n") + out


# ─────────────────────────── dispatch ───────────────────────────

_IMPL = {
    "list_tree": lambda c, a: _list_tree(c, a.get("path", ""), a.get("depth", 2)),
    "read_summary": lambda c, a: (db.get_summary(c.root, a["folder"])
                                  or f"No summary for '{a['folder']}'. "
                                     f"Known folders: {sorted(db.all_summaries(c.root))}"),
    "deps_of": lambda c, a: "\n".join(db.deps_of(c.root, a["path"])) or "imports nothing in-repo",
    "dependents_of": lambda c, a: "\n".join(db.dependents_of(c.root, a["path"]))
                                  or "nothing imports this file",
    "grep": lambda c, a: _grep(c, a["pattern"], a.get("glob") or "**/*.py"),
    "read_file": lambda c, a: _read_file(c, a["path"], a.get("start"), a.get("end")),
    "write_plan": lambda c, a: _write_plan(c, a["path"], a["verdict"], a.get("plan", ""),
                                           a.get("writes"), a.get("deletes")),
    "read_plan": lambda c, a: _read_plan(c, a["path"]),
    "write_file": lambda c, a: _write_file(c, a["path"], a["content"]),
    "str_replace": lambda c, a: _str_replace(c, a["path"], a["old_str"], a["new_str"]),
    "delete_file": lambda c, a: _delete_file(c, a["path"]),
    "run_tests": lambda c, a: _run_tests(c, a.get("target", "")),
    "skip_unit": lambda c, a: _skip_unit(c, a["reason"]),
}


def dispatch(ctx: Ctx, name: str, args_json: str) -> str:
    """
    Run one tool call. ALWAYS returns a string; never raises at the model.

    A raised exception would kill the loop. An error string goes back into the
    conversation and the model corrects itself on the next turn.
    """
    allowed = PLAN_TOOLS if ctx.phase == "PLAN" else EXECUTE_TOOLS
    if name not in allowed:
        return f"ERROR: '{name}' is not available during {ctx.phase}. Available: {allowed}"
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError as exc:
        return f"ERROR: could not parse arguments: {exc}"
    try:
        return _IMPL[name](ctx, args)
    except Refused as exc:
        return f"REFUSED: {exc}"
    except KeyError as exc:
        return f"ERROR: missing required argument {exc}"
    except FileNotFoundError:
        return f"ERROR: no such file: {args.get('path') or args.get('target')}"
    except Exception as exc:                       # never let a tool kill the run
        return f"ERROR: {type(exc).__name__}: {exc}"
