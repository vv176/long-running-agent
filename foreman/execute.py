"""
PHASE 3 — EXECUTE. One file's plan at a time, in its own conversation.

Each unit is: snapshot -> fresh conversation -> tools -> accept -> snapshot.
A failed unit is ROLLED BACK to its pre-attempt snapshot before the retry, so
attempt 2 starts from exactly the bytes attempt 1 started from. That is how a
retry stays idempotent without keeping a second copy of the tree.

WHAT THE HARNESS GATES ON, and what it deliberately does not:

    accepted when   every declared write exists, and every written .py parses
    NOT gated on    the full test suite

Because mid-migration the suite is legitimately red: a ported test needs a ported
router that may not exist yet. Gating each unit on a green suite would deadlock
the run. The model still has `run_tests` and is told to use it; the suite is the
gate at VERIFY, not here. Recording test status per unit as information rather
than as a verdict is the honest split.
"""
import ast
import json

from . import checkpoint, config, db, index, llm, tools

SYSTEM_PROMPT = """You are a senior engineer carrying out ONE step of an approved plan.

The plan for this file was written earlier, by you, with the whole repository in view.
Follow it. If the plan is wrong, do the correct thing.

If, after reading the file, this file genuinely needs NO change — the plan overreached —
call skip_unit(reason) and stop. Do not invent an edit to satisfy the plan. Do not use
skip_unit to dodge hard work: the whole suite still has to pass at the end.

Rules:
- You may only write or delete the paths the plan declared. Anything else is blocked.
- Prefer str_replace over write_file for edits to an existing file: it is surgical and
  keeps the rest of the file byte-identical. Use write_file for new files or a rewrite.
- Read before you write. Use read_file on the file itself and on anything it imports.
- Run run_tests when it will tell you something. Mid-change the suite may be red for
  reasons outside this file — that is expected, and not your problem to fix here.
- Keep behaviour identical unless the plan says otherwise.
- Stop when the plan for THIS file is done. Do not start on other files.

Be brief. The work is the tool calls."""


def available_packages() -> str:
    """
    What is importable in this environment.

    The agent cannot install anything — there is no tool for it, deliberately. On a
    real run it wrote an async pytest fixture, correctly worked out that it needed
    pytest-asyncio, tried to edit requirements.txt, and was blocked by the write
    guard. Its reasoning was right; it simply had no lever. Telling it the truth up
    front costs one line and removes the whole failure class.
    """
    from importlib.metadata import distributions
    names = sorted({d.metadata["Name"].lower() for d in distributions()
                    if d.metadata.get("Name")})
    return ", ".join(names)


def _brief(root, st, unit) -> str:
    """The user message for one unit. Everything the model needs, nothing more.

    Six blocks: the file and its edges, the plan PLAN wrote for it, the declared scope
    it will be held to, the previous failure if this is a retry, the units already
    finished (so it codes against the NEW contracts, not the old ones), the installed
    packages, and the repo map.

    Note what is NOT here: the conversation from the previous attempt. Each attempt
    starts fresh — a failed attempt's reasoning is exactly what we do not want to
    carry forward, and dropping it is also what keeps the context flat across units.
    """
    deps = db.deps_of(root, unit.path)
    parts = [
        f"## The file\n{unit.path}",
        f"imports (in-repo): {', '.join(deps) or 'nothing'}",
        f"imported by: {', '.join(db.dependents_of(root, unit.path)) or 'nothing'}",
        "",
        f"## Your plan for it\n{unit.plan}",
        "",
        f"## Declared scope (enforced)\nwrites: {unit.writes}\ndeletes: {unit.deletes}",
    ]
    if unit.error:
        parts += ["", "## The previous attempt failed", unit.error,
                  "The tree has been rolled back, so you are starting clean."]
    # What was already done, so it can code against new contracts rather than old ones.
    done = [f.path for f in st.planned() if f.status == "done"]
    if done:
        parts += ["", f"## Already completed in this run\n{', '.join(done)}",
                  "Those files are in their NEW form — read them if you depend on them."]
    parts += ["", "## Installed packages (you CANNOT install anything — design for these)",
              available_packages()]
    parts += ["", "## Repo map", index.repo_map(root)]
    return "\n".join(parts)


def accept(root, unit, ctx) -> tuple[bool, str]:
    """
    Deterministic gate: is this unit acceptable? Returns (ok, why_not).

    The model's own opinion that it is finished does not count. Four mechanical checks:
      1. every declared write exists on disk
      2. every declared delete is actually gone
      3. something changed at all (an empty attempt is a failure, not a success)
      4. every written .py file parses

    What this deliberately does NOT do is run the whole suite. A unit is one file; the
    suite can only go green once the units that share a contract are all finished. That
    is what the REPAIR phase is for.
    """
    missing = [w for w in unit.writes if not (root / w).exists()]
    if missing:
        return False, f"declared writes that do not exist: {missing}"
    still_there = [d for d in unit.deletes if (root / d).exists()]
    if still_there:
        return False, f"declared deletes that still exist: {still_there}"
    if not ctx.wrote and not ctx.deleted:
        return False, "the attempt ended without changing anything"
    for w in sorted(ctx.wrote):
        if not w.endswith(".py"):
            continue
        try:
            ast.parse((root / w).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            return False, f"{w} does not parse: line {exc.lineno}: {exc.msg}"
    return True, ""

# unit_started event payload : 
# path      = 'app/__init__.py'
# plan      = 'Change the Flask app creation to FastAPI. Update blueprint
#             registration to FastAPI router inclusion.'
# verdict   = 'plan'
# version   = 1
# status    = 'done'
# attempts  = 1
# sha       = '6811ca9c0dc1'
# error     = None
# writes    = ['app/__init__.py']
# deletes   = []
def run_unit(run_id, root, unit, log=print) -> str:
    """One attempt at one file. Returns the new status."""
    before = checkpoint.head(root)
    db.append(run_id, "unit_started", unit.path)
    st = db.fold(run_id)

    ctx = tools.Ctx(run_id, root, "EXECUTE", unit=unit.path,
                    allow_writes=unit.writes, allow_deletes=unit.deletes)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _brief(root, st, unit)}]
    schemas = tools.schemas_for("EXECUTE")
    spent = 0.0

    for turn in range(1, config.MAX_TOOL_CALLS_PER_UNIT + 1):
        try:
            msg, tin, tout, cents = llm.call(messages, tools=schemas)
        except llm.LLMError as exc:
            db.append(run_id, "unit_failed", unit.path, {"error": f"LLM error: {exc}"})
            checkpoint.rollback(root, before)
            return "planned"
        spent += cents
        db.append(run_id, "spend", payload={"cents": cents, "tokens_in": tin, "tokens_out": tout})

        if not msg.tool_calls:
            break
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            result = tools.dispatch(ctx, call.function.name, call.function.arguments)
            _log(ctx, call.function.name, call.function.arguments, result, log)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # The plan can be wrong. A unit that read the file and found nothing to do closes
    # as `skipped`, not as three failed attempts — see tools._skip_unit.
    if ctx.skipped and not ctx.wrote and not ctx.deleted:
        db.append(run_id, "unit_skipped", unit.path, {"reason": ctx.skipped})
        log(f"    SKIP {unit.path} — {ctx.skipped[:100]}")
        return "skipped"

    ok, why = accept(root, unit, ctx)
    if not ok:
        # Roll back so the retry starts from the same bytes this attempt started from.
        checkpoint.rollback(root, before)
        db.append(run_id, "unit_failed", unit.path, {"error": why})
        log(f"    FAILED {unit.path}: {why}  (rolled back)")
        return "planned"

    sha = checkpoint.snapshot(root, f"{unit.path}: {(unit.plan or '')[:60]}")
    db.append(run_id, "checkpointed", unit.path, {"sha": sha})
    db.append(run_id, "unit_done", unit.path)
    tag = "" if ctx.last_tests is None else (" tests:green" if ctx.last_tests else " tests:red")
    log(f"    DONE {unit.path}  wrote={sorted(ctx.wrote)} deleted={sorted(ctx.deleted)} "
        f"@{sha} ${spent / 100:.3f}{tag}")
    return "done"

# order payload in the phase_entered event : {
#    "phase": "EXECUTE",
#    "order": [
#      "app/__init__.py",
#      "app/routes/health.py",
#      "app/routes/invoices.py",
#      "app/schemas.py",
#      "wsgi.py",
#      "tests/conftest.py",
#      "tests/test_invoices.py"
#    ]
#  }

def run(run_id, root, order, log=print) -> dict:
    """
    Drain the planned units. Resumable: the queue is a fold of the ledger, so a
    restart continues rather than restarting.
    """
    db.append(run_id, "phase_entered", payload={"phase": "EXECUTE"})
    checkpoint.init(root)
    rank = {p: i for i, p in enumerate(order)}
    done = failed = dlq = skipped = 0

    while True:
        st = db.fold(run_id)
        ready = [f for f in st.planned()
                 if f.status in ("planned", "running") and f.attempts < config.MAX_ATTEMPTS_PER_FILE]
        if not ready:
            break
        # The plan's order wins; anything unordered goes last, alphabetically.
        unit = min(ready, key=lambda f: (rank.get(f.path, len(rank)), f.path))

        log(f"  [{unit.path}] attempt {unit.attempts + 1}/{config.MAX_ATTEMPTS_PER_FILE}")
        status = run_unit(run_id, root, unit, log)
        done += status == "done"
        skipped += status == "skipped"
        failed += status not in ("done", "skipped")

        st = db.fold(run_id)
        f = st.files[unit.path]
        if f.status == "planned" and f.attempts >= config.MAX_ATTEMPTS_PER_FILE:
            db.append(run_id, "unit_dlq", unit.path, {"error": f.error or "out of attempts"})
            dlq += 1
            log(f"    DLQ {unit.path} after {f.attempts} attempts — needs a human")

    st = db.fold(run_id)
    log(f"  {sum(1 for f in st.planned() if f.status == 'done')} done · "
        f"{skipped} skipped · {dlq} dlq")
    return {"done": sum(1 for f in st.planned() if f.status == "done"),
            "skipped": skipped, "dlq": dlq, "attempts": failed + done}


def _log(ctx, name, args_json, result, log) -> None:
    """Print one tool call and record it as a `tool_call` event. See plan._log_call.

    Same helper, different phase tag — EXECUTE has three extra arg names to pull the
    hint from (`target` for run_tests), which is why it is not shared code.
    """
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        args = {}
    hint = args.get("path") or args.get("target") or args.get("pattern") or args.get("folder") or ""
    first = (result or "").strip().splitlines()[:1]
    log(f"      {name}({hint})" + (f"  -> {first[0][:76]}" if first else ""))
    db.append(ctx.run_id, "tool_call", path=hint or None,
              payload={"phase": ctx.phase, "tool": name, "args": args,
                       "result": (result or "")[:400]})


# ─────────────────────────── PHASE 4 — REPAIR ───────────────────────────

REPAIR_PROMPT = """You are fixing a test suite that broke during a multi-file change.

Every file was changed correctly ON ITS OWN, but two of them may disagree with each
other — one exports a factory while another imports an instance, a fixture returns the
wrong type, an import path moved. Cross-file contract mismatches are the expected
failure here, so read BOTH sides before editing either.

How to work:
1. Read the failure. Identify the two files whose contracts disagree.
2. read_file both. Decide which one is right, then change the other.
3. run_tests to confirm. Repeat until green.

You may only touch files the plan already looked at. Keep behaviour identical to the
original service — the tests are the specification."""


def repair(run_id, root, log=print) -> dict:
    """
    The generate-test-repair loop the pipeline was missing.

    Rombaut §5.4: "the ReAct loop, while foundational, is typically insufficient on
    its own; layering a retry, test-repair, or planning primitive on top addresses
    failure modes a single feedback loop cannot handle." Our per-unit gate checks
    that each file did what it promised; it CANNOT see that two files promised
    incompatible things. Only the suite can, and only after both are done.

    Scope is every path PLAN gave a verdict on — not just the ones it decided to
    change. That widening is deliberate: REPAIR runs only after the suite is RED, and
    at that point a `no_change` verdict has been disproved by evidence. A run failed
    exactly here — PLAN called tests/test_invoices.py `no_change` while it still used
    Flask's `r.get_json()`; REPAIR found the line by grep and was refused three rounds
    running because the file was outside scope. Correct diagnosis, no lever.

    Still bounded: only files PLAN actually looked at, that exist on disk, plus any
    manifest already in the repo.
    """
    st = db.fold(run_id)
    scope = {w for f in st.planned() for w in f.writes}
    scope |= {p for p in st.files if p and (root / p).exists()}
    MANIFESTS = ("requirements.txt", "pyproject.toml", "setup.cfg", "setup.py")
    scope |= {m for m in MANIFESTS if (root / m).exists()}
    scope = sorted(scope)
    deletes = sorted({d for f in st.planned() for d in f.deletes})

    for rnd in range(1, config.MAX_REPAIR_ROUNDS + 1):
        passed, out = tools.run_pytest(root)
        db.append(run_id, "repair_round", payload={"round": rnd, "tests_passed": passed})
        if passed:
            log(f"  round {rnd}: suite already green")
            return {"rounds": rnd - 1, "tests_passed": True}

        log(f"  round {rnd}/{config.MAX_REPAIR_ROUNDS}: suite red — repairing")
        ctx = tools.Ctx(run_id, root, "EXECUTE", unit="<repair>",
                        allow_writes=scope, allow_deletes=deletes)
        messages = [
            {"role": "system", "content": REPAIR_PROMPT},
            {"role": "user", "content":
             f"## Goal of the change\n{db.get_run(run_id)['goal']}\n\n"
             f"## Files the plan changed (the only ones you may touch)\n"
             + "\n".join(f"  {p}" for p in scope)
             + "\n\n## Installed packages (you CANNOT install anything)\n"
             + available_packages()
             + f"\n\n## Failing suite\n```\n{out[-2500:]}\n```"},
        ]
        for _ in range(config.MAX_TOOL_CALLS_PER_UNIT):
            try:
                msg, tin, tout, cents = llm.call(messages, tools=tools.schemas_for("EXECUTE"))
            except llm.LLMError as exc:
                log(f"    ! repair call failed: {exc}")
                break
            db.append(run_id, "spend",
                      payload={"cents": cents, "tokens_in": tin, "tokens_out": tout})
            if not msg.tool_calls:
                break
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                result = tools.dispatch(ctx, call.function.name, call.function.arguments)
                _log(ctx, call.function.name, call.function.arguments, result, log)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        passed, _ = tools.run_pytest(root)
        if passed:
            sha = checkpoint.snapshot(root, f"repair round {rnd}")
            db.append(run_id, "checkpointed", None, {"sha": sha})
            log(f"    suite GREEN after round {rnd} @{sha}")
            return {"rounds": rnd, "tests_passed": True}

    log(f"  still red after {config.MAX_REPAIR_ROUNDS} rounds — needs a human")
    return {"rounds": config.MAX_REPAIR_ROUNDS, "tests_passed": False}
