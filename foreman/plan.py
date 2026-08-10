"""
PHASE 2 — PLAN. Goal-directed traversal of the repo, one plan per file.

The agent is NOT handed a file list. It gets the goal, the repo map, and the
in/out edge tools, and has to work out for itself which files matter. That is the
localisation problem, and the literature says it is where coding agents actually
fail: "bug localization is the primary bottleneck" (Ceka et al.), "repository
navigation dominates agent activity over patch writing" (Majgaonkar et al.).

Three bounds, because an unbounded traversal will happily spend the whole budget
before a single line of code is written:

    MAX_PLAN_HOPS   hard ceiling on turns; then it must finish with what it has
    MAX_REVISITS    a file can be re-planned at most this many times
    verdict         'no_change' is a cheap answer, and the prompt insists it is fine

Output: one `plan_written` event per visited file. Nothing is edited here — PLAN
does not have a single write tool (see tools.PLAN_TOOLS).
"""
import json

from . import config, db, index, llm, tools

# The terminal tool. An explicit signal beats "the model stopped calling tools",
# because we want to distinguish finishing from giving up.
FINISH = {
    "type": "function",
    "function": {
        "name": "finish_planning",
        "description": "Call this ONLY when every file that needs a plan has one.",
        "parameters": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string",
                              "description": "2-3 sentences: the shape of the change overall"},
                "order": {"type": "array", "items": {"type": "string"},
                          "description": "the files to execute, in the order they should be "
                                         "done. Dependencies before the things that use them."},
            },
            "required": ["rationale", "order"],
        },
    },
}


SYSTEM_PROMPT = """You are a senior engineer planning a change to a codebase you have not seen before.

You will NOT edit anything in this phase. Your only output is a plan per file, recorded with write_plan.

## How to find the files that matter

You are not given a file list. Work it out. In rough order:

1. Read the repo map you were given. Its Conventions section tells you where things
   belong in THIS codebase — trust it over your own habits.
2. Form a hypothesis about which layer the change lives in, then confirm it with grep.
   Search for the mechanism, not the goal: to find pagination, grep for `limit`,
   `offset`, `page`; to find a framework, grep for its import name.
3. read_summary on a folder before read_file on its contents. A summary is ~10x cheaper
   and usually tells you whether the folder is relevant at all.
4. For every file you decide to change, call dependents_of BEFORE you write its plan.
   Anything that imports it may need to change too, and that is how you find work you
   did not know existed.
5. Call deps_of to find what a file needs. If a file's plan depends on a helper gaining
   a new capability, that helper needs its own plan, and it must be executed first.

## What kinds of files a change usually touches

Generic patterns. Confirm each one against the actual repo; do not assume.

- **Framework or library migration**: entry points; every file that imports the old
  framework; dependency manifests (requirements.txt, pyproject.toml); and every test that
  exercises the framework. Pure domain logic that never touches the framework is usually
  UNTOUCHED — verify, then record no_change.
- **New feature**: find the nearest existing analogue. If the repo already does something
  similar, read it and copy its shape. Touch the layer that owns that concern, plus its
  tests. Adding a field usually means: schema/validation, the handler, the service, tests.
- **Bug fix**: localise from the symptom, then call dependents_of to check whether the same
  mistake exists in sibling call sites.
- **Rename or signature change**: mechanical but wide. Every dependent must be in the plan.
- **Dependency upgrade**: files importing the dependency, the pin in the manifest, and any
  test asserting the old behaviour.
- **Validation or security hardening**: the boundary layer where untrusted input arrives,
  not the core. Trace inward from the entry points.

## An absent import is NOT proof of independence

This is the trap that catches real agents. A file can be tightly coupled to something it
never imports:

- **Test files** receive an app or client through a FIXTURE, so they import nothing, yet
  they call framework-specific methods on what they are handed. Flask responses use
  `.get_json()` and `.data`; httpx/FastAPI responses use `.json()` and `.text`. A test
  that asserts on `.get_json()` MUST change in a Flask-to-FastAPI migration even though
  `deps_of` on it returns nothing at all.
- Anything wired by dependency injection, a plugin registry, a settings string, or
  `importlib` is invisible to the dependency graph.

So for a test file, never judge by its imports. READ ITS ASSERTIONS and ask: would this
code still run, and still mean the same thing, after the change? If the answer is no, it
needs a plan.

## Rules

- One write_plan call per file you visit. Say concretely WHAT changes and HOW.
- `verdict: "no_change"` is a good, cheap answer. Most files in a repo are irrelevant to
  any given change. Recording no_change is how you prove you considered a file.
- `writes` and `deletes` are a contract. During execution you will be BLOCKED from
  touching any path you did not declare, so declare every file you will create, modify,
  or remove — including new tests.
- Do not plan the same file twice unless new information genuinely changes it.
- If a decision needs a human (undefined behaviour, a security question, an intentional-
  looking bug), still write the plan, and say plainly in the plan text what the open
  question is. Do not guess silently.
- When you are done, call finish_planning with the execution order: dependencies first.

Be brief in your messages. The work is the tool calls."""


def _seed_prompt(goal: str, root, repo_map: str, n_files: int) -> str:
    """The first user message of the traversal: goal + repo map + the budget.

    The turn budget is stated out loud on purpose. Told how many moves it has, the
    model rules folders out from summaries instead of opening files one by one.
    """
    return (
        f"## Goal\n{goal}\n\n"
        f"## Repository map (always available to you)\n{repo_map}\n\n"
        f"## Scale\n{n_files} python files. You have at most {config.MAX_PLAN_HOPS} turns, "
        f"so do not read files you can rule out from a folder summary.\n\n"
        f"Start by deciding which folders can possibly be involved, then confirm with grep."
    )


def coverage_gap(root, st) -> list[str]:
    """
    Files that MUST have been considered but have no verdict at all.

    Two closures over the files being changed:

      1. dependents — anything importing a changed file may break
      2. same-folder siblings — because the import graph cannot see coupling that
         is not an import. On the first real run the agent planned
         tests/conftest.py but never looked at tests/test_invoices.py, which calls
         `r.get_json()` (Flask-only). pytest injects the `client` fixture BY NAME,
         so there is no edge to follow and `dependents_of(conftest)` is empty.
         Folder membership is the only signal that survives that.

    Returns paths with no plan and no no_change verdict — a genuine blind spot,
    not merely an unvisited file.
    """
    from pathlib import Path as _P
    changed = {f.path for f in st.planned()}
    every = set(db.get_files(root))
    must: set[str] = set()
    for p in changed:
        must |= set(db.dependents_of(root, p))
        folder = str(_P(p).parent)
        must |= {q for q in every if str(_P(q).parent) == folder}
    return sorted(must - set(st.files) - changed)


def run(run_id: str, root, goal: str, log=print) -> dict:
    """
    Plan the change. Returns coverage stats.

    In     run_id, repo root, the goal in the user's own words
    Out    {visited, to_change, no_change, unvisited, hops, order}
    Writes one `plan_written` event per visited file, plus spend events
    """
    db.append(run_id, "phase_entered", payload={"phase": "PLAN"})
    all_files = set(db.get_files(root))
    ctx = tools.Ctx(run_id, root, "PLAN")
    ctx.plan_versions = {}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _seed_prompt(goal, root, index.repo_map(root), len(all_files))},
    ]
    schemas = tools.schemas_for("PLAN") + [FINISH]
    order, rationale, hops, gap_rounds = [], "", 0, 0

    while hops < config.MAX_PLAN_HOPS:
        hops += 1
        # On the last turn there is no time left to explore, so demand the wrap-up.
        force = ({"type": "function", "function": {"name": "finish_planning"}}
                 if hops == config.MAX_PLAN_HOPS else None)
        try:
            msg, tin, tout, cents = llm.call(messages, tools=schemas, tool_choice=force)
        except llm.LLMError as exc:
            log(f"  ! planning call failed: {exc}")
            break
        db.append(run_id, "spend", payload={"cents": cents, "tokens_in": tin, "tokens_out": tout})

        if not msg.tool_calls:
            log("  (no tool calls — nudging for finish_planning)")
            messages.append(msg.model_dump(exclude_none=True))
            messages.append({"role": "user", "content":
                             "Call finish_planning now, or keep planning with the tools."})
            continue

        messages.append(msg.model_dump(exclude_none=True))
        finished = False
        for call in msg.tool_calls:
            name = call.function.name
            if name == "finish_planning":
                args = json.loads(call.function.arguments or "{}")
                rationale = args.get("rationale", "")
                order = [p for p in args.get("order", []) if p in all_files]
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": "Planning closed."})
                finished = True
                continue

            result = _dispatch_with_revisit_cap(ctx, name, call.function.arguments, log)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        if finished:
            # THE COVERAGE GATE. One chance to close blind spots before we accept
            # the plan. Bounded to a single round so it cannot become a loop.
            gap = coverage_gap(root, db.fold(run_id)) if gap_rounds == 0 else []
            if gap:
                gap_rounds += 1
                log(f"  ! coverage gap — {len(gap)} file(s) never considered: {', '.join(gap)}")
                messages.append({"role": "user", "content":
                    "Before this plan is accepted: these files are dependents or "
                    "same-folder siblings of files you are changing, and you have given "
                    "no verdict on any of them:\n" +
                    "\n".join(f"  {g}" for g in gap) +
                    "\n\nRead what you need, then call write_plan for EACH of them "
                    "(verdict 'plan' or 'no_change'), then call finish_planning again. "
                    "A same-folder sibling of a changed file often needs the same change "
                    "even when nothing imports it — a pytest fixture is injected by name, "
                    "not by import, so the dependency graph cannot see it."})
                continue
            break

    st = db.fold(run_id)
    to_change = [f.path for f in st.planned()]
    no_change = [p for p, f in st.files.items() if f.verdict == "no_change"]
    unvisited = sorted(all_files - set(st.files))

    # Order: what the model said, then anything it planned but forgot to order.
    order = [p for p in order if p in to_change]
    order += [p for p in to_change if p not in order]

    log(f"  {hops} turns · {len(to_change)} to change · {len(no_change)} no-change "
        f"· {len(unvisited)} never visited"
        + (f" · {gap_rounds} coverage round" if gap_rounds else ""))
    if rationale:
        log(f"  rationale: {rationale}")
    for p in order:
        f = st.files[p]
        log(f"    CHANGE {p}")
        log(f"           writes={f.writes} deletes={f.deletes}")
    if unvisited:
        # Not an error, but not a silent pass either: these files were never
        # considered, and the final report must not imply they were cleared.
        log(f"  ! never visited ({len(unvisited)}): {', '.join(unvisited[:8])}"
            f"{' …' if len(unvisited) > 8 else ''}")

    return {"visited": len(st.files), "to_change": len(to_change), "no_change": len(no_change),
            "unvisited": len(unvisited), "hops": hops, "order": order, "rationale": rationale,
            "gap_rounds": gap_rounds, "gap_remaining": coverage_gap(root, st)}


def _dispatch_with_revisit_cap(ctx, name, args_json, log) -> str:
    """
    Wraps dispatch to enforce MAX_REVISITS on write_plan.

    Revisit policy lives here rather than in tools.py: the tools should not know
    about planning strategy, only about what they are allowed to touch.
    """
    if name == "write_plan":
        try:
            path = json.loads(args_json or "{}").get("path", "")
        except json.JSONDecodeError:
            path = ""
        seen = ctx.plan_versions.get(path, 0)
        if seen > config.MAX_REVISITS:
            refusal = (f"REFUSED: {path} has already been planned {seen} times "
                       f"(limit {config.MAX_REVISITS} revisits). Keep the existing plan and "
                       f"move on, or call finish_planning.")
            # Log it too. A refusal is the single most useful line in a trace — it
            # shows a bound doing its job, and the first version returned it to the
            # model without ever printing it.
            _log_call(ctx, name, args_json, refusal, log)
            return refusal
        ctx.plan_versions[path] = seen + 1

    result = tools.dispatch(ctx, name, args_json)
    _log_call(ctx, name, args_json, result, log)
    return result


def _log_call(ctx, name, args_json, result, log) -> None:
    """Print one tool call, and record it in the ledger as a `tool_call` event.

    Two audiences. The printed line is for whoever is watching the run; the event is
    for `--events`, which can replay the whole loop after the fact. `hint` is
    whichever identifying argument this tool happens to use, so one line format fits
    all eight PLAN tools.
    """
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        args = {}
    hint = args.get("path") or args.get("folder") or args.get("pattern") or ""
    first = (result or "").strip().splitlines()[:1]
    tail = f"  -> {first[0][:80]}" if first else ""
    log(f"      {name}({hint}){tail}")
    db.append(ctx.run_id, "tool_call", path=hint or None,
              payload={"phase": "PLAN", "tool": name, "args": args,
                       "result": (result or "")[:400]})
