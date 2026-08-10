"""
CLI. Wires the phases and nothing else.

    python -m foreman.run "<goal>" --repo <path>      start a run (copies the repo)
    python -m foreman.run --resume <run_id>            continue after Ctrl-C
    python -m foreman.run --list                       every run in the ledger
    python -m foreman.run --index-only --repo <path>   just build the map (no goal needed)
    python -m foreman.run --events <run_id>             replay the ledger (the trace)

The phase order lives in `runs.phase` via events, so a resume reads the ledger and
carries on. Nothing is rehydrated: no conversation is restored, because every unit
had its own and they were thrown away deliberately.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from . import checkpoint, config, db, execute, index, plan, tools

PHASES = ["INDEX", "PLAN", "EXECUTE", "REPAIR", "VERIFY", "DONE"]


def _work_root(repo: Path) -> Path:
    """Where the working COPY lives: <repo>_foreman, beside the original.

    A sibling, not a tempdir, so a failed run is still there to inspect afterwards.
    """
    return repo.parent / f"{repo.name}_foreman"


def verify(run_id, root, log=print) -> dict:
    """Run the WHOLE suite one last time. Returns {"tests_passed", "tail"}.

    REPAIR already ran the suite, so this is usually a formality — but it is the only
    number the report is allowed to print, so it is measured here rather than inherited.
    """
    db.append(run_id, "phase_entered", payload={"phase": "VERIFY"})
    passed, out = tools.run_pytest(root)
    tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
    log(f"  suite: {'GREEN' if passed else 'RED'} — {tail}")
    return {"tests_passed": passed, "tail": tail}


def report(run_id, root, tests_passed, log=print) -> None:
    """Print the run summary. Reads state by folding the ledger; returns nothing.

    Counts come from four disjoint buckets so they cannot flatter the run:
      done      unit finished and its accept gate passed
      no_change PLAN looked and decided nothing was needed
      dlq       hit the attempt cap — needs a human
      stuck     still planned/running, i.e. the run was interrupted
    The last line is the honest gate: any dlq, any stuck, or a red suite means
    "do not ship", regardless of how many units succeeded.
    """
    st = db.fold(run_id)
    done = [f for f in st.planned() if f.status == "done"]
    dlq = [f for f in st.planned() if f.status == "dlq"]
    skipped = [f for f in st.planned() if f.status == "skipped"]
    stuck = [f for f in st.planned() if f.status in ("planned", "running")]
    no_change = [p for p, f in st.files.items() if f.verdict == "no_change"]

    log("")
    log(f"  changed        : {len(done)}")
    log(f"  left alone     : {len(no_change)}")
    log(f"  needs a human  : {len(dlq)}")
    if skipped:
        # Not a failure: EXECUTE read the file and overruled a wrong plan. Shown so
        # the plan's mistakes are visible rather than averaged into "changed".
        log(f"  plan overruled : {len(skipped)}")
        for f in skipped:
            log(f"    SKIP {f.path} — {(f.note or '')[:100]}")
    if stuck:
        log(f"  ! outstanding  : {len(stuck)} (run was interrupted)")
    for f in dlq:
        log(f"    DLQ {f.path} — {(f.error or '')[:110]}")
    log(f"  suite          : {'GREEN' if tests_passed else 'RED'}")
    log(f"  spend          : ${st.cents / 100:.3f} · "
        f"{st.tokens_in + st.tokens_out:,} tokens")
    log(f"  work tree      : {root}")
    if not tests_passed or dlq or stuck:
        log("  NOT complete — do not ship this tree")
    db.append(run_id, "run_finished", payload={"tests_passed": tests_passed})


def drive(run_id: str, log=print) -> None:
    """Run the phase machine from wherever the ledger says this run is.

    The `if phase == X` blocks deliberately FALL THROUGH rather than being elif:
    a fresh run enters at INDEX and walks all the way to REPORT, while a resume
    enters at whichever phase the fold reported and continues from there. One code
    path serves both, so resume cannot drift from the happy path.
    """
    run = db.get_run(run_id)
    root = Path(run["repo_root"]) # ← where to work
    goal = run["goal"] # ← what to do
    st = db.fold(run_id) # ← how far it got
    phase = st.phase if st.phase in PHASES else "INDEX"
    order = db.last_plan_order(run_id)      # To find order details in the latest plan written event
    #db.last_plan_order("run_ef48dec1")
    #['app/__init__.py',  
   #'app/routes/health.py',
   #'app/routes/invoices.py',
   #'app/schemas.py',
   #'wsgi.py',
   #'tests/conftest.py',
   #'tests/test_invoices.py']

    try:
        if phase == "INDEX":
            log("\n━━ INDEX")
            index.build(run_id, root, log)
            phase = "PLAN"

        if phase == "PLAN":
            log("\n━━ PLAN")
            out = plan.run(run_id, root, goal, log)
            order = out["order"]
            phase = "EXECUTE"

        if phase == "EXECUTE":
            if not order:
                # Belt and braces. Reaching here means PLAN planned nothing: on a
                # fresh run plan.run() rebuilds `order` from to_change even if
                # finish_planning was never called, and a resume that never recorded
                # an EXECUTE phase_entered re-enters at PLAN, not here. So this
                # normally yields [] too — it exists so a future caller that skips
                # PLAN still gets a deterministic order rather than none.
                order = [f.path for f in db.fold(run_id).planned()]
                if order:
                    log(f"  ! no recorded plan order — falling back to path order "
                        f"({len(order)} unit(s))")
            db.append(run_id, "phase_entered", payload={"phase": "EXECUTE", "order": order})
            log(f"\n━━ EXECUTE  {len(order)} unit(s)")
            execute.run(run_id, root, order, log)
            phase = "REPAIR"

        if phase == "REPAIR":
            db.append(run_id, "phase_entered", payload={"phase": "REPAIR", "order": order})
            log("\n━━ REPAIR  (generate-test-repair: the per-unit gate cannot see "
                "cross-file mismatches)")
            execute.repair(run_id, root, log)
            phase = "VERIFY"

        if phase == "VERIFY":
            log("\n━━ VERIFY")
            v = verify(run_id, root, log)
            log("\n━━ REPORT")
            report(run_id, root, v["tests_passed"], log)

    except KeyboardInterrupt:
        # Nothing to clean up. append() was the only writer, so no state is half-done.
        log("\n\n  interrupted — nothing lost")
        log(f"  resume with:  python -m foreman.run --resume {run_id}")
        raise SystemExit(130)


KIND_MARK = {"tool_call": "  ->", "tool_result": "  <-", "plan_written": "  plan",
             "unit_started": "UNIT", "unit_done": "  ok", "unit_failed": "  FAIL",
             "phase_entered": "PHASE", "spend": "  $"}


def show_events(run_id: str, kinds: str = "") -> None:
    """Replay a run's ledger to stdout — the trace of what the agent actually did.

    This is not logging. The rows printed here ARE the run's state; fold() reads the
    same rows to resume. `kinds` is an optional comma-separated filter, used as
    --events <id> --only tool_call,tool_result.
    """
    rows = db.events(run_id)
    if not rows:
        sys.exit(f"no events for {run_id}")
    wanted = {k.strip() for k in kinds.split(",") if k.strip()}
    t0 = rows[0]["ts"]
    shown = 0
    for r in rows:
        if wanted and r["kind"] not in wanted:
            continue
        shown += 1
        # One line per event: seconds-since-start, a marker, the kind, the path, and a
        # squashed payload. " ".join(split()) collapses every newline and run of spaces
        # so a 200-line tool result stays one grep-able row.
        body = " ".join(json.dumps(json.loads(r["payload"]), default=str).split())
        mark = KIND_MARK.get(r["kind"], "   .")
        path = f" {r['path']}" if r["path"] else ""
        print(f"[{r['ts'] - t0:7.1f}s] {mark:<5} {r['kind']}{path}  {body[:220]}")
    print(f"\n  {shown} shown of {len(rows)} events - "
          f"fold() replays exactly these to rebuild state")


def main() -> None:
    """Parse args and dispatch. Five modes: --list, --events, --resume, --index-only,
    or a fresh run from a goal + --repo."""
    ap = argparse.ArgumentParser(prog="foreman")
    ap.add_argument("goal", nargs="?", help="what to do, in plain English")
    ap.add_argument("--repo", help="path to the repository")
    ap.add_argument("--resume", metavar="RUN_ID")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--events", metavar="RUN_ID", help="replay a run's event ledger")
    ap.add_argument("--only", default="", help="with --events: comma-separated event kinds")
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--in-place", action="store_true",
                    help="work on --repo directly instead of a copy (dangerous)")
    a = ap.parse_args()
    db.init()

    if a.list:
        for r in db.list_runs():
            st = db.fold(r["run_id"])
            print(f"{r['run_id']}  {st.phase:<8} ${st.cents / 100:.3f}  "
                  f"{len([f for f in st.planned() if f.status == 'done'])}/"
                  f"{len(st.planned())} done  \"{r['goal'][:50]}\"")
        return

    if a.events:
        show_events(a.events, a.only)
        return

    if a.resume:
        if db.get_run(a.resume) is None:
            sys.exit(f"no such run: {a.resume}")
        st = db.fold(a.resume)
        print(f"━━ RESUME {a.resume}  (was in {st.phase}, ${st.cents / 100:.3f} spent)")
        drive(a.resume)
        return

    # Everything below makes model calls. Check the key HERE, not inside a worker
    # thread four minutes in — the read-only modes above deliberately work without one.
    if not config.OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY missing. Copy .env.example to .env and fill it in.")

    if not a.repo:
        ap.error("--repo is required")
    repo = Path(a.repo).resolve()
    if not repo.is_dir():
        sys.exit(f"not a directory: {repo}")

    if a.in_place:
        root = repo
    else:
        root = _work_root(repo)
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(repo, root, ignore=shutil.ignore_patterns(
            *config.IGNORE_DIRS, "*.pyc")) # Ignore all files in the config.IGNORE_DIRS list, and all .pyc files
        print(f"working on a copy: {root}")

    if a.index_only:
        rid = db.create_run("index only", root)
        print("\n━━ INDEX")
        index.build(rid, root)
        return

    if not a.goal:
        ap.error("a goal is required (or use --index-only)")
    rid = db.create_run(a.goal, root)
    print(f"━━ FOREMAN {rid}\n  goal: {a.goal}\n  repo: {root}\n"
          f"  caps: {config.MAX_PLAN_HOPS} plan turns · "
          f"{config.MAX_ATTEMPTS_PER_FILE} attempts/file · "
          f"{config.MAX_TOOL_CALLS_PER_UNIT} tool calls/unit")
    checkpoint.init(root)
    drive(rid)


if __name__ == "__main__":
    main()
