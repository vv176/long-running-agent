"""
Step 3 test: the ledger. The property that matters is that fold() reconstructs
state from events alone, so a "crash" loses nothing.

Run: python -m pytest tests/test_db.py -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foreman import db  # noqa: E402

REPO = Path("/tmp/fakerepo")


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.sqlite")
    db.init()


def test_create_and_fold_empty():
    rid = db.create_run("migrate to fastapi", REPO)
    st = db.fold(rid)
    assert st.phase == "INIT" and st.files == {} and not st.finished


def test_unknown_event_kind_is_refused():
    rid = db.create_run("g", REPO)
    with pytest.raises(ValueError, match="unknown event kind"):
        db.append(rid, "oops_typo", path="a.py")


def test_plan_then_execute_lifecycle():
    rid = db.create_run("g", REPO)
    db.append(rid, "phase_entered", payload={"phase": "PLAN"})
    db.append(rid, "plan_written", "a.py", {"plan": "port it", "verdict": "plan"})
    db.append(rid, "plan_written", "b.py", {"plan": None, "verdict": "no_change"})

    st = db.fold(rid)
    assert st.phase == "PLAN"
    assert [f.path for f in st.planned()] == ["a.py"]      # b.py excluded
    assert st.files["b.py"].status == "no_change"

    db.append(rid, "unit_started", "a.py")
    db.append(rid, "unit_done", "a.py")
    db.append(rid, "checkpointed", "a.py", {"sha": "deadbeef"})
    st = db.fold(rid)
    assert st.files["a.py"].status == "done"
    assert st.files["a.py"].attempts == 1
    assert st.files["a.py"].sha == "deadbeef"


def test_failure_returns_the_file_to_the_queue():
    rid = db.create_run("g", REPO)
    db.append(rid, "plan_written", "a.py", {"plan": "x", "verdict": "plan"})
    db.append(rid, "unit_started", "a.py")
    db.append(rid, "unit_failed", "a.py", {"error": "tests red"})

    st = db.fold(rid)
    assert st.files["a.py"].status == "planned"       # re-queued, not lost
    assert st.files["a.py"].attempts == 1
    assert st.files["a.py"].error == "tests red"
    assert db.next_ready(st, max_attempts=3).path == "a.py"


def test_attempt_cap_stops_reissuing():
    rid = db.create_run("g", REPO)
    db.append(rid, "plan_written", "a.py", {"plan": "x", "verdict": "plan"})
    for _ in range(3):
        db.append(rid, "unit_started", "a.py")
        db.append(rid, "unit_failed", "a.py", {"error": "nope"})
    st = db.fold(rid)
    assert st.files["a.py"].attempts == 3
    assert db.next_ready(st, max_attempts=3) is None


def test_replanning_bumps_version_and_keeps_history():
    """A revisit during traversal overwrites the plan but not the audit trail."""
    rid = db.create_run("g", REPO)
    db.append(rid, "plan_written", "a.py", {"plan": "v1 idea", "verdict": "plan"})
    db.append(rid, "plan_written", "a.py", {"plan": "v2 idea", "verdict": "plan"})
    st = db.fold(rid)
    assert st.files["a.py"].version == 2
    assert st.files["a.py"].plan == "v2 idea"
    # both writes are still on the log — nothing was overwritten in storage
    assert sum(1 for e in db.events(rid) if e["kind"] == "plan_written") == 2


def test_fold_is_the_whole_resume_mechanism():
    """Simulate a crash: build state, throw the object away, fold again."""
    rid = db.create_run("g", REPO)
    for i in range(6):
        db.append(rid, "plan_written", f"f{i}.py", {"plan": "p", "verdict": "plan"})
    for i in range(3):
        db.append(rid, "unit_started", f"f{i}.py")
        db.append(rid, "unit_done", f"f{i}.py")
    db.append(rid, "unit_started", "f3.py")        # killed mid-unit, no terminal event

    before = db.fold(rid)
    after = db.fold(rid)                          # a "restart" is just this call
    assert [f.status for f in before.planned()] == [f.status for f in after.planned()]

    done = [f.path for f in after.planned() if f.status == "done"]
    assert done == ["f0.py", "f1.py", "f2.py"]
    # f3 was interrupted mid-flight; it is 'running' with 1 attempt spent, and it
    # is picked up again because 1 < max_attempts. Nothing to reclaim.
    assert after.files["f3.py"].status == "running"
    assert db.next_ready(after, max_attempts=3).path == "f4.py"


def test_spend_accumulates():
    rid = db.create_run("g", REPO)
    db.append(rid, "spend", payload={"cents": 1.5, "tokens_in": 100, "tokens_out": 20})
    db.append(rid, "spend", payload={"cents": 2.0, "tokens_in": 50, "tokens_out": 10})
    st = db.fold(rid)
    assert st.cents == 3.5 and st.tokens_in == 150 and st.tokens_out == 30


def test_questions_pair_up():
    rid = db.create_run("g", REPO)
    db.append(rid, "question_asked", "admin.py", {"q": "is missing auth intentional?"})
    st = db.fold(rid)
    assert st.questions[0]["answer"] is None
    db.append(rid, "question_answered", "admin.py", {"answer": "no, add auth"})
    assert db.fold(rid).questions[0]["answer"] == "no, add auth"


# ─────────────── caches ───────────────

def test_edges_both_directions():
    db.save_edges(REPO, [("a.py", "b.py"), ("c.py", "b.py"), ("b.py", "d.py")])
    assert db.deps_of(REPO, "b.py") == ["d.py"]
    assert db.dependents_of(REPO, "b.py") == ["a.py", "c.py"]
    assert db.edge_count(REPO) == 3


def test_save_edges_replaces_not_appends():
    db.save_edges(REPO, [("a.py", "b.py")])
    db.save_edges(REPO, [("x.py", "y.py")])
    assert db.edge_count(REPO) == 1


def test_files_upsert_by_sha():
    db.save_files(REPO, [("a.py", "sha1", 10)])
    db.save_files(REPO, [("a.py", "sha2", 12)])
    assert db.get_files(REPO) == {"a.py": "sha2"}


def test_summary_staleness_cycle():
    db.save_summary(REPO, "app", "does app things")
    assert db.stale_folders(REPO) == []
    db.mark_stale(REPO, "app")
    assert db.stale_folders(REPO) == ["app"]
    db.save_summary(REPO, "app", "updated")        # re-saving clears stale
    assert db.stale_folders(REPO) == []
    assert db.get_summary(REPO, "app") == "updated"


def test_root_summary_uses_empty_folder_key():
    db.save_summary(REPO, "", "the repo overall")
    assert db.get_summary(REPO, "") == "the repo overall"
    assert "" in db.all_summaries(REPO)


# ─────────────── recovering the plan order on resume ───────────────

def test_last_plan_order_skips_payloads_without_an_order():
    """
    The bug this pins: several phases append `phase_entered`, and only the one
    drive() writes carries an `order`. Taking the LATEST payload returned {}, so a
    resumed run fell back to alphabetical and silently discarded the sequence PLAN
    reasoned about.
    """
    rid = db.create_run("g", REPO)
    db.append(rid, "phase_entered", payload={"phase": "PLAN"})
    db.append(rid, "phase_entered", payload={"phase": "EXECUTE",
                                            "order": ["b.py", "a.py", "c.py"]})
    db.append(rid, "phase_entered", payload={"phase": "EXECUTE"})       # execute.run's
    db.append(rid, "phase_entered", payload={"phase": "REPAIR"})        # no order
    assert db.last_plan_order(rid) == ["b.py", "a.py", "c.py"]
    # the naive version returned {} here, which is what caused the fallback
    assert db.last_payload(rid, "phase_entered").get("order") is None


def test_last_plan_order_is_empty_on_a_fresh_run():
    rid = db.create_run("g", REPO)
    assert db.last_plan_order(rid) == []


def test_last_plan_order_prefers_the_most_recent_order():
    """A re-plan must win over the order recorded before it."""
    rid = db.create_run("g", REPO)
    db.append(rid, "phase_entered", payload={"phase": "EXECUTE", "order": ["old.py"]})
    db.append(rid, "phase_entered", payload={"phase": "EXECUTE", "order": ["new.py"]})
    db.append(rid, "phase_entered", payload={"phase": "VERIFY"})
    assert db.last_plan_order(rid) == ["new.py"]
