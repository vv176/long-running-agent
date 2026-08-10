"""
Step 8 test: EXECUTE. Fake model. What must hold: the acceptance gate, rollback
before retry, the attempt cap into DLQ, and resumability.

Run: python -m pytest tests/test_execute.py -q
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foreman import checkpoint, config, db, execute, graph, index, llm  # noqa: E402


class FakeFn:
    def __init__(self, n, a): self.name, self.arguments = n, json.dumps(a)


class FakeCall:
    _n = 0

    def __init__(self, n, a):
        FakeCall._n += 1
        self.id, self.type, self.function = f"c{FakeCall._n}", "function", FakeFn(n, a)


class FakeMsg:
    def __init__(self, calls=None, content=""):
        self.content, self.role, self.tool_calls = content, "assistant", calls

    def model_dump(self, exclude_none=False):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments}}
                               for c in self.tool_calls]
        return d


def use(monkeypatch, steps):
    it = iter(steps)

    def fake(messages, tools=None, tool_choice=None, model=None, max_retries=4):
        try:
            turn = next(it)
        except StopIteration:
            return FakeMsg(content="done"), 10, 5, 0.01
        return FakeMsg([FakeCall(n, a) for n, a in turn]), 10, 5, 0.01
    monkeypatch.setattr(execute.llm, "call", fake)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    dst = tmp_path / "miniledger"
    shutil.copytree(ROOT / "fixtures" / "miniledger", dst,
                    ignore=shutil.ignore_patterns(".foreman", "__pycache__"))
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.sqlite")
    db.init()
    pairs, _ = graph.edges(dst)
    db.save_edges(dst, pairs)
    files, folders = index.scan(dst)
    db.save_files(dst, [(r, s, l) for r, (s, l) in files.items()])
    (dst / config.SIDECAR).mkdir(exist_ok=True)
    (dst / config.SIDECAR / "REPO_MAP.md").write_text("# map\n")
    return dst


def planned(rid, path, plan="do it", writes=None, deletes=None):
    db.append(rid, "plan_written", path,
              {"plan": plan, "verdict": "plan",
               "writes": writes if writes is not None else [path], "deletes": deletes or []})


GOOD = "X = 1\n"
BROKEN = "def f(:\n"


def test_unit_succeeds_and_checkpoints(repo, monkeypatch):
    rid = db.create_run("g", repo)
    planned(rid, "app/utils/money.py")
    use(monkeypatch, [[("write_file", {"path": "app/utils/money.py", "content": GOOD})]])
    out = execute.run(rid, repo, ["app/utils/money.py"], log=lambda *_: None)
    assert out["done"] == 1
    f = db.fold(rid).files["app/utils/money.py"]
    assert f.status == "done" and f.sha


def test_unparseable_write_is_rejected_and_rolled_back(repo, monkeypatch):
    original = (repo / "app/utils/money.py").read_text()
    rid = db.create_run("g", repo)
    planned(rid, "app/utils/money.py")
    use(monkeypatch, [[("write_file", {"path": "app/utils/money.py", "content": BROKEN})]] * 5)
    out = execute.run(rid, repo, ["app/utils/money.py"], log=lambda *_: None)
    assert out["done"] == 0 and out["dlq"] == 1
    assert (repo / "app/utils/money.py").read_text() == original, "must be rolled back"


def test_rollback_means_the_retry_starts_clean(repo, monkeypatch):
    """Attempt 1 writes garbage; attempt 2 must see the ORIGINAL bytes."""
    rid = db.create_run("g", repo)
    planned(rid, "app/utils/money.py")
    seen = []

    def fake(messages, tools=None, tool_choice=None, model=None, max_retries=4):
        first_turn = len(messages) == 2          # system + brief only
        if not first_turn:
            return FakeMsg(content="done"), 10, 5, 0.01     # stop, so accept() runs
        seen.append((repo / "app/utils/money.py").read_text())
        content = BROKEN if len(seen) == 1 else GOOD
        return FakeMsg([FakeCall("write_file",
                                 {"path": "app/utils/money.py", "content": content})]), 10, 5, 0.01
    monkeypatch.setattr(execute.llm, "call", fake)
    execute.run(rid, repo, ["app/utils/money.py"], log=lambda *_: None)
    assert len(seen) == 2, f"expected 2 attempts, saw {len(seen)}"
    assert seen[0] == seen[1], "attempt 2 saw different bytes than attempt 1"


def test_missing_declared_write_is_rejected(repo, monkeypatch):
    rid = db.create_run("g", repo)
    planned(rid, "app/routes/health.py", writes=["app/routers/health.py"])
    use(monkeypatch, [[("read_file", {"path": "app/routes/health.py"})]] * 5)
    out = execute.run(rid, repo, ["app/routes/health.py"], log=lambda *_: None)
    assert out["done"] == 0 and out["dlq"] == 1


def test_doing_nothing_is_rejected(repo, monkeypatch):
    rid = db.create_run("g", repo)
    planned(rid, "app/utils/money.py")
    use(monkeypatch, [[("read_file", {"path": "app/utils/money.py"})]] * 5)
    assert execute.run(rid, repo, [], log=lambda *_: None)["done"] == 0


def test_declared_delete_must_actually_happen(repo, monkeypatch):
    rid = db.create_run("g", repo)
    planned(rid, "wsgi.py", writes=[], deletes=["wsgi.py"])
    use(monkeypatch, [[("delete_file", {"path": "wsgi.py"})]])
    assert execute.run(rid, repo, ["wsgi.py"], log=lambda *_: None)["done"] == 1
    assert not (repo / "wsgi.py").exists()


def test_write_outside_the_declared_scope_is_blocked(repo, monkeypatch):
    rid = db.create_run("g", repo)
    planned(rid, "app/utils/money.py")
    use(monkeypatch, [[("write_file", {"path": "app/schemas.py", "content": "hacked"}),
                       ("write_file", {"path": "app/utils/money.py", "content": GOOD})]])
    execute.run(rid, repo, ["app/utils/money.py"], log=lambda *_: None)
    assert "hacked" not in (repo / "app/schemas.py").read_text()


def test_execution_follows_the_plan_order(repo, monkeypatch):
    rid = db.create_run("g", repo)
    for p in ("app/utils/money.py", "app/schemas.py", "wsgi.py"):
        planned(rid, p)
    seen = []

    def fake(messages, tools=None, tool_choice=None, model=None, max_retries=4):
        if len(messages) != 2:
            return FakeMsg(content="done"), 10, 5, 0.01
        path = messages[1]["content"].split("## The file\n")[1].splitlines()[0]
        if path not in seen:
            seen.append(path)
        return FakeMsg([FakeCall("write_file", {"path": path, "content": GOOD})]), 10, 5, 0.01
    monkeypatch.setattr(execute.llm, "call", fake)
    order = ["wsgi.py", "app/schemas.py", "app/utils/money.py"]
    execute.run(rid, repo, order, log=lambda *_: None)
    assert seen == order


def test_resume_skips_completed_units(repo, monkeypatch):
    rid = db.create_run("g", repo)
    for p in ("app/utils/money.py", "app/schemas.py"):
        planned(rid, p)
    db.append(rid, "unit_started", "app/utils/money.py")
    db.append(rid, "unit_done", "app/utils/money.py")

    touched = []

    def fake(messages, tools=None, tool_choice=None, model=None, max_retries=4):
        if len(messages) != 2:
            return FakeMsg(content="done"), 10, 5, 0.01
        path = messages[1]["content"].split("## The file\n")[1].splitlines()[0]
        touched.append(path)
        return FakeMsg([FakeCall("write_file", {"path": path, "content": GOOD})]), 10, 5, 0.01
    monkeypatch.setattr(execute.llm, "call", fake)
    execute.run(rid, repo, ["app/utils/money.py", "app/schemas.py"], log=lambda *_: None)
    assert touched == ["app/schemas.py"], "a completed unit must not be redone"


def test_completed_units_are_named_in_the_brief(repo, monkeypatch):
    rid = db.create_run("g", repo)
    for p in ("app/utils/money.py", "app/schemas.py"):
        planned(rid, p)
    db.append(rid, "unit_started", "app/utils/money.py")
    db.append(rid, "unit_done", "app/utils/money.py")
    briefs = []

    def fake(messages, tools=None, tool_choice=None, model=None, max_retries=4):
        briefs.append(messages[1]["content"])
        return FakeMsg(content="done"), 10, 5, 0.01
    monkeypatch.setattr(execute.llm, "call", fake)
    execute.run(rid, repo, ["app/schemas.py"], log=lambda *_: None)
    assert "Already completed in this run" in briefs[0]
    assert "app/utils/money.py" in briefs[0]


def test_previous_error_is_fed_back(repo, monkeypatch):
    rid = db.create_run("g", repo)
    planned(rid, "app/utils/money.py")
    db.append(rid, "unit_started", "app/utils/money.py")
    db.append(rid, "unit_failed", "app/utils/money.py", {"error": "does not parse"})
    briefs = []

    def fake(messages, tools=None, tool_choice=None, model=None, max_retries=4):
        briefs.append(messages[1]["content"])
        return FakeMsg([FakeCall("write_file",
                                 {"path": "app/utils/money.py", "content": GOOD})]), 10, 5, 0.01
    monkeypatch.setattr(execute.llm, "call", fake)
    execute.run(rid, repo, ["app/utils/money.py"], log=lambda *_: None)
    assert "does not parse" in briefs[0] and "rolled back" in briefs[0]


# ─────────── skip_unit: the escape hatch for a wrong plan ───────────

def test_a_file_that_needs_no_change_is_skipped_not_dlqd(repo, monkeypatch):
    """
    The failure this pins: a real run planned wsgi.py, EXECUTE read it and correctly
    found nothing to do — and accept() failed it three times ("ended without changing
    anything") straight into the DLQ. Punished for being right.
    """
    original = (repo / "wsgi.py").read_text()
    rid = db.create_run("g", repo)
    planned(rid, "wsgi.py")
    use(monkeypatch, [
        [("read_file", {"path": "wsgi.py"})],
        [("skip_unit", {"reason": "app still exports create_app, so this file is correct"})],
    ])
    out = execute.run(rid, repo, ["wsgi.py"], log=lambda *_: None)

    assert out["skipped"] == 1 and out["dlq"] == 0
    f = db.fold(rid).files["wsgi.py"]
    assert f.status == "skipped"
    assert "create_app" in f.note
    assert (repo / "wsgi.py").read_text() == original, "a skip must not touch the file"


def test_skip_is_terminal_and_never_retried(repo, monkeypatch):
    rid = db.create_run("g", repo)
    planned(rid, "wsgi.py")
    use(monkeypatch, [[("skip_unit", {"reason": "nothing to do"})]] * 5)
    out = execute.run(rid, repo, ["wsgi.py"], log=lambda *_: None)
    assert out["skipped"] == 1
    assert db.fold(rid).files["wsgi.py"].attempts == 1, "one attempt, not three"


def test_skip_after_writing_is_refused(repo, monkeypatch):
    """You cannot edit a file and then claim it needed nothing."""
    rid = db.create_run("g", repo)
    planned(rid, "app/utils/money.py")
    seen = []
    use(monkeypatch, [
        [("write_file", {"path": "app/utils/money.py", "content": GOOD})],
        [("skip_unit", {"reason": "changed my mind"})],
    ])
    monkeypatch.setattr(execute, "_log",
                        lambda ctx, n, a, r, log: seen.append((n, r)))
    out = execute.run(rid, repo, ["app/utils/money.py"], log=lambda *_: None)
    assert any(n == "skip_unit" and r.startswith("REFUSED") for n, r in seen)
    assert out["done"] == 1 and out["skipped"] == 0


def test_plan_cannot_skip(repo):
    """skip_unit is an EXECUTE tool. PLAN has no business closing units."""
    from foreman import tools
    assert "skip_unit" not in tools.PLAN_TOOLS
    ctx = tools.Ctx("r", repo, "PLAN")
    assert "not available during PLAN" in tools.dispatch(ctx, "skip_unit", '{"reason": "x"}')


# ─────────── REPAIR scope: a wrong no_change must be fixable ───────────

def test_repair_may_touch_files_plan_called_no_change(repo, monkeypatch):
    """
    The failure this pins: PLAN said tests/test_invoices.py needed no change while it
    still called Flask's r.get_json(). REPAIR grepped the exact line and was REFUSED
    three rounds running, because the file was outside scope. Correct diagnosis, no
    lever. After the suite is red, a `no_change` verdict has been disproved.
    """
    rid = db.create_run("g", repo)
    planned(rid, "app/__init__.py")
    db.append(rid, "plan_written", "tests/test_invoices.py",
              {"plan": "", "verdict": "no_change", "writes": [], "deletes": []})
    checkpoint.init(repo)

    monkeypatch.setattr(execute.tools, "run_pytest",
                        lambda root, target="": (False, "AssertionError: get_json"))
    seen = []
    fixed = "def test_create_and_get(client):\n    assert client.get('/x').json()\n"
    use(monkeypatch, [[("write_file", {"path": "tests/test_invoices.py",
                                       "content": fixed})]])
    monkeypatch.setattr(execute, "_log", lambda ctx, n, a, r, log: seen.append(r))
    execute.repair(rid, repo, log=lambda *_: None)

    assert seen, "the repair agent made no tool call"
    assert not seen[0].startswith("REFUSED"), f"still refused: {seen[0]}"
    assert "get_json" not in (repo / "tests/test_invoices.py").read_text()


def test_repair_still_refuses_files_plan_never_saw(repo, monkeypatch):
    """Widened, not unbounded: a file with no verdict at all stays out of scope."""
    rid = db.create_run("g", repo)
    planned(rid, "app/__init__.py")
    checkpoint.init(repo)
    monkeypatch.setattr(execute.tools, "run_pytest", lambda root, target="": (False, "boom"))
    seen = []
    use(monkeypatch, [[("write_file", {"path": "app/utils/money.py", "content": GOOD})]])
    monkeypatch.setattr(execute, "_log", lambda ctx, n, a, r, log: seen.append(r))
    execute.repair(rid, repo, log=lambda *_: None)
    assert seen and seen[0].startswith("REFUSED")
