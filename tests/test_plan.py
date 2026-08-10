"""
Step 6 test: the PLAN traversal. Fake model, so we test the HARNESS: bounds,
termination, revisit caps, ordering, coverage accounting.

Prompt quality is a separate question, exercised by scripts/try_plan.py against
the real model.

Run: python -m pytest tests/test_plan.py -q
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foreman import config, db, graph, index, llm, plan  # noqa: E402


# ─────────────── fake model plumbing ───────────────

class FakeFn:
    def __init__(self, name, args): self.name, self.arguments = name, json.dumps(args)


class FakeCall:
    _n = 0

    def __init__(self, name, args):
        FakeCall._n += 1
        self.id, self.type = f"c{FakeCall._n}", "function"
        self.function = FakeFn(name, args)


class FakeMsg:
    def __init__(self, calls=None, content=""):
        self.content, self.role = content, "assistant"
        self.tool_calls = calls

    def model_dump(self, exclude_none=False):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments}}
                               for c in self.tool_calls]
        return d


def scripted(steps):
    """steps: list of lists of (tool_name, args). One list per turn."""
    it = iter(steps)

    def fake(messages, tools=None, tool_choice=None, model=None, max_retries=4):
        try:
            turn = next(it)
        except StopIteration:
            turn = [("finish_planning", {"rationale": "ran out of script", "order": []})]
        return FakeMsg([FakeCall(n, a) for n, a in turn]), 10, 5, 0.01
    return fake


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
    for f in folders:
        db.save_summary(dst, f, f"# {f}\nsummary of {f}\n")
    (dst / config.SIDECAR).mkdir(exist_ok=True)
    (dst / config.SIDECAR / "REPO_MAP.md").write_text("# Repository map\ntiny flask app\n")
    return dst


def use(monkeypatch, steps):
    monkeypatch.setattr(llm, "call", scripted(steps))
    monkeypatch.setattr(plan.llm, "call", scripted(steps))


# ─────────────── happy path ───────────────

def test_plans_then_finishes(repo, monkeypatch):
    use(monkeypatch, [
        [("read_summary", {"folder": "app/routes"})],
        [("dependents_of", {"path": "app/schemas.py"})],
        [("write_plan", {"path": "app/schemas.py", "verdict": "plan",
                         "plan": "pydantic", "writes": ["app/schemas.py"]})],
        [("write_plan", {"path": "app/routes/invoices.py", "verdict": "plan",
                         "plan": "APIRouter", "writes": ["app/routes/invoices.py"]})],
        [("write_plan", {"path": "app/utils/money.py", "verdict": "no_change"})],
        [("finish_planning", {"rationale": "schemas first, then routers",
                              "order": ["app/schemas.py", "app/routes/invoices.py"]})],
        # the coverage gate fires here; clear it, then finish again
        [("write_plan", {"path": "app/__init__.py", "verdict": "no_change"}),
         ("write_plan", {"path": "app/routes/__init__.py", "verdict": "no_change"}),
         ("write_plan", {"path": "app/routes/health.py", "verdict": "no_change"})],
        [("finish_planning", {"rationale": "schemas first, then routers",
                              "order": ["app/schemas.py", "app/routes/invoices.py"]})],
    ])
    out = plan.run(db.create_run("g", repo), repo, "migrate to fastapi", log=lambda *_: None)
    assert out["to_change"] == 2 and out["no_change"] == 4
    assert out["order"] == ["app/schemas.py", "app/routes/invoices.py"]
    assert out["gap_rounds"] == 1 and out["gap_remaining"] == []


def test_unvisited_files_are_counted_not_hidden(repo, monkeypatch):
    use(monkeypatch, [
        [("write_plan", {"path": "app/schemas.py", "verdict": "plan",
                         "plan": "x", "writes": ["app/schemas.py"]})],
        [("finish_planning", {"rationale": "done", "order": ["app/schemas.py"]})],
    ])
    out = plan.run(db.create_run("g", repo), repo, "g", log=lambda *_: None)
    # the gate adds verdicts for app/ siblings, but most of the repo stays unvisited
    assert out["unvisited"] >= 10, "unvisited files must be reported, not silently cleared"


# ─────────────── the coverage gate ───────────────

def test_gate_catches_the_fixture_blind_spot(repo, monkeypatch):
    """
    tests/test_invoices.py has NO imports — pytest injects `client` by name. So no
    edge points at it and dependents_of(conftest) is empty. Only same-folder
    closure can find it, and a real run missed it before this gate existed.
    """
    use(monkeypatch, [
        [("write_plan", {"path": "tests/conftest.py", "verdict": "plan",
                         "plan": "async fixture", "writes": ["tests/conftest.py"]})],
        [("finish_planning", {"rationale": "r", "order": ["tests/conftest.py"]})],
        [("write_plan", {"path": "tests/test_invoices.py", "verdict": "plan",
                         "plan": "httpx", "writes": ["tests/test_invoices.py"]}),
         ("write_plan", {"path": "tests/test_money.py", "verdict": "no_change"})],
        [("finish_planning", {"rationale": "r",
                              "order": ["tests/conftest.py", "tests/test_invoices.py"]})],
    ])
    lines = []
    out = plan.run(db.create_run("g", repo), repo, "g", log=lines.append)
    assert out["gap_rounds"] == 1
    assert "tests/test_invoices.py" in out["order"]
    assert any("coverage gap" in ln and "test_invoices" in ln for ln in lines)


def test_gate_runs_at_most_once(repo, monkeypatch):
    """It must not become a loop if the model ignores the gate."""
    use(monkeypatch, [
        [("write_plan", {"path": "tests/conftest.py", "verdict": "plan",
                         "plan": "x", "writes": ["tests/conftest.py"]})],
        [("finish_planning", {"rationale": "r", "order": ["tests/conftest.py"]})],
        [("finish_planning", {"rationale": "still ignoring you", "order": []})],
    ])
    out = plan.run(db.create_run("g", repo), repo, "g", log=lambda *_: None)
    assert out["gap_rounds"] == 1
    assert out["gap_remaining"], "an ignored gate is reported, not retried forever"


def test_no_gate_when_nothing_is_planned(repo, monkeypatch):
    use(monkeypatch, [[("finish_planning", {"rationale": "nothing to do", "order": []})]])
    out = plan.run(db.create_run("g", repo), repo, "g", log=lambda *_: None)
    assert out["gap_rounds"] == 0


def test_unvisited_is_logged_loudly(repo, monkeypatch):
    use(monkeypatch, [[("finish_planning", {"rationale": "r", "order": []})]])
    lines = []
    plan.run(db.create_run("g", repo), repo, "g", log=lines.append)
    assert any("never visited" in ln for ln in lines)


# ─────────────── bounds ───────────────

def test_hop_ceiling_stops_the_traversal(repo, monkeypatch):
    monkeypatch.setattr(config, "MAX_PLAN_HOPS", 4)
    # never calls finish_planning on its own
    use(monkeypatch, [[("read_summary", {"folder": "app"})]] * 20)
    out = plan.run(db.create_run("g", repo), repo, "g", log=lambda *_: None)
    assert out["hops"] == 4


def test_last_turn_forces_finish(repo, monkeypatch):
    monkeypatch.setattr(config, "MAX_PLAN_HOPS", 3)
    seen = {}

    def fake(messages, tools=None, tool_choice=None, model=None, max_retries=4):
        seen["choice"] = tool_choice
        return FakeMsg([FakeCall("read_summary", {"folder": "app"})]), 10, 5, 0.01

    monkeypatch.setattr(plan.llm, "call", fake)
    plan.run(db.create_run("g", repo), repo, "g", log=lambda *_: None)
    assert seen["choice"] == {"type": "function", "function": {"name": "finish_planning"}}


def test_revisit_cap_refuses_further_replanning(repo, monkeypatch):
    monkeypatch.setattr(config, "MAX_REVISITS", 1)
    p = {"path": "app/schemas.py", "verdict": "plan", "plan": "v", "writes": ["app/schemas.py"]}
    use(monkeypatch, [[("write_plan", p)]] * 5 +
        [[("finish_planning", {"rationale": "r", "order": ["app/schemas.py"]})]])
    lines = []
    plan.run(db.create_run("g", repo), repo, "g", log=lines.append)
    assert any("REFUSED" in ln and "already been planned" in ln for ln in lines)


def test_plan_cannot_edit_a_file(repo, monkeypatch):
    """The structural guardrail: PLAN has no write tools at all."""
    use(monkeypatch, [
        [("write_file", {"path": "app/schemas.py", "content": "hacked"})],
        [("finish_planning", {"rationale": "r", "order": []})],
    ])
    lines = []
    plan.run(db.create_run("g", repo), repo, "g", log=lines.append)
    assert "hacked" not in (repo / "app" / "schemas.py").read_text()


# ─────────────── ordering & robustness ───────────────

def test_planned_but_unordered_files_are_appended(repo, monkeypatch):
    use(monkeypatch, [
        [("write_plan", {"path": "app/schemas.py", "verdict": "plan", "plan": "a",
                         "writes": ["app/schemas.py"]})],
        [("write_plan", {"path": "wsgi.py", "verdict": "plan", "plan": "b",
                         "writes": ["wsgi.py"]})],
        # order omits wsgi.py
        [("finish_planning", {"rationale": "r", "order": ["app/schemas.py"]})],
    ])
    out = plan.run(db.create_run("g", repo), repo, "g", log=lambda *_: None)
    assert out["order"] == ["app/schemas.py", "wsgi.py"]


def test_invented_paths_are_dropped_from_the_order(repo, monkeypatch):
    use(monkeypatch, [
        [("write_plan", {"path": "app/schemas.py", "verdict": "plan", "plan": "a",
                         "writes": ["app/schemas.py"]})],
        [("finish_planning", {"rationale": "r",
                              "order": ["app/schemas.py", "app/does_not_exist.py"]})],
    ])
    out = plan.run(db.create_run("g", repo), repo, "g", log=lambda *_: None)
    assert out["order"] == ["app/schemas.py"]


def test_text_only_reply_is_nudged_not_fatal(repo, monkeypatch):
    turns = [FakeMsg(content="I think I should look at routes."),
             FakeMsg([FakeCall("finish_planning", {"rationale": "r", "order": []})])]
    it = iter(turns)
    monkeypatch.setattr(plan.llm, "call",
                        lambda *a, **k: (next(it), 10, 5, 0.01))
    lines = []
    out = plan.run(db.create_run("g", repo), repo, "g", log=lines.append)
    assert out["hops"] == 2
    assert any("nudging" in ln for ln in lines)


def test_llm_failure_ends_planning_gracefully(repo, monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("429 forever")
    monkeypatch.setattr(plan.llm, "call", boom)
    lines = []
    out = plan.run(db.create_run("g", repo), repo, "g", log=lines.append)
    assert out["to_change"] == 0
    assert any("planning call failed" in ln for ln in lines)


def test_spend_is_recorded_per_turn(repo, monkeypatch):
    use(monkeypatch, [
        [("read_summary", {"folder": "app"})],
        [("finish_planning", {"rationale": "r", "order": []})],
    ])
    rid = db.create_run("g", repo)
    plan.run(rid, repo, "g", log=lambda *_: None)
    st = db.fold(rid)
    assert st.cents == pytest.approx(0.02)
    assert st.phase == "PLAN"
