"""
Step 5 test: the tools, and above all the GUARDS. No model involved.

The guards are the part that must be provably airtight — everything else the
model can recover from by reading an error string.

Run: python -m pytest tests/test_tools.py -q
"""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foreman import config, db, graph, tools  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    dst = tmp_path / "miniledger"
    shutil.copytree(ROOT / "fixtures" / "miniledger", dst,
                    ignore=shutil.ignore_patterns(".foreman", "__pycache__"))
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.sqlite")
    db.init()
    pairs, _ = graph.edges(dst)
    db.save_edges(dst, pairs)
    db.save_summary(dst, "app/routes", "# app/routes\nFlask routes.\n")
    return dst


@pytest.fixture
def plan_ctx(repo):
    return tools.Ctx(db.create_run("g", repo), repo, "PLAN")


def exec_ctx(repo, unit, writes=(), deletes=()):
    return tools.Ctx(db.create_run("g", repo), repo, "EXECUTE",
                     unit=unit, allow_writes=list(writes), allow_deletes=list(deletes))


# ─────────────── phase scoping ───────────────

def test_plan_has_no_edit_tools():
    names = {s["function"]["name"] for s in tools.schemas_for("PLAN")}
    assert "write_file" not in names and "delete_file" not in names and "run_tests" not in names
    assert "write_plan" in names


def test_execute_has_no_write_plan():
    names = {s["function"]["name"] for s in tools.schemas_for("EXECUTE")}
    assert "write_plan" not in names
    assert {"write_file", "str_replace", "delete_file", "run_tests"} <= names


def test_calling_an_out_of_phase_tool_is_refused(plan_ctx):
    out = tools.dispatch(plan_ctx, "write_file", '{"path":"x.py","content":"y"}')
    assert out.startswith("ERROR: 'write_file' is not available during PLAN")


# ─────────────── read tools ───────────────

def test_list_tree(plan_ctx):
    out = tools.dispatch(plan_ctx, "list_tree", '{"path":"app","depth":1}')
    assert "routes/" in out and "schemas.py" in out


def test_list_tree_hides_sidecar(repo, plan_ctx):
    (repo / config.SIDECAR).mkdir(exist_ok=True)
    (repo / config.SIDECAR / "REPO_MAP.md").write_text("x")
    assert config.SIDECAR not in tools.dispatch(plan_ctx, "list_tree", '{"path":""}')


def test_read_summary_hit_and_miss(plan_ctx):
    assert "Flask routes" in tools.dispatch(plan_ctx, "read_summary", '{"folder":"app/routes"}')
    assert "No summary" in tools.dispatch(plan_ctx, "read_summary", '{"folder":"nope"}')


def test_deps_and_dependents(plan_ctx):
    deps = tools.dispatch(plan_ctx, "deps_of", '{"path":"app/services/invoice.py"}')
    assert "app/models/invoice.py" in deps and "app/utils/money.py" in deps
    used = tools.dispatch(plan_ctx, "dependents_of", '{"path":"app/utils/money.py"}')
    assert "app/schemas.py" in used and "tests/test_money.py" in used


def test_deps_of_a_leaf(plan_ctx):
    assert tools.dispatch(plan_ctx, "deps_of",
                          '{"path":"app/utils/money.py"}') == "imports nothing in-repo"


def test_read_file_with_and_without_range(plan_ctx):
    whole = tools.dispatch(plan_ctx, "read_file", '{"path":"app/utils/money.py"}')
    assert "class Money" in whole and whole.lstrip().startswith("1 |")
    part = tools.dispatch(plan_ctx, "read_file",
                          '{"path":"app/utils/money.py","start":1,"end":2}')
    assert len(part.splitlines()) == 2


def test_grep(plan_ctx):
    out = tools.dispatch(plan_ctx, "grep", '{"pattern":"Blueprint"}')
    assert "app/routes/invoices.py" in out and "app/routes/health.py" in out
    assert tools.dispatch(plan_ctx, "grep", '{"pattern":"zzz_nope"}') == "no matches"


def test_grep_bad_regex_is_an_error_not_a_crash(plan_ctx):
    assert tools.dispatch(plan_ctx, "grep", '{"pattern":"[unclosed"}').startswith("ERROR: bad regex")


# ─────────────── path guards ───────────────

@pytest.mark.parametrize("path", ["../escape.py", "app/../../escape.py"])
def test_paths_outside_the_repo_are_refused(plan_ctx, path):
    out = tools.dispatch(plan_ctx, "read_file", '{"path":"%s"}' % path)
    assert "REFUSED" in out and "outside the repo" in out


def test_sidecar_is_off_limits(plan_ctx):
    out = tools.dispatch(plan_ctx, "read_file", '{"path":".foreman/REPO_MAP.md"}')
    assert "REFUSED" in out


# ─────────────── write_plan ───────────────

def test_write_plan_records_and_folds(plan_ctx):
    out = tools.dispatch(plan_ctx, "write_plan",
                         '{"path":"app/routes/health.py","verdict":"plan",'
                         '"plan":"port to APIRouter","writes":["app/routers/health.py"]}')
    assert "Plan recorded" in out
    f = db.fold(plan_ctx.run_id).files["app/routes/health.py"]
    assert f.verdict == "plan" and f.writes == ["app/routers/health.py"]


def test_no_change_verdict_is_cheap_and_excluded_from_work(plan_ctx):
    tools.dispatch(plan_ctx, "write_plan",
                   '{"path":"app/utils/money.py","verdict":"no_change"}')
    st = db.fold(plan_ctx.run_id)
    assert st.files["app/utils/money.py"].status == "no_change"
    assert st.planned() == []


def test_plan_with_no_blast_radius_is_rejected(plan_ctx):
    out = tools.dispatch(plan_ctx, "write_plan",
                         '{"path":"app/routes/health.py","verdict":"plan","plan":"do something"}')
    assert out.startswith("ERROR") and "writes or deletes" in out


def test_replanning_overwrites_the_plan_but_not_the_log(plan_ctx):
    for n in (1, 2):
        tools.dispatch(plan_ctx, "write_plan",
                       '{"path":"a.py","verdict":"plan","plan":"v%d","writes":["a.py"]}' % n)
    f = db.fold(plan_ctx.run_id).files["a.py"]
    assert f.plan == "v2" and f.version == 2


def test_read_plan_roundtrip(plan_ctx):
    assert "No plan recorded" in tools.dispatch(plan_ctx, "read_plan", '{"path":"a.py"}')
    tools.dispatch(plan_ctx, "write_plan",
                   '{"path":"a.py","verdict":"plan","plan":"do it","writes":["a.py"]}')
    out = tools.dispatch(plan_ctx, "read_plan", '{"path":"a.py"}')
    assert "verdict: plan" in out and "do it" in out


# ─────────────── the write guard (the important one) ───────────────

def test_write_allowed_only_for_declared_paths(repo):
    ctx = exec_ctx(repo, "app/routes/health.py", writes=["app/routers/health.py"])
    ok = tools.dispatch(ctx, "write_file", '{"path":"app/routers/health.py","content":"x=1"}')
    assert ok.startswith("Created")
    bad = tools.dispatch(ctx, "write_file", '{"path":"app/routers/invoices.py","content":"x=1"}')
    assert "REFUSED" in bad and "declared writes" in bad
    assert not (repo / "app/routers/invoices.py").exists()


def test_str_replace_obeys_the_same_guard(repo):
    ctx = exec_ctx(repo, "app/utils/money.py", writes=["app/utils/money.py"])
    ok = tools.dispatch(ctx, "str_replace",
                        '{"path":"app/utils/money.py","old_str":"INR","new_str":"USD"}')
    assert "Replaced 1" in ok
    bad = tools.dispatch(ctx, "str_replace",
                         '{"path":"app/schemas.py","old_str":"request","new_str":"req"}')
    assert "REFUSED" in bad


def test_str_replace_refuses_ambiguous_and_missing(repo):
    ctx = exec_ctx(repo, "app/utils/money.py", writes=["app/utils/money.py"])
    missing = tools.dispatch(ctx, "str_replace",
                             '{"path":"app/utils/money.py","old_str":"zzz","new_str":"y"}')
    assert "old_str not found" in missing
    dup = tools.dispatch(ctx, "str_replace",
                         '{"path":"app/utils/money.py","old_str":"self","new_str":"this"}')
    assert "appears" in dup and "times" in dup


def test_delete_needs_a_declared_delete(repo):
    ctx = exec_ctx(repo, "wsgi.py", deletes=["wsgi.py"])
    assert "Deleted" in tools.dispatch(ctx, "delete_file", '{"path":"wsgi.py"}')
    assert not (repo / "wsgi.py").exists()
    bad = tools.dispatch(ctx, "delete_file", '{"path":"app/schemas.py"}')
    assert "REFUSED" in bad and (repo / "app/schemas.py").exists()


def test_protected_globs_win_over_a_declared_write(repo, monkeypatch):
    monkeypatch.setattr(config, "PROTECTED_GLOBS", ["app/services/*"])
    ctx = exec_ctx(repo, "app/services/invoice.py", writes=["app/services/invoice.py"])
    out = tools.dispatch(ctx, "write_file",
                         '{"path":"app/services/invoice.py","content":"x=1"}')
    assert "REFUSED" in out and "protected" in out


def test_writing_marks_the_folder_summary_stale(repo):
    db.save_summary(repo, "app/utils", "old summary")
    ctx = exec_ctx(repo, "app/utils/money.py", writes=["app/utils/money.py"])
    tools.dispatch(ctx, "write_file", '{"path":"app/utils/money.py","content":"x=1"}')
    assert db.stale_folders(repo) == ["app/utils"]


# ─────────────── tests tool ───────────────

def test_run_tests_green_then_red(repo):
    ctx = exec_ctx(repo, "app/utils/money.py", writes=["app/utils/money.py"])
    out = tools.dispatch(ctx, "run_tests", '{"target":"tests/test_money.py"}')
    assert out.startswith("PASSED") and ctx.last_tests is True

    tools.dispatch(ctx, "write_file",
                   '{"path":"app/utils/money.py","content":"class Money:\\n    pass\\n"}')
    out = tools.dispatch(ctx, "run_tests", '{"target":"tests/test_money.py"}')
    assert out.startswith("FAILED") and ctx.last_tests is False


# ─────────────── dispatch robustness ───────────────

def test_bad_json_and_missing_args_are_strings(plan_ctx):
    assert tools.dispatch(plan_ctx, "read_file", "{not json").startswith("ERROR: could not parse")
    assert tools.dispatch(plan_ctx, "read_file", "{}").startswith("ERROR: missing required")


def test_unknown_file_is_a_string_not_an_exception(plan_ctx):
    assert tools.dispatch(plan_ctx, "read_file", '{"path":"nope.py"}').startswith("ERROR")


def test_no_tool_ever_raises(plan_ctx):
    """Every tool, called with junk, must return a string."""
    for name in tools.PLAN_TOOLS:
        out = tools.dispatch(plan_ctx, name, '{"path":"???","folder":"???","pattern":"("}')
        assert isinstance(out, str) and out
