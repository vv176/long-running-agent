"""
Step 4 test: INDEX. Uses a FAKE model so the assertions are deterministic and free.
The real-model run is exercised separately by `scripts/try_index.py`.

Run: python -m pytest tests/test_index.py -q
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foreman import config, db, index, llm  # noqa: E402


class FakeFn:
    def __init__(self, args): self.name, self.arguments = "x", json.dumps(args)


class FakeCall:
    def __init__(self, args): self.function = FakeFn(args)


class FakeMsg:
    def __init__(self, args): self.tool_calls = [FakeCall(args)]


@pytest.fixture
def repo(tmp_path):
    """A throwaway copy of miniledger, so tests can edit files."""
    dst = tmp_path / "miniledger"
    shutil.copytree(ROOT / "fixtures" / "miniledger", dst)
    return dst


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    """Counts calls and returns shape-correct canned answers."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.sqlite")
    db.init()
    calls = {"folder": 0, "repo": 0}

    def fake(messages, tools=None, tool_choice=None, model=None, max_retries=4):
        prompt = messages[0]["content"]
        if prompt.startswith("A Python repository has these folders"):
            calls["repo"] += 1
            return FakeMsg({"purpose": "A tiny invoicing service.", "stack": "Flask",
                            "entry_points": ["wsgi.py"],
                            "conventions": ["validation lives in schemas.py"]}), 10, 5, 0.01
        calls["folder"] += 1
        # echo back the file names the prompt showed us
        names = [ln[4:-4] for ln in prompt.splitlines() if ln.startswith("--- ")]
        return FakeMsg({"purpose": "Does a thing.", "one_line": "does a thing",
                        "files": [{"name": n, "does": "does x"} for n in names]}), 10, 5, 0.01

    monkeypatch.setattr(llm, "call", fake)
    monkeypatch.setattr(index.llm, "call", fake)
    return calls


def test_scan_finds_files_and_folders(repo):
    files, folders = index.scan(repo)
    assert len(files) == 15
    assert set(folders) == {"", "app", "app/routes", "app/services",
                            "app/models", "app/utils", "tests"}
    assert folders[""] == ["wsgi.py"]


def test_folder_of():
    assert index.folder_of("app/routes/x.py") == "app/routes"
    assert index.folder_of("wsgi.py") == ""


def test_first_build_summarises_every_folder(repo, fake_llm, capsys):
    out = index.build(db.create_run("g", repo), repo, log=lambda *_: None)
    assert out == {"files": 15, "edges": 11, "folders": 7, "summarised": 7, "skipped": 0}
    assert fake_llm == {"folder": 7, "repo": 1}


def test_second_build_is_free(repo, fake_llm):
    rid = db.create_run("g", repo)
    index.build(rid, repo, log=lambda *_: None)
    fake_llm["folder"] = fake_llm["repo"] = 0

    out = index.build(rid, repo, log=lambda *_: None)
    assert out["summarised"] == 0 and out["skipped"] == 7
    assert fake_llm == {"folder": 0, "repo": 0}, "a no-change reboot must cost nothing"


def test_touching_one_file_reindexes_only_its_folder(repo, fake_llm):
    rid = db.create_run("g", repo)
    index.build(rid, repo, log=lambda *_: None)
    fake_llm["folder"] = fake_llm["repo"] = 0

    (repo / "app" / "utils" / "money.py").write_text("# changed\nX = 1\n")
    out = index.build(rid, repo, log=lambda *_: None)
    assert out["summarised"] == 1, "only app/utils changed"
    assert fake_llm["folder"] == 1


def test_adding_a_file_reindexes_its_folder(repo, fake_llm):
    rid = db.create_run("g", repo)
    index.build(rid, repo, log=lambda *_: None)
    fake_llm["folder"] = 0

    (repo / "app" / "routes" / "payments.py").write_text("from flask import Blueprint\n")
    out = index.build(rid, repo, log=lambda *_: None)
    assert out["files"] == 16 and out["summarised"] == 1
    assert fake_llm["folder"] == 1


def test_deleting_a_file_reindexes_its_folder(repo, fake_llm):
    rid = db.create_run("g", repo)
    index.build(rid, repo, log=lambda *_: None)
    fake_llm["folder"] = 0

    (repo / "app" / "routes" / "health.py").unlink()
    out = index.build(rid, repo, log=lambda *_: None)
    assert out["files"] == 14 and out["summarised"] == 1
    assert fake_llm["folder"] == 1


def test_edges_land_in_the_db_both_ways(repo, fake_llm):
    index.build(db.create_run("g", repo), repo, log=lambda *_: None)
    assert db.deps_of(repo, "app/services/invoice.py") == [
        "app/models/invoice.py", "app/utils/money.py"]
    assert db.dependents_of(repo, "app/utils/money.py") == [
        "app/models/invoice.py", "app/schemas.py",
        "app/services/invoice.py", "tests/test_money.py"]


def test_repo_map_is_written_and_bounded(repo, fake_llm):
    index.build(db.create_run("g", repo), repo, log=lambda *_: None)
    p = repo / config.SIDECAR / "REPO_MAP.md"
    assert p.exists()
    text = p.read_text()
    assert "## Layout" in text and "## Conventions" in text
    assert "app/routes/" in text
    # the always-in-context block must stay small
    assert len(text) < 4000, f"repo map is {len(text)} chars — too big for every prompt"
    assert index.repo_map(repo) == text


def test_folder_summaries_are_on_disk_and_fetchable(repo, fake_llm):
    index.build(db.create_run("g", repo), repo, log=lambda *_: None)
    assert (repo / config.SIDECAR / "folders" / "app__routes.md").exists()
    assert (repo / config.SIDECAR / "folders" / "_root.md").exists()
    body = db.get_summary(repo, "app/routes")
    assert "invoices.py" in body


def test_sidecar_is_not_indexed_as_source(repo, fake_llm):
    """.foreman/ must never appear in files or folders on a re-run."""
    rid = db.create_run("g", repo)
    index.build(rid, repo, log=lambda *_: None)
    files, folders = index.scan(repo)
    assert not any(config.SIDECAR in f for f in files)
    assert config.SIDECAR not in folders


def test_unparsable_file_is_reported(repo, fake_llm):
    (repo / "app" / "broken.py").write_text("def f(:\n")
    logged = []
    index.build(db.create_run("g", repo), repo, log=logged.append)
    assert any("failed to parse" in line for line in logged)


def test_spend_is_recorded_per_call(repo, fake_llm):
    rid = db.create_run("g", repo)
    index.build(rid, repo, log=lambda *_: None)
    st = db.fold(rid)
    assert st.cents == pytest.approx(0.08)     # 7 folders + 1 repo, 0.01 each
    assert st.phase == "INDEX"


def test_cache_hit_still_writes_the_artefacts(repo, fake_llm):
    """
    The bug this pins: build() returned early on a cache hit WITHOUT writing
    REPO_MAP.md. The DB cache is keyed by repo and survives across runs, but the
    artefacts live inside the work copy, which main() re-creates every run. So a
    second run left PLAN reading "(no repo map — run INDEX first)" and it planned
    badly. Zero model calls must still leave the artefacts on disk.
    """
    index.build(db.create_run("g", repo), repo, log=lambda *_: None)
    assert (repo / config.SIDECAR / "REPO_MAP.md").exists()
    first = dict(fake_llm)
    assert first["folder"] > 0 and first["repo"] == 1

    # simulate main(): the work copy is recreated, so the sidecar is gone
    shutil.rmtree(repo / config.SIDECAR)

    index.build(db.create_run("g", repo), repo, log=lambda *_: None)
    assert fake_llm == first, "a cache hit must make zero model calls"
    body = (repo / config.SIDECAR / "REPO_MAP.md").read_text()
    assert "Repository map" in body and "Entry points" in body
    assert index.repo_map(repo) == body
    assert (repo / config.SIDECAR / "folders" / "app__routes.md").exists()
