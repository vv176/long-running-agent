"""
The ledger. SQLite, and `append()` is the ONLY function that writes run state.

Everything the agent learns or decides becomes an event. Current state is never
stored — it is computed by folding the event log (`fold()`). That is the whole
trick behind resume: there is no mutable state to leave half-written, so Ctrl+C
needs no cleanup and no repair pass.

Borrowed from OpenHands' `EventStream` (Rombaut §4.3.1), and deliberately NOT
Aider's two-list design, where "summarization overwrites done_messages".

Three tables are caches, not state, because they are derived from the filesystem
and cheap to rebuild: `files` (path + sha), `edges` (the import graph), and
`summaries` (folder descriptions). They exist so a second boot can skip INDEX.
"""
import json
import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    goal       TEXT NOT NULL,
    repo_root  TEXT NOT NULL,
    started_at REAL NOT NULL
);

-- Append-only. The source of truth. Never UPDATEd, never DELETEd.
CREATE TABLE IF NOT EXISTS events (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  TEXT NOT NULL,
    kind    TEXT NOT NULL,
    path    TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    ts      REAL NOT NULL
);

-- Caches below: derived from the repo, safe to rebuild, keyed by repo not run.
CREATE TABLE IF NOT EXISTS files (
    repo_root TEXT NOT NULL,
    path      TEXT NOT NULL,
    sha256    TEXT NOT NULL,
    loc       INTEGER NOT NULL,
    PRIMARY KEY (repo_root, path)
);

CREATE TABLE IF NOT EXISTS edges (
    repo_root TEXT NOT NULL,
    src       TEXT NOT NULL,
    dst       TEXT NOT NULL,
    PRIMARY KEY (repo_root, src, dst)
);

CREATE TABLE IF NOT EXISTS summaries (
    repo_root TEXT NOT NULL,
    folder    TEXT NOT NULL,          -- "" = the repo root summary
    body      TEXT NOT NULL,
    stale     INTEGER NOT NULL DEFAULT 0,
    ts        REAL NOT NULL,
    PRIMARY KEY (repo_root, folder)
);

CREATE INDEX IF NOT EXISTS idx_events_run ON events (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges (repo_root, dst);
"""

# Event kinds. Anything not listed is a bug, so fold() raises on unknowns.
KINDS = {
    "run_started", "phase_entered",
    "plan_written",        # payload: {plan, verdict, writes, deletes}
    "unit_started", "unit_done", "unit_failed", "unit_dlq",
    "unit_skipped",        # payload: {reason} — EXECUTE found the plan was wrong
    "checkpointed",        # payload: {sha}
    "repair_round",        # payload: {round, tests_passed}
    "spend",               # payload: {cents, tokens_in, tokens_out}
    "tool_call",           # payload: {phase, tool, args, result}  TRACE ONLY —
                           # fold() ignores it, so it can never affect resume. It
                           # exists so `--events` can show the agentic loop.
    "question_asked", "question_answered",
    "run_finished",
}

DB_PATH = Path(__file__).resolve().parent.parent / "foreman.sqlite"


def connect() -> sqlite3.Connection:
    """A connection with row access by name and WAL enabled.

    WAL matters: a reader (a dashboard, another terminal running --events) sees a
    consistent snapshot while the agent is mid-write.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # a dashboard can read while we write
    return conn


def init() -> None:
    """Create every table if absent. Safe to call on every start."""
    with connect() as c:
        c.executescript(SCHEMA)


# ─────────────────────────── runs ───────────────────────────

def create_run(goal: str, repo_root: Path) -> str:
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    with connect() as c:
        c.execute("INSERT INTO runs (run_id, goal, repo_root, started_at) VALUES (?,?,?,?)",
                  (run_id, goal, str(repo_root), time.time()))
    append(run_id, "run_started", payload={"goal": goal, "repo_root": str(repo_root)})
    return run_id


def get_run(run_id: str):
    """The runs row (goal, repo_root, started_at), or None."""
    with connect() as c:
        return c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()


def list_runs():
    """Every run, newest first. Backs `--list`."""
    with connect() as c:
        return c.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()


# ─────────────────────────── the only writer ───────────────────────────

def append(run_id: str, kind: str, path: str | None = None, payload: dict | None = None) -> int:
    """Append one event. Returns its seq. The only function that writes run state."""
    if kind not in KINDS:
        raise ValueError(f"unknown event kind {kind!r}; add it to db.KINDS deliberately")
    with connect() as c:
        cur = c.execute(
            "INSERT INTO events (run_id, kind, path, payload, ts) VALUES (?,?,?,?,?)",
            (run_id, kind, path, json.dumps(payload or {}), time.time()))
        return cur.lastrowid


def events(run_id: str, after: int = 0):
    """Events in seq order. `after` lets a follower poll for only what is new."""
    with connect() as c:
        return c.execute("SELECT * FROM events WHERE run_id = ? AND seq > ? ORDER BY seq",
                         (run_id, after)).fetchall()


def last_plan_order(run_id: str) -> list[str]:
    """The most recent EXECUTE order recorded for this run, or [].

    NOT `last_payload("phase_entered")`: several phases record a `phase_entered`
    event and only the one `drive()` writes carries an `order`. Taking the latest
    payload therefore returned {} whenever anything appended after it, and a resumed
    run silently fell back to alphabetical order — throwing away the sequence PLAN
    reasoned about. So scan backwards for the most recent payload that actually has
    a non-empty order.
    """
    with connect() as c:
        rows = c.execute("SELECT payload FROM events WHERE run_id = ?"
                         " AND kind = 'phase_entered' ORDER BY seq DESC", (run_id,))
        for r in rows:
            order = json.loads(r["payload"] or "{}").get("order")
            if order:
                return order
    return []


def last_payload(run_id: str, kind: str) -> dict:
    """Most recent payload for an event kind. Used to recover the plan order on resume."""
    with connect() as c:
        r = c.execute("SELECT payload FROM events WHERE run_id = ? AND kind = ?"
                      " ORDER BY seq DESC LIMIT 1", (run_id, kind)).fetchone()
    return json.loads(r["payload"]) if r else {}


# ─────────────────────────── the fold ───────────────────────────

class FileState:
    """Per-file view, derived. `status` drives what EXECUTE picks up next."""
    __slots__ = ("path", "plan", "verdict", "version", "status", "attempts", "sha", "error",
                 "writes", "deletes", "note")

    def __init__(self, path):
        self.path = path
        self.plan = None
        self.verdict = None      # "plan" | "no_change"
        self.version = 0
        # The blast radius the plan declared. EXECUTE refuses anything outside it.
        self.writes: list[str] = []
        self.deletes: list[str] = []
        # unplanned -> planned -> running -> done | dlq | skipped
        # (no "failed" state: a failure writes "planned", which IS the retry)
        self.status = "unplanned"
        self.attempts = 0
        self.sha = None
        self.error = None
        self.note = None         # why a unit was skipped, if it was

    def __repr__(self):
        return f"<{self.path} {self.status} v{self.version}>"


class RunState:
    def __init__(self):
        self.phase = "INIT"
        self.files: dict[str, FileState] = {}
        self.cents = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.questions: list[dict] = []
        self.finished = False
        self.last_seq = 0

    def _f(self, path) -> FileState:
        """Get-or-create the view for one file. `setdefault` means fold() never has
        to check whether it has seen this path before."""
        return self.files.setdefault(path, FileState(path))

    def planned(self) -> list[FileState]:
        """Files that actually need work, in stable order."""
        return sorted((f for f in self.files.values() if f.verdict == "plan"),
                      key=lambda f: f.path)


def fold(run_id: str) -> RunState:
    """
    Replay the event log into current state.

    This is the entire resume mechanism: `--resume` calls fold() and continues.
    No reconciliation, no lease reclaim, no "was that write committed?" — because
    nothing was ever mutated in the first place.
    """
    st = RunState()
    for e in events(run_id):
        st.last_seq = e["seq"]
        kind, path = e["kind"], e["path"]
        p = json.loads(e["payload"] or "{}")

        if kind == "phase_entered":
            st.phase = p["phase"]
        elif kind == "plan_written":
            f = st._f(path)
            f.plan, f.verdict = p.get("plan"), p.get("verdict")
            f.writes = p.get("writes", [])
            f.deletes = p.get("deletes", [])
            f.version += 1
            f.status = "planned" if f.verdict == "plan" else "no_change"
        elif kind == "unit_started":
            f = st._f(path)
            f.status, f.attempts = "running", f.attempts + 1
        elif kind == "unit_done":
            st._f(path).status = "done"
        elif kind == "unit_failed":
            f = st._f(path)
            f.status, f.error = "planned", p.get("error")     # back in the queue
        elif kind == "unit_skipped":
            # EXECUTE looked and found the plan was wrong: this file needs nothing.
            # Terminal, like done — never retried, never DLQ'd.
            f = st._f(path)
            f.status, f.note = "skipped", p.get("reason")
        elif kind == "unit_dlq":
            f = st._f(path)
            f.status, f.error = "dlq", p.get("error")
        elif kind == "checkpointed":
            # REPAIR checkpoints with no path; without this guard a phantom
            # FileState(None) appears in st.files and inflates every count.
            if path:
                st._f(path).sha = p.get("sha")
        elif kind == "spend":
            st.cents += p.get("cents", 0.0)
            st.tokens_in += p.get("tokens_in", 0)
            st.tokens_out += p.get("tokens_out", 0)
        elif kind == "question_asked":
            st.questions.append({"path": path, **p, "answer": None})
        elif kind == "question_answered":
            for q in st.questions:
                if q["path"] == path and q["answer"] is None:
                    q["answer"] = p.get("answer")
                    break
        elif kind == "run_finished":
            st.finished = True
        elif kind == "run_started":
            pass
    return st


def next_ready(state: RunState, max_attempts: int) -> FileState | None:
    """
    The next file to execute: planned, still has attempts left, lowest path.

    Dependency order is NOT enforced here — EXECUTE takes it from the order
    `finish_planning` declared, because the plan knows more about ordering than the
    raw import graph does (it knows which contract changes first).
    """
    for f in state.planned():
        if f.status == "planned" and f.attempts < max_attempts:
            return f
    return None


# ─────────────────────────── caches (files, edges, summaries) ───────────────────────────

def save_files(repo_root: Path, rows: list[tuple[str, str, int]]) -> None:
    """Upsert (path, sha256, loc) for a repo. The sha is what makes INDEX idempotent."""
    with connect() as c:
        c.executemany(
            "INSERT INTO files (repo_root, path, sha256, loc) VALUES (?,?,?,?)"
            " ON CONFLICT(repo_root, path) DO UPDATE SET sha256=excluded.sha256, loc=excluded.loc",
            [(str(repo_root), *r) for r in rows])


def get_files(repo_root: Path) -> dict[str, str]:
    """path -> sha256 as last indexed. Compare against disk to find what changed."""
    with connect() as c:
        return {r["path"]: r["sha256"] for r in c.execute(
            "SELECT path, sha256 FROM files WHERE repo_root = ?", (str(repo_root),))}


def save_edges(repo_root: Path, pairs: list[tuple[str, str]]) -> None:
    """Replace the whole edge set for a repo.

    DELETE-then-insert, not upsert: an edge that disappeared because an import was
    removed must actually go away, and there is no cheap way to diff for that.
    """
    with connect() as c:
        c.execute("DELETE FROM edges WHERE repo_root = ?", (str(repo_root),))
        c.executemany("INSERT OR IGNORE INTO edges (repo_root, src, dst) VALUES (?,?,?)",
                      [(str(repo_root), a, b) for a, b in pairs])


def deps_of(repo_root: Path, path: str) -> list[str]:
    """Outgoing: what `path` imports."""
    with connect() as c:
        return [r[0] for r in c.execute(
            "SELECT dst FROM edges WHERE repo_root = ? AND src = ? ORDER BY dst",
            (str(repo_root), path))]


def dependents_of(repo_root: Path, path: str) -> list[str]:
    """Incoming: who imports `path`."""
    with connect() as c:
        return [r[0] for r in c.execute(
            "SELECT src FROM edges WHERE repo_root = ? AND dst = ? ORDER BY src",
            (str(repo_root), path))]


def edge_count(repo_root: Path) -> int:
    """How many in-repo import edges are stored. Used for the repo map header."""
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM edges WHERE repo_root = ?",
                         (str(repo_root),)).fetchone()[0]


def save_summary(repo_root: Path, folder: str, body: str) -> None:
    """Store a folder summary and clear its stale flag. Folder "" is the repo root."""
    with connect() as c:
        c.execute(
            "INSERT INTO summaries (repo_root, folder, body, stale, ts) VALUES (?,?,?,0,?)"
            " ON CONFLICT(repo_root, folder) DO UPDATE SET body=excluded.body, stale=0,"
            " ts=excluded.ts", (str(repo_root), folder, body, time.time()))


def get_summary(repo_root: Path, folder: str) -> str | None:
    """One folder's summary body, or None. This is what read_summary serves."""
    with connect() as c:
        r = c.execute("SELECT body FROM summaries WHERE repo_root = ? AND folder = ?",
                      (str(repo_root), folder)).fetchone()
        return r["body"] if r else None


def all_summaries(repo_root: Path) -> dict[str, str]:
    """folder -> body for every summarised folder."""
    with connect() as c:
        return {r["folder"]: r["body"] for r in c.execute(
            "SELECT folder, body FROM summaries WHERE repo_root = ? ORDER BY folder",
            (str(repo_root),))}


def mark_stale(repo_root: Path, folder: str) -> None:
    """A file changed -> its folder's summary is out of date. Refresh in a batch later."""
    with connect() as c:
        c.execute("UPDATE summaries SET stale = 1 WHERE repo_root = ? AND folder = ?",
                  (str(repo_root), folder))


def stale_folders(repo_root: Path) -> list[str]:
    """Folders whose summary is out of date because a file in them was written."""
    with connect() as c:
        return [r[0] for r in c.execute(
            "SELECT folder FROM summaries WHERE repo_root = ? AND stale = 1 ORDER BY folder",
            (str(repo_root),))]
