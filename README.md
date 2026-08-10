# Foreman — a long-running coding agent

Give it a codebase and a goal in plain English. It figures out **which files matter**,
changes them one at a time, runs the tests, and repairs what broke.

```bash
python -m foreman.run \
  "migrate this service from Flask to FastAPI. Keep every endpoint's behaviour and status codes identical." \
  --repo fixtures/miniledger
```

Nobody tells it which files to touch. On the 15-file fixture it typically changes 7,
leaves the framework-agnostic code byte-identical, and finishes with the suite green for
about **$0.50** in ~7 minutes.

Ctrl-C at any point and nothing is lost — `--resume <run_id>` continues from where it
stopped.

---

## Why this exists

Most agent demos are one prompt, one answer. This one is about the problems that only
show up when an agent runs for **minutes across many files**:

| Problem | What this repo does about it |
|---|---|
| the context fills up as work proceeds | a three-tier repo map; each file gets a **fresh** conversation |
| you can't put a whole codebase in a prompt | INDEX builds ~1.4k chars of map, constant regardless of repo size |
| nobody said which files to change | PLAN walks the real import graph (`ast`, no LLM) to find them |
| a failed attempt poisons the retry | shadow git snapshot per unit, rolled back before each retry |
| "I'm done" is not evidence | a deterministic accept gate per file, then the whole test suite |
| the process dies at minute 4 | an append-only event ledger; state is replayed, never stored |

---

## Install

Python 3.11+.

```bash
git clone https://github.com/vv176/long-running-agent.git
cd long-running-agent

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put your key in it
```

`.env`:

```
OPENAI_API_KEY=sk-...
```

Check the harness before spending anything:

```bash
python -m pytest tests/ -q         # 116 tests, no API calls, no network
```

Also needs `git` on your PATH — checkpointing shells out to it.

---

## Run it

The fixture is a small Flask invoicing service with 4 passing tests
(`fixtures/miniledger`). The goal below migrates it to FastAPI.

```bash
python -u -m foreman.run \
  "migrate this service from Flask to FastAPI. Keep every endpoint's behaviour and status codes identical." \
  --repo fixtures/miniledger
```

`-u` keeps the log streaming. **The fixture is never modified** — the agent works on a
copy at `fixtures/miniledger_foreman/`.

Cheaper first look (~$0.02, no changes made):

```bash
python -m foreman.run --index-only --repo fixtures/miniledger
cat fixtures/miniledger_foreman/.foreman/REPO_MAP.md
```

Run that twice and the second time prints `index is current — 0 model calls`.

### Other commands

| Command | What it does |
|---|---|
| `--list` | every run: phase, spend, units done |
| `--resume <run_id>` | continue after Ctrl-C |
| `--events <run_id>` | replay the event ledger — the full trace |
| `--events <run_id> --only tool_call` | just the agentic loop |
| `--index-only --repo <path>` | build the map, no goal needed |
| `--in-place` | work on `--repo` directly. **Dangerous** — see below |

### Point it at your own repo

```bash
python -u -m foreman.run "add request-id logging to every endpoint" --repo ../my-service
```

It works on Python repos, uses `pytest` for verification, and copies the repo first. It
was built and tested on small repos — expect PLAN to get expensive on anything large (see
Known limits).

---

## How it works

Five phases. Each announces itself to the ledger before doing any work, so a resume knows
exactly where to re-enter.

```
INDEX     build a map of the repo, so the repo itself never enters the context
PLAN      decide WHICH files change, by walking the import graph
EXECUTE   change them, one file per fresh conversation, rollback on failure
REPAIR    fix disagreements BETWEEN files — only the test suite can see those
VERIFY    run the whole suite; REPORT refuses to flatter the result
```

**INDEX** — three-tier context:

```
tier 1   .foreman/REPO_MAP.md     ALWAYS in the prompt     ~1.4k chars, constant
tier 2   read_summary(folder)     on demand                one summary per folder
tier 3   file contents            only via read_file        never preloaded
```

Tier 1 is one line *per folder*, not per file — that's what keeps it constant as a repo
grows. Indexing is idempotent: a second boot with unchanged files makes **zero** model
calls.

**PLAN** — the LLM navigates with `deps_of` / `dependents_of` over a real `ast`-derived
import graph, and records a verdict per file (`plan` or `no_change`) plus the blast radius
it intends to touch. It has **8 read-only tools and no way to edit a file** — a structural
guarantee, not an instruction.

Before a plan is accepted, a **coverage gate** checks that every dependent and same-folder
sibling of a changed file got a verdict. That catches coupling the import graph cannot
see: a pytest fixture is injected *by name*, so `tests/test_invoices.py` has no imports
and no edges, and a pure graph walk misses it.

**EXECUTE** — one file = one unit = one **fresh** conversation. Snapshot the tree first;
if the unit's deterministic gate fails, roll back and retry from identical bytes (3
attempts, then DLQ). Because nothing accumulates, unit 7 costs what unit 1 cost.

The gate is mechanical — declared writes exist, declared deletes are gone, something
actually changed, every written `.py` parses. It deliberately does **not** run the suite,
because one file can't make the suite green.

**REPAIR** — which is what this phase is for. Each file can be individually correct and
still disagree with another (one exports a factory, another imports an instance). Only the
suite sees that, and only after both are done. Up to 3 rounds.

**REPORT** — four buckets that cannot overlap: changed / left alone / needs a human /
outstanding. Any DLQ, any outstanding unit, or a red suite prints
`NOT complete — do not ship this tree`.

### State: an append-only ledger

`db.append()` is the only writer. Current state is never stored — it's replayed by
`fold()`:

```
seq 109  plan_written   app/__init__.py   {"verdict": "plan", "writes": [...]}
seq 150  unit_started   app/__init__.py   {}
seq 159  checkpointed   app/__init__.py   {"sha": "1e5b6b9e950c"}
seq 160  unit_done      app/__init__.py   {}
```

Kill the process between 150 and 160 and the replay stops at `running`, with attempts
left, so the resume re-runs that file. No lease to reclaim, no lock to release, no "was
that write committed?" — nothing was ever mutated. The crash handler is four lines.

`--events <run_id>` prints those same rows. The trace and the recovery mechanism are one
artefact.

### Checkpoints: shadow git

```bash
git --git-dir=<repo>/.foreman/shadow.git --work-tree=<repo> commit -am "..."
```

Splitting those two flags versions the tree without putting a `.git` inside it and without
touching any real history. One commit per unit:

```bash
cd fixtures/miniledger_foreman
git --git-dir=.foreman/shadow.git log --stat --oneline
```

---

## Layout

```
foreman/
  run.py         CLI + the phase machine (this is where a resume re-enters)
  config.py      every bound in the system, on one screen
  db.py          the append-only ledger + three caches (files, edges, summaries)
  graph.py       the import graph. Pure ast, no LLM, no network
  index.py       INDEX
  plan.py        PLAN — localization by graph traversal
  execute.py     EXECUTE + REPAIR
  tools.py       12 tools, scoped by phase, plus the guards
  checkpoint.py  shadow git
  llm.py         one model call, retries, measured cost
fixtures/
  miniledger/    a deliberately tricky 15-file Flask app, 4 green tests
tests/           116 tests. Fake model, so they run offline and free
scripts/         try_index.py · try_plan.py · try_resume.py (real-API experiments)
```

Suggested reading order, bottom-up: `config.py` → `graph.py` → `db.py` → `llm.py` →
`checkpoint.py` → `tools.py` → `index.py` → `plan.py` → `execute.py` → `run.py`.

If you only read four things: `config.py` in full, `db.KINDS` + `db.fold`, the module
docstring at the top of `tools.py`, and `run.drive`.

---

## The knobs

All in `foreman/config.py`:

| Constant | Default | Bounds |
|---|---|---|
| `STRONG_MODEL` | `gpt-4o` | planning + edits |
| `WEAK_MODEL` | `gpt-4o-mini` | folder summaries |
| `MAX_PLAN_HOPS` | 60 | PLAN turns |
| `MAX_REVISITS` | 2 | re-plans per file |
| `MAX_ATTEMPTS_PER_FILE` | 3 | then DLQ |
| `MAX_TOOL_CALLS_PER_UNIT` | 20 | one unit's ReAct loop |
| `MAX_REPAIR_ROUNDS` | 3 | then "needs a human" |
| `PROTECTED_GLOBS` | `[]` | opt-in paths no phase may write |

Everything that could run away has a number here, and exceeding one is always *reported*,
never hidden.

---

## Known limits

Stated plainly, because they're the interesting part.

- **PLAN is the expensive phase.** It keeps one shared conversation, so it has its own
  token snowball — ~$0.33 of a $0.50 run, ~140k tokens on a *15-file* repo. On a large
  codebase you'll hit the context limit before the hop limit. There is no compaction yet.
- **Localization varies between runs.** At temperature 0.1, two runs on the same fixture
  can produce different plans, and a wrong `no_change` verdict is not caught by the
  coverage gate — the gate catches *omission*, not a confident wrong answer. Downstream
  phases can recover from it (EXECUTE may `skip_unit`, REPAIR may touch any file PLAN
  gave a verdict to), which is the design: make each wrong answer recoverable at the next
  stage rather than pretending the model is reliable.
- **Tests are writable.** REPAIR can edit test files inside its scope — necessary for a
  framework migration, since the test client changes. Nothing verifies that assertions
  survived. A gutted test still parses. Read the diff.
- **`--in-place` is genuinely dangerous.** It edits your real files and a rollback
  overwrites them. Your git history is never touched (that's what the shadow repo is
  for), but uncommitted work in the tree is at risk. Safe on a clean, committed tree.
- **No conflict detection.** If two plans declare the same file, both units may write it
  and the last one wins.
- **Python + pytest only.** The import graph is `ast`-based and verification is a pytest
  subprocess.
- **`MAX_CONTEXT_TOKENS` is declared but not enforced.** Context stays flat because each
  unit gets a fresh conversation, so nothing needs a budget yet.

---

## Prior art

The design borrows deliberately, mostly from *An Empirical Study of Coding Agents*
(Rombaut et al.) and the agents it surveys:

- **event ledger** — OpenHands' `EventStream` (§4.3.1), specifically *not* a destructive
  two-list design
- **shadow git checkpoints** — Cline's `CheckpointTracker` (§4.2.5): rollback without
  touching real history and without Docker
- **per-phase tool binding** — Prometheus (§4.2.1): a phase can only hold the tools its
  job needs
- **loop composition** — §5.4: plan-execute over per-unit ReAct over generate-test-repair.
  11 of 13 surveyed agents layer primitives this way
- **`str_replace` as the edit primitive** — 5 of 13 agents converged on it independently

---

## License

MIT.
