"""Control panel. Constants only."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

STRONG_MODEL = "gpt-4o"        # planning + edits
WEAK_MODEL = "gpt-4o-mini"     # summaries, one-liners, commit messages
TEMPERATURE = 0.1

# NOT ENFORCED YET — declared here on purpose. Foreman keeps context flat by giving
# every unit a fresh conversation, so nothing currently needs a token budget. Class 2
# (compaction) is where this constant starts doing work.
MAX_CONTEXT_TOKENS = 16_000

MAX_ATTEMPTS_PER_FILE = 3
MAX_TOOL_CALLS_PER_UNIT = 20
MAX_PLAN_HOPS = 60             # hard bound on the PLAN traversal
MAX_REVISITS = 2               # per file, during PLAN
MAX_REPAIR_ROUNDS = 3          # generate-test-repair rounds after VERIFY goes red

READ_MAX_CHARS = 20_000
SUMMARY_HEAD_LINES = 60        # lines of each file shown to the summariser

SIDECAR = ".foreman"           # where REPO_MAP.md and folder summaries live
IGNORE_DIRS = {"__pycache__", ".venv", "venv", ".git", ".pytest_cache", "node_modules", SIDECAR}

# Empty by default, opt-in per repo: paths no phase may ever write (vendored code,
# generated files, a licence header). Enforced in tools._check_write.
PROTECTED_GLOBS: list[str] = []


def require() -> None:
    """Fail loudly and early if the API key is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Copy .env.example to .env and fill it in.")
