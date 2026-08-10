"""
Shadow git. A per-unit snapshot of the work tree, in a repo that lives OUTSIDE it.

    git --git-dir=<sidecar>/shadow.git --work-tree=<repo> ...

Borrowed from Cline's `CheckpointTracker` (Rombaut §4.2.5): "diff-based rollback
without touching the user's real git history, and without requiring Docker."

Why it matters here: a failed unit is rolled back before its retry, so attempt 2
starts from exactly the bytes attempt 1 started from. That is what makes a retry
idempotent without keeping a second pristine copy of the tree.
"""
import subprocess
from pathlib import Path

from . import config


def _git(root: Path, *args, check=True):
    """Run git against the SHADOW repo but the REAL work tree.

    The two flags are the whole trick: --git-dir points at .foreman/shadow.git while
    --work-tree points at the repo, so we version the tree without putting a .git
    inside it and without touching any real history.
    """
    gitdir = root / config.SIDECAR / "shadow.git"
    cmd = ["git", f"--git-dir={gitdir}", f"--work-tree={root}", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:200]}")
    return r.stdout.strip()

# git --git-dir=<repo>/.foreman/shadow.git  --work-tree=<repo>  commit -am "..."
#        └─ store history HERE                └─ track THESE files
# we are creating shadow git instead of using the real .git because we want to track the files in the repo without touching the real .git
# Helpful in case --in-place is used because it would otherwise touch the real .git of the repo
def init(root: Path) -> None:
    """Create the shadow repo and take a baseline commit. Idempotent."""
    gitdir = root / config.SIDECAR / "shadow.git"
    if not gitdir.exists():
        gitdir.parent.mkdir(parents=True, exist_ok=True)
        # `git init` refuses --work-tree, so create the bare repo directly and then
        # clear core.bare — a bare repo will not accept work-tree operations.
        r = subprocess.run(["git", "init", "--quiet", "--bare", str(gitdir)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git init: {r.stderr.strip()[:200]}")
        _git(root, "config", "core.bare", "false") # because we are not using a bare repo
        # The real .git (if any) must never be captured or disturbed.
        info = gitdir / "info" # info directory is used to store the configuration for the git repository for example the exclude file
        info.mkdir(exist_ok=True) # create the info directory if it doesn't exist
        (info / "exclude").write_text(f".git/\n{config.SIDECAR}/\n__pycache__/\n*.pyc\n") # write the exclude file
        _git(root, "config", "user.email", "foreman@local") # set the user email
        _git(root, "config", "user.name", "Foreman") # set the user name, it is compulsory to set the user name and email for the git repository, else git will not commit
        snapshot(root, "baseline") # create a baseline commit, this is the first commit in the shadow git repository and it practically points to no changes


def snapshot(root: Path, message: str) -> str:
    """Commit the current tree state; returns a short sha.

    --allow-empty matters: a unit may legitimately change nothing yet still need a
    checkpoint boundary, and git would otherwise refuse the commit.
    """
    _git(root, "add", "-A") # add all the files to the staging area
    _git(root, "commit", "--quiet", "--allow-empty", "-m", message) # commit the changes to the shadow git repository
    return _git(root, "rev-parse", "HEAD")[:12] # return the short sha of the commit


def rollback(root: Path, sha: str) -> None:
    """Restore the work tree to a snapshot. Used before a retry."""
    _git(root, "checkout", "-f", sha, "--", ".") # checkout the commit with the given sha


def head(root: Path) -> str:
    """Short sha of the latest snapshot. Captured before a unit so it can roll back."""
    return _git(root, "rev-parse", "HEAD")[:12]


def diff_stat(root: Path, a: str, b: str = "HEAD") -> str:
    """`git diff --stat` between two snapshots — what a unit actually changed."""
    return _git(root, "diff", "--stat", a, b, check=False)
