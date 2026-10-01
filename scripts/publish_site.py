"""Publish the site: commit the page and prophecy to `gh-pages`, and push.

GitHub Pages serves the orphan `gh-pages` branch, which only this script
writes, so `main` never gets daily commits. The branch is checked out as a
git worktree in PAGES_DIR, inside the ignored `data/`, sharing this repo's
objects. Each run:
1. Creates the branch and worktree if they're missing. The branch takes
   `origin`'s if it has one; otherwise it starts from an empty commit, since
   git before 2.42 can't add an orphan worktree.
2. Builds the site into the worktree (`build_site.build`): the page, the
   latest prophecy and record, and dated copies that accumulate into an
   archive.
3. Commits whatever changed as "Prophecy for D", and pushes.

Usage:
    python scripts/publish_site.py [--no-push]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_site import WEB_DIR, build
from scripts.daily_predictions import PREDICTIONS_DIR

PAGES_DIR = Path("data/processed/enwiki/gh-pages")
BRANCH = "gh-pages"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's well-known empty tree


def git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def _ok(*args: str, cwd: Path) -> bool:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True).returncode == 0


def ensure_worktree(repo: Path, pages: Path, branch: str = BRANCH) -> None:
    """Make `pages` a worktree on `branch`, creating the branch if needed."""
    if not _ok("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", cwd=repo):
        if git("ls-remote", "--heads", "origin", branch, cwd=repo):
            git("fetch", "origin", f"{branch}:{branch}", cwd=repo)
        else:
            first = git("commit-tree", EMPTY_TREE, "-m", "Start the gh-pages branch", cwd=repo)
            git("branch", branch, first, cwd=repo)
    if not (pages / ".git").exists():
        git("worktree", "prune", cwd=repo)
        git("worktree", "add", str(pages.resolve()), branch, cwd=repo)


def publish(repo: Path, pages: Path, predictions: Path = PREDICTIONS_DIR, web: Path = WEB_DIR,
            push: bool = True) -> str | None:
    """Build, commit and push the site; returns the prophecy's date, or None if nothing changed."""
    ensure_worktree(repo, pages)
    used = build(pages, predictions, web)
    (pages / ".nojekyll").touch()  # serve the files as they are
    git("add", "-A", cwd=pages)
    if not git("status", "--porcelain", cwd=pages):
        return None
    day = (used["latest.json"] or "no prophecy")[:10]
    git("commit", "-m", f"Prophecy for {day}", cwd=pages)
    if push:
        git("push", "--set-upstream", "origin", BRANCH, cwd=pages)
    return day


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args(argv)
    day = publish(Path.cwd(), PAGES_DIR, push=not args.no_push)
    print(f"published the prophecy for {day}" if day else "site unchanged; nothing to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
