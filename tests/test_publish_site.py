import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.publish_site import publish as _publish


def publish(*args, **kwargs):
    return _publish(*args, forecasts=Path("no-such-forecasts.json"), **kwargs)  # not this machine's data


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class PublishTest(unittest.TestCase):
    def test_orphan_branch_pushed_and_only_changes_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            origin, repo, pages = tmp / "origin.git", tmp / "repo", tmp / "pages"
            web, predictions = tmp / "web", tmp / "predictions"
            git("init", "--bare", "-q", str(origin), cwd=tmp)
            git("clone", "-q", str(origin), str(repo), cwd=tmp)
            for key, value in (("user.name", "Test"), ("user.email", "test@example.invalid")):
                git("config", key, value, cwd=repo)
            (repo / "README.md").write_text("code")
            git("add", "README.md", cwd=repo)
            git("commit", "-q", "-m", "code", cwd=repo)
            git("push", "-q", "origin", "HEAD:main", cwd=repo)
            web.mkdir(), predictions.mkdir()
            (web / "index.html").write_text("<html></html>")
            (predictions / "2026-10-01.json").write_text(json.dumps({"date": "2026-10-01"}))

            self.assertEqual(publish(repo, pages, predictions, web), "2026-10-01")
            files = git("ls-tree", "-r", "--name-only", "gh-pages", cwd=origin).splitlines()
            self.assertEqual(sorted(files), [".nojekyll", "data/2026-10-01.json", "data/latest.json", "index.html"])
            self.assertNotIn("README.md", files)  # nothing from main
            self.assertNotEqual(subprocess.run(["git", "merge-base", "main", "gh-pages"], cwd=origin,
                                               capture_output=True).returncode, 0)  # no shared history

            self.assertIsNone(publish(repo, pages, predictions, web))  # nothing changed, nothing committed
            (predictions / "2026-10-02.json").write_text(json.dumps({"date": "2026-10-02"}))
            self.assertEqual(publish(repo, pages, predictions, web), "2026-10-02")
            self.assertEqual(git("log", "--format=%s", "gh-pages", cwd=origin).splitlines(),
                             ["Prophecy for 2026-10-02", "Prophecy for 2026-10-01", "Start the gh-pages branch"])

            # A commit made on GitHub, as setting a custom domain does, is kept.
            other = tmp / "other"
            git("clone", "-q", "--branch", "gh-pages", str(origin), str(other), cwd=tmp)
            for key, value in (("user.name", "Test"), ("user.email", "test@example.invalid")):
                git("config", key, value, cwd=other)
            (other / "CNAME").write_text("example.invalid")
            git("add", "CNAME", cwd=other)
            git("commit", "-q", "-m", "Create CNAME", cwd=other)
            git("push", "-q", "origin", "gh-pages", cwd=other)
            (predictions / "2026-10-03.json").write_text(json.dumps({"date": "2026-10-03"}))
            self.assertEqual(publish(repo, pages, predictions, web), "2026-10-03")
            self.assertEqual(git("log", "--format=%s", "-3", "gh-pages", cwd=origin).splitlines(),
                             ["Prophecy for 2026-10-03", "Create CNAME", "Prophecy for 2026-10-02"])
            self.assertEqual(git("show", "gh-pages:CNAME", cwd=origin), "example.invalid")

            # A commit whose push failed goes out on the next run, even with nothing new.
            (predictions / "2026-10-04.json").write_text(json.dumps({"date": "2026-10-04"}))
            self.assertEqual(publish(repo, pages, predictions, web, push=False), "2026-10-04")
            self.assertEqual(publish(repo, pages, predictions, web), "2026-10-04")
            self.assertEqual(git("log", "--format=%s", "-1", "gh-pages", cwd=origin), "Prophecy for 2026-10-04")
            self.assertIsNone(publish(repo, pages, predictions, web))


if __name__ == "__main__":
    unittest.main()
