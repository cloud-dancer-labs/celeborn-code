#!/usr/bin/env python3
"""Test suite for the Celeborn CLI.

Stdlib `unittest` only — no third-party deps, matching the project's "boring technology"
constraint. Run with:

    python -m unittest discover -s tests        # from the repo root
    python tests/test_celeborn.py                # direct

The suite has three layers:
  1. Unit tests for the pure parsing helpers (the parts most likely to regress silently).
  2. End-to-end command tests that drive the real argparse entrypoint `main([...])`.
  3. The PLAN.md §10 success criteria, encoded as executable acceptance tests.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

# Import the single-file CLI from ../scripts/celeborn.py without installing it.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import celeborn as cb  # noqa: E402
import celeborn_sync as cs  # noqa: E402


# --------------------------------------------------------------------------- helpers

class Run:
    """Result of invoking the CLI: captured stdout/stderr and exit behaviour."""

    def __init__(self, out: str, err: str, exit_code: int | None):
        self.out = out
        self.err = err
        self.exit_code = exit_code  # None unless the command called sys.exit / die

    @property
    def all(self) -> str:
        return self.out + self.err


def run_cli(*argv: str) -> Run:
    """Invoke `celeborn <argv>` through the real argparse entrypoint, capturing output.

    `die()` raises SystemExit; we catch it and expose the code rather than letting it
    abort the test process. argparse usage errors (SystemExit too) are captured the same way.
    """
    out, err = io.StringIO(), io.StringIO()
    code: int | None = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cb.main(list(argv))
        except SystemExit as e:  # die() / argparse / doctor failures
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return Run(out.getvalue(), err.getvalue(), code)


class CelebornTestCase(unittest.TestCase):
    """Base case: a fresh temp project with `.context/` already scaffolded."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ctx = self.root / ".context"
        # Isolate machine-global state (fleet registry + credentials live under XDG_CONFIG_HOME) so a
        # hook's orient self-register (CELE-t124) — and any other fleet/creds write — lands in this
        # test's sandbox, never the developer's real ~/.config/celeborn. Without this, every test that
        # drives session-start would register its temp project into the real fleet registry.
        self._old_xdg = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root / "_xdg")
        # Every test starts from a real `celeborn init` against the shipped templates.
        r = self.init()
        self.assertIsNone(r.exit_code, f"init failed: {r.all}")
        # Unit tests must never boot the real Next.js board: REPO_ROOT/board has live node_modules,
        # so a hook's ensure-on-orient / per-turn re-ensure (CELE-t99) would actually spawn a dev
        # server from a temp project. Disable autostart in the shared fixture; the autostart decision
        # tree is covered directly in TestEnsureOnOrient (which stubs the real process launch).
        rc_path = self.ctx / ".celebornrc"
        rc = json.loads(rc_path.read_text())
        rc["board_autostart"] = False
        rc_path.write_text(json.dumps(rc, indent=2) + "\n")
        # CELE-t153 — belt-and-suspenders: no test may spawn a REAL detached board supervisor.
        # The suite's anti-spawn protection otherwise rests entirely on every board-touching test
        # remembering to stub _spawn_board; a single forgotten stub launches a `next dev` that
        # outlives the run (the zombie supervisors that motivated this card). Default _spawn_board to
        # a hard failure so an accidental real launch is impossible — a test that reaches it fails
        # loudly instead of silently leaking a process. Tests that exercise the launch path opt in:
        # they assign their own capturing stub (TestEnsureOnOrient.setUp / _board_stubs) or restore
        # self._real_spawn_board (the genuine impl, with subprocess.Popen faked).
        self._real_spawn_board = cb._spawn_board
        self.addCleanup(setattr, cb, "_spawn_board", self._real_spawn_board)
        def _no_real_board_spawn(*a, **k):
            raise AssertionError(
                "test reached the REAL _spawn_board — it would boot a detached `next dev` that "
                "outlives the suite. Stub it (see TestEnsureOnOrient.setUp / _board_stubs), or "
                "restore self._real_spawn_board if you mean to exercise the launch path.")
        cb._spawn_board = _no_real_board_spawn

    def tearDown(self):
        if self._old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._old_xdg
        self._tmp.cleanup()

    # thin wrappers that always target this test's temp project via --path
    def init(self) -> Run:
        # --no-scan keeps the shared fixture fast + deterministic (no per-test git shell-out);
        # smart-init's repo-reading behaviour is covered directly in TestSmartInit. --no-cmm keeps
        # the fixture free of CMM auto-engage side-effects (settings.json/.mcp.json/North Star);
        # the auto-engage-on-init behaviour is covered directly in TestCmmInitAutoEngage.
        return run_cli("--path", str(self.root), "init", "--no-scan", "--no-cmm")

    def cli(self, *argv: str) -> Run:
        return run_cli("--path", str(self.root), *argv)

    def write(self, rel: str, text: str):
        p = self.ctx / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def read(self, rel: str) -> str:
        return (self.ctx / rel).read_text()


# --------------------------------------------------------------------------- 1. unit: parsing helpers

class TestParsingHelpers(unittest.TestCase):

    def test_slugify(self):
        self.assertEqual(cb.slugify("Hello, World!"), "hello-world")
        self.assertEqual(cb.slugify("  Multiple   Spaces  "), "multiple-spaces")
        self.assertEqual(cb.slugify("snake_case and-dash"), "snake-case-and-dash")
        self.assertEqual(cb.slugify("!!!???"), "")  # pure punctuation -> empty
        self.assertEqual(cb.slugify("café"), "café")  # \w is Unicode-aware; letters kept

    def test_strip_frontmatter(self):
        fm, body = cb.strip_frontmatter("---\nname: x\ntags: a b\n---\nhello\n")
        self.assertEqual(fm["name"], "x")
        self.assertEqual(fm["tags"], "a b")
        self.assertEqual(body, "hello\n")

    def test_strip_frontmatter_absent(self):
        fm, body = cb.strip_frontmatter("no frontmatter here")
        self.assertEqual(fm, {})
        self.assertEqual(body, "no frontmatter here")

    def test_parse_sections_splits_on_headings(self):
        secs = cb.parse_sections("# Title\npreamble\n## A\nbody a\n## B\nbody b\n")
        titles = [s["title"] for s in secs]
        self.assertEqual(titles, ["Title", "A", "B"])
        b = next(s for s in secs if s["title"] == "B")
        self.assertEqual(b["body"], "body b")
        self.assertEqual(b["anchor"], "b")
        self.assertEqual(b["level"], 2)

    def test_parse_sections_strips_html_comments(self):
        # Regression: HTML comments (template boilerplate) must never reach the index.
        secs = cb.parse_sections("## Real\n<!-- ## Fake heading inside a comment -->\nbody\n")
        titles = [s["title"] for s in secs]
        self.assertEqual(titles, ["Real"])
        self.assertNotIn("Fake heading", secs[0]["body"])

    def test_parse_sections_tags_and_links(self):
        secs = cb.parse_sections("## S\nsome #alpha text with [[other-note]] link #beta\n")
        s = secs[0]
        self.assertEqual(s["tags"], "alpha beta")
        self.assertEqual(s["links"], ["other-note"])

    def test_split_journal_ignores_comment_headings(self):
        text = (
            "# Journal\n<!--\n## YYYY-MM-DD template heading\n-->\n"
            "## 2026-01-01 entry one\nbody\n## 2026-01-02 entry two\nbody\n"
        )
        header, entries = cb.split_journal(text)
        # The `##` inside the comment is NOT treated as an entry — only the two real ones are.
        self.assertEqual(len(entries), 2)
        self.assertIn("entry one", entries[0])
        self.assertIn("entry two", entries[1])
        # The comment legitimately stays in the header block; parse_sections strips it before
        # anything reaches the index, so it never becomes searchable.
        self.assertNotIn("template heading", "".join(entries))

    def test_split_journal_no_entries(self):
        header, entries = cb.split_journal("# Journal\njust a header\n")
        self.assertEqual(entries, [])

    def test_est_tokens(self):
        self.assertEqual(cb._est_tokens("", 4), 0)
        self.assertEqual(cb._est_tokens("abcd", 4), 1)
        self.assertEqual(cb._est_tokens("abcde", 4), 2)  # ceil division


# --------------------------------------------------------------------------- 1b. identity / disambiguation

class TestAboutIdentity(unittest.TestCase):
    """`celeborn about` is the install-time identity check: an agent that ran `pip install celeborn`
    mid-conversation runs it to confirm it grabbed the coding-agent context substrate — not one of
    the same-named projects (Apache Celeborn; the frkngksl/Celeborn Windows tool). Guard the
    disambiguation so it can't silently rot."""

    def test_about_self_identifies_as_celeborn_code(self):
        r = run_cli("about")
        self.assertIsNone(r.exit_code, f"about errored: {r.all}")
        self.assertIn("Celeborn Code", r.out)
        self.assertIn("uv tool install celeborn", r.out)
        self.assertIn("cloud-dancer-labs/celeborn", r.out)

    def test_about_disambiguates_from_namesakes(self):
        out = run_cli("about").out
        self.assertIn("Apache Celeborn", out)
        self.assertIn("frkngksl/Celeborn", out)

    def test_top_level_help_carries_brand_and_disambiguation(self):
        help_text = run_cli("--help").all
        self.assertIn("Celeborn Code", help_text)
        self.assertIn("Apache Celeborn", help_text)


# --------------------------------------------------------------------------- 2. init

class TestInit(CelebornTestCase):

    def test_creates_required_files(self):
        for rel in cb.REQUIRED_FILES:
            self.assertTrue((self.ctx / rel).is_file(), f"missing {rel}")
        self.assertTrue((self.ctx / "session.json").is_file())
        self.assertTrue((self.ctx / "metrics.json").is_file())
        self.assertTrue((self.ctx / "journal-archive").is_dir())

    def test_session_json_gets_live_timestamp(self):
        data = json.loads(self.read("session.json"))
        self.assertIsNotNone(data["updated_at"])  # template had null

    def test_gitignores_the_index(self):
        gi = (self.root / ".gitignore").read_text()
        self.assertIn(".context/index.db", gi)

    def test_is_idempotent(self):
        # Mutate a file, re-init, and confirm the existing copy is preserved (not clobbered).
        self.write("state.md", "MY EDITS\n")
        r = self.init()
        self.assertIsNone(r.exit_code)
        self.assertEqual(self.read("state.md"), "MY EDITS\n")
        self.assertIn("exists, kept", r.all)

    def test_gitignore_not_duplicated(self):
        self.init()
        gi = (self.root / ".gitignore").read_text()
        self.assertEqual(gi.count(".context/index.db\n"), 1)

    def test_private_gitignores_whole_context(self):
        # --private keeps the working memory out of git entirely (root-anchored).
        root = Path(tempfile.mkdtemp())
        try:
            r = run_cli("--path", str(root), "init", "--private")
            self.assertIsNone(r.exit_code, r.all)
            gi = (root / ".gitignore").read_text()
            self.assertIn("/.context/", gi)
            self.assertIn("celeborn sync", r.all)  # points the user at sync
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_private_is_idempotent(self):
        root = Path(tempfile.mkdtemp())
        try:
            run_cli("--path", str(root), "init", "--private")
            run_cli("--path", str(root), "init", "--private")
            gi = (root / ".gitignore").read_text()
            self.assertEqual(gi.count("/.context/\n"), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_public_flag_tracks_context(self):
        # --public keeps the default behaviour (only the index is ignored, not all of .context/).
        root = Path(tempfile.mkdtemp())
        try:
            run_cli("--path", str(root), "init", "--public")
            gi = (root / ".gitignore").read_text()
            self.assertIn(".context/index.db", gi)
            self.assertNotIn("/.context/\n", gi)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_repo_visibility_none_without_git(self):
        # No .git -> never shells out; unknown visibility.
        self.assertIsNone(cb._repo_visibility(self.root))

    def test_annotates_claude_md(self):
        # init (run in setUp) drops a managed block in CLAUDE.md so Claude Code auto-loads the orient.
        cm = (self.root / "CLAUDE.md")
        self.assertTrue(cm.is_file())
        text = cm.read_text()
        self.assertIn(cb.CLAUDE_MD_BEGIN, text)
        self.assertIn(cb.CLAUDE_MD_END, text)
        self.assertIn(".context/state.md", text)   # the orient instruction is present
        self.assertIn("context-health notice", text)   # surface-this channel for surfaces that hide hooks
        self.assertIn("do NOT surface", text)           # heartbeat is context-only, not reprinted
        self.assertIn("Multi-agent kanban", text)
        self.assertIn("celeborn claim", text)

    def test_annotates_agents_md(self):
        am = self.root / "AGENTS.md"
        self.assertTrue(am.is_file())
        text = am.read_text()
        self.assertIn(cb.AGENTS_MD_BEGIN, text)
        self.assertIn(cb.AGENTS_MD_END, text)
        self.assertIn("Multi-agent kanban", text)

    def test_claude_md_idempotent(self):
        before = (self.root / "CLAUDE.md").read_text()
        self.init()
        after = (self.root / "CLAUDE.md").read_text()
        self.assertEqual(after, before)
        self.assertEqual(after.count(cb.CLAUDE_MD_BEGIN), 1)

    def test_claude_md_appends_preserving_existing(self):
        root = Path(tempfile.mkdtemp())
        try:
            (root / "CLAUDE.md").write_text("# My Project\n\nHand-written guidance.\n")
            run_cli("--path", str(root), "init")
            text = (root / "CLAUDE.md").read_text()
            self.assertIn("Hand-written guidance.", text)          # original preserved
            self.assertIn("maintained by Celeborn", text)          # block appended
            self.assertEqual(text.count(cb.CLAUDE_MD_BEGIN), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_no_claude_md_optout(self):
        root = Path(tempfile.mkdtemp())
        try:
            run_cli("--path", str(root), "init", "--no-claude-md")
            self.assertFalse((root / "CLAUDE.md").exists())
            self.assertTrue((root / "AGENTS.md").is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestSmartInit(unittest.TestCase):
    """celeborn init smart-scan — read the repo (README, manifest, git) to pre-seed the Hot tier."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ctx = self.root / ".context"

    def tearDown(self):
        self._tmp.cleanup()

    def _git(self, *argv):
        import subprocess
        subprocess.run(["git", "-C", str(self.root), *argv], capture_output=True, text=True, check=True)

    def _make_repo(self):
        (self.root / "README.md").write_text(
            "# Acme Widget\n\n"
            "[![build](https://img.shields.io/badge/x)](http://x)\n\n"
            "Acme Widget turns sprockets into widgets, fast and with zero config.\n\n"
            "## Install\n...\n"
        )
        (self.root / "package.json").write_text(
            '{ "name": "acme-widget", "description": "Turn sprockets into widgets." }\n')
        (self.root / "index.js").write_text("console.log(1)\n")
        (self.root / "lib.ts").write_text("export const x = 1\n")
        self._git("init", "-q")
        self._git("config", "user.email", "t@celeborn.local")
        self._git("config", "user.name", "T")
        self._git("add", "-A")
        self._git("commit", "-qm", "initial commit: scaffold widget CLI")

    def _init(self, *extra):
        return run_cli("--path", str(self.root), "init", "--no-claude-md", "--no-agents-md", *extra)

    def test_seeds_state_from_readme_and_manifest(self):
        self._make_repo()
        r = self._init()
        self.assertIsNone(r.exit_code, r.all)
        state = (self.ctx / "state.md").read_text()
        self.assertIn("acme-widget", state)                       # manifest name wins
        self.assertIn("turns sprockets into widgets", state)      # README description
        self.assertIn("Node/JS", state)                           # detected stack
        self.assertNotIn("<what we are working on", state)         # template placeholder replaced

    def test_seeds_session_focus_and_branch(self):
        self._make_repo()
        self._init()
        data = json.loads((self.ctx / "session.json").read_text())
        self.assertIn("acme-widget", data["focus"])
        self.assertTrue(data["branch"])                            # detected git branch
        self.assertIn("first task", data["next_action"])

    def test_repo_snapshot_lands_in_notes_with_commits(self):
        self._make_repo()
        self._init()
        notes = (self.ctx / "notes.md").read_text()
        self.assertIn("Repo snapshot", notes)
        self.assertIn("scaffold widget CLI", notes)                # recent commit subject
        self.assertIn("package.json", notes)                       # detected manifest

    def test_no_scan_leaves_template(self):
        self._make_repo()
        r = self._init("--no-scan")
        self.assertIsNone(r.exit_code, r.all)
        state = (self.ctx / "state.md").read_text()
        self.assertIn("<what we are working on", state)            # untouched template
        self.assertNotIn("acme-widget", state)
        self.assertNotIn("smart init", r.all)

    def test_bare_dir_degrades_gracefully(self):
        # No git, no README, no manifest — init must still succeed and fall back to the dir name.
        r = self._init()
        self.assertIsNone(r.exit_code, r.all)
        state = (self.ctx / "state.md").read_text()
        self.assertIn(self.root.name, state)                       # dir name as the project fallback
        self.assertIn("no git history yet", state)

    def test_smart_seed_never_clobbers_existing_state(self):
        self._make_repo()
        self.ctx.mkdir(parents=True)
        (self.ctx / "state.md").write_text("MY HAND-WRITTEN STATE\n")
        self._init()
        self.assertEqual((self.ctx / "state.md").read_text(), "MY HAND-WRITTEN STATE\n")


# --------------------------------------------------------------------------- 3. index + search

class TestIndexSearch(CelebornTestCase):

    def test_index_then_search_finds_section(self):
        self.write("state.md", "# State\n## Auth decision\nWe chose JWT over sessions.\n")
        self.assertIsNone(self.cli("index").exit_code)
        self.assertTrue((self.ctx / "index.db").is_file())

        r = self.cli("search", "JWT")
        self.assertIsNone(r.exit_code)
        self.assertIn("match", r.out)
        self.assertIn("JWT", r.out)
        self.assertIn("state.md", r.out)  # pointer to the source file
        self.assertIn("#auth-decision", r.out)  # anchor pointer

    def test_search_without_index_errors(self):
        r = self.cli("search", "anything")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("no index", r.err.lower())

    def test_search_no_match(self):
        self.cli("index")
        r = self.cli("search", "zzzznotpresentanywhere")
        self.assertIsNone(r.exit_code)
        self.assertIn("No matches", r.out)

    def test_bad_fts_query_exits_one(self):
        self.cli("index")
        r = self.cli("search", '"unbalanced')
        self.assertEqual(r.exit_code, 1)
        self.assertIn("bad FTS query", r.err)

    def test_index_excludes_comment_text(self):
        # Template files are full of HTML comments; none of that boilerplate should be searchable.
        self.cli("index")
        r = self.cli("search", "Chronological")  # a word only in journal.md's comment block
        self.assertIn("No matches", r.out)

    def _bump_mtime(self, rel: str):
        """Force `.context/<rel>` to be newer than index.db (well past the +1s grace)."""
        db = self.ctx / cb.INDEX_NAME
        future = db.stat().st_mtime + 1000
        p = self.ctx / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("# touched\n")
        os.utime(p, (future, future))

    def test_stale_ignores_mechanical_auto_files(self):
        # Rewriting the per-turn mechanical capture files must NOT report the index stale —
        # `celeborn capture` rewrites them every turn, so counting them would make the warning noise.
        self.cli("index")
        self.assertFalse(cb._index_is_stale(self.ctx))
        self._bump_mtime("activity.md")
        self._bump_mtime("auto/2026-06-04.md")
        self.assertFalse(cb._index_is_stale(self.ctx),
                         "mechanical auto-capture churn must not mark the index stale")

    def test_stale_detects_durable_change(self):
        # A genuinely re-indexable change (durable content) must still report stale.
        self.cli("index")
        self.assertFalse(cb._index_is_stale(self.ctx))
        self._bump_mtime("decisions.md")
        self.assertTrue(cb._index_is_stale(self.ctx),
                        "touching a durable file must still mark the index stale")

    def test_stale_indexes_mechanical_files_still_searchable(self):
        # The staleness heuristic ignores mechanical files, but they must remain searchable.
        self.write("auto/2026-06-04.md", "# Auto\n## Snapshot\nuniquemechanicaltoken here.\n")
        self.cli("index")
        r = self.cli("search", "uniquemechanicaltoken")
        self.assertIn("match", r.out)
        self.assertIn("auto/2026-06-04.md", r.out)


# --------------------------------------------------------------------------- 4. archive (forgetting)

class TestArchive(CelebornTestCase):

    def _journal_with(self, n: int) -> str:
        head = "# Journal\n\n"
        entries = "".join(f"## 2026-01-{i:02d} entry {i}\n- did thing {i}\n\n" for i in range(1, n + 1))
        return head + entries

    def test_archive_moves_overflow_keeps_budget(self):
        self.write("journal.md", self._journal_with(25))
        r = self.cli("archive", "--keep", "20")
        self.assertIsNone(r.exit_code)
        _, kept = cb.split_journal(self.read("journal.md"))
        self.assertEqual(len(kept), 20)
        arch = self.read("journal-archive/archive.md")
        # The 5 oldest entries moved; the newest 20 stayed.
        self.assertIn("entry 1", arch)
        self.assertIn("entry 5", arch)
        self.assertNotIn("## 2026-01-01 entry 1", self.read("journal.md"))

    def test_archive_noop_under_budget(self):
        self.write("journal.md", self._journal_with(3))
        r = self.cli("archive", "--keep", "20")
        self.assertIn("nothing to archive", r.out)
        self.assertFalse((self.ctx / "journal-archive" / "archive.md").is_file())

    def test_nothing_lost_archived_entries_still_searchable(self):
        self.write("journal.md", self._journal_with(25))
        self.cli("archive", "--keep", "20")
        self.cli("index")
        r = self.cli("search", "entry 1")  # an archived (cold-tier) entry
        self.assertIsNone(r.exit_code)
        self.assertIn("match", r.out)
        self.assertIn("[cold]", r.out)  # tagged as cold tier, not lost


# --------------------------------------------------------------------------- 5. promote (distillation)

class TestPromote(CelebornTestCase):

    def test_promote_to_learnings(self):
        r = self.cli("promote", "--to", "learnings", "--title", "Cache the index",
                     "--note", "It's regenerable so durability is irrelevant.")
        self.assertIsNone(r.exit_code)
        text = self.read("learnings.md")
        self.assertIn("## Cache the index", text)
        self.assertIn("durability is irrelevant", text)

    def test_promote_to_durable_creates_doc_and_manifest_line(self):
        r = self.cli("promote", "--to", "durable", "--doc", "gotchas",
                     "--title", "SQLite tuning", "--note", "synchronous=OFF is safe here.")
        self.assertIsNone(r.exit_code)
        self.assertTrue((self.ctx / "durable" / "gotchas.md").is_file())
        self.assertIn("SQLite tuning", self.read("durable/gotchas.md"))
        # The manifest must gain a pointer so the new doc is discoverable from the Hot tier.
        self.assertIn("(gotchas.md)", self.read("durable/manifest.md"))

    def test_promote_durable_manifest_not_duplicated(self):
        self.cli("promote", "--to", "durable", "--doc", "gotchas", "--title", "A", "--note", "x")
        self.cli("promote", "--to", "durable", "--doc", "gotchas", "--title", "B", "--note", "y")
        self.assertEqual(self.read("durable/manifest.md").count("(gotchas.md)"), 1)


# --------------------------------------------------------------------------- 6. handoff

class TestWire(CelebornTestCase):
    """`celeborn wire` merges the collapsed `celeborn hook <event>` commands + statusLine into a
    settings.json, idempotently — and migrates a legacy bash-based install in place."""

    def _settings(self) -> Path:
        return self.root / ".claude" / "settings.json"

    def test_wire_creates_settings_with_hook_commands(self):
        self.cli("wire")
        d = json.loads(self._settings().read_text())
        # Collapsed form: no $CELEBORN_HOME env, no bash wrappers — bare in-process commands.
        self.assertNotIn("env", d)
        self.assertEqual(d["statusLine"]["command"], "celeborn hook statusline")
        expected = {
            "SessionStart": "celeborn hook session-start",
            "UserPromptSubmit": "celeborn hook user-prompt-submit",
            "PreCompact": "celeborn hook pre-compact",
            "SessionEnd": "celeborn hook session-end",
            "Stop": "celeborn hook stop",
        }
        for ev, cmd in expected.items():
            cmds = [h["command"] for g in d["hooks"][ev] for h in g["hooks"]]
            self.assertIn(cmd, cmds)
            self.assertFalse(any("bash" in c for c in cmds), f"{ev} still bash-wrapped")

    def test_wire_is_idempotent(self):
        self.cli("wire")
        self.cli("wire")
        d = json.loads(self._settings().read_text())
        self.assertEqual(len(d["hooks"]["Stop"]), 1)   # no duplicate Celeborn group on re-run

    def test_wire_migrates_legacy_bash_install(self):
        # An older install wired to the bash scripts + $CELEBORN_HOME statusLine.
        s = self._settings()
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({
            "env": {"CELEBORN_HOME": "/old/clone"},
            "statusLine": {"type": "command", "command": 'bash "$CELEBORN_HOME/hooks/statusline.sh"'},
            "hooks": {
                "Stop": [{"hooks": [{"type": "command",
                                     "command": 'bash "$CELEBORN_HOME/hooks/capture.sh"'}]}],
            },
        }))
        self.cli("wire")
        d = json.loads(s.read_text())
        self.assertEqual(d["statusLine"]["command"], "celeborn hook statusline")
        cmds = [h["command"] for g in d["hooks"]["Stop"] for h in g["hooks"]]
        self.assertEqual(cmds, ["celeborn hook stop"])          # migrated in place, not duplicated
        self.assertEqual(len(d["hooks"]["Stop"]), 1)

    def test_wire_preserves_existing_hooks_and_skips_foreign_statusline(self):
        s = self._settings()
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({
            "statusLine": {"type": "command", "command": "my-own-statusline"},
            "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo mine"}]}]},
        }))
        self.cli("wire")
        d = json.loads(s.read_text())
        self.assertEqual(d["statusLine"]["command"], "my-own-statusline")   # not clobbered
        cmds = [h["command"] for g in d["hooks"]["Stop"] for h in g["hooks"]]
        self.assertIn("echo mine", cmds)                                    # existing hook kept
        self.assertIn("celeborn hook stop", cmds)                           # Celeborn hook added

    def test_wire_force_replaces_foreign_statusline(self):
        s = self._settings()
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({"statusLine": {"type": "command", "command": "my-own-statusline"}}))
        self.cli("wire", "--force")
        d = json.loads(s.read_text())
        self.assertEqual(d["statusLine"]["command"], "celeborn hook statusline")

    def test_wire_refuses_invalid_json(self):
        s = self._settings()
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text("{ not json")
        r = self.cli("wire")
        self.assertEqual(r.exit_code, 1)

    def test_wire_installs_pretooluse_guard(self):
        # One PreToolUse group fans out in dispatch_hook: Bash → cd+redirect guard (t101);
        # Edit/Write/NotebookEdit → card-less-work gate (t131). Matcher lists exactly those tools.
        self.cli("wire")
        d = json.loads(self._settings().read_text())
        groups = d["hooks"]["PreToolUse"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["matcher"], "Bash|Edit|Write|NotebookEdit")
        cmds = [h["command"] for h in groups[0]["hooks"]]
        self.assertEqual(cmds, ["celeborn hook pre-tool-use"])

    def test_wire_pretooluse_idempotent(self):
        self.cli("wire")
        self.cli("wire")
        d = json.loads(self._settings().read_text())
        self.assertEqual(len(d["hooks"]["PreToolUse"]), 1)   # no duplicate guard group on re-run

    def test_wire_migrates_outdated_pretooluse_matcher(self):
        # An install wired before t131 carries the Bash-only matcher; re-wiring widens it in place
        # (so the card-less-work gate starts firing on Edit/Write) without duplicating the group.
        s = self._settings()
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({"hooks": {"PreToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "celeborn hook pre-tool-use"}]}]}}))
        self.cli("wire")
        groups = json.loads(s.read_text())["hooks"]["PreToolUse"]
        self.assertEqual(len(groups), 1)                     # migrated in place, not duplicated
        self.assertEqual(groups[0]["matcher"], "Bash|Edit|Write|NotebookEdit")


class TestCdRedirectGuard(CelebornTestCase):
    """t101 — the PreToolUse guard turns an un-approvable `cd … > relative/file` compound into a
    deny-with-correction, while leaving everything else (and an explicit bypass) untouched."""

    def _decide(self, command, tool_name="Bash"):
        return cb.dispatch_hook(
            "pre-tool-use",
            {"tool_name": tool_name, "tool_input": {"command": command}},
            str(self.root),
        )

    def _is_deny(self, command, **kw) -> bool:
        out = self._decide(command, **kw)
        if not out:
            return False
        return json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_blocks_cd_plus_relative_redirect(self):
        self.assertTrue(self._is_deny('cd /Users/x/proj && echo "hi" > notes.txt'))
        self.assertTrue(self._is_deny("cd sub && cmd >> log.txt"))          # append counts too
        out = self._decide("cd sub && echo x > out.txt")
        self.assertIn("Write/Edit tool", out)                               # corrective message present
        self.assertIn("celeborn:allow-redirect", out)                       # escape hatch advertised

    def test_bypass_marker_auto_allows(self):
        # An explicit trailing comment → guard returns `allow` → the write runs with NO prompt
        # (operator accepted the path-resolution risk for marked writes).
        out = self._decide("cd sub && echo x > out.txt  # celeborn:allow-redirect: tee needs a real fd")
        self.assertTrue(out)
        self.assertEqual(json.loads(out)["hookSpecificOutput"]["permissionDecision"], "allow")

    def test_absolute_and_devnull_targets_pass(self):
        # cwd-independent targets aren't the path-resolution risk — never blocked.
        self.assertFalse(self._is_deny("cd sub && echo x > /tmp/out.txt"))
        self.assertFalse(self._is_deny("cd sub && noisy 2>/dev/null"))
        self.assertFalse(self._is_deny("cd sub && noisy > /dev/null 2>&1"))

    def test_fd_dup_and_no_redirect_pass(self):
        self.assertFalse(self._is_deny("cd sub && run 2>&1 | tee"))         # 2>&1 is not a file write
        self.assertFalse(self._is_deny("cd sub && ls -la"))                 # no redirect at all

    def test_relative_redirect_without_cd_passes(self):
        # No directory change → the target is unambiguous → not our concern (normal flow handles it).
        self.assertFalse(self._is_deny('echo "hi" > notes.txt'))

    def test_only_bash_tool_runs_the_redirect_guard(self):
        # The cd+redirect guard is Bash-only. An Edit/Write also flows through PreToolUse now (for the
        # t131 card gate), but with the fixture's EMPTY board that gate is exempt, so a redirect-shaped
        # input on a non-Bash tool is still a clean pass-through here.
        self.assertFalse(self._is_deny("cd sub && echo x > out.txt", tool_name="Write"))
        self.assertEqual(self._decide("cd sub && echo x > out.txt", tool_name="Edit"), "")


class TestCardlessWorkGate(CelebornTestCase):
    """t131 — the card-less-work gate. A session that owns no board card is steered onto one: a
    top-priority UserPromptSubmit directive (lever 1) and a PreToolUse soft-deny of Edit/Write/
    NotebookEdit (lever 2). Exempt when the board is empty or the bypass env is armed; the CLI is
    never gated."""

    def _transcript(self) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        f.write(json.dumps({"message": {"usage": {"input_tokens": 1000}}}) + "\n")
        f.close()
        return f.name

    def _decide(self, tool_name, session_id="sgate", **tool_input):
        return cb.dispatch_hook(
            "pre-tool-use",
            {"tool_name": tool_name, "tool_input": tool_input, "session_id": session_id},
            str(self.root))

    def _denied(self, tool_name, **kw) -> bool:
        out = self._decide(tool_name, **kw)
        return bool(out) and json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"

    def _ups_ctx(self, prompt, sid="sgate") -> str:
        tp = self._transcript()
        payload = {"session_id": sid, "transcript_path": tp, "prompt": prompt}
        try:
            with mock.patch.dict(os.environ, {"CELEBORN_AGENT": ""}), \
                    mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload))):
                r = run_cli("--path", str(self.root), "hook", "user-prompt-submit")
        finally:
            os.unlink(tp)
        return (json.loads(r.out)["hookSpecificOutput"]["additionalContext"]
                if r.out.strip() else "")

    # --- lever 2: PreToolUse soft-deny -------------------------------------------------------------

    def test_empty_board_is_exempt(self):
        # Nothing to claim → never gate (init seeds an empty board).
        self.assertFalse(self._denied("Edit", file_path="x.py"))
        self.assertFalse(self._denied("Write", file_path="x.py"))

    def test_open_unowned_card_blocks_edits(self):
        self.cli("tasks", "add", "Wire the adapter")          # t1 todo, unowned
        self.assertTrue(self._denied("Edit", file_path="x.py"))
        self.assertTrue(self._denied("Write", file_path="x.py"))
        self.assertTrue(self._denied("NotebookEdit", notebook_path="x.ipynb"))
        out = self._decide("Edit", file_path="x.py")
        self.assertIn("a card is MANDATORY", out)             # corrective message present
        self.assertIn("celeborn claim", out)
        self.assertIn("CELEBORN_ALLOW_NO_CARD", out)          # escape hatch advertised

    def test_research_and_subagent_tools_blocked(self):
        # CELE-t134: the gate now has teeth beyond file edits — web research and subagent spawn are the
        # tools a card-less *research/design* turn runs through (the gap the user hit). Bash/Read stay
        # ungated (asserted in test_cli_and_readonly_tools_never_gated) so a session can still orient+claim.
        self.cli("tasks", "add", "Wire the adapter")          # t1 todo, unowned
        for tool in ("WebFetch", "WebSearch", "Task", "Agent"):
            self.assertTrue(self._denied(tool), f"{tool} should be gated for a card-less session")
        out = self._decide("WebFetch", url="https://example.com")
        self.assertIn("a card is MANDATORY", out)             # tool-agnostic corrective message
        self.assertIn("research", out)

    def test_claiming_with_session_clears_the_gate(self):
        self.cli("tasks", "add", "Wire the adapter")          # t1 todo
        self.assertTrue(self._denied("Edit", file_path="x.py"))
        self.cli("claim", "t1", "--session", "sgate")          # owner ← sgate, link recorded
        self.assertFalse(self._denied("Edit", file_path="x.py"))

    def test_owner_short_id_fallback_clears_gate(self):
        # A card owned by this session's short id (no recorded session→card link) still clears the gate.
        self.cli("tasks", "add", "Wire the adapter")          # t1
        self.cli("claim", "t1", "--by", "sgateX")             # sgateX[:6] == this session's short id
        self.assertFalse(self._denied("Edit", session_id="sgateXYZ", file_path="x.py"))

    def test_bypass_env_lifts_the_gate(self):
        self.cli("tasks", "add", "Wire the adapter")          # open, unowned
        with mock.patch.dict(os.environ, {"CELEBORN_ALLOW_NO_CARD": "1"}):
            self.assertFalse(self._denied("Edit", file_path="x.py"))

    def test_cli_and_readonly_tools_never_gated(self):
        self.cli("tasks", "add", "Wire the adapter")          # open card exists
        # Bash (incl. the `celeborn` CLI) is not card-gated; read-only tools never match.
        self.assertEqual(self._decide("Bash", command="celeborn claim t1"), "")
        self.assertEqual(self._decide("Read", file_path="x.py"), "")
        self.assertEqual(self._decide("Grep", pattern="foo"), "")

    def test_gate_noop_outside_context(self):
        with tempfile.TemporaryDirectory() as bare:           # no .context/
            out = cb.dispatch_hook(
                "pre-tool-use",
                {"tool_name": "Edit", "tool_input": {"file_path": "x.py"}, "session_id": "s"},
                bare)
            self.assertEqual(out, "")

    # --- lever 1: UserPromptSubmit directive -------------------------------------------------------

    def test_directive_injected_when_cardless_with_open_cards(self):
        self.cli("tasks", "add", "Wire the adapter")          # open, unowned
        ctx = self._ups_ctx("let's get started")              # no card named → no prose claim
        self.assertIn("NO TASK CLAIMED", ctx)
        self.assertIn("TOP PRIORITY", ctx)

    def test_no_directive_when_board_empty(self):
        ctx = self._ups_ctx("let's get started")              # empty board → exempt
        self.assertNotIn("NO TASK CLAIMED", ctx)

    def test_no_directive_after_prose_claim_this_turn(self):
        self.cli("tasks", "add", "Wire the adapter")          # t1
        slug = cb.project_slug(self.ctx)
        ctx = self._ups_ctx(f"work on {slug.upper()}-t1")     # claims this turn → owns a card now
        self.assertIn("card claim", ctx)
        self.assertNotIn("NO TASK CLAIMED", ctx)

    def test_directive_suppressed_by_bypass_env(self):
        self.cli("tasks", "add", "Wire the adapter")
        with mock.patch.dict(os.environ, {"CELEBORN_ALLOW_NO_CARD": "1"}):
            ctx = self._ups_ctx("let's get started")
        self.assertNotIn("NO TASK CLAIMED", ctx)


class TestPermissionBaseline(CelebornTestCase):
    """t100 — `wire --global` merges the SAFE 'big three' permission baseline into
    ~/.claude/settings.json: read-only built-ins + safe Bash prefixes → permissions.allow, and
    defaultMode=acceptEdits. Proactive + universal; idempotent; ask-wins; opt-out; global-only."""

    def setUp(self):
        super().setUp()
        # t115 — `wire --global` now also installs the Matt Pocock skills default-on. Stub the installer
        # so these permission-baseline tests never shell out to `npx` / the network.
        p = mock.patch.object(cb, "_install_mattpocock",
                              return_value={"ok": True, "count": 0, "installed": [], "cwd": ""})
        p.start()
        self.addCleanup(p.stop)

    def _use_tmp_home(self) -> Path:
        home = tempfile.mkdtemp()
        old = os.environ.get("HOME")
        self.addCleanup(lambda: os.environ.__setitem__("HOME", old) if old is not None
                        else os.environ.pop("HOME", None))
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        os.environ["HOME"] = home
        return Path(home)

    def _gsettings(self, home: Path) -> Path:
        return home / ".claude" / "settings.json"

    def test_applies_full_baseline_to_empty_settings(self):
        home = self._use_tmp_home()
        r = self.cli("wire", "--global")
        self.assertIsNone(r.exit_code, r.all)
        d = json.loads(self._gsettings(home).read_text())
        allow = d["permissions"]["allow"]
        for tool in ("Read", "Glob", "Grep"):                       # part 1: read-only built-ins
            self.assertIn(tool, allow)
        self.assertIn("Bash(grep:*)", allow)                        # part 2: safe Bash prefixes
        self.assertIn("Bash(git log:*)", allow)
        self.assertIn("Bash(curl -sS http://localhost:*)", allow)
        self.assertEqual(d["permissions"]["defaultMode"], "acceptEdits")   # part 3
        # NEVER ships a write/delete/non-localhost command.
        joined = "\n".join(allow)
        for danger in ("Bash(sed", "Bash(awk", "Bash(rm", "Bash(echo", "Bash(curl:*)"):
            self.assertNotIn(danger, joined)

    def test_merges_without_clobbering_existing_allow(self):
        home = self._use_tmp_home()
        s = self._gsettings(home)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({"permissions": {"allow": ["Bash(grep:*)", "Bash(my-tool:*)"]}}))
        self.cli("wire", "--global")
        allow = json.loads(s.read_text())["permissions"]["allow"]
        self.assertEqual(allow.count("Bash(grep:*)"), 1)            # overlap not duplicated
        self.assertIn("Bash(my-tool:*)", allow)                     # user's own rule preserved
        self.assertEqual(allow[:2], ["Bash(grep:*)", "Bash(my-tool:*)"])  # originals kept, in order
        self.assertIn("Read", allow)                                # baseline still added

    def test_does_not_override_existing_default_mode(self):
        home = self._use_tmp_home()
        s = self._gsettings(home)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({"permissions": {"defaultMode": "default"}}))
        self.cli("wire", "--global")
        self.assertEqual(json.loads(s.read_text())["permissions"]["defaultMode"], "default")

    def test_deny_wins_rule_not_added_to_allow(self):
        home = self._use_tmp_home()
        s = self._gsettings(home)
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({"permissions": {"allow": [], "deny": ["Bash(curl -sS http://localhost:*)"]}}))
        self.cli("wire", "--global")
        d = json.loads(s.read_text())
        self.assertNotIn("Bash(curl -sS http://localhost:*)", d["permissions"]["allow"])
        self.assertIn("Bash(curl -sS http://localhost:*)", d["permissions"]["deny"])  # deny intact
        self.assertIn("Bash(grep:*)", d["permissions"]["allow"])    # other rules still added

    def test_idempotent_byte_identical_on_rerun(self):
        home = self._use_tmp_home()
        self.cli("wire", "--global")
        first = self._gsettings(home).read_text()
        self.cli("wire", "--global")
        self.assertEqual(self._gsettings(home).read_text(), first)  # second run changes nothing

    def test_opt_out_leaves_permissions_untouched(self):
        home = self._use_tmp_home()
        self.cli("wire", "--global", "--no-permission-baseline")
        d = json.loads(self._gsettings(home).read_text())
        self.assertNotIn("permissions", d)                          # no baseline block written
        self.assertIn("hooks", d)                                   # but hooks still wired

    def test_project_wire_does_not_apply_baseline(self):
        # iron rule 6: global settings only — a project-scoped wire never touches permissions.
        self.cli("wire")
        d = json.loads((self.root / ".claude" / "settings.json").read_text())
        self.assertNotIn("permissions", d)


class TestPermissionsSettingsCli(CelebornTestCase):
    """t115 — the board Settings page CLI surface: `permissions --json` (read state), the safe-baseline
    apply/remove, the Danger Zone arm/disarm, and `skills --json`. All target the project settings here
    (--shared) so the tests stay inside the temp project."""

    def _shared(self) -> Path:
        return self.root / ".claude" / "settings.json"

    def test_permissions_json_shape(self):
        r = self.cli("permissions", "--json")
        self.assertIsNone(r.exit_code, r.all)
        d = json.loads(r.out)
        for key in ("effective_default_mode", "baseline", "danger", "current_allow", "scopes"):
            self.assertIn(key, d)
        self.assertEqual(d["danger"]["confirm_phrase"], "DISABLE ALL SAFETY")
        self.assertIn("Bash(*)", [x["rule"] for x in d["danger"]["spectrum"]])
        self.assertEqual(d["baseline"]["default_mode"]["value"], "acceptEdits")

    def test_baseline_apply_then_remove(self):
        r = self.cli("permissions", "--baseline", "--shared")
        self.assertIsNone(r.exit_code, r.all)
        perms = json.loads(self._shared().read_text())["permissions"]
        self.assertIn("Read", perms["allow"])
        self.assertIn("Bash(grep:*)", perms["allow"])
        self.assertEqual(perms["defaultMode"], "acceptEdits")
        # remove strips exactly the baseline rules + reverts the mode it set
        self.cli("permissions", "--baseline", "--shared", "--remove")
        perms = json.loads(self._shared().read_text())["permissions"]
        self.assertNotIn("Read", perms["allow"])
        self.assertNotIn("Bash(grep:*)", perms["allow"])
        self.assertNotEqual(perms.get("defaultMode"), "acceptEdits")

    def test_danger_refuses_without_yes(self):
        r = self.cli("permissions", "--danger-zone", "--shared")
        self.assertIsNotNone(r.exit_code)            # die()
        self.assertNotEqual(r.exit_code, 0)
        # nothing armed
        self.assertFalse(self._shared().exists() and "Bash(*)" in
                         (json.loads(self._shared().read_text()).get("permissions") or {}).get("allow", []))

    def test_danger_arm_then_disarm(self):
        r = self.cli("permissions", "--danger-zone", "--yes", "--shared")
        self.assertIsNone(r.exit_code, r.all)
        perms = json.loads(self._shared().read_text())["permissions"]
        self.assertIn("Bash(*)", perms["allow"])
        self.assertEqual(perms["defaultMode"], "bypassPermissions")
        # disarm removes the spectrum + restores acceptEdits
        self.cli("permissions", "--danger-zone", "--disarm", "--shared")
        perms = json.loads(self._shared().read_text())["permissions"]
        self.assertNotIn("Bash(*)", perms["allow"])
        self.assertEqual(perms["defaultMode"], "acceptEdits")

    def test_danger_arm_keeps_a_backup(self):
        self._shared().parent.mkdir(parents=True, exist_ok=True)
        self._shared().write_text(json.dumps({"permissions": {"allow": ["Bash(my-tool:*)"]}}))
        self.cli("permissions", "--danger-zone", "--yes", "--shared")
        self.assertTrue((self._shared().parent / "settings.json.celeborn-bak").is_file())

    def test_skills_json_shape(self):
        # _skills_dirs() scans ~/.claude/skills (where global skills install), so isolate HOME to
        # a fresh temp dir — otherwise a developer machine with the suite already installed leaks
        # those skills in and installed_count != 0.
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        with mock.patch.dict(os.environ, {"HOME": home}):
            r = self.cli("skills", "--json")
        self.assertIsNone(r.exit_code, r.all)
        d = json.loads(r.out)
        self.assertEqual(len(d["core"]), 5)
        self.assertEqual(len(d["recommended"]), 6)
        self.assertEqual(d["mattpocock"]["total"], len(cb.MATTPOCOCK_SKILLS))
        self.assertEqual(d["mattpocock"]["installed_count"], 0)   # fresh temp project has no skills dir
        self.assertIn("npx", d["mattpocock"]["install_cmd"])

    def test_wire_global_installs_skills_default_on(self):
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        old = os.environ.get("HOME")
        self.addCleanup(lambda: os.environ.__setitem__("HOME", old) if old is not None
                        else os.environ.pop("HOME", None))
        os.environ["HOME"] = home
        with mock.patch("shutil.which", return_value="/usr/bin/npx"), \
                mock.patch.object(cb, "_install_mattpocock",
                                  return_value={"ok": True, "count": 21, "installed": [], "cwd": home}) as inst:
            self.cli("wire", "--global")
            self.assertEqual(inst.call_count, 1)             # default-on
            inst.reset_mock()
            self.cli("wire", "--global", "--no-skills")
            self.assertEqual(inst.call_count, 0)             # opt-out honored


class TestSkillsAutoUpdate(CelebornTestCase):
    """t116 — Matt Pocock 'stay updated': the weekly throttle, the detached SessionStart refresh
    (Claude-only), and the `skills update` verb. Global skills state is isolated via XDG_CONFIG_HOME."""

    def _tmp_config(self) -> Path:
        cfg = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, cfg, ignore_errors=True)
        p = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": cfg})
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("CELEBORN_NO_SKILLS", None)
        return Path(cfg) / "celeborn"

    def _days_ago(self, n: int) -> str:
        return (cb._dt.datetime.now() - cb._dt.timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%S")

    def test_autoupdate_due_states(self):
        self._tmp_config()
        self.assertTrue(cb._skills_autoupdate_due({}))                                   # never -> due
        self.assertFalse(cb._skills_autoupdate_due({"last_refresh": cb.now_iso()}))      # fresh
        self.assertTrue(cb._skills_autoupdate_due({"last_refresh": self._days_ago(10)})) # stale
        self.assertFalse(cb._skills_autoupdate_due(
            {"last_refresh": self._days_ago(10), "autoupdate": False}))                  # opted out
        with mock.patch.dict(os.environ, {"CELEBORN_NO_SKILLS": "1"}):
            self.assertFalse(cb._skills_autoupdate_due({}))                              # env opt-out

    def test_skills_update_stamps_refresh(self):
        self._tmp_config()
        with mock.patch.object(cb, "_install_mattpocock",
                               return_value={"ok": True, "count": 3, "installed": [], "cwd": ""}):
            r = self.cli("skills", "update", "--global")
        self.assertIsNone(r.exit_code, r.all)
        self.assertTrue(cb._load_skills_state().get("last_refresh"))

    def test_ensure_fresh_spawns_when_due_then_not_when_fresh(self):
        self._tmp_config()
        with mock.patch.dict(os.environ, {"CELEBORN_HARNESS": "claude"}), \
                mock.patch.object(cb, "_spawn_skills_refresh", return_value=True) as spawn:
            cb._ensure_skills_fresh(self.ctx)               # no state -> due
            self.assertEqual(spawn.call_count, 1)
            cb._save_skills_state({"last_refresh": cb.now_iso()})
            spawn.reset_mock()
            cb._ensure_skills_fresh(self.ctx)               # fresh -> skip
            self.assertEqual(spawn.call_count, 0)

    def test_ensure_fresh_skips_non_claude_harness(self):
        self._tmp_config()
        with mock.patch.dict(os.environ, {"CELEBORN_HARNESS": "grok"}), \
                mock.patch.object(cb, "_spawn_skills_refresh", return_value=True) as spawn:
            cb._ensure_skills_fresh(self.ctx)
            self.assertEqual(spawn.call_count, 0)

    def test_spawn_refresh_noop_without_npx(self):
        self._tmp_config()
        with mock.patch("shutil.which", return_value=None):
            self.assertFalse(cb._spawn_skills_refresh())
        self.assertIsNone(cb._load_skills_state().get("last_refresh"))

    def test_spawn_refresh_stamps_and_launches_detached(self):
        self._tmp_config()
        with mock.patch("shutil.which", return_value="/usr/bin/npx"), \
                mock.patch("subprocess.Popen") as popen:
            self.assertTrue(cb._spawn_skills_refresh())
            self.assertEqual(popen.call_count, 1)
            self.assertTrue(popen.call_args.kwargs.get("start_new_session"))   # detached
        self.assertTrue(cb._load_skills_state().get("last_refresh"))           # stamped optimistically

    def test_skills_json_exposes_harness_scope(self):
        self._tmp_config()
        d = json.loads(self.cli("skills", "--json").out)
        self.assertEqual(d["harness"], "claude")
        self.assertTrue(d["mattpocock"]["claude_only"])
        self.assertEqual(d["mattpocock"]["refresh_days"], cb.SKILLS_REFRESH_DAYS)


class TestConsent(unittest.TestCase):
    """`celeborn consent` (t102) — the install-time opt-out screen + agreement recorder. Records to
    ~/.context/consent.json under an isolated tmp HOME so the real one is never touched."""

    def setUp(self):
        self.home = tempfile.mkdtemp()
        self._old = os.environ.get("HOME")
        os.environ["HOME"] = self.home
        self.addCleanup(lambda: os.environ.__setitem__("HOME", self._old) if self._old is not None
                        else os.environ.pop("HOME", None))
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)

    def _file(self) -> Path:
        return Path(self.home) / ".context" / "consent.json"

    def test_records_name_with_all_enabled_by_default(self):
        r = run_cli("consent", "--name", "Grace Hopper")
        self.assertIsNone(r.exit_code, r.all)
        d = json.loads(self._file().read_text())
        self.assertTrue(d["agreed"])
        self.assertEqual(d["name"], "Grace Hopper")
        self.assertEqual(d["opted_out"], [])
        self.assertEqual(d["enabled"], cb.CONSENT_KEYS)          # opt-out model: everything on
        self.assertEqual(d["agreement_url"], cb.AGREEMENT_URL)

    def test_opt_out_accepts_numbers_and_keys(self):
        run_cli("consent", "--name", "Dev", "--opt-out", "5,cd-redirect-guard")
        d = json.loads(self._file().read_text())
        self.assertIn("cd-redirect-autoallow", d["opted_out"])   # item #5
        self.assertIn("cd-redirect-guard", d["opted_out"])       # by key
        self.assertNotIn("cd-redirect-autoallow", d["enabled"])

    def test_show_outputs_recorded_consent(self):
        run_cli("consent", "--name", "Dev")
        r = run_cli("consent", "--show")
        self.assertIn("\"agreed\": true", r.out)
        self.assertIn("Dev", r.out)

    def test_no_name_non_interactive_declines_and_writes_nothing(self):
        r = run_cli("consent", "--yes")           # not a tty + no name → graceful decline
        self.assertEqual(r.exit_code, 1)
        self.assertFalse(self._file().exists())

    def test_checklist_shows_every_automation_and_the_agreement_link(self):
        r = run_cli("consent", "--name", "Dev")
        self.assertIn(cb.AGREEMENT_URL, r.out)
        for _key, what, _why, _risk in cb.CONSENT_ITEMS:
            self.assertIn(what, r.out)


class TestWireQuality(CelebornTestCase):
    """`celeborn wire-quality` (t70 Phase 2) — opt-in deterministic quality gates routed through the
    active adapter: a PostToolUse + Stop hook group on Claude (shared by default), an AGENTS.md
    instruction on a harness with no hooks. Idempotent; never installed by plain `wire`."""

    def _shared(self) -> Path:
        return self.root / ".claude" / "settings.json"

    def _local(self) -> Path:
        return self.root / ".claude" / "settings.local.json"

    def test_writes_post_edit_and_stop_groups_shared_by_default(self):
        self.cli("wire-quality")
        d = json.loads(self._shared().read_text())
        post = [h["command"] for g in d["hooks"]["PostToolUse"] for h in g["hooks"]]
        self.assertIn("celeborn hook post-edit", post)
        # the PostToolUse group carries the Edit/Write matcher so it only fires on edits
        self.assertTrue(any("Edit" in (g.get("matcher") or "") for g in d["hooks"]["PostToolUse"]))
        stop = [h["command"] for g in d["hooks"]["Stop"] for h in g["hooks"]]
        self.assertIn("celeborn hook quality-stop", stop)
        self.assertFalse(self._local().exists())          # shared, not personal, by default

    def test_local_flag_targets_settings_local(self):
        self.cli("wire-quality", "--local")
        self.assertTrue(self._local().exists())
        self.assertFalse(self._shared().exists())
        d = json.loads(self._local().read_text())
        self.assertIn("PostToolUse", d["hooks"])

    def test_is_idempotent(self):
        self.cli("wire-quality")
        self.cli("wire-quality")
        d = json.loads(self._shared().read_text())
        self.assertEqual(len(d["hooks"]["PostToolUse"]), 1)   # no duplicate group on re-run
        self.assertEqual(len(d["hooks"]["Stop"]), 1)

    def test_preserves_existing_settings(self):
        s = self._shared()
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({
            "permissions": {"allow": ["Bash(ls)"]},
            "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "celeborn hook stop"}]}]},
        }))
        self.cli("wire-quality")
        d = json.loads(s.read_text())
        self.assertEqual(d["permissions"]["allow"], ["Bash(ls)"])     # untouched
        stop = [h["command"] for g in d["hooks"]["Stop"] for h in g["hooks"]]
        self.assertIn("celeborn hook stop", stop)                    # capture hook kept
        self.assertIn("celeborn hook quality-stop", stop)            # quality hook added alongside

    def test_plain_wire_does_not_install_quality_hooks(self):
        self.cli("wire")
        d = json.loads(self._shared().read_text())
        post = [h["command"] for g in d["hooks"].get("PostToolUse", []) for h in g["hooks"]]
        self.assertNotIn("celeborn hook post-edit", post)            # opt-in only

    def test_refuses_invalid_json(self):
        s = self._shared()
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text("{ not json")
        r = self.cli("wire-quality")
        self.assertEqual(r.exit_code, 1)

    def test_neutral_harness_falls_back_to_agents_md(self):
        with mock.patch.dict(os.environ, {"CELEBORN_HARNESS": "neutral"}):
            self.cli("wire-quality")
        agents = self.root / "AGENTS.md"
        self.assertTrue(agents.is_file())
        body = agents.read_text()
        self.assertIn(cb.QUALITY_MD_BEGIN, body)
        self.assertIn("next build", body)                  # the explicit "NEVER `next build`" guard
        self.assertIn("tsc --noEmit", body)
        self.assertFalse(self._shared().exists())          # no hooks json on a hookless harness

    def test_agents_md_fallback_is_idempotent(self):
        with mock.patch.dict(os.environ, {"CELEBORN_HARNESS": "neutral"}):
            self.cli("wire-quality")
            self.cli("wire-quality")
        body = (self.root / "AGENTS.md").read_text()
        self.assertEqual(body.count(cb.QUALITY_MD_BEGIN), 1)   # managed block written once


class TestQualityGate(CelebornTestCase):
    """The runtime half of the gates: `celeborn hook post-edit` (cheap per-edit check + dirty marker)
    and `celeborn hook quality-stop` (the deferred full suite, once per turn). Surfaces, never blocks."""

    def setUp(self):
        super().setUp()
        cb._save_metrics(self.ctx, dict(cb.METRICS_TEMPLATE))

    def _edit(self, rel: str, body: str, session: str = "S1") -> str:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        return cb.dispatch_hook(
            "post-edit",
            {"tool_input": {"file_path": str(p)}, "session_id": session},
            str(self.root),
        )

    def test_gate_resolution(self):
        q = cb._quality_config(self.ctx)
        self.assertEqual(cb._quality_gate_for("scripts/celeborn.py", q), "test")
        self.assertEqual(cb._quality_gate_for("tests/test_x.py", q), "test")
        self.assertEqual(cb._quality_gate_for("board/app/Page.tsx", q), "type")
        self.assertIsNone(cb._quality_gate_for("README.md", q))
        self.assertIsNone(cb._quality_gate_for("scripts/notes.txt", q))   # right dir, wrong suffix

    def test_clean_python_edit_marks_dirty_and_is_quiet(self):
        out = self._edit("scripts/good.py", "x = 1\n")
        self.assertEqual(out, "")                                         # py_compile passes -> silent
        self.assertEqual(cb._load_metrics(self.ctx)["quality"]["dirty_session"], "S1")

    def test_broken_python_edit_surfaces_failure(self):
        out = self._edit("scripts/bad.py", "def (:\n")
        self.assertIn("additionalContext", out)                          # surfaced into context
        self.assertIn("FAILED", out)
        env = json.loads(out)
        self.assertEqual(env["hookSpecificOutput"]["hookEventName"], "PostToolUse")

    def test_non_matching_edit_is_noop(self):
        out = self._edit("docs/readme.md", "# hi\n")
        self.assertEqual(out, "")
        self.assertIsNone(cb._load_metrics(self.ctx)["quality"]["dirty_session"])

    def test_quality_stop_wrong_session_does_not_run_or_clear(self):
        self._edit("scripts/good.py", "x = 1\n", session="S1")           # marks S1 dirty
        with mock.patch.object(cb, "_run_quality_cmd") as runner:
            out = cb.dispatch_hook("quality-stop", {"session_id": "OTHER"}, str(self.root))
        runner.assert_not_called()                                       # not this turn's session
        self.assertEqual(out, "")
        self.assertEqual(cb._load_metrics(self.ctx)["quality"]["dirty_session"], "S1")

    def test_quality_stop_right_session_runs_suite_and_clears(self):
        self._edit("scripts/good.py", "x = 1\n", session="S1")
        with mock.patch.object(cb, "_run_quality_cmd", return_value=None) as runner:
            out = cb.dispatch_hook("quality-stop", {"session_id": "S1"}, str(self.root))
        runner.assert_called_once()                                      # full suite ran once
        self.assertEqual(out, "")                                        # passed -> silent
        self.assertIsNone(cb._load_metrics(self.ctx)["quality"]["dirty_session"])   # marker cleared

    def test_quality_stop_surfaces_suite_failure(self):
        self._edit("tests/test_x.py", "x = 1\n", session="S1")
        with mock.patch.object(cb, "_run_quality_cmd", return_value="🏹 ... FAILED"):
            out = cb.dispatch_hook("quality-stop", {"session_id": "S1"}, str(self.root))
        self.assertIn("FAILED", out)

    def test_quality_stop_is_noop_when_nothing_dirty(self):
        with mock.patch.object(cb, "_run_quality_cmd") as runner:
            out = cb.dispatch_hook("quality-stop", {"session_id": "S1"}, str(self.root))
        runner.assert_not_called()
        self.assertEqual(out, "")


class TestQualifiedTaskIds(CelebornTestCase):
    """t79/t84 — project-qualified card ids (SLUG-tN). Stored ids stay bare `tN`; qualification is
    display + input-acceptance. t84: qualified display is the DEFAULT (opt-out via
    `qualified_task_ids: false`); the prefix is a short 4-char derivation of the folder name
    (`project_slug` overrides verbatim)."""

    def _qualify(self, slug="cele"):
        self.write(".celebornrc", json.dumps({"qualified_task_ids": True, "project_slug": slug}))

    # -- short-prefix derivation (t84) --------------------------------------
    def test_short_slug_derivation(self):
        self.assertEqual(cb._short_slug("celeborn"), "cele")
        self.assertEqual(cb._short_slug("DrugAtlas"), "drug")
        self.assertEqual(cb._short_slug("ab"), "ab")               # shorter than n → kept
        self.assertEqual(cb._short_slug("my-proj"), "mypr")        # separators dropped
        self.assertEqual(cb._short_slug("!!!"), "project")         # all-symbol → sane fallback

    def test_explicit_slug_not_shortened(self):
        # An explicit project_slug is the user's authority — used verbatim, never truncated to 4.
        self.write(".celebornrc", json.dumps({"project_slug": "drugatlas"}))
        self.assertEqual(cb.project_slug(self.ctx), "drugatlas")

    # -- parser --------------------------------------------------------------
    def test_split_qualified_tid_forms(self):
        f = cb._split_qualified_tid
        self.assertEqual(f("t79"), (None, "t79"))
        self.assertEqual(f("CELE-t79"), ("CELE", "t79"))      # displayed form
        self.assertEqual(f("cele/t79"), ("cele", "t79"))      # marker form
        self.assertEqual(f("drug-atlas-t12"), ("drug-atlas", "t12"))  # hyphenated slug
        self.assertEqual(f("T79"), (None, "t79"))             # t-number lower-cased
        self.assertEqual(f("garbage"), (None, "garbage"))     # not an id -> exact-match fallback
        self.assertEqual(f(""), (None, ""))

    # -- display -------------------------------------------------------------
    def test_display_qualified_by_default(self):
        # t84: qualified ids are the default. Pin the slug so the derived prefix is deterministic.
        self.write(".celebornrc", json.dumps({"project_slug": "cele"}))
        self.cli("tasks", "add", "First card")
        out = self.cli("tasks").out
        self.assertIn("[CELE-t1]", out)
        self.assertNotIn("[t1]", out)

    def test_display_bare_when_opted_out(self):
        self.write(".celebornrc", json.dumps({"qualified_task_ids": False}))
        self.cli("tasks", "add", "First card")
        out = self.cli("tasks").out
        self.assertIn("[t1]", out)
        self.assertNotIn("CELE-t1", out)

    def test_display_qualified_when_opted_in(self):
        self.cli("tasks", "add", "First card")
        self._qualify()
        out = self.cli("tasks").out
        self.assertIn("[CELE-t1]", out)
        self.assertNotIn("[t1]", out)

    def test_tasks_md_id_stays_bare(self):
        # Source of truth must never carry the qualifier, or the parser/markers break.
        self._qualify()
        self.cli("tasks", "add", "First card")
        self.assertIn("## [t1]", self.read("tasks.md"))
        self.assertNotIn("CELE-t1", self.read("tasks.md"))

    def test_json_projection_carries_display_id_and_flag(self):
        self.cli("tasks", "add", "First card")
        self._qualify()
        doc = json.loads(self.cli("tasks", "json").out)
        self.assertTrue(doc["qualified_task_ids"])
        self.assertEqual(doc["id_prefix"], "CELE")
        t = doc["tasks"][0]
        self.assertEqual(t["id"], "t1")            # canonical key unchanged
        self.assertEqual(t["display_id"], "CELE-t1")

    def test_json_display_id_bare_when_disabled(self):
        self.write(".celebornrc", json.dumps({"qualified_task_ids": False}))
        self.cli("tasks", "add", "First card")
        doc = json.loads(self.cli("tasks", "json").out)
        self.assertFalse(doc["qualified_task_ids"])
        self.assertEqual(doc["tasks"][0]["display_id"], "t1")

    # -- input acceptance ----------------------------------------------------
    def test_move_accepts_qualified_id(self):
        self.cli("tasks", "add", "First card")
        r = self.cli("tasks", "move", "CELE-t1", "doing")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(cb._load_tasks(self.ctx)[0]["state"], "doing")

    def test_move_accepts_marker_form(self):
        self.cli("tasks", "add", "First card")
        r = self.cli("tasks", "move", "cele/t1", "doing")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(cb._load_tasks(self.ctx)[0]["state"], "doing")

    def test_claim_accepts_qualified_id(self):
        self.cli("tasks", "add", "First card")
        r = self.cli("claim", "CELE-t1", "--by", "claude")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(cb._load_tasks(self.ctx)[0]["owner"], "claude")

    def test_ship_accepts_qualified_id(self):
        self.cli("tasks", "add", "First card")
        r = self.cli("ship", "CELE-t1", "--by", "claude")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(cb._load_tasks(self.ctx)[0]["state"], "done")

    def test_rm_accepts_qualified_id(self):
        self.cli("tasks", "add", "First card")
        r = self.cli("tasks", "rm", "CELE-t1")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(cb._load_tasks(self.ctx), [])

    def test_wrong_project_qualifier_warns_but_resolves(self):
        self.cli("tasks", "add", "First card")
        r = self.cli("tasks", "show", "atlas-t1")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("≠ this board", r.all)
        self.assertIn("First card", r.out)


class TestQualityRecommendations(CelebornTestCase):
    """t70 Phase 3 — portable quality recommendations. Change-derived advisor signals (sensitive paths
    → security review; substantial code changes → code review + verify), rendered as Claude slash
    commands or neutral checklists, surfaced via `advise` and a once-per-session Stop nudge."""

    def _git_init(self):
        import subprocess
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)

    def _make(self, *rels):
        for rel in rels:
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x\n")

    # -- pure helpers --------------------------------------------------------
    def test_is_sensitive_matches_path_and_basename(self):
        globs = ["supabase/**", "*auth*", "*billing*"]
        self.assertTrue(cb._is_sensitive("supabase/migrations/x.sql", globs))
        self.assertTrue(cb._is_sensitive("app/lib/auth.ts", globs))   # basename glob
        self.assertTrue(cb._is_sensitive("server/billing_webhook.py", globs))
        self.assertFalse(cb._is_sensitive("app/page.tsx", globs))

    def test_is_code_file(self):
        self.assertTrue(cb._is_code_file("a.py"))
        self.assertTrue(cb._is_code_file("b.TSX"))
        self.assertFalse(cb._is_code_file("README.md"))

    def test_changed_files_empty_outside_git(self):
        self.assertEqual(cb._changed_files(self.root), [])  # no .git → silent, no nag

    # -- advise surfacing ----------------------------------------------------
    def test_advise_surfaces_security_and_code_review(self):
        self._git_init()
        self._make("a.py", "b.ts", "c.py", "lib/auth.ts")
        out = self.cli("advise").out
        self.assertIn("security-review-changes", out)
        self.assertIn("/security-review", out)
        self.assertIn("review-changes", out)
        self.assertIn("/code-review", out)

    def test_advise_clean_tree_silent(self):
        import subprocess
        self._git_init()
        subprocess.run(["git", "-C", str(self.root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.email=t@t",
                        "-c", "user.name=t", "commit", "-qm", "init"], check=True)
        out = self.cli("advise").all
        self.assertIn("No friction detected", out)

    def test_neutral_harness_emits_portable_checklist(self):
        self._git_init()
        self._make("a.py", "b.ts", "c.py")
        doc = json.loads(self.cli("advise", "--harness", "neutral", "--json").out)
        rec = next(r for r in doc["recommendations"] if r["intent"] == "review-changes")
        self.assertEqual(rec["channel"], "instruction")     # neutral = injected instruction
        self.assertIn("Code review", rec["text"])
        self.assertNotIn("/code-review", rec["text"])        # neutral host has no slash command

    def test_review_min_files_threshold(self):
        # Two code files is below the default review_min_files (3) → no code-review signal.
        self._git_init()
        self._make("a.py", "b.ts")
        sigs = cb.active_adapter(self.ctx).friction_signals(self.ctx)
        self.assertFalse(any(s["signal"] == "uncommitted-changes" for s in sigs))

    def test_dismiss_silences_security_review(self):
        self._git_init()
        self._make("lib/auth.ts", "a.py", "b.ts", "c.py")
        self.cli("advise", "--dismiss", "security-review-changes")
        out = self.cli("advise").out
        self.assertNotIn("security-review-changes", out)
        self.assertIn("review-changes", out)                 # the other rec still shows

    # -- Stop nudge ----------------------------------------------------------
    def test_quality_stop_nudges_once_then_dedupes(self):
        self._git_init()
        self._make("lib/auth.ts")
        first = cb.dispatch_hook("quality-stop", {"session_id": "S1"}, str(self.root))
        self.assertIn("/security-review", first)
        # same session → throttled (no repeat nag)
        again = cb.dispatch_hook("quality-stop", {"session_id": "S1"}, str(self.root))
        self.assertEqual(again, "")

    def test_quality_stop_silent_on_clean_tree(self):
        import subprocess
        self._git_init()
        subprocess.run(["git", "-C", str(self.root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.email=t@t",
                        "-c", "user.name=t", "commit", "-qm", "init"], check=True)
        self.assertEqual(cb.dispatch_hook("quality-stop", {"session_id": "S1"}, str(self.root)), "")


class TestThroughputRecommendations(CelebornTestCase):
    """t70 Phase 4 — throughput/autonomy. A large changeset auto-fires a subagent/Workflow hint;
    spawn_task + /loop+/elves are on-demand (surfaced only via `advise --throughput`, never nagged)."""

    def _git_init(self):
        import subprocess
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)

    def _make_code(self, n):
        for i in range(n):
            (self.root / f"f{i}.py").write_text("x\n")

    def test_large_changeset_fires_parallelize(self):
        self._git_init()
        self._make_code(14)                                  # ≥ parallelize_min_files (12)
        out = self.cli("advise").out
        self.assertIn("parallelize-large-changeset", out)
        self.assertIn("review-changes", out)                 # both fire on a big changeset

    def test_below_parallelize_threshold_no_signal(self):
        self._git_init()
        self._make_code(5)                                   # ≥ review (3), < parallelize (12)
        sigs = cb.active_adapter(self.ctx).friction_signals(self.ctx)
        triggers = {s["signal"] for s in sigs}
        self.assertIn("uncommitted-changes", triggers)
        self.assertNotIn("large-changeset", triggers)

    def test_on_demand_intents_hidden_without_flag(self):
        self._git_init()
        self._make_code(4)
        out = self.cli("advise").out
        self.assertNotIn("spawn-tangent", out)
        self.assertNotIn("unattended-run", out)

    def test_throughput_flag_lists_on_demand(self):
        self._git_init()
        self._make_code(4)
        doc = json.loads(self.cli("advise", "--throughput", "--json").out)
        intents = {r["intent"] for r in doc["recommendations"]}
        self.assertIn("spawn-tangent", intents)
        self.assertIn("unattended-run", intents)

    def test_throughput_neutral_renders_generic(self):
        # Neutral host has no /loop or spawn_task — still gets a generic, non-empty recommendation.
        self._git_init()
        doc = json.loads(self.cli("advise", "--throughput", "--harness", "neutral", "--json").out)
        rec = next(r for r in doc["recommendations"] if r["intent"] == "unattended-run")
        self.assertTrue(rec["text"])
        self.assertNotIn("/loop", rec["text"])               # no harness-specific command on neutral

    def test_throughput_dismiss_silences_on_demand(self):
        self._git_init()
        self.cli("advise", "--dismiss", "spawn-tangent")
        doc = json.loads(self.cli("advise", "--throughput", "--json").out)
        intents = {r["intent"] for r in doc["recommendations"]}
        self.assertNotIn("spawn-tangent", intents)
        self.assertIn("unattended-run", intents)


class TestFleetSlugDedup(unittest.TestCase):
    """Project-qualified ids (SLUG-tN) must be unambiguous across the machine. Registering a project
    whose qualifier collides with another auto-suffixes it (-2, -3 …) and persists the resolved slug;
    an explicit project_slug is kept but warned. (t79 follow-up.)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_xdg = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self._tmp.name      # isolate the fleet registry

    def tearDown(self):
        if self._old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._old_xdg
        self._tmp.cleanup()

    def _project(self, folder, slug=None):
        """A fresh initialized project at <newtmp>/<folder>, optionally with an explicit project_slug."""
        root = Path(tempfile.mkdtemp()) / folder
        root.mkdir()
        self.addCleanup(lambda: shutil.rmtree(root.parent, ignore_errors=True))
        run_cli("--path", str(root), "init", "--no-scan")
        if slug is not None:
            (root / ".context" / ".celebornrc").write_text(json.dumps({"project_slug": slug}))
        return root

    def test_dedupe_slug_unit(self):
        self.assertEqual(cb._dedupe_slug("cele", []), "cele")
        self.assertEqual(cb._dedupe_slug("cele", ["other"]), "cele")
        self.assertEqual(cb._dedupe_slug("cele", ["cele"]), "cele-2")
        self.assertEqual(cb._dedupe_slug("cele", ["CELE"]), "cele-2")   # case-insensitive
        self.assertEqual(cb._dedupe_slug("cele", ["cele", "cele-2"]), "cele-3")

    def test_collision_gets_numeric_suffix_persisted(self):
        a = self._project("myproj")
        b = self._project("myproj")
        ra = cb._fleet_register_path(a)
        rb = cb._fleet_register_path(b)
        # the qualifier is the short 4-char prefix derived from the folder ("myproj" → "mypr")
        self.assertEqual(ra["slug"], "mypr")
        self.assertEqual(rb["slug"], "mypr-2")
        # persisted to B's rc so display + markers agree
        self.assertEqual(cb.project_slug(b / ".context"), "mypr-2")
        self.assertEqual(json.loads((b / ".context/.celebornrc").read_text())["project_slug"], "mypr-2")

    def test_qualified_ids_differ_after_dedup(self):
        a = self._project("myproj"); b = self._project("myproj")
        cb._fleet_register_path(a); cb._fleet_register_path(b)
        cfg = {"qualified_task_ids": True}
        self.assertEqual(cb._display_tid(a / ".context", "t1", cfg=cfg), "MYPR-t1")
        self.assertEqual(cb._display_tid(b / ".context", "t1", cfg=cfg), "MYPR-2-t1")

    def test_three_way_collision(self):
        slugs = [cb._fleet_register_path(self._project("p"))["slug"] for _ in range(3)]
        self.assertEqual(slugs, ["p", "p-2", "p-3"])

    def test_idempotent_reregister_keeps_slug(self):
        b = self._project("myproj")
        cb._fleet_register_path(self._project("myproj"))   # claim "myproj" first
        first = cb._fleet_register_path(b)["slug"]
        again = cb._fleet_register_path(b)["slug"]
        self.assertEqual(first, again)                      # idempotent on path

    def test_explicit_slug_kept_and_warned(self):
        cb._fleet_register_path(self._project("alpha", slug="cele"))
        beta = self._project("beta", slug="cele")
        err = io.StringIO()
        with contextlib.redirect_stdout(err), contextlib.redirect_stderr(err):
            row = cb._fleet_register_path(beta)
        self.assertEqual(row["slug"], "cele")               # explicit slug is NOT auto-changed
        self.assertIn("ambiguous", err.getvalue())          # but the clash is surfaced
        self.assertEqual(json.loads((beta / ".context/.celebornrc").read_text())["project_slug"], "cele")


class TestFleetRepair(unittest.TestCase):
    """`celeborn fleet repair` — one-shot re-dedup of the whole registry (the repair t85 deferred).
    t85 only deduped at register time, so pre-t85/pre-t84 rows kept a stale slug that never went
    through dedup; a later project's short qualifier could then collide undetected. Repair recomputes
    every project's current qualifier, re-dedups, and reconciles both the registry and .celebornrc."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_xdg = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self._tmp.name      # isolate the fleet registry

    def tearDown(self):
        if self._old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._old_xdg
        self._tmp.cleanup()

    def _project(self, folder, slug=None):
        root = Path(tempfile.mkdtemp()) / folder
        root.mkdir()
        self.addCleanup(lambda: shutil.rmtree(root.parent, ignore_errors=True))
        run_cli("--path", str(root), "init", "--no-scan")
        if slug is not None:
            (root / ".context" / ".celebornrc").write_text(json.dumps({"project_slug": slug}))
        return root

    def _seed_registry(self, *rows):
        """Write fleet.json directly with (path, stale_slug) pairs — bypasses register-time dedup to
        simulate a pre-t85/pre-t84 registry."""
        cb._save_fleet_registry({"projects": [
            {"path": str(p), "slug": s, "name": Path(p).name, "added": "2026-01-01T00:00:00"}
            for p, s in rows]})

    def _registry_slugs(self):
        return [p["slug"] for p in cb._load_fleet_registry()["projects"]]

    def test_repair_syncs_stale_long_slugs(self):
        # Pre-t84 rows held the full folder name; repair syncs them to the short qualifier (no collision).
        a = self._project("drugatlas")
        self._seed_registry((a, "drugatlas"))
        changes = cb._fleet_repair()
        self.assertEqual(self._registry_slugs(), ["drug"])
        self.assertEqual(changes[0]["old"], "drugatlas")
        self.assertEqual(changes[0]["new"], "drug")
        self.assertFalse(changes[0]["rc_written"])          # derived + unique → no rc needed

    def test_repair_dedupes_preexisting_collision(self):
        # Two folders that both short-slug to "drug" — distinct as long names, colliding as qualifiers.
        a = self._project("drugatlas")
        b = self._project("drugzone")
        self._seed_registry((a, "drugatlas"), (b, "drugzone"))
        cb._fleet_repair()
        self.assertEqual(self._registry_slugs(), ["drug", "drug-2"])
        # the loser's resolved qualifier is persisted to its rc so display + markers agree
        self.assertEqual(cb.project_slug(b / ".context"), "drug-2")
        self.assertEqual(json.loads((b / ".context/.celebornrc").read_text())["project_slug"], "drug-2")

    def test_repair_is_idempotent(self):
        a = self._project("drugatlas"); b = self._project("drugzone")
        self._seed_registry((a, "drugatlas"), (b, "drugzone"))
        cb._fleet_repair()
        self.assertEqual(cb._fleet_repair(), [])            # second pass: nothing left to change

    def test_repair_dry_run_writes_nothing(self):
        a = self._project("drugatlas"); b = self._project("drugzone")
        self._seed_registry((a, "drugatlas"), (b, "drugzone"))
        changes = cb._fleet_repair(apply=False)
        self.assertEqual(len(changes), 2)
        self.assertEqual(self._registry_slugs(), ["drugatlas", "drugzone"])   # registry untouched
        self.assertIsNone(json.loads((b / ".context/.celebornrc").read_text()).get("project_slug"))

    def test_repair_keeps_explicit_slug_but_flags_clash(self):
        a = self._project("alpha", slug="cele")
        b = self._project("beta", slug="cele")
        self._seed_registry((a, "cele"), (b, "cele"))
        changes = cb._fleet_repair()
        # explicit slug is authority — NOT auto-suffixed; both stay "cele"
        self.assertEqual(self._registry_slugs(), ["cele", "cele"])
        self.assertEqual(json.loads((b / ".context/.celebornrc").read_text())["project_slug"], "cele")
        self.assertTrue(any(c.get("collision") for c in changes))

    def test_repair_skips_unreachable_project(self):
        a = self._project("drugatlas")
        gone = Path(self._tmp.name) / "deleted-project"     # path with no .context/
        self._seed_registry((a, "drugatlas"), (gone, "ghost"))
        changes = cb._fleet_repair()
        skips = [c for c in changes if c.get("action") == "skip"]
        self.assertEqual(len(skips), 1)
        self.assertEqual(self._registry_slugs()[0], "drug")  # reachable project still repaired


class TestFleetAutoRegister(unittest.TestCase):
    """Orient self-registers the active project into the fleet (CELE-t124) so the board's savings bar
    and `celeborn fleet` reflect every project that actually runs Celeborn — not just hand-added ones
    (the 'why are there only 3 of my 6 projects?' bug). Best-effort, idempotent, skips the global sink."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_xdg = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self._tmp.name      # isolate the fleet registry

    def tearDown(self):
        if self._old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._old_xdg
        self._tmp.cleanup()

    def _project(self, folder):
        root = Path(tempfile.mkdtemp()) / folder
        root.mkdir()
        self.addCleanup(lambda: shutil.rmtree(root.parent, ignore_errors=True))
        run_cli("--path", str(root), "init", "--no-scan")
        return root

    def _registry_paths(self):
        return [r["path"] for r in cb._load_fleet_registry().get("projects", [])]

    def test_autoregister_adds_unregistered_project(self):
        root = self._project("drugatlas.ai")
        self.assertEqual(self._registry_paths(), [])        # not registered yet
        cb._fleet_autoregister(root / ".context")
        self.assertEqual(self._registry_paths(), [str(root.resolve())])

    def test_autoregister_is_idempotent(self):
        root = self._project("peptoids.ai")
        cb._fleet_autoregister(root / ".context")
        cb._fleet_autoregister(root / ".context")           # second orient — no duplicate row
        self.assertEqual(self._registry_paths(), [str(root.resolve())])

    def test_autoregister_skips_global_sink(self):
        """The home-level ~/.context capture sink is not a project — orienting from home must not
        register it (or it would pollute the fleet with a phantom 'project')."""
        home = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(home, ignore_errors=True))
        gctx = home / cb.CONTEXT_DIRNAME
        gctx.mkdir()
        with mock.patch.object(cb.Path, "home", staticmethod(lambda: home)):
            cb._fleet_autoregister(gctx)
        self.assertEqual(self._registry_paths(), [])

    def test_orient_hook_self_registers(self):
        """End-to-end: a SessionStart orient through dispatch_hook registers the project."""
        root = self._project("living-eamon")
        with mock.patch.object(cb, "ensure_board", lambda *a, **k: None), \
             mock.patch.object(cb, "_ensure_skills_fresh", lambda *a, **k: None):
            cb.dispatch_hook("session-start", {"session_id": "S1"}, str(root))
        self.assertIn(str(root.resolve()), self._registry_paths())


class TestHook(CelebornTestCase):
    """`celeborn hook <event>` — the collapsed in-process hook entry point (executable-app §3).

    Each test drives the real argparse entrypoint with a mocked stdin payload and asserts parity with
    the behavior the bash hooks produced."""

    def _transcript(self, *entries) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for e in entries:
            f.write(json.dumps(e) + "\n")
        f.close()
        return f.name

    def hook(self, event, payload=None, path=None) -> Run:
        p = path if path is not None else str(self.root)
        with mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload or {}))):
            return run_cli("--path", p, "hook", event)

    def test_session_start_prints_orient_load(self):
        r = self.hook("session-start", {"session_id": "s1"})
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("## Celeborn memory (Orient load)", r.out)
        self.assertIn("Orient load (Hot tier)", r.out)

    def test_session_start_carries_shell_hygiene_rule(self):
        # t101 lever 1: the soft directive rides the orient channel every session.
        r = self.hook("session-start", {"session_id": "s1"})
        self.assertIn("Celeborn shell rule", r.out)
        self.assertIn("Write/Edit tool", r.out)

    def test_session_start_records_orient(self):
        self.hook("session-start", {"session_id": "s1"})
        m = cb._load_metrics(self.ctx)
        self.assertGreaterEqual(m.get("orient_events", 0), 1)
        self.assertEqual(m.get("last_session_id"), "s1")

    def test_noop_outside_context(self):
        with tempfile.TemporaryDirectory() as bare:          # no .context/ here
            r = self.hook("session-start", {"session_id": "s1"}, path=bare)
            self.assertIsNone(r.exit_code, r.all)
            self.assertEqual(r.out.strip(), "")

    def test_pre_compact_prints_checkpoint_and_counts(self):
        r = self.hook("pre-compact", {})
        self.assertIn("Compaction imminent", r.out)
        self.assertIn("state.md", r.out)
        self.assertEqual(cb._load_metrics(self.ctx).get("compactions_bridged", 0), 1)

    def test_session_end_writes_handoff(self):
        self.hook("session-end", {})
        self.assertTrue((self.ctx / "handoff.md").is_file())

    def test_stop_captures_and_emits_system_message(self):
        tp = self._transcript(
            {"type": "user", "uuid": "u1", "sessionId": "s1",
             "message": {"role": "user", "content": "do the thing"}},
            {"type": "assistant", "uuid": "a1", "sessionId": "s1",
             "message": {"role": "assistant",
                         "content": [{"type": "tool_use", "id": "t1", "name": "Edit",
                                      "input": {"file_path": "app/x.py"}}]}},
        )
        try:
            r = self.hook("stop", {"session_id": "s1", "transcript_path": tp})
        finally:
            os.unlink(tp)
        self.assertIsNone(r.exit_code, r.all)
        env = json.loads(r.out)                               # the Stop hook emits a JSON systemMessage
        self.assertIn("systemMessage", env)
        self.assertTrue((self.ctx / "activity.md").is_file())

    def test_stop_without_transcript_is_silent(self):
        r = self.hook("stop", {"session_id": "s1"})
        self.assertEqual(r.out.strip(), "")

    def test_user_prompt_submit_emits_envelope(self):
        # A transcript over the 150k soft line → heartbeat + an urgent nudge in one JSON envelope.
        tp = self._transcript({"message": {"usage": {"input_tokens": 161_000}}})
        try:
            self.hook("stop", {"session_id": "s1", "transcript_path": self._transcript(
                {"type": "user", "uuid": "u1", "sessionId": "s1",
                 "message": {"role": "user", "content": "hi"}})})   # bank a capture for the heartbeat
            r = self.hook("user-prompt-submit", {"session_id": "s1", "transcript_path": tp})
        finally:
            os.unlink(tp)
        self.assertIsNone(r.exit_code, r.all)
        env = json.loads(r.out)
        ctx = env["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(env["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit")
        self.assertIn("do NOT surface", ctx)                  # heartbeat block
        self.assertIn("SURFACE THIS TO THE USER", ctx)        # nudge block
        self.assertIn("/clear", ctx)
        # The clear-OK nudge must ALSO instruct the model to freshen the Hot tier first, so the
        # "without need to rehydrate" promise actually holds when the user clears.
        self.assertIn("FRESHEN THE HOT TIER FIRST", ctx)
        self.assertIn("state.md", ctx)
        self.assertIn("session.json", ctx)

    def test_user_prompt_submit_without_transcript_is_silent(self):
        r = self.hook("user-prompt-submit", {"session_id": "s1"})
        self.assertEqual(r.out.strip(), "")

    def test_user_prompt_submit_drains_handoff_into_envelope(self):
        # A card queued via the board (outbox) rides into the next turn as a work instruction,
        # then the pending queue is emptied so it fires exactly once.
        self.cli("tasks", "add", "Refactor the parser", "--note", "split it in two")
        self.cli("outbox", "push", "--task", "t1")
        tp = self._transcript({"message": {"usage": {"input_tokens": 1000}}})
        try:
            r = self.hook("user-prompt-submit", {"session_id": "s1", "transcript_path": tp})
        finally:
            os.unlink(tp)
        ctx = json.loads(r.out)["hookSpecificOutput"]["additionalContext"]
        self.assertIn("task hand-off", ctx)
        self.assertIn("Refactor the parser", ctx)
        self.assertIn("split it in two", ctx)
        self.assertIn("Outbox empty", self.cli("outbox", "list").out)  # drained exactly once

    def test_user_prompt_submit_claims_a_pasted_card(self):
        # Claim-on-receipt: a pasted card (its marker in the prompt) is claimed by the receiving
        # session — owner ← session id (no CELEBORN_AGENT here), TODO → DOING — and the envelope
        # tells the model it now owns the card.
        self.cli("tasks", "add", "Wire the adapter")          # t1, todo
        tp = self._transcript({"message": {"usage": {"input_tokens": 1000}}})
        try:
            with mock.patch.dict(os.environ, {"CELEBORN_AGENT": ""}):
                r = self.hook("user-prompt-submit", {
                    "session_id": "sess-7", "transcript_path": tp,
                    "prompt": "start on this please\n\n⟨celeborn:t1⟩",
                })
        finally:
            os.unlink(tp)
        ctx = json.loads(r.out)["hookSpecificOutput"]["additionalContext"]
        self.assertIn("card claim", ctx)
        show = self.cli("tasks", "show", "t1").out
        self.assertIn("owner:      sess-7", show)
        self.assertIn("state:      doing", show)

    def test_prose_task_mention_claims_card_as_session(self):
        # CELE-t131: naming a project-qualified card in PROSE (no pasted marker) on a fresh session is a
        # strong, intentional signal — Celeborn CLAIMS the card, owned by the SESSION short id (the
        # session is the agent's name, never a model handle), and advances TODO → DOING.
        self.cli("tasks", "add", "Wire the adapter")          # t1, todo, no owner
        slug = cb.project_slug(self.ctx)
        tp = self._transcript({"message": {"usage": {"input_tokens": 1000}}})
        try:
            with mock.patch.dict(os.environ, {"CELEBORN_AGENT": ""}):
                self.hook("user-prompt-submit", {
                    "session_id": "sess-9", "transcript_path": tp,
                    "prompt": f"work on {slug.upper()}-t1",
                })
        finally:
            os.unlink(tp)
        show = self.cli("tasks", "show", "t1").out
        self.assertIn("owner:      sess-9", show)             # owned by the SESSION short id …
        self.assertIn("state:      doing", show)             # … and advanced to DOING
        link = (cb._load_metrics(self.ctx).get("agent_sessions") or {}).get("sess-9") or {}
        self.assertEqual(link.get("task"), "t1")             # session→card bridge recorded

    def test_prose_claim_is_vacuum_fill_only(self):
        # CELE-t131: once a session has claimed a live card, a later casual prose mention of a DIFFERENT
        # card must not thrash the board off the card it's actually working.
        self.cli("tasks", "add", "First card")                # t1
        self.cli("tasks", "add", "Second card")               # t2
        slug = cb.project_slug(self.ctx)
        tp = self._transcript({"message": {"usage": {"input_tokens": 1000}}})
        try:
            with mock.patch.dict(os.environ, {"CELEBORN_AGENT": ""}):
                self.hook("user-prompt-submit", {           # first mention claims t1 …
                    "session_id": "sess-x", "transcript_path": tp,
                    "prompt": f"work on {slug.upper()}-t1",
                })
                self.hook("user-prompt-submit", {           # … a casual mention of t2 must NOT thrash
                    "session_id": "sess-x", "transcript_path": tp,
                    "prompt": f"by the way {slug.upper()}-t2 is related",
                })
        finally:
            os.unlink(tp)
        link = (cb._load_metrics(self.ctx).get("agent_sessions") or {}).get("sess-x") or {}
        self.assertEqual(link.get("task"), "t1")              # still on t1, not thrashed to t2
        self.assertIn("state:      todo", self.cli("tasks", "show", "t2").out)  # t2 untouched

    def test_find_prose_card_refs_matches_qualified_open_cards_only(self):
        claimable = {"t1", "t2"}
        self.assertEqual(
            cb._find_prose_card_refs("continue with CELE-t1", expected_slug="cele", claimable_ids=claimable),
            ["t1"])
        # Jira-style id (no leading 't'), wrong project, bare number, and shipped card are all ignored.
        self.assertEqual(cb._find_prose_card_refs("SCRUM-115 done", expected_slug="cele", claimable_ids=claimable), [])
        self.assertEqual(cb._find_prose_card_refs("FOO-t1 elsewhere", expected_slug="cele", claimable_ids=claimable), [])
        self.assertEqual(cb._find_prose_card_refs("bare t1", expected_slug="cele", claimable_ids=claimable), [])
        self.assertEqual(cb._find_prose_card_refs("CELE-t9 shipped", expected_slug="cele", claimable_ids=claimable), [])

    def test_statusline_prints_line(self):
        r = self.hook("statusline", {"session_id": "s1"})
        self.assertIn("🏹 Celeborn", r.out)

    def test_malformed_stdin_does_not_crash(self):
        # Garbage on stdin parses to {} — the hook degrades gracefully instead of raising.
        with mock.patch.object(sys, "stdin", io.StringIO("{ not json")):
            r = run_cli("--path", str(self.root), "hook", "session-start")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Orient load", r.out)     # still produced the payload; just no session_id


class TestHandoff(CelebornTestCase):

    def test_handoff_regenerates_from_session_json(self):
        session = {
            "schema": "celeborn/1", "updated_at": cb.now_iso(),
            "focus": "shipping the parser", "next_action": "write tests",
            "branch": "feat/parser", "status": "in-progress",
            "stop_allowed": True, "open_threads": ["thread one", "thread two"],
        }
        self.write("session.json", json.dumps(session))
        r = self.cli("handoff")
        self.assertIsNone(r.exit_code)
        h = self.read("handoff.md")
        self.assertIn("shipping the parser", h)
        self.assertIn("write tests", h)
        self.assertIn("feat/parser", h)
        self.assertIn("- thread one", h)
        self.assertIn("- thread two", h)
        self.assertIn("Resume prompt", h)

    def test_handoff_increments_metric(self):
        before = json.loads(self.read("metrics.json"))["handoffs_written"]
        self.cli("handoff")
        after = json.loads(self.read("metrics.json"))["handoffs_written"]
        self.assertEqual(after, before + 1)


# --------------------------------------------------------------------------- 7. doctor (health + secrets)

class TestDoctor(CelebornTestCase):

    def test_doctor_green_on_fresh_init(self):
        self.cli("index")  # avoid the "index absent" warning
        r = self.cli("doctor")
        self.assertIsNone(r.exit_code, f"doctor should pass: {r.all}")
        self.assertIn("0 problem(s)", r.out)

    def test_doctor_fails_on_missing_required_file(self):
        (self.ctx / "decisions.md").unlink()
        r = self.cli("doctor")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("MISSING required file: decisions.md", r.out)

    def test_doctor_detects_secret_and_exits_one(self):
        self.write("state.md", "# State\nleaked key sk-abcdefghijklmnopqrstuvwxyz0123\n")
        r = self.cli("doctor")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("POSSIBLE SECRET", r.out)

    def test_doctor_flags_invalid_session_json(self):
        self.write("session.json", "{ not valid json ")
        r = self.cli("doctor")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("INVALID JSON", r.out)

    def test_doctor_warns_when_state_over_budget(self):
        self.write("state.md", "# State\n" + "\n".join(f"line {i}" for i in range(200)))
        self.cli("index")
        r = self.cli("doctor")
        # Over-budget is a warning, not a problem: doctor still exits 0.
        self.assertIsNone(r.exit_code)
        self.assertIn("condense it", r.out)

    def test_doctor_warns_open_card_missing_stop(self):
        self.cli("tasks", "add", "Needs a stop")
        self.cli("tasks", "edit", "t1", "--stop", "")  # clear the auto-filled default
        self.cli("index")
        r = self.cli("doctor")
        # Missing stop is a warning, not a problem: doctor still exits 0.
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("no Stop condition", r.out)

    def test_doctor_notes_default_stop_as_info(self):
        # A card carrying the auto-filled default is an informational nudge, not a warning.
        self.cli("tasks", "add", "Default stop")
        self.cli("index")
        r = self.cli("doctor")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("generic default Stop condition", r.out)

    def test_doctor_notes_gh_banner_when_unauthenticated(self):
        # When gh is installed-but-unauthed, doctor prints an informational heads-up about Claude
        # Code's PR-status banner — but it's NOT a Celeborn problem and must not change the counts.
        self.addCleanup(setattr, cb, "_gh_unauthenticated", cb._gh_unauthenticated)
        cb._gh_unauthenticated = lambda: True
        self.cli("index")
        r = self.cli("doctor")
        self.assertIsNone(r.exit_code, f"the gh note must not fail doctor: {r.all}")
        self.assertIn("gh auth login", r.out)
        self.assertIn("0 problem(s)", r.out)

    def test_doctor_silent_about_gh_when_authenticated(self):
        self.addCleanup(setattr, cb, "_gh_unauthenticated", cb._gh_unauthenticated)
        cb._gh_unauthenticated = lambda: False
        self.cli("index")
        r = self.cli("doctor")
        self.assertNotIn("PR-status panel", r.out)

    def test_gh_unauthenticated_false_when_gh_absent(self):
        # No gh on PATH → never invent a problem, regardless of how the subprocess would behave.
        # The helper does a local `import shutil`, so patch the real module it resolves to.
        self.addCleanup(setattr, shutil, "which", shutil.which)
        shutil.which = lambda _name: None
        self.assertFalse(cb._gh_unauthenticated())

class TestMemoryDrift(CelebornTestCase):
    """`doctor` keeps live memory HONEST: state.md/notes.md must not point at files the repo no
    longer has. A stale reference is a warning (not a hard failure) — the next session would
    otherwise rehydrate a path that's been deleted or renamed out from under it."""

    def test_extract_memory_paths_precision(self):
        # Only backtick-wrapped, slash+extension tokens count. Bare words, commands, URLs,
        # <placeholders>, and dotfiles are all ignored so we never flag a non-path.
        text = ("touch `scripts/celeborn.py` and `board/app/Foo.tsx`; run `celeborn tasks move t1` "
                "see `https://github.com/a/b.md` and `<file>/x.py` and `.context/.board.pid` "
                "plain scripts/celeborn.py outside backticks")
        got = cb._extract_memory_paths(text)
        self.assertEqual(got, ["scripts/celeborn.py", "board/app/Foo.tsx"])

    def test_doctor_flags_stale_reference(self):
        self.write("state.md", "# State\n## Now\n- core lives in `src/ghost.py` (renamed away)\n")
        self.cli("index")
        r = self.cli("doctor")
        self.assertIsNone(r.exit_code, f"drift is a warning, not a problem: {r.all}")
        self.assertIn("memory drift", r.out)
        self.assertIn("src/ghost.py", r.out)

    def test_doctor_clean_when_reference_exists(self):
        # A referenced file that actually exists in the repo must NOT be flagged.
        (self.root / "src").mkdir()
        (self.root / "src" / "real.py").write_text("x = 1\n")
        self.write("state.md", "# State\n## Now\n- core lives in `src/real.py`\n")
        self.cli("index")
        r = self.cli("doctor")
        self.assertIsNone(r.exit_code, r.all)
        self.assertNotIn("memory drift", r.out)
        self.assertIn("memory matches repo", r.out)

    def test_doctor_ignores_history_tiers(self):
        # journal.md is append-only history: a since-deleted file named there is correct, not drift.
        self.write("journal.md", "## 2026-01-01\nremoved `legacy/old.py` today\n")
        self.cli("index")
        r = self.cli("doctor")
        self.assertNotIn("legacy/old.py", r.out)


class TestHotTierBudget(CelebornTestCase):
    """The Orient load is injected as SessionStart additionalContext; if it outgrows the host's
    inline budget the host persists it to a file and the model gets only a preview — silently
    killing automatic rehydration. `status` must clip oversized Hot files (with a pointer) so the
    payload stays small, and `doctor` must surface the over-budget condition rather than hide it."""

    def _big_state(self, n: int) -> str:
        return "# State\n\n## Now\n" + "\n".join(f"- historical bullet {i} " + "x" * 80 for i in range(n))

    def test_status_clips_oversized_state(self):
        self.write("state.md", self._big_state(300))  # well over hot_state_max_chars
        r = self.cli("status")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Hot tier clipped", r.out)
        # The clipped Orient load must be far smaller than the raw file.
        self.assertLess(len(r.out), len(self.read("state.md")))

    def test_status_full_bypasses_clip(self):
        self.write("state.md", self._big_state(300))
        r = self.cli("status", "--full")
        self.assertIsNone(r.exit_code, r.all)
        self.assertNotIn("Hot tier clipped", r.out)
        self.assertIn("historical bullet 299", r.out)  # tail of the file is present, unclipped

    def test_status_does_not_clip_small_state(self):
        # A within-budget Hot tier rehydrates verbatim — no pointer noise.
        r = self.cli("status")
        self.assertIsNone(r.exit_code, r.all)
        self.assertNotIn("Hot tier clipped", r.out)

    def test_status_clips_long_focus(self):
        self.write("session.json", json.dumps({"focus": "F" * 5000, "branch": "main"}))
        r = self.cli("status")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Hot tier clipped", r.out)
        self.assertNotIn("F" * 5000, r.out)

    def test_doctor_warns_when_hot_over_char_budget(self):
        self.write("state.md", self._big_state(300))
        self.cli("index")
        r = self.cli("doctor")
        self.assertIsNone(r.exit_code, f"over-budget is a warning, not a failure: {r.all}")
        self.assertIn("over char budget", r.out)

    def test_doctor_ok_when_hot_within_char_budget(self):
        self.cli("index")
        r = self.cli("doctor")
        self.assertIn("Hot tier within char budget", r.out)


class TestHeadlineNotesSplit(CelebornTestCase):
    """state.md is the small Hot headline that loads on Orient; notes.md is the unbounded working
    detail that is NOT auto-loaded (so it never needs trimming) and is read on demand."""

    def test_init_creates_notes(self):
        self.assertTrue((self.ctx / "notes.md").is_file())

    def test_status_lists_notes_on_demand_not_inlined(self):
        # A distinctive marker in notes.md must NOT appear in the Orient load — only a pointer to it.
        self.write("notes.md", "# Working notes\n\nUNIQUE_NOTES_MARKER_ZZZ deep detail here\n")
        r = self.cli("status")
        self.assertIsNone(r.exit_code, r.all)
        self.assertNotIn("UNIQUE_NOTES_MARKER_ZZZ", r.out)  # body not inlined
        self.assertIn("notes.md", r.out)                    # but pointed to
        self.assertIn("read it for depth", r.out)

    def test_notes_can_be_huge_without_clip_warning(self):
        # The whole point: notes.md has no size budget, so a large one is fine — doctor stays green
        # (the Hot char-budget check looks at state.md/activity.md, never notes.md).
        self.write("notes.md", "# Working notes\n\n" + "\n".join(f"- detail {i}" for i in range(2000)))
        self.cli("index")
        r = self.cli("doctor")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Hot tier within char budget", r.out)

    def test_doctor_flags_missing_notes(self):
        (self.ctx / "notes.md").unlink()
        r = self.cli("doctor")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("MISSING required file: notes.md", r.out)

    def test_notes_is_searchable(self):
        self.write("notes.md", "# Working notes\n\n## Open threads\n- the flux capacitor needs calibration\n")
        self.cli("index")
        r = self.cli("search", "flux capacitor")
        self.assertIn("notes.md", r.out)


# --------------------------------------------------------------------------- 8. metrics (read-only invariant)

class TestMetrics(CelebornTestCase):

    def test_status_does_not_mutate_metrics(self):
        # Invariant from conventions: status/metrics are READ-ONLY; only record/hooks mutate.
        before = self.read("metrics.json")
        self.cli("status")
        self.cli("metrics")
        self.assertEqual(self.read("metrics.json"), before)

    def test_record_orient_credits_a_resume_once_per_session(self):
        # Give the project some non-Hot memory so there is something to "save".
        self.write("journal.md", "# Journal\n## e\n" + "x " * 500 + "\n")
        r1 = self.cli("record", "orient", "--session", "sess-A")
        m1 = json.loads(self.read("metrics.json"))
        self.assertEqual(m1["sessions_resumed"], 1)
        self.assertGreater(m1["tokens_saved_estimate"], 0)

        # A second orient in the SAME session must not double-count a resume.
        self.cli("record", "orient", "--session", "sess-A")
        m2 = json.loads(self.read("metrics.json"))
        self.assertEqual(m2["sessions_resumed"], 1)
        self.assertEqual(m2["orient_events"], 2)

        # A new session id credits another resume.
        self.cli("record", "orient", "--session", "sess-B")
        m3 = json.loads(self.read("metrics.json"))
        self.assertEqual(m3["sessions_resumed"], 2)

    def test_record_compaction_bridged(self):
        self.write("journal.md", "# Journal\n## e\n" + "y " * 500 + "\n")
        self.cli("record", "compaction")
        m = json.loads(self.read("metrics.json"))
        self.assertEqual(m["compactions_bridged"], 1)


# --------------------------------------------------------------------------- 9. PLAN §10 success criteria

class TestSuccessCriteria(CelebornTestCase):
    """The five acceptance criteria from PLAN.md §10, as executable tests."""

    def test_1_bounded_rehydration(self):
        # Hot-tier load cost stays ~constant as total memory grows. Measure Hot tokens with a
        # small journal, then with a large one; the Hot delta should be ~zero.
        cpt = cb.DEFAULTS["chars_per_token"]
        self.write("journal.md", "# Journal\n## e\nsmall\n")
        hot_small, total_small = cb._measure(self.ctx, cpt)
        self.write("journal.md", "# Journal\n## e\n" + ("lots of warm-tier history. " * 2000))
        hot_large, total_large = cb._measure(self.ctx, cpt)

        self.assertEqual(hot_small, hot_large, "Hot tier must not grow with warm-tier history")
        self.assertGreater(total_large, total_small * 5, "total memory did grow a lot")

    def test_2_targeted_recall_returns_snippet_not_whole_file(self):
        big = "# State\n" + "\n".join(
            f"## section {i}\nfiller filler filler {i}\n" for i in range(50)
        )
        big += "\n## buried decision\nWe picked Postgres for the ledger.\n"
        self.write("state.md", big)
        self.cli("index")
        r = self.cli("search", "Postgres ledger")
        self.assertIsNone(r.exit_code)
        self.assertIn("Postgres", r.out)
        self.assertIn("#buried-decision", r.out)
        # The snippet is far smaller than the whole file (recall is targeted, not a full load).
        self.assertLess(len(r.out), len(big) // 2)

    def test_3_compaction_immunity_resume_from_hot_alone(self):
        # Simulate a fresh thread: an agent that can read ONLY the Hot tier must still get the
        # focus + next action. `status` is exactly that Orient load.
        session = json.loads(self.read("session.json"))
        session["focus"] = "rebuild the auth flow"
        session["next_action"] = "wire the OAuth callback"
        self.write("session.json", json.dumps(session))
        self.write("state.md", "# State\n## Now\nFocus: rebuild the auth flow\n")
        r = self.cli("status")
        self.assertIsNone(r.exit_code)
        self.assertIn("rebuild the auth flow", r.out)
        self.assertIn("wire the OAuth callback", r.out)

    def test_4_forgetting_holds_the_line_without_loss(self):
        # Over a long run, archiving keeps journal.md under budget while nothing is lost.
        head = "# Journal\n\n"
        entries = "".join(f"## 2026-02-{i:02d} entry {i}\n- work {i}\n\n" for i in range(1, 41))
        self.write("journal.md", head + entries)
        self.cli("archive", "--keep", "20")

        _, kept = cb.split_journal(self.read("journal.md"))
        self.assertLessEqual(len(kept), 20, "journal held under budget")

        self.cli("index")
        # An old entry that left the Hot/Warm path is still recallable from cold storage.
        r = self.cli("search", "entry 2")
        self.assertIn("match", r.out)

    def test_5_degrades_gracefully_markdown_only(self):
        # Everything core must work with NO database present: status, archive, promote, handoff,
        # doctor all operate on markdown alone. (Search is the only DB-dependent verb.)
        self.assertFalse((self.ctx / "index.db").is_file())
        self.assertIsNone(self.cli("status").exit_code)
        self.assertIsNone(self.cli("handoff").exit_code)
        self.assertIsNone(self.cli("promote", "--to", "learnings",
                                   "--title", "x", "--note", "y").exit_code)
        self.write("journal.md", "# Journal\n" + "".join(
            f"## e{i}\nb\n\n" for i in range(25)))
        self.assertIsNone(self.cli("archive", "--keep", "20").exit_code)
        # doctor warns about the absent index but does not fail (it's a warning, not a problem).
        r = self.cli("doctor")
        self.assertIsNone(r.exit_code)


# --------------------------------------------------------------------------- 10. context discovery

class TestContextDiscovery(CelebornTestCase):

    def test_commands_walk_up_to_find_context(self):
        # Running from a nested subdirectory should still locate the project's .context/.
        nested = self.root / "src" / "deep"
        nested.mkdir(parents=True)
        r = run_cli("--path", str(nested), "status")
        self.assertIsNone(r.exit_code)
        self.assertIn("CELEBORN", r.out)

    def test_no_context_anywhere_errors_clearly(self):
        with tempfile.TemporaryDirectory() as empty:
            r = run_cli("--path", empty, "status")
            self.assertEqual(r.exit_code, 1)
            self.assertIn("No .context/", r.err)


class TestRemind(CelebornTestCase):

    def test_generic_when_no_tokens(self):
        r = self.cli("remind")
        self.assertIsNone(r.exit_code)
        self.assertIn("Celeborn", r.out)
        self.assertIn("re-explain", r.out)        # call-to-action line ("nothing to re-explain")
        self.assertNotIn("stale tokens", r.out)   # no count → no "Carrying ~Nk" weight

    def test_message_names_token_count(self):
        # The wording is uniform across milestones now (no escalating verse); it just names
        # the live token count (rounded to ~Nk) as the stale weight to shed.
        self.assertIn("~100,000 stale tokens", self.cli("remind", "--tokens", "100000").out)
        self.assertIn("~250,000 stale tokens", self.cli("remind", "--tokens", "250000").out)
        for tok in ("100000", "350000", "900000"):
            self.assertIn("nothing to re-explain", self.cli("remind", "--tokens", tok).out)

    def test_silent_within_same_increment(self):
        # 120k and 110k are both in the [100k,200k) band → no new milestone → stay silent.
        r = self.cli("remind", "--tokens", "120000", "--last", "110000")
        self.assertIsNone(r.exit_code)
        self.assertNotIn("Celeborn", r.out)
        self.assertEqual(r.out.strip(), "")

    def test_speaks_when_new_increment_crossed(self):
        r = self.cli("remind", "--tokens", "210000", "--last", "190000")  # band 1 -> 2
        self.assertIn("~210,000 stale tokens", r.out)

    def test_force_overrides_suppression(self):
        r = self.cli("remind", "--tokens", "120000", "--last", "110000", "--force")
        self.assertIn("Celeborn", r.out)

    def test_clear_cmd_is_shown(self):
        r = self.cli("remind", "--tokens", "100000", "--clear-cmd", "/clear")
        self.assertIn("/clear", r.out)

    def test_closer_rotation_is_deterministic_and_one_in_ten_wellness(self):
        # The sign-off cycles: exactly one wellness tidbit per 10 firings, the other 9 are flow words.
        # Pure function of the firing index — same n always yields the same closer (reproducible).
        flow = set(cb.REMIND_FLOW_CLOSERS)
        wellness = set(cb.REMIND_WELLNESS_CLOSERS)
        got = [cb._remind_closer(n) for n in range(100)]
        self.assertEqual(got, [cb._remind_closer(n) for n in range(100)])   # deterministic
        well_hits = [n for n in range(100) if got[n] in wellness]
        self.assertEqual(len(well_hits), 10)                                # 1 in 10 over 100
        self.assertEqual(well_hits, list(range(9, 100, 10)))                # at 9,19,29,...
        self.assertTrue(all(got[n] in flow for n in range(100) if n % 10 != 9))
        # The persisted counter advances the live closer once per firing.
        self.cli("remind", "--tokens", "100000")
        self.assertEqual(json.loads(self.read("metrics.json"))["remind_fire_count"], 1)

    def test_custom_increment_size(self):
        # With --every 50k, 60k is already past the first milestone → it speaks.
        self.assertIn("~60,000 stale tokens", self.cli("remind", "--tokens", "60000", "--every", "50000").out)

    def test_auto_uses_rolling_estimate(self):
        # `record turn` accumulates a context estimate; `remind --auto` reads it.
        self.cli("record", "turn", "--tokens", "60000")
        self.cli("record", "turn", "--tokens", "60000")  # 120k total → past the 100k band
        self.assertEqual(json.loads(self.read("metrics.json"))["context_estimate"], 120000)
        r = self.cli("remind", "--auto")
        self.assertIn("~120,000 stale tokens", r.out)
        # A second --auto in the same band stays silent (it recorded where it last spoke).
        self.assertNotIn("Celeborn", self.cli("remind", "--auto").out)

    def test_record_clear_resets_estimate(self):
        self.cli("record", "turn", "--tokens", "250000")
        self.cli("record", "clear")
        self.assertLess(json.loads(self.read("metrics.json"))["context_estimate"], 100_000)
        # Back below the first milestone → --auto stays silent.
        self.assertNotIn("Celeborn", self.cli("remind", "--auto").out)


# --------------------------------------------------------------------------- 12. sync (Phase 8b)

class TestSyncRedaction(unittest.TestCase):
    PATTERNS = ["ghp_[A-Za-z0-9]{36}", "xai-[A-Za-z0-9]{20,}"]

    def test_redacts_and_labels(self):
        text = "before ghp_" + "a" * 36 + " after"
        out, found = cs.redact(text, self.PATTERNS)
        self.assertNotIn("ghp_aaaa", out)
        self.assertIn("[REDACTED:github_pat]", out)
        self.assertEqual(found, ["github_pat"])

    def test_clean_text_untouched(self):
        out, found = cs.redact("nothing secret here", self.PATTERNS)
        self.assertEqual(out, "nothing secret here")
        self.assertEqual(found, [])

    def test_label_for_known_and_fallback(self):
        self.assertEqual(cs._label_for("AKIAEXAMPLE0000"), "aws_key")
        self.assertEqual(cs._label_for("xai-deadbeef"), "xai_key")
        self.assertEqual(cs._label_for("opaque-blob"), "secret")


class TestSyncConfig(unittest.TestCase):
    def test_env_overrides_and_strips_slash(self):
        os.environ["CELEBORN_SUPABASE_URL"] = "https://proj.supabase.co/"
        try:
            cfg = cs.sync_config(None)
            self.assertEqual(cfg["url"], "https://proj.supabase.co")
        finally:
            del os.environ["CELEBORN_SUPABASE_URL"]


class TestSyncCreds(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self._tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._old
        self._tmp.cleanup()

    def test_roundtrip_and_0600_perms(self):
        cs.save_creds({"github_token": "x", "access_token": "y"})
        self.assertEqual(cs.load_creds()["access_token"], "y")
        self.assertEqual(cs._creds_path().stat().st_mode & 0o777, 0o600)


class TestSyncPush(CelebornTestCase):
    def test_build_push_rows_redacts_and_filters(self):
        self.write("state.md", "leaked ghp_" + "b" * 36 + "\n")
        patterns = cb.load_config(self.ctx).get("secret_patterns", [])
        rows, redactions = cs.build_push_rows(self.ctx, "pid", patterns)
        paths = {r["path"] for r in rows}
        self.assertIn("state.md", paths)
        self.assertNotIn(".celebornrc", paths)               # rc never synced
        self.assertFalse(any(p.endswith("index.db") for p in paths))  # derived index never synced
        state = next(r for r in rows if r["path"] == "state.md")
        self.assertIn("[REDACTED:github_pat]", state["content"])
        self.assertNotIn("ghp_bbbb", state["content"])       # raw secret is gone from the upload
        self.assertGreaterEqual(redactions, 1)

    def test_board_files_excluded_from_raw_file_channel(self):
        # tasks.md / tasks.json sync ONLY through the structured table channel; the raw-file channel
        # must NOT carry them, or a stale file-pull would clobber a table-level reconcile.
        self.write("state.md", "headline\n")
        cb._tasks_path(self.ctx).write_text("## [t1] x\n- state: todo\n")
        cb._tasks_json_path(self.ctx).write_text("{}\n")
        rows, _ = cs.build_push_rows(self.ctx, "pid", [])
        names = {r["path"] for r in rows}
        self.assertIn("state.md", names)
        self.assertNotIn("tasks.md", names)
        self.assertNotIn("tasks.json", names)

    def test_push_403_exits_with_upgrade_code(self):
        # An unsubscribed user's push is refused; the CLI exits 2 (upgrade-required).
        orig = cs._http
        cs._http = lambda *a, **k: (403, {"error": "not_subscribed", "checkout_url": "u"})
        try:
            with self.assertRaises(SystemExit) as ctx:
                cs._push(self.ctx, {"url": "https://x", "anon": "a"}, "jwt", "pid",
                         cb.load_config(self.ctx).get("secret_patterns", []))
            self.assertEqual(ctx.exception.code, 2)
        finally:
            cs._http = orig

    def test_build_task_rows_projects_tasks_and_redacts(self):
        # The hosted `tasks` table (0006/4b) is the synced projection of tasks.md. One row per task,
        # keyed on (project_id, task_id), with title/notes/stop redacted like the file push.
        cb._tasks_path(self.ctx).write_text(
            "## [t1] Ship the thing\n- state: doing\n- owner: claude\n- tags: backend, infra\n"
            "- blocked-by: t2\n- phase: p1\n- stop: tests green\n"
            "- created: 2026-06-09T00:00:00\n- updated: 2026-06-10T00:00:00\n\n"
            "leaked ghp_" + "c" * 36 + "\n\n"
        )
        patterns = cb.load_config(self.ctx).get("secret_patterns", [])
        rows = cs.build_task_rows(self.ctx, "pid-9", patterns)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["project_id"], "pid-9")
        self.assertEqual(r["task_id"], "t1")
        self.assertEqual(r["state"], "doing")
        self.assertEqual(r["owner"], "claude")
        self.assertEqual(r["tags"], ["backend", "infra"])
        self.assertEqual(r["blocked_by"], ["t2"])
        self.assertEqual(r["stop"], "tests green")
        self.assertIn("[REDACTED:github_pat]", r["notes"])   # secret in card body never uploaded
        self.assertNotIn("ghp_cccc", r["notes"])

    def test_build_task_rows_no_board_is_noop(self):
        # No tasks.md → empty (callers skip push + prune so an empty board never wipes the table).
        self.assertEqual(cs.build_task_rows(self.ctx, "pid", []), [])

    def test_hosted_push_upserts_changed_and_deletes_gone(self):
        # The live push (spawned after a CLI mutation) upserts ONLY the requested ids that still exist
        # locally, and DELETEs a requested id that's gone locally (a card rm) so the hosted board drops
        # it. A card not named in --ids is never touched. Network + auth + project resolution mocked.
        import argparse as _ap
        cb._tasks_path(self.ctx).write_text(
            "## [t1] Alpha\n- state: doing\n- updated: 2026-06-10T00:00:00\n\n"
            "## [t2] Beta\n- state: todo\n- updated: 2026-06-10T00:00:00\n\n"
        )
        calls = []
        orig = (cs._http, cs._session_quiet, cs._ensure_project, cs.sync_config)
        cs._http = lambda method, url, headers=None, body=None, timeout=30: (
            calls.append((method, url, body)) or (204, None)
        )
        cs._session_quiet = lambda cfg: "jwt"
        cs._ensure_project = lambda ctx, cfg, jwt: "pid"
        cs.sync_config = lambda ctx: {"url": "https://x", "anon": "a"}
        try:
            cs.cmd_hosted_push(_ap.Namespace(path=str(self.root), ids="t1,t3"))
        finally:
            cs._http, cs._session_quiet, cs._ensure_project, cs.sync_config = orig
        # Filter to the /tasks calls — a card push ALSO refreshes the hosted active_agents (CELE-t131),
        # which adds its own POST/DELETE against /active_agents; this test only cares about the task rows.
        task_posts = [c for c in calls if c[0] == "POST" and "/rest/v1/tasks" in c[1]]
        task_dels = [c for c in calls if c[0] == "DELETE" and "/rest/v1/tasks" in c[1]]
        # t1 upserted (requested + present locally); t2 untouched (not requested).
        self.assertEqual(len(task_posts), 1)
        self.assertEqual({r["task_id"] for r in task_posts[0][2]}, {"t1"})
        # t3 deleted (requested but gone locally → drop the hosted row).
        self.assertEqual(len(task_dels), 1)
        self.assertIn("task_id=eq.t3", task_dels[0][1])

    def test_hosted_push_silent_when_unconfigured(self):
        # Not configured → no network, no raise (free / offline users pay nothing on every mutation).
        import argparse as _ap
        orig = (cs._http, cs.sync_config)
        hit = []
        cs._http = lambda *a, **k: (hit.append(1) or (204, None))
        cs.sync_config = lambda ctx: {"url": "https://REPLACE-with-your-project.supabase.co", "anon": "a"}
        try:
            cs.cmd_hosted_push(_ap.Namespace(path=str(self.root), ids="t1"))
        finally:
            cs._http, cs.sync_config = orig
        self.assertEqual(hit, [])

    def test_pull_tasks_reconciles_web_edit_into_tasks_md(self):
        # End-to-end of the gate (network mocked): a hosted state-change lands back in tasks.md.
        cb._tasks_path(self.ctx).write_text(
            "## [t1] Ship the thing\n- state: todo\n- owner: claude\n- tags: \n- blocked-by: \n"
            "- phase: \n- stop: tests green\n"
            "- created: 2026-06-09T00:00:00\n- updated: 2026-06-10T00:00:00\n\nbody\n\n"
        )
        # Remote row is newer (a web drag to done) and a web-created card.
        remote = [
            {"task_id": "t1", "title": "Ship the thing", "state": "done", "owner": "claude",
             "tags": [], "blocked_by": [], "phase": "", "stop": "tests green", "notes": "body",
             "created": "2026-06-09T00:00:00", "updated": "2026-06-11T00:00:00"},
            {"task_id": "t2", "title": "Made on the web", "state": "todo", "owner": "grok",
             "tags": [], "blocked_by": [], "phase": "", "stop": "", "notes": "",
             "created": "2026-06-11T00:00:00", "updated": "2026-06-11T00:00:00"},
        ]
        orig = cs._http
        cs._http = lambda method, url, **k: (200, remote)
        try:
            changed, conflicts = cs._pull_tasks(self.ctx, {"url": "https://x", "anon": "a"}, "jwt", "pid")
        finally:
            cs._http = orig
        self.assertEqual(changed, 2)                                    # t1 won + t2 adopted
        tasks = {t["id"]: t for t in cb._load_tasks(self.ctx)}
        self.assertEqual(tasks["t1"]["state"], "done")                  # web drag landed in tasks.md
        self.assertIn("t2", tasks)                                      # web-created card adopted
        self.assertEqual(tasks["t2"]["owner"], "grok")

    def test_pull_tasks_empty_both_sides_is_noop(self):
        # No local board AND no remote rows → no file is created (a "no board here" project stays empty).
        orig = cs._http
        cs._http = lambda *a, **k: (200, [])
        try:
            self.assertEqual(cs._pull_tasks(self.ctx, {"url": "https://x", "anon": "a"}, "jwt", "pid"),
                             (0, []))
        finally:
            cs._http = orig
        self.assertFalse(cb._tasks_path(self.ctx).is_file())

    def test_pull_tasks_materializes_board_on_fresh_device(self):
        # The board syncs only through the table now, so a fresh device (no local tasks.md) with a
        # non-empty hosted board must materialize it locally.
        self.assertFalse(cb._tasks_path(self.ctx).is_file())
        remote = [{"task_id": "t1", "title": "From the hub", "state": "doing", "owner": "claude",
                   "tags": [], "blocked_by": [], "phase": "", "stop": "", "notes": "",
                   "created": "2026-06-11T00:00:00", "updated": "2026-06-11T00:00:00"}]
        orig = cs._http
        cs._http = lambda *a, **k: (200, remote)
        try:
            changed, _ = cs._pull_tasks(self.ctx, {"url": "https://x", "anon": "a"}, "jwt", "pid")
        finally:
            cs._http = orig
        self.assertEqual(changed, 1)
        self.assertTrue(cb._tasks_path(self.ctx).is_file())
        tasks = {t["id"]: t for t in cb._load_tasks(self.ctx)}
        self.assertEqual(tasks["t1"]["state"], "doing")


class TestReconcile(unittest.TestCase):
    """Pure LWW reconcile of local tasks.md against the hosted `tasks` rows (t61 Phase 2). The whole
    write-back risk lives here, so it is tested in isolation before any web/network code (no IO)."""

    @staticmethod
    def _local(tid, updated, **kw):
        t = {"id": tid, "title": tid, "state": "todo", "owner": "", "tags": [], "blocked_by": [],
             "phase": "", "stop": "", "jira": "", "created": "", "updated": updated, "notes": ""}
        t.update(kw)
        return t

    @staticmethod
    def _remote(tid, updated, **kw):
        r = {"task_id": tid, "title": tid, "state": "todo", "owner": "", "tags": [], "blocked_by": [],
             "phase": "", "stop": "", "notes": "", "created": "", "updated": updated,
             # board-only enrichment that must be dropped on the way back into tasks.md:
             "display_id": "CELE-" + tid, "owner_family": "Claude", "owner_model": "Opus 4.8"}
        r.update(kw)
        return r

    def test_only_local_is_kept(self):
        # A card never pushed (or pruned remotely) must survive a pull — a pull never deletes local.
        local = [self._local("t1", "2026-06-10T00:00:00")]
        merged, conflicts = cs.reconcile_tasks(local, [])
        self.assertEqual(merged, local)
        self.assertEqual(conflicts, [])

    def test_only_remote_is_adopted(self):
        # A card created on the hosted board is adopted into tasks.md, mapped to the local shape.
        remote = [self._remote("t2", "2026-06-11T00:00:00", state="doing", owner="grok")]
        merged, conflicts = cs.reconcile_tasks([], remote)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["id"], "t2")
        self.assertEqual(merged[0]["state"], "doing")
        self.assertEqual(merged[0]["owner"], "grok")
        self.assertNotIn("display_id", merged[0])      # board-only enrichment dropped
        self.assertNotIn("task_id", merged[0])          # mapped to the local `id` key
        self.assertEqual([c["id"] for c in conflicts], ["t2"])
        self.assertEqual(conflicts[0]["changed"], ["*new*"])

    def test_remote_newer_wins_whole_task(self):
        # A web drag (newer `updated`) flips state on pull and is surfaced as a conflict.
        local = [self._local("t1", "2026-06-10T00:00:00", state="todo")]
        remote = [self._remote("t1", "2026-06-11T00:00:00", state="done")]
        merged, conflicts = cs.reconcile_tasks(local, remote)
        self.assertEqual(merged[0]["state"], "done")
        self.assertEqual([c["id"] for c in conflicts], ["t1"])
        self.assertIn("state", conflicts[0]["changed"])

    def test_local_newer_wins_no_conflict(self):
        local = [self._local("t1", "2026-06-12T00:00:00", state="doing")]
        remote = [self._remote("t1", "2026-06-11T00:00:00", state="done")]
        merged, conflicts = cs.reconcile_tasks(local, remote)
        self.assertEqual(merged[0]["state"], "doing")    # local wins
        self.assertEqual(conflicts, [])

    def test_tie_local_wins(self):
        # Equal `updated` → tasks.md is canonical; local wins and nothing is surfaced.
        local = [self._local("t1", "2026-06-11T00:00:00", state="doing")]
        remote = [self._remote("t1", "2026-06-11T00:00:00", state="done")]
        merged, conflicts = cs.reconcile_tasks(local, remote)
        self.assertEqual(merged[0]["state"], "doing")
        self.assertEqual(conflicts, [])

    def test_missing_timestamp_sorts_oldest(self):
        # Empty `updated` loses; never crashes. Remote stamped, local blank → remote wins.
        local = [self._local("t1", "", state="todo")]
        remote = [self._remote("t1", "2026-06-11T00:00:00", state="done")]
        merged, _ = cs.reconcile_tasks(local, remote)
        self.assertEqual(merged[0]["state"], "done")
        # Reverse: local stamped, remote blank → local wins (blank remote sorts oldest).
        local = [self._local("t1", "2026-06-11T00:00:00", state="doing")]
        remote = [self._remote("t1", "", state="done")]
        merged, _ = cs.reconcile_tasks(local, remote)
        self.assertEqual(merged[0]["state"], "doing")

    def test_redacted_title_coming_back_keeps_local_text(self):
        # The sharp edge: remote is newer ONLY because of a state drag, but carries a redacted title.
        # The merge must take the remote `state` WITHOUT clobbering the real local title/notes/stop.
        local = [self._local("t1", "2026-06-10T00:00:00", state="todo",
                              title="Rotate ghp_realsecret key", notes="real body", stop="real stop")]
        remote = [self._remote("t1", "2026-06-11T00:00:00", state="done",
                               title="[REDACTED:github_pat]", notes="[REDACTED:github_pat]",
                               stop="[REDACTED:github_pat]")]
        merged, conflicts = cs.reconcile_tasks(local, remote)
        self.assertEqual(merged[0]["state"], "done")             # state drag applied
        self.assertEqual(merged[0]["title"], "Rotate ghp_realsecret key")  # local text preserved
        self.assertEqual(merged[0]["notes"], "real body")
        self.assertEqual(merged[0]["stop"], "real stop")
        self.assertIn("state", conflicts[0]["changed"])
        self.assertNotIn("title", conflicts[0]["changed"])       # title did not change (kept local)

    def test_postgrest_timestamptz_roundtrip_is_a_tie(self):
        # The sharp edge: the hosted column is timestamptz, so PostgREST returns '…+00:00' while
        # tasks.md stores tz-naive '…'. A clean push→pull MUST be a tie (local wins), not a phantom
        # remote-win from the longer string. Also tolerate fractional seconds.
        local = [self._local("t1", "2026-06-10T00:00:00", state="doing")]
        for remote_ts in ("2026-06-10T00:00:00+00:00", "2026-06-10T00:00:00.482931+00:00",
                           "2026-06-10 00:00:00+00"):
            remote = [self._remote("t1", remote_ts, state="done")]
            merged, conflicts = cs.reconcile_tasks(local, remote)
            self.assertEqual(merged[0]["state"], "doing", f"round-trip {remote_ts!r} should tie→local")
            self.assertEqual(conflicts, [], f"round-trip {remote_ts!r} must not surface a conflict")
        # And a genuine later web edit (tz-suffixed) still wins and is normalized clean into tasks.md.
        remote = [self._remote("t1", "2026-06-11T09:30:00+00:00", state="done")]
        merged, _ = cs.reconcile_tasks(local, remote)
        self.assertEqual(merged[0]["state"], "done")
        self.assertEqual(merged[0]["updated"], "2026-06-11T09:30:00")   # tz stripped for tasks.md

    def test_new_card_from_web_alongside_existing(self):
        # A pull must adopt a web-created card without disturbing existing local cards or their order.
        local = [self._local("t1", "2026-06-10T00:00:00")]
        remote = [self._remote("t1", "2026-06-10T00:00:00"),
                  self._remote("t9", "2026-06-11T00:00:00", state="todo")]
        merged, conflicts = cs.reconcile_tasks(local, remote)
        self.assertEqual([t["id"] for t in merged], ["t1", "t9"])
        self.assertEqual([c["id"] for c in conflicts], ["t9"])

    def test_archived_card_is_not_re_adopted(self):
        # A done card aged off the board (in done-archive.md) still exists on the hub. Without the
        # archive tombstone it would be re-adopted on every sync (and re-archived → endless bounce).
        local = [self._local("t1", "2026-06-12T00:00:00")]
        remote = [self._remote("t1", "2026-06-12T00:00:00"),
                  self._remote("t5", "2026-06-09T00:00:00", state="done")]  # archived, lingering on hub
        merged, conflicts = cs.reconcile_tasks(local, remote, archived_ids={"t5"})
        self.assertEqual([t["id"] for t in merged], ["t1"])     # t5 NOT resurrected
        self.assertEqual(conflicts, [])
        # …and without the tombstone it WOULD be adopted (guards against the param silently no-op'ing).
        merged2, _ = cs.reconcile_tasks(local, remote)
        self.assertIn("t5", [t["id"] for t in merged2])


class TestSyncMetrics(CelebornTestCase):
    def test_push_metrics_sends_cumulative_counters(self):
        m = cb._load_metrics(self.ctx)
        m["tokens_saved_estimate"] = 12345
        m["sessions_resumed"] = 2
        cb._save_metrics(self.ctx, m)
        captured = {}
        orig = cs._http

        def fake(method, url, headers=None, body=None, **k):
            captured.update(method=method, url=url, body=body)
            return (204, None)
        cs._http = fake
        try:
            cs._push_metrics(self.ctx, {"url": "https://x", "anon": "a"}, "jwt", "pid-123")
        finally:
            cs._http = orig
        self.assertEqual(captured["method"], "PATCH")
        self.assertIn("projects?id=eq.pid-123", captured["url"])
        self.assertEqual(captured["body"]["tokens_saved"], 12345)   # cumulative, not a delta
        self.assertEqual(captured["body"]["sessions_resumed"], 2)
        self.assertIn("metrics_updated_at", captured["body"])

    def test_fetch_user_total_parses_row_and_handles_empty(self):
        orig = cs._http
        cs._http = lambda *a, **k: (200, [{"tokens_saved": 80000, "projects": 2}])
        try:
            t = cs._fetch_user_total({"url": "https://x", "anon": "a"}, "jwt")
            self.assertEqual(t["tokens_saved"], 80000)
            cs._http = lambda *a, **k: (200, [])
            self.assertIsNone(cs._fetch_user_total({"url": "https://x", "anon": "a"}, "jwt"))
        finally:
            cs._http = orig


class TestGithubLink(CelebornTestCase):
    """`celeborn github link` binds a repo to the hosted project via the gh-link Edge Function (which
    enforces ownership server-side); `celeborn sync` then pulls captured GitHub threads into
    journal.md using a per-device cursor (so every linked member's local .context/ receives them)."""

    def setUp(self):
        super().setUp()
        self._http, self._sess, self._proj, self._fn = (
            cs._http, cs._ensure_session, cs._ensure_project, cs._fn)
        os.environ["CELEBORN_SUPABASE_URL"] = "https://x"
        os.environ["CELEBORN_SUPABASE_ANON_KEY"] = "anon"
        cs._ensure_session = lambda cfg: "jwt"
        cs._ensure_project = lambda ctx, cfg, jwt: "pid-1"

    def tearDown(self):
        cs._http, cs._ensure_session, cs._ensure_project, cs._fn = (
            self._http, self._sess, self._proj, self._fn)
        os.environ.pop("CELEBORN_SUPABASE_URL", None)
        os.environ.pop("CELEBORN_SUPABASE_ANON_KEY", None)
        super().tearDown()

    def _args(self, **kw):
        base = dict(path=str(self.root), repo="octo/myrepo", installation="12345")
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_link_success_writes_rc(self):
        captured = {}
        cs._fn = lambda cfg, name, jwt, body: (captured.update(name=name, body=body) or (200, {"ok": True}))
        cs.cmd_github_link(self._args())
        self.assertEqual(captured["name"], "gh-link")
        self.assertEqual(captured["body"]["repo_full_name"], "octo/myrepo")
        self.assertEqual(captured["body"]["installation_id"], 12345)        # coerced to int
        self.assertEqual(captured["body"]["project_id"], "pid-1")
        rc = json.loads((self.ctx / cb.RC_NAME).read_text())
        self.assertEqual(rc["sync"]["github_repo"], "octo/myrepo")

    def test_link_bad_repo_format_exits(self):
        with self.assertRaises(SystemExit):
            cs.cmd_github_link(self._args(repo="not-a-repo"))

    def test_link_requires_installation(self):
        with self.assertRaises(SystemExit):
            cs.cmd_github_link(self._args(installation=None))

    def test_link_refused_exits_2(self):
        cs._fn = lambda *a, **k: (403, {"error": "not_project_owner"})
        with self.assertRaises(SystemExit) as ctx:
            cs.cmd_github_link(self._args())
        self.assertEqual(ctx.exception.code, 2)

    def test_pull_ingested_appends_and_advances_cursor(self):
        self.write("journal.md", "# Journal\n")
        rows = [
            {"gh_event": "issue_comment", "author_login": "alice", "body": "first thread",
             "source_url": "https://github.com/o/r/issues/1#c1", "occurred_at": "2026-06-15T10:00:00Z",
             "created_at": "2026-06-15T10:00:01Z"},
            {"gh_event": "pull_request_review", "author_login": "bob", "body": "second",
             "source_url": "https://github.com/o/r/pull/2", "occurred_at": "2026-06-15T11:00:00Z",
             "created_at": "2026-06-15T11:00:02Z"},
        ]
        cs._http = lambda method, url, **k: (200, rows)
        n = cs._pull_ingested(self.ctx, {"url": "https://x", "anon": "a"}, "jwt", "pid-1")
        self.assertEqual(n, 2)
        journal = (self.ctx / "journal.md").read_text()
        self.assertIn("ingested issue_comment by alice", journal)
        self.assertIn("first thread", journal)
        self.assertIn("https://github.com/o/r/pull/2", journal)
        rc = json.loads((self.ctx / cb.RC_NAME).read_text())
        self.assertEqual(rc["sync"]["ingested_cursor"], "2026-06-15T11:00:02Z")  # high-water advanced

    def test_pull_ingested_uses_cursor_in_query(self):
        captured = {}
        cs._http = lambda method, url, **k: (captured.update(url=url) or (200, []))
        cs._set_rc_value(self.ctx, "ingested_cursor", "2026-06-14T00:00:00Z")
        cfg = cs.sync_config(self.ctx)
        cs._pull_ingested(self.ctx, cfg, "jwt", "pid-1")
        self.assertIn("created_at=gt.2026-06-14T00:00:00Z", captured["url"])

    def test_pull_ingested_empty_is_noop(self):
        cs._http = lambda *a, **k: (200, [])
        self.assertEqual(cs._pull_ingested(self.ctx, {"url": "https://x", "anon": "a"}, "jwt", "pid-1"), 0)


class TestProjectRemove(CelebornTestCase):
    """`celeborn project list` / `rm` manage hosted projects directly over PostgREST (t97). Removal needs
    no local .context/ for the target — an orphaned project whose repo was deleted is still removable —
    and the FK cascade clears its tasks/files/links server-side. The destructive path is guarded by a
    typed-name confirm unless --yes; deleting the project this repo points at clears the stale rc id."""

    PID_A = "11111111-1111-1111-1111-111111111111"
    PID_B = "22222222-2222-2222-2222-222222222222"

    def setUp(self):
        super().setUp()
        self._http, self._cfg, self._sess = cs._http, cs._resolve_cfg, cs._ensure_session
        cs._resolve_cfg = lambda args: {"url": "https://x", "anon": "a"}
        cs._ensure_session = lambda cfg: "jwt"
        self.projects = [
            {"id": self.PID_A, "name": "celeborn", "created_at": "t"},
            {"id": self.PID_B, "name": "testrepoforscotch-ingest", "created_at": "t"},
        ]
        self.calls = []

        def fake_http(method, url, headers=None, body=None, timeout=30):
            self.calls.append((method, url))
            if method == "GET":
                return (200, list(self.projects))
            if method == "DELETE":
                return (204, None)
            return (200, None)

        cs._http = fake_http

    def tearDown(self):
        cs._http, cs._resolve_cfg, cs._ensure_session = self._http, self._cfg, self._sess
        super().tearDown()

    def _args(self, **kw):
        base = dict(path=str(self.root), project_cmd=None, name=None, yes=False)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def _run(self, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cs.cmd_project(self._args(**kw))
        return buf.getvalue()

    def _deletes(self):
        return [u for m, u in self.calls if m == "DELETE"]

    def test_list_prints_projects_never_deletes(self):
        out = self._run(project_cmd="list")
        self.assertIn("celeborn", out)
        self.assertIn("testrepoforscotch-ingest", out)
        self.assertIn("2 hosted project(s)", out)
        self.assertEqual(self._deletes(), [])

    def test_rm_by_name_deletes_right_id(self):
        self._run(project_cmd="rm", name="testrepoforscotch-ingest", yes=True)
        self.assertEqual(len(self._deletes()), 1)
        self.assertIn(f"id=eq.{self.PID_B}", self._deletes()[0])

    def test_rm_by_id_deletes(self):
        self._run(project_cmd="rm", name=self.PID_A, yes=True)
        self.assertEqual(len(self._deletes()), 1)
        self.assertIn(f"id=eq.{self.PID_A}", self._deletes()[0])

    def test_rm_unknown_exits_without_delete(self):
        with self.assertRaises(SystemExit):
            self._run(project_cmd="rm", name="ghost", yes=True)
        self.assertEqual(self._deletes(), [])

    def test_rm_ambiguous_name_exits(self):
        self.projects.append({"id": "33333333-3333-3333-3333-333333333333",
                              "name": "celeborn", "created_at": "t"})
        with self.assertRaises(SystemExit):
            self._run(project_cmd="rm", name="celeborn", yes=True)
        self.assertEqual(self._deletes(), [])

    def test_rm_confirm_typed_name_proceeds(self):
        with mock.patch("builtins.input", lambda *a: "testrepoforscotch-ingest"):
            self._run(project_cmd="rm", name="testrepoforscotch-ingest", yes=False)
        self.assertEqual(len(self._deletes()), 1)

    def test_rm_confirm_mismatch_aborts(self):
        with mock.patch("builtins.input", lambda *a: "nope"):
            with self.assertRaises(SystemExit):
                self._run(project_cmd="rm", name="testrepoforscotch-ingest", yes=False)
        self.assertEqual(self._deletes(), [])

    def test_rm_clears_stale_local_project_id(self):
        cs._set_rc_value(self.ctx, "project_id", self.PID_B)
        self._run(project_cmd="rm", name="testrepoforscotch-ingest", yes=True)
        rc = json.loads((self.ctx / cb.RC_NAME).read_text())
        self.assertIsNone(rc["sync"].get("project_id"))

    def test_rm_keeps_unrelated_local_project_id(self):
        other = "99999999-9999-9999-9999-999999999999"
        cs._set_rc_value(self.ctx, "project_id", other)
        self._run(project_cmd="rm", name="testrepoforscotch-ingest", yes=True)
        rc = json.loads((self.ctx / cb.RC_NAME).read_text())
        self.assertEqual(rc["sync"]["project_id"], other)


class TestSyncLogin(unittest.TestCase):
    """Supabase Auth (GoTrue) login/register. Identity is email+password (+TOTP) or GitHub OAuth;
    the GitHub device flow is retired. _resolve_cfg is stubbed so commands skip the REPLACE guard."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self._tmp.name
        self._http, self._cfg, self._tier = cs._http, cs._resolve_cfg, cs._tier_line
        cs._resolve_cfg = lambda a: {"url": "https://x", "anon": "anon"}
        cs._tier_line = lambda *a, **k: None  # avoid the entitlements network call

    def tearDown(self):
        cs._http, cs._resolve_cfg, cs._tier_line = self._http, self._cfg, self._tier
        if self._old is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._old
        self._tmp.cleanup()

    def test_login_success_stores_session(self):
        cs._http = lambda method, url, **k: (
            200, {"access_token": "JWT", "refresh_token": "RT", "expires_in": 3600,
                  "user": {"email": "frodo@shire.me", "id": "u1", "factors": []}})
        cs.cmd_login(types.SimpleNamespace(path=".", email="frodo@shire.me", password="pw", github=False))
        creds = cs.load_creds()
        self.assertEqual(creds["access_token"], "JWT")
        self.assertEqual(creds["refresh_token"], "RT")
        self.assertEqual(creds["email"], "frodo@shire.me")
        self.assertNotIn("github_token", creds)  # device-flow token is gone

    def test_login_bad_credentials_exits_2(self):
        cs._http = lambda method, url, **k: (400, {"error_description": "Invalid login credentials"})
        with self.assertRaises(SystemExit) as ctx:
            cs.cmd_login(types.SimpleNamespace(path=".", email="x@y.z", password="bad", github=False))
        self.assertEqual(ctx.exception.code, 2)

    def test_register_pending_confirmation_does_not_store_session(self):
        # Signup with email-confirmation on returns no session; we should NOT store one.
        cs._http = lambda method, url, **k: (200, {"id": "u1", "email": "sam@shire.me"})
        cs.cmd_register(types.SimpleNamespace(
            path=".", email="sam@shire.me", username="sam", password="pw"))
        self.assertEqual(cs.load_creds(), {})

    def test_ensure_session_refreshes_with_refresh_token(self):
        cs.save_creds({"refresh_token": "RT", "access_token": "old", "expires_at": 0})
        cs._http = lambda method, url, **k: (
            200, {"access_token": "NEW", "refresh_token": "RT2", "expires_in": 3600})
        tok = cs._ensure_session({"url": "https://x", "anon": "anon"})
        self.assertEqual(tok, "NEW")
        self.assertEqual(cs.load_creds()["refresh_token"], "RT2")

    # --- billing: upgrade (create-checkout) + billing (portal) ----------------------------------
    def _no_browser(self):
        """Stub webbrowser.open so command tests never pop a real browser. Returns the opened-URL dict."""
        import webbrowser
        opened = {}
        orig = webbrowser.open
        webbrowser.open = lambda u, *a, **k: (opened.__setitem__("url", u), True)[1]
        self.addCleanup(lambda: setattr(webbrowser, "open", orig))
        return opened

    def test_upgrade_calls_checkout_and_opens_url(self):
        opened = self._no_browser()
        cs.save_creds({"access_token": "JWT", "expires_at": 9999999999})  # avoid a refresh round-trip
        seen = {}
        cs._http = lambda method, url, **k: (
            seen.update(method=method, url=url, body=k.get("body")) or (200, {"url": "https://checkout.test"}))
        cs.cmd_upgrade(types.SimpleNamespace(path=".", tier="team", annual=True, seats=3))
        self.assertIn("/functions/v1/create-checkout", seen["url"])
        self.assertEqual(seen["body"], {"tier": "team", "interval": "year", "seats": 3})
        self.assertEqual(opened["url"], "https://checkout.test")

    def test_upgrade_defaults_pro_monthly_single_seat(self):
        self._no_browser()
        cs.save_creds({"access_token": "JWT", "expires_at": 9999999999})
        seen = {}
        cs._http = lambda method, url, **k: (seen.update(body=k.get("body")) or (200, {"url": "u"}))
        cs.cmd_upgrade(types.SimpleNamespace(path=".", tier="pro", annual=False, seats=1))
        self.assertEqual(seen["body"], {"tier": "pro", "interval": "month", "seats": 1})

    def test_billing_opens_portal_url(self):
        opened = self._no_browser()
        cs.save_creds({"access_token": "JWT", "expires_at": 9999999999})
        cs._http = lambda method, url, **k: (200, {"url": "https://portal.test"})
        cs.cmd_billing(types.SimpleNamespace(path="."))
        self.assertEqual(opened["url"], "https://portal.test")

    def test_billing_without_subscription_exits_2(self):
        cs.save_creds({"access_token": "JWT", "expires_at": 9999999999})
        cs._http = lambda method, url, **k: (409, {"error": "no_subscription"})
        with self.assertRaises(SystemExit) as ctx:
            cs.cmd_billing(types.SimpleNamespace(path="."))
        self.assertEqual(ctx.exception.code, 2)


class TestIdentitySplit(unittest.TestCase):
    """CELE-t107: the email-vs-GitHub identity split — provider detection, the whoami/sync warning, and
    `celeborn account migrate` (dual-token reassign). _resolve_cfg/_tier_line stubbed like TestSyncLogin."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self._tmp.name
        self._http, self._cfg, self._tier = cs._http, cs._resolve_cfg, cs._tier_line
        cs._resolve_cfg = lambda a: {"url": "https://x", "anon": "anon"}
        cs._tier_line = lambda *a, **k: None

    def tearDown(self):
        cs._http, cs._resolve_cfg, cs._tier_line = self._http, self._cfg, self._tier
        os.environ.pop("CELEBORN_PASSWORD", None)
        if self._old is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._old
        self._tmp.cleanup()

    # --- provider detection (unit) --------------------------------------------------------------
    def test_provider_of_prefers_app_metadata(self):
        self.assertEqual(cs._provider_of({"app_metadata": {"provider": "github"}}), "github")

    def test_provider_of_falls_back_to_identities_then_email(self):
        self.assertEqual(cs._provider_of({"identities": [{"provider": "github"}]}), "github")
        self.assertEqual(cs._provider_of({}), "email")

    # --- the warning surface --------------------------------------------------------------------
    def test_warn_identity_split_silent_for_github(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cs._warn_identity_split("github")
        self.assertEqual(out.getvalue().strip(), "")

    def test_warn_identity_split_warns_for_email(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cs._warn_identity_split("email")
        self.assertIn("celeborn login --github", out.getvalue())
        self.assertIn("celeborn account migrate", out.getvalue())

    def test_whoami_shows_provider_and_warns_on_email(self):
        cs.save_creds({"access_token": "JWT", "expires_at": 9999999999})
        cs._http = lambda method, url, **k: (
            200, {"email": "frodo@shire.me", "id": "u1", "factors": [],
                  "app_metadata": {"provider": "email"}})
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cs.cmd_whoami(types.SimpleNamespace(path="."))
        self.assertIn("sign-in:  email", out.getvalue())
        self.assertIn("account migrate", out.getvalue())

    # --- migrate flow ---------------------------------------------------------------------------
    def _route(self, keeper_user, migrate_result, source_user=None):
        """Stub cs._http to answer the three calls cmd_account_migrate makes, by URL."""
        def http(method, url, **k):
            if "/auth/v1/user" in url:
                return (200, keeper_user)
            if "/auth/v1/token" in url:
                return (200, {"access_token": "SRC_JWT", "user": source_user or {"id": "old", "email": "frodo@shire.me"}})
            if "/functions/v1/account-migrate" in url:
                return migrate_result
            raise AssertionError(f"unexpected call: {method} {url}")
        cs._http = http

    def test_account_migrate_moves_projects(self):
        cs.save_creds({"access_token": "KEEP_JWT", "expires_at": 9999999999})
        os.environ["CELEBORN_PASSWORD"] = "oldpw"
        self._route(
            keeper_user={"id": "ghuid", "email": "frodo@shire.me", "app_metadata": {"provider": "github"}},
            migrate_result=(200, {"moved": 2, "keeper": {"id": "ghuid"}, "source": {"id": "old"}}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cs.cmd_account_migrate(types.SimpleNamespace(path=".", email="frodo@shire.me", yes=False))
        self.assertIn("moved 2 project(s)", out.getvalue())

    def test_account_migrate_keeper_session_not_clobbered_by_source_login(self):
        cs.save_creds({"access_token": "KEEP_JWT", "refresh_token": "KEEP_RT", "expires_at": 9999999999})
        os.environ["CELEBORN_PASSWORD"] = "oldpw"
        self._route(
            keeper_user={"id": "ghuid", "email": "frodo@shire.me", "app_metadata": {"provider": "github"}},
            migrate_result=(200, {"moved": 1}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cs.cmd_account_migrate(types.SimpleNamespace(path=".", email="frodo@shire.me", yes=False))
        # signing in to the OLD account must NOT overwrite the keeper credentials on disk
        self.assertEqual(cs.load_creds()["access_token"], "KEEP_JWT")

    def test_account_migrate_same_identity_exits(self):
        cs.save_creds({"access_token": "KEEP_JWT", "expires_at": 9999999999})
        os.environ["CELEBORN_PASSWORD"] = "oldpw"
        self._route(
            keeper_user={"id": "ghuid", "email": "frodo@shire.me", "app_metadata": {"provider": "github"}},
            migrate_result=(400, {"error": "same_identity"}))
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cs.cmd_account_migrate(types.SimpleNamespace(path=".", email="frodo@shire.me", yes=False))


# --------------------------------------------------------------------------- 13. context visibility

class TestTranscriptEstimate(unittest.TestCase):
    def _write(self, objs) -> Path:
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for o in objs:
            f.write(json.dumps(o) + "\n")
        f.close()
        return Path(f.name)

    def test_uses_latest_usage_record(self):
        p = self._write([
            {"message": {"role": "assistant", "usage": {"input_tokens": 100, "output_tokens": 10}}},
            {"message": {"role": "assistant", "usage": {
                "input_tokens": 50000, "cache_read_input_tokens": 120000, "output_tokens": 2000}}},
        ])
        try:
            self.assertEqual(cb._estimate_transcript_tokens(p, 4), 172000)  # last turn's window
        finally:
            p.unlink()

    def test_char_fallback_without_usage(self):
        p = self._write([{"message": {"role": "user", "content": "x" * 400}}])
        try:
            self.assertEqual(cb._estimate_transcript_tokens(p, 4), 100)  # 400 chars / 4
        finally:
            p.unlink()

    def test_missing_file_is_zero(self):
        self.assertEqual(cb._estimate_transcript_tokens(Path("/no/such/file.jsonl"), 4), 0)


class TestRemindTranscript(CelebornTestCase):
    def _tx(self, total: int) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        f.write(json.dumps({"message": {"usage": {"input_tokens": total}}}) + "\n")
        f.close()
        return f.name

    def test_over_soft_limit_is_urgent_and_persists(self):
        tp = self._tx(200000)
        try:
            out = self.cli("remind", "--transcript", tp, "--soft-limit", "150000", "--every", "50000").out
            self.assertIn("~200,000 stale tokens", out)        # uniform line; names the live weight
            self.assertIn("nothing to re-explain", out)
            self.assertEqual(json.loads(self.read("metrics.json"))["context_estimate"], 200000)
        finally:
            os.unlink(tp)

    def test_once_per_band_then_silent(self):
        tp = self._tx(120000)
        try:
            self.assertIn("Celeborn", self.cli("remind", "--transcript", tp, "--every", "50000").out)
            self.assertEqual(self.cli("remind", "--transcript", tp, "--every", "50000").out.strip(), "")
        finally:
            os.unlink(tp)


# --------------------------------------------------------------------------- 14. automatic capture

class TestCapture(CelebornTestCase):
    def _transcript(self, *entries) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for e in entries:
            f.write(json.dumps(e) + "\n")
        f.close()
        return f.name

    def _prompt(self, sid, text, ts="2026-06-02T10:00:00Z", uuid="u1"):
        return {"type": "user", "uuid": uuid, "sessionId": sid, "timestamp": ts,
                "message": {"role": "user", "content": text}}

    def _assist(self, sid, blocks, uuid="a1"):
        return {"type": "assistant", "uuid": uuid, "sessionId": sid,
                "message": {"role": "assistant", "content": blocks}}

    def _result(self, sid, tuid, out, uuid="r1"):
        return {"type": "user", "uuid": uuid, "sessionId": sid,
                "toolUseResult": {"stdout": out, "stderr": ""},
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tuid, "content": out}]}}

    def _auto_files(self):
        d = self.ctx / "auto"
        return sorted(d.glob("*.md")) if d.is_dir() else []

    def _auto_text(self):
        return "\n".join(p.read_text() for p in self._auto_files())

    def test_extracts_files_commands_commits(self):
        tp = self._transcript(
            self._prompt("sessA", "do the thing"),
            self._assist("sessA", [
                {"type": "tool_use", "id": "t1", "name": "Edit", "input": {"file_path": "app/x.py"}},
                {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "git commit -m feat"}},
            ]),
            self._result("sessA", "t2", "[main a1b2c3d] feat"),
        )
        try:
            r = self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertIsNone(r.exit_code, r.all)
            txt = self._auto_text()
            self.assertIn("app/x.py", txt)
            self.assertIn("git commit", txt)
            self.assertIn("a1b2c3d", txt)
        finally:
            os.unlink(tp)

    def test_test_signals(self):
        tp = self._transcript(
            self._prompt("sessA", "run tests"),
            self._assist("sessA", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "python -m unittest"}}]),
            self._result("sessA", "t1", "Ran 12 tests\n12 passed"),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertIn("(pass)", self._auto_text())
        finally:
            os.unlink(tp)

    def test_cursor_idempotency(self):
        tp = self._transcript(self._prompt("sessA", "hello"))
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            n1 = self._auto_text().count("## turn")
            r2 = self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertIn("no new entries", r2.all)
            self.assertEqual(self._auto_text().count("## turn"), n1)
        finally:
            os.unlink(tp)

    def test_incremental_capture_advances_offset(self):
        e1 = self._prompt("sessA", "first")
        tp = self._transcript(e1)
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            off1 = json.loads(self.read("metrics.json"))["capture"]["offset"]
            with open(tp, "a") as f:
                f.write(json.dumps(self._prompt("sessA", "second", uuid="u2")) + "\n")
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertEqual(self._auto_text().count("## turn"), 2)
            self.assertGreater(json.loads(self.read("metrics.json"))["capture"]["offset"], off1)
        finally:
            os.unlink(tp)

    def test_redacts_secret_in_prompt(self):
        secret = "ghp_" + "a" * 36
        tp = self._transcript(self._prompt("sessA", f"use {secret} now"))
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertIn("[REDACTED:", self._auto_text())
            self.assertNotIn(secret, self._auto_text())
            self.assertNotIn(secret, (self.ctx / "activity.md").read_text())
        finally:
            os.unlink(tp)

    def test_new_session_starts_new_file(self):
        a = self._transcript(self._prompt("sessA", "a"))
        b = self._transcript(self._prompt("sessB", "b", uuid="u9"))
        try:
            self.cli("capture", "--transcript", a, "--session", "sessA")
            self.cli("capture", "--transcript", b, "--session", "sessB")
            self.assertEqual(len(self._auto_files()), 2)
            self.assertEqual(json.loads(self.read("metrics.json"))["capture"]["session_id"], "sessB")
        finally:
            os.unlink(a)
            os.unlink(b)

    def test_interleaved_sessions_keep_independent_cursors(self):
        # The bug: a single shared cursor (e.g. the global ~/.context sink) let one session reset
        # another's offset, so re-capturing the first re-read its whole transcript from byte 0 —
        # re-recording old turns and resetting its running total every turn. Per-session cursors fix it.
        a = self._transcript(self._prompt("sessA", "alpha one"))
        b = self._transcript(self._prompt("sessB", "beta one", uuid="ub"))
        try:
            self.cli("capture", "--transcript", a, "--session", "sessA")
            self.cli("capture", "--transcript", b, "--session", "sessB")     # interleave a 2nd session
            with open(a, "a") as f:                                          # sessA takes another turn
                f.write(json.dumps(self._prompt("sessA", "alpha two", uuid="ua2")) + "\n")
            self.cli("capture", "--transcript", a, "--session", "sessA")
            # sessA's first turn was NOT re-read (its cursor survived sessB) → recorded exactly once.
            self.assertEqual(self._auto_text().count("alpha one"), 1)
            self.assertEqual(self._auto_text().count("alpha two"), 1)
            m = json.loads(self.read("metrics.json"))
            self.assertIn("sessA", m["captures"])
            self.assertIn("sessB", m["captures"])     # both cursors coexist
        finally:
            os.unlink(a)
            os.unlink(b)

    def test_heartbeat_reads_the_named_session_not_the_last(self):
        a = self._transcript(self._prompt("sessA", "a short turn"))
        b = self._transcript(self._prompt("sessB", "a much, much longer turn " * 12, uuid="ub"))
        try:
            self.cli("capture", "--transcript", a, "--session", "sessA", "--quiet")
            self.cli("capture", "--transcript", b, "--session", "sessB", "--quiet")   # sessB captured last
            ta = json.loads(self.read("metrics.json"))["captures"]["sessA"]["tokens_session"]
            tb = json.loads(self.read("metrics.json"))["captures"]["sessB"]["tokens_session"]
            self.assertGreater(tb, ta)
            # Without --session the heartbeat shows the most-recent session (sessB)...
            self.assertIn(f"{tb:,} tokens recorded", self.cli("heartbeat").out)
            # ...but --session sessA reports sessA's own total, not whichever captured last.
            self.assertIn(f"{ta:,} tokens recorded", self.cli("heartbeat", "--session", "sessA").out)
        finally:
            os.unlink(a)
            os.unlink(b)

    def test_skips_meta_sidechain_snapshot(self):
        tp = self._transcript(
            {"type": "user", "uuid": "m1", "sessionId": "sessA", "isMeta": True, "message": {"role": "user", "content": "meta"}},
            {"type": "file-history-snapshot", "uuid": "s1", "sessionId": "sessA", "snapshot": {}},
            {"type": "user", "uuid": "sc", "sessionId": "sessA", "isSidechain": True, "message": {"role": "user", "content": "side"}},
        )
        try:
            r = self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertIn("no new entries", r.all)
            self.assertEqual(self._auto_files(), [])           # no turns -> no files created
        finally:
            os.unlink(tp)

    def test_activity_digest_regenerated_not_appended(self):
        tp = self._transcript(
            self._prompt("sessA", "one"),
            self._assist("sessA", [{"type": "tool_use", "id": "t1", "name": "Write", "input": {"file_path": "a.py"}}]),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            ap = self.ctx / "activity.md"
            self.assertTrue(ap.is_file())
            self.assertIn("Recently touched files", ap.read_text())
            self.assertLessEqual(len(ap.read_text().splitlines()), 40)
            with open(tp, "a") as f:
                f.write(json.dumps(self._prompt("sessA", "two", uuid="u2")) + "\n")
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertEqual(ap.read_text().count("# Automatic Context Record"), 1)  # overwritten
        finally:
            os.unlink(tp)

    def test_capture_does_not_touch_authored_tiers(self):
        before_state = self.read("state.md")
        before_session = self.read("session.json")
        tp = self._transcript(self._prompt("sessA", "hi"))
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertEqual(self.read("state.md"), before_state)
            self.assertEqual(self.read("session.json"), before_session)
        finally:
            os.unlink(tp)

    def test_truncated_last_line_tolerated(self):
        tp = self._transcript(self._prompt("sessA", "good"))
        with open(tp, "a") as f:
            f.write('{"type": "user", "message": {bad json\n')
        try:
            r = self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertIsNone(r.exit_code, r.all)
            self.assertIn("good", self._auto_text())
        finally:
            os.unlink(tp)

    def test_gitignore_auto_tier_public_and_idempotent(self):
        root = Path(tempfile.mkdtemp())
        try:
            run_cli("--path", str(root), "init", "--public")
            gi = (root / ".gitignore").read_text()
            self.assertIn(".context/auto/", gi)
            self.assertIn(".context/activity.md", gi)
            run_cli("--path", str(root), "init", "--public")  # idempotent
            self.assertEqual((root / ".gitignore").read_text().count(".context/auto/\n"), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_indexed_and_searchable(self):
        tp = self._transcript(
            self._prompt("sessA", "work"),
            self._assist("sessA", [{"type": "tool_use", "id": "t1", "name": "Edit", "input": {"file_path": "uniquefile_zztop.py"}}]),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.cli("index")
            out = self.cli("search", "uniquefile_zztop").out
            self.assertIn("uniquefile_zztop", out)
        finally:
            os.unlink(tp)

    def test_status_shows_activity(self):
        tp = self._transcript(self._prompt("sessA", "hey"))
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertIn("activity.md", self.cli("status").out)
        finally:
            os.unlink(tp)

    # --- faithful capture (the cold auto file is now near-complete, not a lossy digest) ----------

    def test_captures_assistant_text_and_non_bash_tool(self):
        tp = self._transcript(
            self._prompt("sessA", "look around"),
            self._assist("sessA", [
                {"type": "text", "text": "Scanning the codebase now."},
                {"type": "tool_use", "id": "g1", "name": "Grep", "input": {"pattern": "needle_xyz", "path": "src/"}},
            ]),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            txt = self._auto_text()
            self.assertIn("Scanning the codebase now.", txt)  # assistant text — previously dropped
            self.assertIn("Grep", txt)                        # non-Bash tool — previously dropped
            self.assertIn("needle_xyz", txt)
        finally:
            os.unlink(tp)

    def test_captures_tool_result_output(self):
        tp = self._transcript(
            self._prompt("sessA", "build"),
            self._assist("sessA", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "make"}}]),
            self._result("sessA", "t1", "BUILD_OK_marker_42"),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertIn("BUILD_OK_marker_42", self._auto_text())  # output body — previously dropped
        finally:
            os.unlink(tp)

    def test_full_bash_command_body_captured(self):
        tp = self._transcript(
            self._prompt("sessA", "multi"),
            self._assist("sessA", [{"type": "tool_use", "id": "t1", "name": "Bash",
                                    "input": {"command": "echo one\necho TWO_marker"}}]),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            self.assertIn("TWO_marker", self._auto_text())  # 2nd line — previously truncated off
        finally:
            os.unlink(tp)

    def test_faithful_redaction_in_result(self):
        secret = "ghp_" + "b" * 36
        tp = self._transcript(
            self._prompt("sessA", "go"),
            self._assist("sessA", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "printenv"}}]),
            self._result("sessA", "t1", f"TOKEN={secret}"),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            txt = self._auto_text()
            self.assertNotIn(secret, txt)
            self.assertIn("[REDACTED:", txt)
        finally:
            os.unlink(tp)

    def test_output_size_cap_redacts_then_truncates(self):
        self.write(".celebornrc", json.dumps({"capture_output_max_chars": 50}))
        big = "X" * 500
        tp = self._transcript(
            self._prompt("sessA", "big"),
            self._assist("sessA", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "dump"}}]),
            self._result("sessA", "t1", big),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            txt = self._auto_text()
            self.assertIn("…[truncated", txt)
            self.assertNotIn("X" * 200, txt)  # full body not persisted
        finally:
            os.unlink(tp)

    def test_window_json_is_facts_only(self):
        tp = self._transcript(
            self._prompt("sessA", "w"),
            self._assist("sessA", [{"type": "text", "text": "hello"},
                                   {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            win = json.loads((self.ctx / "auto" / "window.json").read_text())
            self.assertTrue(win)
            self.assertNotIn("events", win[0])   # the faithful stream stays out of the bounded window
            self.assertIn("prompt", win[0])
            self.assertIn("commands", win[0])
        finally:
            os.unlink(tp)

    def test_activity_bounded_despite_large_output(self):
        big = "BIGOUTPUT " * 500
        tp = self._transcript(
            self._prompt("sessA", "huge"),
            self._assist("sessA", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "dump"}}]),
            self._result("sessA", "t1", big),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")
            act = (self.ctx / "activity.md").read_text()
            self.assertLessEqual(len(act.splitlines()), 40)        # Hot tier stays bounded
            self.assertNotIn("BIGOUTPUT BIGOUTPUT", act)            # big body never reaches the digest
        finally:
            os.unlink(tp)

    # --- per-turn `--note` systemMessage (always speaks: tokens, or "nothing material") ----------

    def test_note_reports_recorded_tokens(self):
        tp = self._transcript(
            self._prompt("sessA", "do a reasonably long thing so the recorded tokens are clearly > 0"),
            self._assist("sessA", [{"type": "text", "text": "working on it now"}]),
        )
        try:
            r = self.cli("capture", "--transcript", tp, "--session", "sessA", "--quiet", "--note")
            msg = json.loads(r.out.strip())["systemMessage"]
            self.assertTrue(msg.startswith("🏹 Celeborn —> +"), msg)
            self.assertIn("tokens this turn", msg)
            self.assertIn("this session", msg)            # running total present
        finally:
            os.unlink(tp)

    def test_note_reports_idle_when_no_new_entries(self):
        tp = self._transcript(self._prompt("sessA", "hello"))
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")          # consume the turn
            r = self.cli("capture", "--transcript", tp, "--session", "sessA", "--quiet", "--note")  # nothing new
            msg = json.loads(r.out.strip())["systemMessage"]
            self.assertIn("idle ×", msg)
            self.assertIn("this session", msg)
        finally:
            os.unlink(tp)

    def test_note_stays_unique_across_consecutive_idle_turns(self):
        # Claude Code drops a Stop systemMessage identical to the one before it, so consecutive idle
        # turns MUST produce distinct strings or the heartbeat goes invisible.
        tp = self._transcript(self._prompt("sessA", "hello"))
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessA")          # consume the turn
            seen = set()
            for _ in range(3):
                r = self.cli("capture", "--transcript", tp, "--session", "sessA", "--quiet", "--note")
                seen.add(json.loads(r.out.strip())["systemMessage"])
            self.assertEqual(len(seen), 3, f"idle notes must all differ, got {seen}")
        finally:
            os.unlink(tp)

    def test_note_session_total_accumulates(self):
        # Two active turns: the running session total in the second note must exceed the first.
        def total(msg):
            return int(msg.split("·")[-1].strip().split()[0].replace(",", ""))
        tp1 = self._transcript(
            self._prompt("sessB", "first"),
            self._assist("sessB", [{"type": "text", "text": "alpha " * 40}]),
        )
        try:
            r1 = self.cli("capture", "--transcript", tp1, "--session", "sessB", "--quiet", "--note")
            t1 = total(json.loads(r1.out.strip())["systemMessage"])
        finally:
            os.unlink(tp1)
        tp2 = self._transcript(
            self._prompt("sessB", "first"),
            self._assist("sessB", [{"type": "text", "text": "alpha " * 40}]),
            self._prompt("sessB", "second"),
            self._assist("sessB", [{"type": "text", "text": "beta " * 40}]),
        )
        try:
            r2 = self.cli("capture", "--transcript", tp2, "--session", "sessB", "--quiet", "--note")
            t2 = total(json.loads(r2.out.strip())["systemMessage"])
        finally:
            os.unlink(tp2)
        self.assertGreater(t2, t1, "session total should accumulate across turns")

    # --- heartbeat: the UserPromptSubmit-surfaced, app-visible per-turn line ---------------------

    def test_heartbeat_silent_before_any_capture(self):
        self.assertEqual(self.cli("heartbeat").out.strip(), "")     # nothing banked yet → no line

    def test_heartbeat_reports_session_total_after_capture(self):
        tp = self._transcript(
            self._prompt("sessH", "do work"),
            self._assist("sessH", [{"type": "text", "text": "alpha " * 40}]),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessH", "--quiet")
            msg = self.cli("heartbeat").out.strip()
            self.assertTrue(msg.startswith("🏹 Celeborn —> "), msg)
            self.assertIn("tokens recorded this session", msg)
            self.assertIn("last turn", msg)                          # delta > 0 on the captured turn
        finally:
            os.unlink(tp)

    def test_heartbeat_drops_last_turn_suffix_when_idle(self):
        tp = self._transcript(
            self._prompt("sessH", "do work"),
            self._assist("sessH", [{"type": "text", "text": "alpha " * 40}]),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sessH", "--quiet")  # delta > 0
            self.cli("capture", "--transcript", tp, "--session", "sessH", "--quiet")  # idle → delta 0
            msg = self.cli("heartbeat").out.strip()
            self.assertIn("tokens recorded this session", msg)
            self.assertNotIn("last turn", msg)
        finally:
            os.unlink(tp)

    # --- statusline: the persistent, unsuppressable UI-chrome line -------------------------------

    def test_statusline_always_prints_minimal_line(self):
        # statusLine output replaces the default line, so it must always emit something.
        self.assertEqual(self.cli("statusline").out.strip(), "🏹 Celeborn —>")

    def test_statusline_shows_banked_after_capture(self):
        tp = self._transcript(
            self._prompt("sS", "work"),
            self._assist("sS", [{"type": "text", "text": "alpha " * 40}]),
        )
        try:
            self.cli("capture", "--transcript", tp, "--session", "sS", "--quiet")
            msg = self.cli("statusline").out.strip()
            self.assertTrue(msg.startswith("🏹 Celeborn —>"), msg)
            self.assertIn("tokens recorded", msg)
        finally:
            os.unlink(tp)

    def test_statusline_adds_live_context_with_transcript(self):
        tp = self._transcript(
            self._prompt("sS", "work"),
            self._assist("sS", [{"type": "text", "text": "alpha " * 80}]),
        )
        try:
            msg = self.cli("statusline", "--transcript", tp).out.strip()
            self.assertIn("ctx ~", msg)
        finally:
            os.unlink(tp)

    # --- universal capture (hybrid: repo .context/ else global ~/.context) ----------------------

    def _use_tmp_home(self) -> Path:
        home = tempfile.mkdtemp()
        old = os.environ.get("HOME")
        self.addCleanup(lambda: os.environ.__setitem__("HOME", old) if old is not None
                        else os.environ.pop("HOME", None))
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        os.environ["HOME"] = home
        return Path(home)

    def test_global_fallback_when_no_repo_context(self):
        home = self._use_tmp_home()
        nodir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, nodir, ignore_errors=True)
        tp = self._transcript(self._prompt("sg", "global hello"))
        try:
            r = run_cli("--path", nodir, "capture", "--transcript", tp, "--session", "sg")
            self.assertIsNone(r.exit_code, r.all)
            auto = list((home / ".context" / "auto").glob("*.md"))
            self.assertTrue(auto, "global ~/.context/auto should be created")
            self.assertIn("global hello", "\n".join(p.read_text() for p in auto))
            cur = json.loads((home / ".context" / "metrics.json").read_text())["capture"]
            self.assertEqual(cur["session_id"], "sg")
        finally:
            os.unlink(tp)

    def test_global_scaffold_is_minimal(self):
        home = self._use_tmp_home()
        nodir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, nodir, ignore_errors=True)
        tp = self._transcript(self._prompt("sg3", "x"))
        try:
            run_cli("--path", nodir, "capture", "--transcript", tp, "--session", "sg3", "--global")
            gctx = home / ".context"
            self.assertTrue((gctx / "auto").is_dir())
            self.assertTrue((gctx / "metrics.json").is_file())
            for authored in ("state.md", "session.json", "journal.md", "handoff.md"):
                self.assertFalse((gctx / authored).exists(), f"{authored} should not be scaffolded")
        finally:
            os.unlink(tp)

    def test_global_flag_forces_global_inside_repo(self):
        home = self._use_tmp_home()
        tp = self._transcript(self._prompt("sg2", "forced global"))
        try:
            r = self.cli("capture", "--transcript", tp, "--session", "sg2", "--global")
            self.assertIsNone(r.exit_code, r.all)
            ghome = "\n".join(p.read_text() for p in (home / ".context" / "auto").glob("*.md"))
            self.assertIn("forced global", ghome)
            self.assertEqual(self._auto_files(), [])  # the repo's own auto tier is untouched
        finally:
            os.unlink(tp)


class TestGlobalSyncIdentity(unittest.TestCase):
    """The global ~/.context record gets a stable sync identity ('global') instead of the home-dir name."""

    def test_sync_config_exposes_project_name(self):
        import celeborn_sync as cs
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        ctx = Path(d) / ".context"
        ctx.mkdir()
        (ctx / ".celebornrc").write_text(json.dumps({"sync": {"project_name": "global"}}))
        self.assertEqual(cs.sync_config(ctx)["project_name"], "global")

    def test_ensure_project_prefers_rc_name_else_parent(self):
        import celeborn_sync as cs
        captured = {}

        def fake_http(method, url, **k):
            captured.clear()
            captured.update(k.get("body") or {})
            return 201, [{"id": "pid1"}]

        orig = (cs._http, cs._rest_headers, cs._set_rc_value)
        cs._http = fake_http
        cs._rest_headers = lambda *a, **k: {}
        cs._set_rc_value = lambda *a, **k: None
        self.addCleanup(lambda: setattr(cs, "_http", orig[0]))
        self.addCleanup(lambda: setattr(cs, "_rest_headers", orig[1]))
        self.addCleanup(lambda: setattr(cs, "_set_rc_value", orig[2]))

        cs._ensure_project(Path("/tmp/anything/.context"), {"url": "http://x", "project_name": "global"}, "jwt")
        self.assertEqual(captured.get("name"), "global")
        cs._ensure_project(Path("/tmp/myrepo/.context"), {"url": "http://x"}, "jwt")
        self.assertEqual(captured.get("name"), "myrepo")


class TestVersion(unittest.TestCase):
    """`celeborn version` and the GitHub update check. Network is always monkeypatched."""

    def setUp(self):
        self._head, self._fetch, self._lv = cb._git_head, cb._fetch_url, cb._local_version
        self.addCleanup(self._restore)

    def _restore(self):
        cb._git_head, cb._fetch_url, cb._local_version = self._head, self._fetch, self._lv

    def test_plain_version_is_offline(self):
        def boom(*a, **k):
            raise AssertionError("plain `version` must not touch the network")
        cb._fetch_url = boom
        r = run_cli("version")
        self.assertIn("Celeborn", r.out)

    def test_check_up_to_date_git(self):
        cb._git_head = lambda root: "a" * 40
        cb._fetch_url = lambda url, **k: json.dumps({"sha": "a" * 40})
        r = run_cli("version", "--check")
        self.assertIn("up to date", r.all)

    def test_check_behind_git_reports_and_suggests_pull(self):
        def fake(url, **k):
            if "/commits/main" in url:
                return json.dumps({"sha": "b" * 40})
            if "/compare/" in url:
                return json.dumps({"status": "behind", "ahead_by": 3})
            return "{}"
        cb._git_head = lambda root: "a" * 40
        cb._fetch_url = fake
        r = run_cli("version", "--check")
        self.assertIn("newer Celeborn", r.all)
        self.assertIn("3 commit(s) behind", r.all)
        self.assertIn("git -C", r.all)  # the update command

    def test_check_offline_is_graceful(self):
        import urllib.error
        cb._git_head = lambda root: "a" * 40

        def boom(url, **k):
            raise urllib.error.URLError("no network")
        cb._fetch_url = boom
        r = run_cli("version", "--check")
        self.assertIsNone(r.exit_code)              # never crashes
        self.assertIn("update check skipped", r.all)

    def test_check_pip_install_compares_version(self):
        cb._git_head = lambda root: None            # non-git install
        cb._local_version = lambda: "0.1.0"
        cb._fetch_url = lambda url, **k: 'version = "0.2.0"\n'
        r = run_cli("version", "--check")
        self.assertIn("0.2.0", r.all)
        self.assertIn("newer Celeborn", r.all)


class TestNoAlertWindows(CelebornTestCase):
    """t62 (and t47/t50 before it): native OS alert/dialog windows were repeatedly flagged as annoying
    and have been removed entirely. The reassurance/heartbeat now rides text channels only (stdout +
    hook context). This class is the regression guard that the GUI-modal subsystem stays gone."""

    def test_alert_subsystem_is_gone(self):
        # The helper, the CLI verb's handler, and the icon renderer must not exist.
        for name in ("_gui_alert", "cmd_alert", "_ensure_bow_icon", "_BOW_JXA"):
            self.assertFalse(hasattr(cb, name), f"{name} should have been removed")

    def test_alert_verb_not_registered(self):
        r = self.cli("alert", "boom")
        self.assertEqual(r.exit_code, 2)            # argparse: no such command
        self.assertIn("invalid choice", r.all)

    def test_remind_has_no_alarm_flags(self):
        r = self.cli("remind", "--tokens", "250000", "--alarm-limit", "200000")
        self.assertEqual(r.exit_code, 2)            # --alarm-limit no longer exists
        # The plain reminder path still works, with no dialog side effect.
        ok = self.cli("remind", "--tokens", "250000")
        self.assertIsNone(ok.exit_code, ok.all)


# --------------------------------------------------------------------------- tasks (Phase 11)

class TestTasksParsing(unittest.TestCase):
    """Unit tests for the tasks.md <-> dict round trip — the part most likely to regress silently."""

    def test_parse_minimal(self):
        tasks = cb._parse_tasks(
            "# Tasks\n\n## [t1] Do the thing\n- state: doing\n- owner: claude\n"
            "- tags: a, b\n- blocked-by: t2 t3\n- created: X\n- updated: Y\n\nsome notes\n")
        self.assertEqual(len(tasks), 1)
        t = tasks[0]
        self.assertEqual(t["id"], "t1")
        self.assertEqual(t["title"], "Do the thing")
        self.assertEqual(t["state"], "doing")
        self.assertEqual(t["owner"], "claude")
        self.assertEqual(t["tags"], ["a", "b"])
        self.assertEqual(t["blocked_by"], ["t2", "t3"])
        self.assertEqual(t["notes"], "some notes")

    def test_parse_defaults_state_todo(self):
        # A heading with no `- state:` line defaults to todo, not a crash.
        t = cb._parse_tasks("## [t9] bare\n")[0]
        self.assertEqual(t["state"], "todo")
        self.assertEqual(t["tags"], [])
        self.assertEqual(t["blocked_by"], [])

    def test_render_parse_roundtrip(self):
        tasks = [{
            "id": "t1", "title": "Round trip", "state": "doing", "owner": "x",
            "tags": ["ui", "phase-11"], "blocked_by": ["t2"], "phase": "p11",
            "stop": "All tabs render and the e2e test passes",
            "progress": 0, "engine_floor": 0, "jira": "SCRUM-2",
            "created": "C", "updated": "U", "subtasks": [], "notes": "line one\nline two",
        }]
        reparsed = cb._parse_tasks(cb._render_tasks(tasks))
        self.assertEqual(reparsed, tasks)

    def test_parse_stop_field(self):
        t = cb._parse_tasks(
            "## [t1] Card\n- state: doing\n- stop: tests green and committed\n")[0]
        self.assertEqual(t["stop"], "tests green and committed")

    def test_parse_missing_stop_defaults_empty(self):
        # Legacy cards predating the field parse with an empty stop, not a crash.
        t = cb._parse_tasks("## [t1] Legacy\n- state: todo\n")[0]
        self.assertEqual(t["stop"], "")

    def test_render_idempotent_with_stop(self):
        # render → parse → render must be stable (no double-written `- stop:` line).
        once = cb._render_tasks(cb._parse_tasks("## [t1] X\n- state: todo\n- stop: ship it\n"))
        twice = cb._render_tasks(cb._parse_tasks(once))
        self.assertEqual(once, twice)
        self.assertEqual(once.count("- stop:"), 1)

    def test_next_id_monotonic(self):
        tasks = [{"id": "t1"}, {"id": "t7"}, {"id": "weird"}]
        self.assertEqual(cb._next_task_id(tasks), "t8")
        self.assertEqual(cb._next_task_id([]), "t1")

    def test_parse_note_headings_do_not_split_cards(self):
        """`##` section headings inside a card's notes must not become orphan id-less cards."""
        md = (
            "# Tasks\n\n"
            "## [t1] Parent card\n- state: todo\n- owner: \n- tags: \n- blocked-by: \n"
            "- phase: \n- stop: ship it\n- created: C\n- updated: U\n\n"
            "# CELE-t1 — spec\n\n"
            "## Mission\n\nDo the thing.\n\n"
            "## Architecture\n\nSplit control/data planes.\n\n"
            "## [t2] Real second card\n- state: todo\n- owner: \n- tags: \n- blocked-by: \n"
            "- phase: \n- stop: \n- created: C\n- updated: U\n\n"
        )
        tasks = cb._parse_tasks(md)
        self.assertEqual([t["id"] for t in tasks], ["t1", "t2"])
        self.assertIn("## Mission", tasks[0]["notes"])
        self.assertIn("## Architecture", tasks[0]["notes"])

    def test_parse_metadata_only_at_block_head(self):
        """`- state:` lines inside notes must not overwrite the card's metadata fields."""
        md = (
            "## [t1] Parent\n- state: doing\n- stop: real stop\n- created: C\n- updated: U\n\n"
            "notes start\n\n"
            "- state: todo\n- stop: fake\n- created: X\n"
        )
        t = cb._parse_tasks(md)[0]
        self.assertEqual(t["state"], "doing")
        self.assertEqual(t["stop"], "real stop")
        self.assertEqual(t["created"], "C")
        self.assertIn("- state: todo", t["notes"])

    def test_render_parse_roundtrip_preserves_note_headings(self):
        notes = "## Mission\n\nGoal.\n\n## Build order\n\n1. First\n"
        tasks = [{
            "id": "t9", "title": "Spec card", "state": "todo", "owner": "",
            "tags": [], "blocked_by": [], "phase": "", "stop": "done",
            "progress": 0, "jira": "", "created": "C", "updated": "U",
            "subtasks": [], "notes": notes,
        }]
        reparsed = cb._parse_tasks(cb._render_tasks(tasks))
        self.assertEqual(len(reparsed), 1)
        self.assertEqual(reparsed[0]["notes"].strip(), notes.strip())

    def test_save_rejects_idless_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = Path(d) / ".context"
            ctx.mkdir()
            cb._tasks_path(ctx).write_text(cb.TASKS_HEADER)
            with self.assertRaises(SystemExit):
                cb._save_tasks(ctx, [{"id": "", "title": "[] [] Mission", "state": "todo",
                                      "owner": "", "tags": [], "blocked_by": [], "phase": "",
                                      "stop": "", "progress": 0, "jira": "",
                                      "created": "", "updated": "", "subtasks": [], "notes": ""}])

    def test_relative_context_path_derives_repo_slug_not_proj(self):
        """Path('.context') must resolve before slug derivation — never fall back to PROJ-tN."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "celeborn"
            ctx = root / ".context"
            ctx.mkdir(parents=True)
            (ctx / ".celebornrc").write_text("{}")
            orig = os.getcwd()
            try:
                os.chdir(root)
                self.assertEqual(cb.project_slug(Path(".context")), "cele")
                doc = cb._tasks_doc(ctx.resolve(), [])
                self.assertEqual(doc["id_prefix"], "CELE")
            finally:
                os.chdir(orig)


class TestTasksCommands(CelebornTestCase):
    """End-to-end command tests driving the real argparse entrypoint."""

    def test_add_creates_files_and_assigns_ids(self):
        r = self.cli("tasks", "add", "First task")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("t1]", r.out)   # slug-agnostic: qualified `[SLUG-t1]` is now the default
        self.cli("tasks", "add", "Second task")
        self.assertTrue((self.ctx / "tasks.md").is_file())
        self.assertTrue((self.ctx / "tasks.json").is_file())
        doc = json.loads(self.read("tasks.json"))
        # IDs are assigned in creation order (t1, t2); the newest card renders at the TOP of its
        # column (_bring_to_state_front), so the list order is newest-first within the todo group.
        self.assertEqual({t["id"] for t in doc["tasks"]}, {"t1", "t2"})
        self.assertEqual([t["id"] for t in doc["tasks"]], ["t2", "t1"])
        self.assertEqual(doc["states"], ["todo", "doing", "done"])

    def test_add_with_metadata(self):
        self.cli("tasks", "add", "Meta", "--state", "doing", "--owner", "claude",
                 "--tags", "a,b", "--blocked-by", "t9", "--note", "hello")
        t = json.loads(self.read("tasks.json"))["tasks"][0]
        self.assertEqual(t["state"], "doing")
        self.assertEqual(t["owner"], "claude")
        self.assertEqual(t["tags"], ["a", "b"])
        self.assertEqual(t["blocked_by"], ["t9"])
        self.assertEqual(t["notes"], "hello")

    def test_add_with_explicit_stop(self):
        self.cli("tasks", "add", "Stoppable", "--stop", "PR merged and deployed")
        t = json.loads(self.read("tasks.json"))["tasks"][0]
        self.assertEqual(t["stop"], "PR merged and deployed")

    def test_add_auto_fills_default_stop(self):
        # No --stop → the generic default is auto-filled so no card is ever stop-less.
        self.cli("tasks", "add", "No stop given")
        t = json.loads(self.read("tasks.json"))["tasks"][0]
        self.assertEqual(t["stop"], cb.DEFAULT_STOP)
        self.assertTrue(t["stop"])

    def test_progress_add_edit_clamp_and_roundtrip(self):
        # add with --progress, edit it, clamp out-of-range, and confirm tasks.md round-trips (CELE-t106).
        self.cli("tasks", "add", "Bar card", "--state", "doing", "--progress", "40")
        t = json.loads(self.read("tasks.json"))["tasks"][0]
        self.assertEqual(t["id"], "t1")
        self.assertEqual(t["progress"], 40)
        self.assertIn("- progress: 40", self.read("tasks.md"))
        # edit + clamp above 100. The raw clamp pins to 100, but this is a DOING card and the 100%=Done
        # invariant (CELE-t131) caps an unshipped card at 99 — only a card in Done ever reads 100.
        self.cli("tasks", "edit", "t1", "--progress", "150")
        self.assertEqual(json.loads(self.read("tasks.json"))["tasks"][0]["progress"], 99)
        # clamp below 0
        self.cli("tasks", "edit", "t1", "--progress", "-5")
        self.assertEqual(json.loads(self.read("tasks.json"))["tasks"][0]["progress"], 0)

    def test_subtasks_set_check_derive_weighted_progress(self):
        # set a weighted checklist, check items → progress is the weighted done fraction (CELE-t106).
        self.cli("tasks", "add", "Feature", "--state", "doing")
        self.cli("tasks", "subtasks", "t1", "set", "design", "build*2", "ship")  # weights 1,2,1 = total 4
        t = json.loads(self.read("tasks.json"))["tasks"][0]
        self.assertEqual([s["text"] for s in t["subtasks"]], ["design", "build", "ship"])
        self.assertEqual([s["weight"] for s in t["subtasks"]], [1, 2, 1])
        self.assertEqual(t["progress"], 0)
        self.cli("tasks", "check", "t1", "2")  # check 'build' (weight 2) → 2/4 = 50%
        self.assertEqual(json.loads(self.read("tasks.json"))["tasks"][0]["progress"], 50)
        self.cli("tasks", "check", "t1", "1")  # + 'design' (weight 1) → 3/4 = 75%
        self.assertEqual(json.loads(self.read("tasks.json"))["tasks"][0]["progress"], 75)
        self.cli("tasks", "uncheck", "t1", "2")  # uncheck 'build' → 1/4 = 25%
        self.assertEqual(json.loads(self.read("tasks.json"))["tasks"][0]["progress"], 25)

    def test_subtasks_roundtrip_through_tasks_md(self):
        self.cli("tasks", "add", "RT", "--state", "doing")
        self.cli("tasks", "subtasks", "t1", "set", "a*3", "b")
        self.cli("tasks", "check", "t1", "1")
        # the `### Subtasks` checkbox block is the on-disk form; re-parsing must preserve it
        md = self.read("tasks.md")
        self.assertIn("### Subtasks", md)
        self.assertIn("- [x] a ×3", md)
        self.assertIn("- [ ] b", md)
        reparsed = cb._parse_tasks(md)[0]
        self.assertEqual(len(reparsed["subtasks"]), 2)
        self.assertTrue(reparsed["subtasks"][0]["done"])
        self.assertEqual(reparsed["subtasks"][0]["weight"], 3)
        self.assertEqual(reparsed["progress"], 75)  # 3/4 derived on parse

    def test_check_out_of_range_and_no_subtasks_error(self):
        self.cli("tasks", "add", "E", "--state", "doing")
        self.assertIsNotNone(self.cli("tasks", "check", "t1", "1").exit_code)   # no subtasks yet
        self.cli("tasks", "subtasks", "t1", "add", "only one")
        self.assertIsNotNone(self.cli("tasks", "check", "t1", "5").exit_code)   # out of range

    def test_progress_defaults_zero_and_omitted_from_md(self):
        # a card with no --progress defaults to 0 and writes NO progress line (legacy cards stay byte-identical).
        self.cli("tasks", "add", "Plain")
        self.assertEqual(json.loads(self.read("tasks.json"))["tasks"][0]["progress"], 0)
        self.assertNotIn("progress:", self.read("tasks.md"))

    def test_edit_updates_stop(self):
        self.cli("tasks", "add", "Editable")
        self.cli("tasks", "edit", "t1", "--stop", "real condition here")
        t = json.loads(self.read("tasks.json"))["tasks"][0]
        self.assertEqual(t["stop"], "real condition here")

    def test_move_updates_state_and_json(self):
        self.cli("tasks", "add", "Movable")
        r = self.cli("tasks", "move", "t1", "done")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(json.loads(self.read("tasks.json"))["tasks"][0]["state"], "done")

    def test_move_unknown_id_errors(self):
        r = self.cli("tasks", "move", "t99", "done")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("no task", r.all)

    def test_move_invalid_state_rejected(self):
        self.cli("tasks", "add", "x")
        r = self.cli("tasks", "move", "t1", "wip")  # not a valid state
        self.assertEqual(r.exit_code, 2)  # argparse choices error

    def _done_order(self) -> list[str]:
        return [t["id"] for t in json.loads(self.read("tasks.json"))["tasks"] if t["state"] == "done"]

    def test_completing_a_task_puts_it_atop_the_done_column(self):
        for n in ("First", "Second", "Third"):
            self.cli("tasks", "add", n)
        self.cli("tasks", "move", "t1", "done")
        self.cli("tasks", "move", "t2", "done")
        self.cli("tasks", "move", "t3", "done")
        # newest-done lands on top; older done cards get pushed down.
        self.assertEqual(self._done_order(), ["t3", "t2", "t1"])

    def test_edit_into_done_also_brings_to_front(self):
        self.cli("tasks", "add", "A")
        self.cli("tasks", "add", "B")
        self.cli("tasks", "move", "t1", "done")            # done: [t1]
        self.cli("tasks", "edit", "t2", "--state", "done")  # done: [t2, t1]
        self.assertEqual(self._done_order(), ["t2", "t1"])

    def test_editing_a_done_task_does_not_reorder_it(self):
        self.cli("tasks", "add", "A")
        self.cli("tasks", "add", "B")
        self.cli("tasks", "move", "t1", "done")
        self.cli("tasks", "move", "t2", "done")             # done: [t2, t1]
        self.cli("tasks", "edit", "t1", "--note", "tweaked")  # already done → no reorder
        self.assertEqual(self._done_order(), ["t2", "t1"])

    def test_done_overflow_auto_archives_bottom_cards(self):
        self.write(".celebornrc", json.dumps({"done_keep_cards": 2, "done_archive_keep_cards": 100}))
        for n in ("One", "Two", "Three"):
            self.cli("tasks", "add", n)
        self.cli("tasks", "move", "t1", "done")
        self.cli("tasks", "move", "t2", "done")
        self.cli("tasks", "move", "t3", "done")  # done: [t3, t2, t1] → t1 falls off bottom
        self.assertEqual(self._done_order(), ["t3", "t2"])
        arch = cb._parse_tasks(self.read("done-archive.md"))
        self.assertEqual([t["id"] for t in arch], ["t1"])
        self.assertEqual(arch[0]["title"], "One")

    def test_done_archive_fifo_drops_oldest_past_cap(self):
        self.write(".celebornrc", json.dumps({"done_keep_cards": 0, "done_archive_keep_cards": 2}))
        for title in ("First", "Second", "Third", "Fourth"):
            self.cli("tasks", "add", title)
            tid = json.loads(self.read("tasks.json"))["tasks"][0]["id"]
            self.cli("tasks", "move", tid, "done")
        arch = cb._parse_tasks(self.read("done-archive.md"))
        self.assertEqual(len(arch), 2)
        self.assertEqual([t["title"] for t in arch], ["Third", "Fourth"])

    def test_tasks_archive_manual_command(self):
        self.write(".celebornrc", json.dumps({"done_keep_cards": 2}))
        self.cli("tasks", "add", "Old")
        self.cli("tasks", "add", "New")
        self.cli("tasks", "move", "t1", "done")
        self.cli("tasks", "move", "t2", "done")
        self.assertEqual(self._done_order(), ["t2", "t1"])
        r = self.cli("tasks", "archive", "--keep", "1")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Archived 1 done card(s)", r.out)
        self.assertEqual(self._done_order(), ["t2"])
        self.assertEqual(cb._parse_tasks(self.read("done-archive.md"))[0]["id"], "t1")

    def test_done_archive_noop_under_budget(self):
        self.cli("tasks", "add", "Only")
        self.cli("tasks", "move", "t1", "done")
        r = self.cli("tasks", "archive")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("nothing to archive", r.out)
        self.assertFalse((self.ctx / "done-archive.md").is_file())

    def test_edit_only_changes_passed_fields(self):
        self.cli("tasks", "add", "Orig", "--owner", "a", "--tags", "keep")
        self.cli("tasks", "edit", "t1", "--title", "New")
        t = json.loads(self.read("tasks.json"))["tasks"][0]
        self.assertEqual(t["title"], "New")
        self.assertEqual(t["owner"], "a")       # untouched
        self.assertEqual(t["tags"], ["keep"])   # untouched

    def test_rm_removes_task(self):
        self.cli("tasks", "add", "Gone")
        self.cli("tasks", "add", "Stays")
        self.cli("tasks", "rm", "t1")
        ids = [t["id"] for t in json.loads(self.read("tasks.json"))["tasks"]]
        self.assertEqual(ids, ["t2"])

    def test_list_default_groups_by_state(self):
        self.cli("tasks", "add", "A", "--state", "doing")
        self.cli("tasks", "add", "B", "--state", "done")
        r = self.cli("tasks")
        self.assertIn("DOING (1)", r.out)
        self.assertIn("DONE (1)", r.out)
        self.assertIn("t1] A", r.out)

    def test_list_empty(self):
        r = self.cli("tasks")
        self.assertIn("No tasks yet", r.out)

    def test_list_json_flag(self):
        self.cli("tasks", "add", "J")
        r = self.cli("tasks", "list", "--json")
        self.assertEqual(json.loads(r.out)["tasks"][0]["title"], "J")

    def test_json_subcommand_writes_and_prints(self):
        self.cli("tasks", "add", "K")
        r = self.cli("tasks", "json")
        self.assertEqual(json.loads(r.out)["tasks"][0]["title"], "K")

    def test_show_renders_fields(self):
        self.cli("tasks", "add", "Shown", "--owner", "claude", "--note", "deep detail")
        r = self.cli("tasks", "show", "t1")
        self.assertIn("Shown", r.out)
        self.assertIn("claude", r.out)
        self.assertIn("deep detail", r.out)

    def test_tasks_json_gitignored(self):
        gi = (self.root / ".gitignore").read_text()
        self.assertIn(".context/tasks.json", gi)

    def test_tasks_md_is_searchable(self):
        # tasks.md is in TIER_GLOBS, so it gets indexed and surfaces in search.
        self.cli("tasks", "add", "Findable kanban task")
        self.cli("index")
        r = self.cli("search", "kanban")
        self.assertIn("Findable", r.out)

    def test_task_phase_link(self):
        self.cli("tasks", "add", "Phased", "--phase", "p11")
        self.assertEqual(json.loads(self.read("tasks.json"))["tasks"][0]["phase"], "p11")
        self.cli("tasks", "edit", "t1", "--phase", "p10")
        self.assertEqual(json.loads(self.read("tasks.json"))["tasks"][0]["phase"], "p10")

    def _todo_ids(self) -> list[str]:
        return [t["id"] for t in json.loads(self.read("tasks.json"))["tasks"] if t["state"] == "todo"]

    def test_add_puts_new_card_at_top_of_column(self):
        self.cli("tasks", "add", "First")
        self.cli("tasks", "add", "Second")
        self.assertEqual(self._todo_ids(), ["t2", "t1"])

    def test_reorder_up_down(self):
        for n in ("A", "B", "C"):
            self.cli("tasks", "add", n)
        self.assertEqual(self._todo_ids(), ["t3", "t2", "t1"])
        self.cli("tasks", "reorder", "t2", "up")
        self.assertEqual(self._todo_ids(), ["t2", "t3", "t1"])
        self.cli("tasks", "reorder", "t1", "up")
        self.assertEqual(self._todo_ids(), ["t2", "t1", "t3"])

    def test_reorder_top_bottom(self):
        for n in ("A", "B", "C"):
            self.cli("tasks", "add", n)
        self.cli("tasks", "reorder", "t3", "top")
        self.assertEqual(self._todo_ids(), ["t3", "t2", "t1"])
        self.cli("tasks", "reorder", "t3", "bottom")
        self.assertEqual(self._todo_ids(), ["t2", "t1", "t3"])

    def test_reorder_is_scoped_to_column(self):
        # Reordering within one column must not perturb tasks in other states.
        self.cli("tasks", "add", "A")                     # t1 todo
        self.cli("tasks", "add", "B", "--state", "doing")  # t2 doing
        self.cli("tasks", "add", "C")                     # t3 todo
        self.cli("tasks", "reorder", "t1", "top")
        self.assertEqual(self._todo_ids(), ["t1", "t3"])
        doing = [t["id"] for t in json.loads(self.read("tasks.json"))["tasks"] if t["state"] == "doing"]
        self.assertEqual(doing, ["t2"])

    def test_reorder_edges_are_noops(self):
        for n in ("A", "B"):
            self.cli("tasks", "add", n)
        self.cli("tasks", "reorder", "t2", "up")    # already first
        self.cli("tasks", "reorder", "t1", "down")  # already last
        self.assertEqual(self._todo_ids(), ["t2", "t1"])

    def test_reorder_unknown_id_errors(self):
        r = self.cli("tasks", "reorder", "t99", "up")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("no task", r.all)

    # --- the board loads on Orient (status shows in-flight doing cards, not todo/done) ------------

    def test_status_surfaces_in_flight_tasks(self):
        self.cli("tasks", "add", "Active card", "--state", "doing")
        self.cli("tasks", "add", "Finished card", "--state", "done")
        self.cli("tasks", "add", "Waiting card")  # todo
        out = self.cli("status").out
        self.assertIn("tasks.md", out)
        self.assertIn("Active card", out)   # doing → surfaced
        # todo/done cards are counted but not listed in the Hot tier (kept lean).
        self.assertNotIn("Finished card", out)
        self.assertNotIn("Waiting card", out)
        self.assertIn("1 doing", out)       # the count line

    def test_status_omits_task_block_when_no_tasks(self):
        self.assertNotIn("tasks.md (board", self.cli("status").out)


class TestOutbox(CelebornTestCase):
    """The per-agent prompt hand-off queue the board's Handoff button feeds and the hook drains."""

    def test_push_task_renders_title_and_notes(self):
        self.cli("tasks", "add", "Do the thing", "--note", "with detail")
        r = self.cli("outbox", "push", "--task", "t1")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("t1]", r.out)
        body = self.read("outbox/_unassigned.md")   # no owner → unassigned queue
        self.assertIn("Do the thing", body)
        self.assertIn("with detail", body)

    def test_push_text(self):
        self.cli("outbox", "push", "--text", "ad-hoc prompt")
        self.assertIn("ad-hoc prompt", self.read("outbox/_unassigned.md"))

    def test_push_unknown_task_errors(self):
        r = self.cli("outbox", "push", "--task", "t99")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("no task", r.all)

    def test_push_without_args_errors(self):
        r = self.cli("outbox", "push")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("nothing to push", r.all)

    def test_drain_returns_and_clears(self):
        self.cli("tasks", "add", "First")
        self.cli("tasks", "add", "Second")
        self.cli("outbox", "push", "--task", "t1")
        self.cli("outbox", "push", "--task", "t2")
        r = self.cli("outbox", "drain")
        self.assertIn("First", r.out)
        self.assertIn("Second", r.out)
        self.assertIn("---", r.out)  # entries joined by a separator
        # Pending queue is emptied, and a second drain yields nothing.
        self.assertIn("Outbox empty", self.cli("outbox", "list").out)
        self.assertEqual(self.cli("outbox", "drain").out.strip(), "")

    def test_drain_archives_to_sent(self):
        self.cli("outbox", "push", "--text", "keep me for provenance")
        self.cli("outbox", "drain")
        self.assertIn("keep me for provenance", self.read("outbox/sent.md"))

    def test_drain_empty_is_silent(self):
        self.assertEqual(self.cli("outbox", "drain").out.strip(), "")

    def test_clear(self):
        self.cli("outbox", "push", "--text", "discard me")
        self.cli("outbox", "clear")
        self.assertIn("Outbox empty", self.cli("outbox", "list").out)

    def test_outbox_gitignored(self):
        gi = (self.root / ".gitignore").read_text()
        self.assertIn(".context/outbox/", gi)

    # --- multi-agent routing (v0: card-assignment.md) ---

    def test_push_for_routes_to_agent_file(self):
        self.cli("outbox", "push", "--text", "for grok", "--for", "grok")
        self.assertIn("for grok", self.read("outbox/grok.md"))
        self.assertFalse((self.ctx / "outbox" / "_unassigned.md").is_file())

    def test_push_task_addresses_to_owner(self):
        self.cli("tasks", "add", "Owned work", "--owner", "opus-a")
        self.cli("outbox", "push", "--task", "t1")          # no --for → defaults to owner
        self.assertIn("Owned work", self.read("outbox/opus-a.md"))

    def test_for_flag_overrides_owner(self):
        self.cli("tasks", "add", "Owned work", "--owner", "opus-a")
        self.cli("outbox", "push", "--task", "t1", "--for", "opus-b")
        self.assertIn("Owned work", self.read("outbox/opus-b.md"))
        self.assertFalse((self.ctx / "outbox" / "opus-a.md").is_file())

    def test_drain_pulls_only_addressed_agent(self):
        self.cli("outbox", "push", "--text", "for A", "--for", "opus-a")
        self.cli("outbox", "push", "--text", "for B", "--for", "opus-b")
        r = self.cli("outbox", "drain", "--for", "opus-a")
        self.assertIn("for A", r.out)
        self.assertNotIn("for B", r.out)
        # opus-b's queue is untouched.
        self.assertIn("for B", self.read("outbox/opus-b.md"))

    def test_drain_identity_from_env(self):
        self.cli("outbox", "push", "--text", "env-routed", "--for", "grok")
        with mock.patch.dict(os.environ, {"CELEBORN_AGENT": "grok"}):
            r = self.cli("outbox", "drain")     # no --for → identity from env
        self.assertIn("env-routed", r.out)

    def test_clear_for_one_agent_leaves_others(self):
        self.cli("outbox", "push", "--text", "keep", "--for", "opus-a")
        self.cli("outbox", "push", "--text", "drop", "--for", "opus-b")
        self.cli("outbox", "clear", "--for", "opus-b")
        self.assertIn("keep", self.read("outbox/opus-a.md"))
        self.assertNotIn("drop", self.read("outbox/opus-b.md"))


# --------------------------------------------------------------------------- claim-on-receipt

class TestClaim(CelebornTestCase):
    """Claim-on-receipt: a copied card carries a ⟨celeborn:tN⟩ marker; receiving it (pasted into a
    model, or `celeborn claim`) assigns the card — owner ← claimer, TODO → DOING, last-claim-wins
    (design: references/card-assignment.md)."""

    def test_task_prompt_carries_card_marker(self):
        self.cli("tasks", "add", "Refactor the parser")
        self.cli("outbox", "push", "--task", "t1")
        body = self.read("outbox/_unassigned.md")
        slug = cb.project_slug(self.ctx)
        self.assertIn(f"⟨celeborn:{slug}/t1⟩", body)
        self.assertIn(cb.AGENT_PROTOCOL_MARKER, body)
        self.assertIn("celeborn touch", body)

    def test_tasks_json_includes_agent_protocol(self):
        self.cli("tasks", "add", "Protocol card")
        doc = json.loads(self.cli("tasks", "json").out)
        self.assertIn("project_slug", doc)
        t = doc["tasks"][-1]
        self.assertIn("agent_protocol", t)
        self.assertIn(cb.AGENT_PROTOCOL_MARKER, t["agent_protocol"])
        self.assertIn("[t1]", t["agent_protocol"])

    def test_find_card_refs_is_tolerant_and_deduped(self):
        ids, rejects = cb._find_card_refs("do it ⟨celeborn:t13⟩")
        self.assertEqual(ids, ["t13"])
        self.assertEqual(rejects, [])
        ids, _ = cb._find_card_refs("[celeborn:t2] then celeborn: t9")
        self.assertEqual(ids, ["t2", "t9"])
        ids, _ = cb._find_card_refs("⟨celeborn:t1⟩ again ⟨celeborn:t1⟩")
        self.assertEqual(ids, ["t1"])
        ids, _ = cb._find_card_refs("bare t13, no marker")
        self.assertEqual(ids, [])

    def test_find_card_refs_rejects_cross_project_markers(self):
        ids, rejects = cb._find_card_refs(
            "work on ⟨celeborn:other-repo/t5⟩", expected_slug="celeborn")
        self.assertEqual(ids, [])
        self.assertEqual(len(rejects), 1)
        self.assertIn("other-repo", rejects[0])
        ids, rejects = cb._find_card_refs(
            "⟨celeborn:celeborn/t5⟩", expected_slug="celeborn")
        self.assertEqual(ids, ["t5"])
        self.assertEqual(rejects, [])
        # Legacy unqualified markers still claim in-repo (back-compat).
        ids, _ = cb._find_card_refs("⟨celeborn:t5⟩", expected_slug="celeborn")
        self.assertEqual(ids, ["t5"])

    def test_card_marker_is_project_qualified(self):
        ctx = self.ctx
        slug = cb.project_slug(ctx)
        self.assertIn("/", cb._card_marker("t3", slug))
        self.assertIn(slug, cb._card_marker("t3", slug))

    def test_claim_sets_owner_and_advances_todo_to_doing(self):
        self.cli("tasks", "add", "Build it")            # starts in todo
        r = self.cli("claim", "t1", "--by", "opus-a")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Claimed", r.out)
        self.assertIn("t1]", r.out)
        show = self.cli("tasks", "show", "t1").out
        self.assertIn("owner:      opus-a", show)
        self.assertIn("state:      doing", show)

    def test_claim_last_wins_and_reports_contention(self):
        self.cli("tasks", "add", "Contended")
        self.cli("claim", "t1", "--by", "grok")
        r = self.cli("claim", "t1", "--by", "opus-a")
        self.assertIn("Reassigned", r.out)
        self.assertIn("t1]", r.out)
        self.assertIn("grok → opus-a", r.out)
        self.assertIn("owner:      opus-a", self.cli("tasks", "show", "t1").out)

    def test_claim_identity_falls_back_to_env(self):
        self.cli("tasks", "add", "Env claim")
        with mock.patch.dict(os.environ, {"CELEBORN_AGENT": "grok"}):
            self.cli("claim", "t1")                      # no --by → $CELEBORN_AGENT
        self.assertIn("owner:      grok", self.cli("tasks", "show", "t1").out)

    def test_claim_unknown_id_is_silent_noop(self):
        r = self.cli("claim", "t99")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(r.out.strip(), "")

    def test_claim_blocks_while_agent_has_other_doing(self):
        self.cli("tasks", "add", "First")
        self.cli("tasks", "add", "Second")
        self.cli("claim", "t1", "--by", "grok")
        r = self.cli("claim", "t2", "--by", "grok")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("already has", r.all)
        self.assertIn("t1]", r.all)

    def test_claim_force_allows_second_doing(self):
        self.cli("tasks", "add", "First")
        self.cli("tasks", "add", "Second")
        self.cli("claim", "t1", "--by", "grok")
        r = self.cli("claim", "t2", "--by", "grok", "--force")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Claimed", r.out)
        self.assertIn("t2]", r.out)

    def test_tasks_add_claim_avoids_id_guess(self):
        self.cli("tasks", "add", "New work", "--claim", "--by", "grok")
        show = self.cli("tasks", "show", "t1").out
        self.assertIn("state:      doing", show)
        self.assertIn("owner:      grok", show)

    def test_orient_marks_stale_doing(self):
        self.cli("tasks", "add", "Zombie", "--state", "doing", "--owner", "grok")
        out = self.cli("status").out
        self.assertIn("stale", out)
        self.assertIn("celeborn ship ", out)   # hint names the card (qualified `SLUG-t1` by default)
        self.assertIn("t1", out)


class TestStandup(CelebornTestCase):
    """standup / changelog + the build-in-public tweet sub-feature."""

    def _add_done(self, title: str):
        self.cli("tasks", "add", title)
        tid = cb._load_tasks(self.ctx)[-1]["id"]
        self.cli("tasks", "move", tid, "done")
        return tid

    def test_standup_lists_completed_card(self):
        tid = self._add_done("Ship the thing")
        r = self.cli("standup")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Ship the thing", r.out)
        self.assertIn(tid, r.out)

    def test_standup_window_excludes_old_done(self):
        # A done card stamped 30 days ago must fall outside the 1-day standup window.
        self.cli("tasks", "add", "Ancient work")
        tasks = cb._load_tasks(self.ctx)
        old = (cb._dt.datetime.now() - cb._dt.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
        tasks[-1]["state"] = "done"
        tasks[-1]["updated"] = old
        cb._save_tasks(self.ctx, tasks)
        self.assertNotIn("Ancient work", self.cli("standup").out)
        # …but a 30-day changelog window includes it.
        self.assertIn("Ancient work", self.cli("changelog", "--days", "31").out)

    def test_tweet_is_under_280_and_branded(self):
        for i in range(4):
            self._add_done(f"A reasonably wordy completed feature number {i} for padding")
        r = self.cli("standup", "--tweet")
        self.assertIsNone(r.exit_code, r.all)
        post = r.out.strip()
        self.assertLessEqual(len(post), 280)
        self.assertIn("🏹", post)
        self.assertIn("#buildinpublic", post)

    def test_dollars_saved_uses_configured_rate(self):
        m = cb._load_metrics(self.ctx)
        m["tokens_saved_estimate"] = 2_000_000
        cb._save_metrics(self.ctx, m)
        # default rate 3.0 $/Mtok → $6 for 2M tokens
        self.assertAlmostEqual(cb.dollars_saved(self.ctx), 6.0, places=2)

    def _seed_economy(self, tokens=120_000_000, resumes=23, compactions=2, loads=40):
        m = cb._load_metrics(self.ctx)
        m["tokens_saved_estimate"] = tokens
        m["sessions_resumed"] = resumes
        m["compactions_bridged"] = compactions
        m["load_events"] = loads
        cb._save_metrics(self.ctx, m)

    def test_flex_card_shows_dollars_restarts_and_brand(self):
        self._seed_economy()
        r = self.cli("flex")
        self.assertIsNone(r.exit_code, r.all)
        out = r.out
        self.assertIn("$ WRAPPED", out)
        self.assertIn("🏹", out)
        self.assertIn("💪", out)
        self.assertIn("$360", out)               # 120M tokens × $3/Mtok = $360
        self.assertIn("120,000,000 tokens", out)
        self.assertIn("25 restarts avoided", out)  # 23 resumes + 2 compactions

    def test_flex_card_box_borders_align(self):
        # Every rendered row must be the same display width — emoji included — or the box looks broken.
        self._seed_economy()
        rows = self.cli("flex").out.strip().splitlines()
        widths = {cb._disp_width(line) for line in rows}
        self.assertEqual(len(widths), 1, (widths, rows))

    def test_flex_tweet_is_under_280_and_branded(self):
        self._seed_economy()
        post = self.cli("flex", "--tweet").out.strip()
        self.assertLessEqual(len(post), 280)
        self.assertIn("🏹💪", post)
        self.assertIn("$360", post)
        self.assertIn("#buildinpublic", post)

    def test_flex_json_carries_the_figures(self):
        self._seed_economy()
        data = json.loads(self.cli("flex", "--json").out)
        self.assertEqual(data["tokens_saved"], 120_000_000)
        self.assertEqual(data["dollars_saved"], 360.0)
        self.assertEqual(data["restarts_avoided"], 25)
        self.assertEqual(data["usd_per_mtok"], 3.0)

    def test_flex_small_balance_shows_cents_not_zero(self):
        self._seed_economy(tokens=1_000, resumes=0, compactions=0, loads=1)
        # 1k tokens × $3/Mtok = $0.003 → rounds to $0.00 at 2dp, but the format path is cents-aware.
        self.assertIn("$0.0", self.cli("flex").out)

    def _isolate_fleet_registry(self):
        """Swap the machine's real fleet registry out for the duration of one test, so the cross-fleet
        aggregate sees only the projects this test registers (TestStandup isn't registry-isolated)."""
        reg = cb._fleet_registry_path()
        backup = reg.read_text() if reg.is_file() else None
        if reg.is_file():
            reg.unlink()

        def restore():
            if backup is not None:
                reg.parent.mkdir(parents=True, exist_ok=True)
                reg.write_text(backup)
            elif reg.is_file():
                reg.unlink()

        self.addCleanup(restore)

    def test_savings_json_aggregates_across_projects(self):
        # The board's economy bar (t68): savings --json exposes this project's figures plus a fleet
        # aggregate that sums every registered Celeborn project (+ this one).
        self._isolate_fleet_registry()
        self._seed_economy()  # this project: 120M tokens, 25 restarts
        data = json.loads(self.cli("savings", "--json").out)
        p = data["project"]
        self.assertEqual(p["tokens_saved"], 120_000_000)
        self.assertEqual(p["dollars_saved"], 360.0)
        self.assertEqual(p["restarts_avoided"], 25)
        # Only this repo on the registry → the aggregate equals the project total.
        self.assertEqual(data["fleet"]["projects"], 1)
        self.assertEqual(data["fleet"]["tokens_saved"], 120_000_000)

        # Register a second project with its own economy; the aggregate must now sum both.
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        run_cli("--path", other.name, "init", "--no-scan")
        octx = Path(other.name) / ".context"
        m = cb._load_metrics(octx)
        m["tokens_saved_estimate"] = 30_000_000
        m["sessions_resumed"] = 5
        cb._save_metrics(octx, m)
        run_cli("--path", str(self.root), "fleet", "register", other.name)

        fl = json.loads(self.cli("savings", "--json").out)["fleet"]
        self.assertEqual(fl["projects"], 2)
        self.assertEqual(fl["tokens_saved"], 150_000_000)   # 120M + 30M
        self.assertEqual(fl["restarts_avoided"], 30)         # 25 + 5
        self.assertEqual(fl["dollars_saved"], 450.0)         # $360 + $90

    def test_savings_human_line_has_emoji_sections(self):
        self._isolate_fleet_registry()
        self._seed_economy()
        out = self.cli("savings").out
        for emoji in ("💰", "🧠", "♻️", "🌐"):
            self.assertIn(emoji, out)
        self.assertIn("$360", out)

    def test_board_port_is_stable_and_in_band(self):
        # Derived port is deterministic per project path and lands in the 3141–3940 band.
        a = cb._derive_board_port(Path("/tmp/project-alpha"))
        b = cb._derive_board_port(Path("/tmp/project-beta"))
        self.assertEqual(a, cb._derive_board_port(Path("/tmp/project-alpha")))  # stable across calls
        self.assertNotEqual(a, b)                                              # different repos differ
        for p in (a, b):
            self.assertTrue(cb.BOARD_PORT_BASE <= p < cb.BOARD_PORT_BASE + cb.BOARD_PORT_SPAN)

    def test_board_port_explicit_config_wins(self):
        self.write(".celebornrc", json.dumps({"board_port": 4242}))
        self.assertEqual(cb.board_port(self.ctx), 4242)
        self.assertIn("http://localhost:4242", self.cli("board", "--url").out)

    def test_board_port_derives_when_unset(self):
        # No board_port in rc → falls back to the per-project derived port (not a crash, not 3000).
        self.assertEqual(cb.board_port(self.ctx), cb._derive_board_port(self.ctx.parent))
        self.assertEqual(self.cli("board", "--port").out.strip(), str(cb.board_port(self.ctx)))

    def test_board_command_reports_url_and_liveness(self):
        self.write(".celebornrc", json.dumps({"board_port": 4242}))
        out = self.cli("board").out
        self.assertIn("http://localhost:4242", out)
        self.assertIn("not running", out)        # nothing is listening on 4242 in the test


class TestEnsureOnOrient(CelebornTestCase):
    """ensure_board() — start the viewer on its port if it's down. Decision logic only; the actual
    detached process launch (_spawn_board) is stubbed so tests never boot Next.js."""

    def setUp(self):
        super().setUp()
        self.write(".celebornrc", json.dumps({"board_port": 4242}))
        # Pretend the board app + deps are installed and a runner resolves, so the decision tree
        # reaches the launch branch. Each test overrides _board_live as needed.
        self._spawned = []
        self.addCleanup(setattr, cb, "_board_live", cb._board_live)
        self.addCleanup(setattr, cb, "_board_runner", cb._board_runner)
        self.addCleanup(setattr, cb, "_spawn_board", cb._spawn_board)
        self.addCleanup(setattr, cb, "_pid_alive", cb._pid_alive)
        cb._board_runner = lambda board_dir: ["npm", "run", "dev"]
        cb._spawn_board = lambda *a: (self._spawned.append(a) or 99999)

    def test_live_port_is_a_noop(self):
        cb._board_live = lambda port, timeout=0.15: True
        st = cb.ensure_board(self.ctx)
        self.assertEqual(st["action"], "live")
        self.assertEqual(st["port"], 4242)
        self.assertEqual(self._spawned, [])                       # nothing launched

    def test_starts_when_down_and_kanban_in_use(self):
        cb._board_live = lambda port, timeout=0.15: False
        self.write("tasks.md", "# Tasks\n")                        # project uses the kanban
        st = cb.ensure_board(self.ctx)
        self.assertEqual(st["action"], "started")
        self.assertEqual(st["pid"], 99999)
        self.assertEqual(len(self._spawned), 1)
        # A pidfile records the launch so the next orient doesn't double-launch while it boots.
        rec = json.loads((self.ctx / cb.BOARD_PIDFILE).read_text())
        self.assertEqual(rec["pid"], 99999)
        self.assertEqual(rec["port"], 4242)

    def test_no_tasks_stays_quiet(self):
        cb._board_live = lambda port, timeout=0.15: False
        st = cb.ensure_board(self.ctx)                            # no tasks.md written
        self.assertEqual(st["action"], "no-tasks")
        self.assertEqual(self._spawned, [])

    def test_autostart_disabled(self):
        cb._board_live = lambda port, timeout=0.15: False
        self.write(".celebornrc", json.dumps({"board_port": 4242, "board_autostart": False}))
        self.write("tasks.md", "# Tasks\n")
        st = cb.ensure_board(self.ctx)
        self.assertEqual(st["action"], "off")
        self.assertEqual(self._spawned, [])

    def test_booting_does_not_double_launch(self):
        cb._board_live = lambda port, timeout=0.15: False
        cb._pid_alive = lambda pid: True                          # a launched board is still coming up
        self.write("tasks.md", "# Tasks\n")
        (self.ctx / cb.BOARD_PIDFILE).write_text(json.dumps({"pid": 4321, "port": 4242}))
        st = cb.ensure_board(self.ctx)
        self.assertEqual(st["action"], "booting")
        self.assertEqual(self._spawned, [])

    def test_unavailable_when_deps_missing(self):
        cb._board_live = lambda port, timeout=0.15: False
        cb._board_runner = lambda board_dir: None                 # no app / node_modules / npm
        self.write("tasks.md", "# Tasks\n")
        st = cb.ensure_board(self.ctx)
        self.assertEqual(st["action"], "unavailable")
        self.assertEqual(self._spawned, [])

    def test_launch_false_reports_down_without_spawning(self):
        cb._board_live = lambda port, timeout=0.15: False
        self.write("tasks.md", "# Tasks\n")
        st = cb.ensure_board(self.ctx, launch=False)
        self.assertEqual(st["action"], "down")
        self.assertEqual(self._spawned, [])

    def test_board_start_cli_is_graceful(self):
        cb._board_live = lambda port, timeout=0.15: False
        self.write("tasks.md", "# Tasks\n")
        r = self.cli("board", "--start")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("started", r.out)


class TestInitNameAndBoard(CelebornTestCase):
    """CELE-t121 — on install, init names the project and (interactively) seeds tasks.md + launches
    and opens the kanban board. The board UX is gated on `_init_is_interactive`, so a headless/CI/test
    install stays side-effect free; the SessionStart ensure-on-orient hook brings the board up later."""

    # ---- name resolution (precedence) -------------------------------------------------------------
    def test_explicit_name_wins(self):
        args = types.SimpleNamespace(name="  Cool Project ")
        self.assertEqual(cb._resolve_init_name(self.root, self.ctx, args), "Cool Project")

    def test_existing_rc_name_is_kept(self):
        self.write(".celebornrc", json.dumps({"project_name": "Already Named"}))
        args = types.SimpleNamespace(name=None)
        # No --name and an rc name already set → return None (caller leaves the rc untouched).
        self.assertIsNone(cb._resolve_init_name(self.root, self.ctx, args))

    def test_headless_no_name_returns_none(self):
        # Not a TTY (StringIO during tests) and no --name → don't prompt, fall back to the folder.
        args = types.SimpleNamespace(name=None)
        self.assertIsNone(cb._resolve_init_name(self.root, self.ctx, args))

    def test_interactive_prompt_default(self):
        self.addCleanup(setattr, cb, "_init_is_interactive", cb._init_is_interactive)
        cb._init_is_interactive = lambda: True
        with mock.patch("builtins.input", return_value=""):     # user accepts the default
            args = types.SimpleNamespace(name=None)
            self.assertEqual(cb._resolve_init_name(self.root, self.ctx, args), self.root.name)
        with mock.patch("builtins.input", return_value="Typed Name"):
            args = types.SimpleNamespace(name=None)
            self.assertEqual(cb._resolve_init_name(self.root, self.ctx, args), "Typed Name")

    # ---- tasks.md seeding -------------------------------------------------------------------------
    def test_ensure_tasks_md_seeds_then_idempotent(self):
        tp = self.ctx / "tasks.md"
        self.assertFalse(tp.is_file())
        self.assertTrue(cb._ensure_tasks_md(self.ctx))
        self.assertTrue(tp.is_file())
        self.assertIn("# Tasks", tp.read_text())
        self.assertFalse(cb._ensure_tasks_md(self.ctx))          # already there → no-op

    # ---- end-to-end via the CLI -------------------------------------------------------------------
    def _board_stubs(self):
        """Pretend the board app is installed and stub the detached launch + browser so no real
        Next.js server boots and no tab opens. Returns (spawned, opened) capture lists."""
        spawned, opened = [], []
        self.addCleanup(setattr, cb, "_board_live", cb._board_live)
        self.addCleanup(setattr, cb, "_board_runner", cb._board_runner)
        self.addCleanup(setattr, cb, "_spawn_board", cb._spawn_board)
        cb._board_live = lambda port, timeout=0.15: False
        cb._board_runner = lambda board_dir: ["npm", "run", "dev"]
        cb._spawn_board = lambda *a: (spawned.append(a) or 99999)
        import webbrowser
        self.addCleanup(setattr, webbrowser, "open", webbrowser.open)
        webbrowser.open = lambda u, *a, **k: (opened.append(u), True)[1]
        # The base fixture disables board autostart; re-enable so the launch branch is reachable.
        rc = self.ctx / ".celebornrc"
        d = json.loads(rc.read_text()); d.pop("board_autostart", None); rc.write_text(json.dumps(d))
        return spawned, opened

    def test_interactive_install_names_seeds_launches_opens(self):
        self.addCleanup(setattr, cb, "_init_is_interactive", cb._init_is_interactive)
        cb._init_is_interactive = lambda: True
        spawned, opened = self._board_stubs()
        r = self.cli("init", "--no-scan", "--no-cmm", "--name", "My Cool Project")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(cb.load_config(self.ctx).get("project_name"), "My Cool Project")
        self.assertTrue((self.ctx / "tasks.md").is_file())
        self.assertEqual(len(spawned), 1)
        self.assertEqual(len(opened), 1)
        self.assertIn(f":{cb.board_port(self.ctx)}", opened[0])

    def test_headless_install_skips_board(self):
        # _init_is_interactive stays its real (False) self — no seed, no launch, no tab.
        spawned, opened = self._board_stubs()
        r = self.cli("init", "--no-scan", "--no-cmm", "--name", "Headless")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(cb.load_config(self.ctx).get("project_name"), "Headless")  # --name still persists
        self.assertFalse((self.ctx / "tasks.md").is_file())
        self.assertEqual(spawned, [])
        self.assertEqual(opened, [])

    def test_no_open_flag_skips_board(self):
        self.addCleanup(setattr, cb, "_init_is_interactive", cb._init_is_interactive)
        cb._init_is_interactive = lambda: True
        spawned, opened = self._board_stubs()
        r = self.cli("init", "--no-scan", "--no-cmm", "--name", "X", "--no-open")
        self.assertIsNone(r.exit_code, r.all)
        self.assertFalse((self.ctx / "tasks.md").is_file())
        self.assertEqual(spawned, [])
        self.assertEqual(opened, [])

    def test_no_browser_launches_without_tab(self):
        self.addCleanup(setattr, cb, "_init_is_interactive", cb._init_is_interactive)
        cb._init_is_interactive = lambda: True
        spawned, opened = self._board_stubs()
        r = self.cli("init", "--no-scan", "--no-cmm", "--name", "X", "--no-browser")
        self.assertIsNone(r.exit_code, r.all)
        self.assertTrue((self.ctx / "tasks.md").is_file())
        self.assertEqual(len(spawned), 1)
        self.assertEqual(opened, [])


class TestSetup(CelebornTestCase):
    """CELE-t120 — `celeborn setup` is the Modal-clean first run: one guided command that orchestrates
    wire → init → login. A thin shell over the existing verbs, so the tests assert the orchestration
    (right steps, right order, idempotent/resumable, login required-but-safe-headless) rather than
    re-testing wire/init/login internals. Login network/browser is always stubbed."""

    def _no_creds(self):
        """Pretend no account is signed in (the dev box running tests may actually be logged in)."""
        self.addCleanup(setattr, cs, "load_creds", cs.load_creds)
        cs.load_creds = lambda: {}

    def _record_login(self):
        """Stub celeborn_sync.cmd_login so no browser/network fires; capture the args it's called with."""
        calls = []
        self.addCleanup(setattr, cs, "cmd_login", cs.cmd_login)
        cs.cmd_login = lambda a: calls.append(a)
        return calls

    def _fresh(self) -> Path:
        sub = self.root / "fresh"
        sub.mkdir()
        return sub

    # ---- orchestration ----------------------------------------------------------------------------
    def test_setup_wires_scaffolds_and_reports_ready(self):
        sub = self._fresh()
        self._no_creds()
        calls = self._record_login()
        r = run_cli("--path", str(sub), "setup", "--project", "--no-login", "--no-open", "--no-cmm")
        self.assertIsNone(r.exit_code, r.all)
        self.assertTrue((sub / ".claude" / "settings.json").is_file())   # wire (project scope) ran
        self.assertTrue((sub / ".context").is_dir())                     # init scaffolded the project
        self.assertIn("Celeborn is ready", r.all)
        self.assertEqual(calls, [])                                      # --no-login: no sign-in attempt

    def test_setup_skips_init_when_already_a_project(self):
        # self.root is already a Celeborn project (base fixture ran init).
        self._no_creds()
        self._record_login()
        r = self.cli("setup", "--project", "--no-login", "--no-open")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("already a Celeborn project", r.all)

    def test_setup_is_idempotent(self):
        sub = self._fresh()
        self._no_creds()
        self._record_login()
        run_cli("--path", str(sub), "setup", "--project", "--no-login", "--no-open", "--no-cmm")
        r = run_cli("--path", str(sub), "setup", "--project", "--no-login", "--no-open", "--no-cmm")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("already a Celeborn project", r.all)               # init no-ops on the re-run
        d = json.loads((sub / ".claude" / "settings.json").read_text())
        self.assertEqual(len(d["hooks"]["Stop"]), 1)                     # wire didn't duplicate

    # ---- login: required by default, safe headless, skippable ------------------------------------
    def test_setup_login_required_but_safe_when_non_interactive(self):
        # No --no-login: login is REQUIRED — but a non-TTY shell can't open a browser, so it warns and
        # finishes rather than aborting (_init_is_interactive stays its real False during tests).
        sub = self._fresh()
        self._no_creds()
        calls = self._record_login()
        r = run_cli("--path", str(sub), "setup", "--project", "--no-open", "--no-cmm")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("non-interactive", r.all.lower())
        self.assertEqual(calls, [])                                      # never tried to open a browser
        self.assertIn("celeborn login --github", r.all)                 # told how to finish

    def test_setup_interactive_login_uses_github_browser(self):
        sub = self._fresh()
        self.addCleanup(setattr, cb, "_init_is_interactive", cb._init_is_interactive)
        cb._init_is_interactive = lambda: True
        self._no_creds()
        calls = self._record_login()
        r = run_cli("--path", str(sub), "setup", "--project", "--no-open", "--no-cmm")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(len(calls), 1)                                  # login attempted
        self.assertTrue(calls[0].github)                                 # via the GitHub browser flow

    def test_setup_email_flag_uses_password_login(self):
        sub = self._fresh()
        self.addCleanup(setattr, cb, "_init_is_interactive", cb._init_is_interactive)
        cb._init_is_interactive = lambda: True
        self._no_creds()
        calls = self._record_login()
        r = run_cli("--path", str(sub), "setup", "--project", "--no-open", "--no-cmm",
                    "--email", "me@example.com")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0].github)                                # email+password, not GitHub
        self.assertEqual(calls[0].email, "me@example.com")

    def test_setup_already_signed_in_skips_login(self):
        sub = self._fresh()
        self.addCleanup(setattr, cs, "load_creds", cs.load_creds)
        cs.load_creds = lambda: {"email": "a@b.c", "access_token": "tok"}
        calls = self._record_login()
        r = run_cli("--path", str(sub), "setup", "--project", "--no-open", "--no-cmm")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("already signed in as a@b.c", r.all)
        self.assertEqual(calls, [])

    def test_setup_failed_login_warns_but_finishes(self):
        sub = self._fresh()
        self.addCleanup(setattr, cb, "_init_is_interactive", cb._init_is_interactive)
        cb._init_is_interactive = lambda: True
        self._no_creds()
        def _boom(a):
            raise SystemExit(2)
        self.addCleanup(setattr, cs, "cmd_login", cs.cmd_login)
        cs.cmd_login = _boom
        r = run_cli("--path", str(sub), "setup", "--project", "--no-open", "--no-cmm")
        self.assertIsNone(r.exit_code, r.all)                            # a login failure doesn't abort setup
        self.assertIn("didn't complete", r.all)
        self.assertIn("Celeborn is ready", r.all)                       # local setup still finished


class TestBoardSupervisor(CelebornTestCase):
    """The self-healing restart-loop supervisor (CELE-t99). _board_supervise() is driven with an
    injected spawn/clock/sleeper so it never boots Next.js and never really sleeps."""

    class _FakeProc:
        """Stand-in for the `next dev` child: wait() returns a fixed rc; terminate() is a no-op."""
        def __init__(self, rc=0):
            self._rc = rc
        def wait(self):
            return self._rc
        def terminate(self):
            pass

    def test_relaunches_child_on_each_exit(self):
        # Three healthy children (each ran long enough to reset the rapid-failure budget), then the
        # next spawn raises to end the loop. The supervisor relaunches after every exit.
        procs = [self._FakeProc(0), self._FakeProc(0), self._FakeProc(0)]
        calls = {"n": 0}
        def spawn():
            if calls["n"] >= len(procs):
                raise RuntimeError("no more children")
            p = procs[calls["n"]]; calls["n"] += 1
            return p
        ticks = iter(range(0, 1_000_000, 100))      # +100s per clock() → each child "ran" 100s
        sleeps = []
        restarts = cb._board_supervise(
            ["npm", "run", "dev"], 4242, Path("/x/tasks.json"), Path("/x/board"),
            spawn=spawn, sleeper=sleeps.append, clock=lambda: next(ticks))
        self.assertEqual(restarts, 3)               # three child exits, each relaunched
        self.assertEqual(calls["n"], 3)             # a 4th spawn was attempted and raised → exit
        self.assertEqual(sleeps, [1.0, 1.0, 1.0])   # healthy runs keep backoff at the floor

    def test_gives_up_after_n_rapid_failures(self):
        # Every child dies instantly (ran < rapid_window) → the supervisor stops hot-looping a
        # broken build after max_rapid deaths instead of relaunching forever.
        restarts = cb._board_supervise(
            ["npm", "run", "dev"], 4242, Path("/x/t.json"), Path("/x/b"),
            spawn=lambda: self._FakeProc(1), sleeper=lambda s: None, clock=lambda: 0.0,
            max_rapid=5, rapid_window_s=10.0)
        self.assertEqual(restarts, 5)               # gave up exactly at the rapid-failure cap

    def test_backoff_doubles_and_caps(self):
        sleeps = []
        cb._board_supervise(
            ["npm", "run", "dev"], 4242, Path("/x/t.json"), Path("/x/b"),
            spawn=lambda: self._FakeProc(1), sleeper=sleeps.append, clock=lambda: 0.0,
            max_rapid=8, rapid_window_s=10.0, backoff_cap_s=30.0)
        # Exponential until it hits the cap, then flat — and never above the cap.
        self.assertEqual(sleeps, [2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0])
        self.assertTrue(all(s <= 30.0 for s in sleeps))

    def test_run_board_supervisor_noop_when_viewer_unavailable(self):
        import types
        self.addCleanup(setattr, cb, "_board_runner", cb._board_runner)
        cb._board_runner = lambda board_dir: None        # no app / deps / npm
        args = types.SimpleNamespace(supervise=True, supervise_port=4242,
                                     supervise_tasks="/x/t.json")
        cb._run_board_supervisor(args)                   # returns quietly, no exception = pass

    def test_spawn_board_launches_supervisor_not_bare_dev(self):
        import subprocess
        captured = {}
        class FakePopen:
            def __init__(self, argv, **kw):
                captured["argv"] = argv; captured["kw"] = kw
                self.pid = 12345
        self.addCleanup(setattr, subprocess, "Popen", subprocess.Popen)
        subprocess.Popen = FakePopen
        cb._spawn_board = self._real_spawn_board   # opt in to the genuine impl (Popen is faked above)
        pid = cb._spawn_board(Path("/x/board"), ["npm", "run", "dev"], 4242,
                              Path("/x/tasks.json"), self.ctx)
        self.assertEqual(pid, 12345)
        argv = captured["argv"]
        self.assertIn("--supervise", argv)               # the supervisor, not bare `npm run dev`
        self.assertIn("board", argv)
        self.assertIn("4242", argv)
        self.assertTrue(captured["kw"].get("start_new_session"))   # detached → outlives the session


class TestBoardSpawnGuard(CelebornTestCase):
    """CELE-t153 — the base fixture makes a REAL detached board spawn impossible, so a future
    board-touching test that forgets to stub _spawn_board fails loudly here instead of silently
    leaking a `next dev` supervisor that outlives the suite and collides with the live board."""

    def test_real_spawn_board_is_blocked_by_default(self):
        # No opt-in → the guard is in force: reaching the real launch raises AssertionError rather
        # than booting Next.js. (A clear message points the next author at the stub seams.)
        with self.assertRaises(AssertionError):
            cb._spawn_board(self.ctx, ["npm", "run", "dev"], 4242,
                            self.ctx / "tasks.json", self.ctx)

    def test_opt_in_restores_the_real_impl(self):
        # The escape hatch a launch-path test uses: restore the genuine impl with subprocess.Popen
        # faked, so the real _spawn_board runs without booting anything.
        import subprocess
        captured = {}
        class FakePopen:
            def __init__(self, argv, **kw):
                captured["argv"] = argv; self.pid = 4242
        self.addCleanup(setattr, subprocess, "Popen", subprocess.Popen)
        subprocess.Popen = FakePopen
        cb._spawn_board = self._real_spawn_board
        pid = cb._spawn_board(self.ctx, ["npm", "run", "dev"], 4242,
                              self.ctx / "tasks.json", self.ctx)
        self.assertEqual(pid, 4242)
        self.assertIn("--supervise", captured["argv"])   # the real impl ran, just with a fake Popen


class TestPerTurnBoardReensure(CelebornTestCase):
    """The UserPromptSubmit hook re-ensures the board every turn (CELE-t99 safety net) — best-effort
    and never breaking the turn. ensure_board is stubbed so no real Next.js boots."""

    def _transcript(self) -> Path:
        tp = self.root / "t.jsonl"
        tp.write_text(json.dumps({"message": {"usage": {"input_tokens": 1000}}}) + "\n")
        return tp

    def test_re_ensures_board_on_every_turn(self):
        calls = []
        self.addCleanup(setattr, cb, "ensure_board", cb.ensure_board)
        cb.ensure_board = lambda ctx, **kw: (calls.append(ctx) or {"action": "live"})
        cb.dispatch_hook("user-prompt-submit",
                         {"session_id": "s1", "transcript_path": str(self._transcript())},
                         str(self.root))
        self.assertEqual(len(calls), 1)                  # re-ensured exactly once this turn
        self.assertEqual(calls[0].resolve(), self.ctx.resolve())   # this project's .context dir

    def test_re_ensure_failure_never_breaks_the_turn(self):
        self.addCleanup(setattr, cb, "ensure_board", cb.ensure_board)
        def boom(ctx, **kw):
            raise RuntimeError("board blew up")
        cb.ensure_board = boom
        # Must not raise — a re-ensure blow-up is swallowed and the hook returns normally.
        out = cb.dispatch_hook("user-prompt-submit",
                               {"session_id": "s1", "transcript_path": str(self._transcript())},
                               str(self.root))
        self.assertIsInstance(out, str)                  # returned cleanly, turn intact


class TestBlame(CelebornTestCase):
    """celeborn blame — git history + memory cross-link."""

    def _git(self, *argv: str):
        import subprocess
        return subprocess.run(["git", "-C", str(self.root), *argv], capture_output=True, text=True, check=True)

    def _init_git_with_file(self, relpath: str, content: str) -> str:
        self._git("init", "-q")
        self._git("config", "user.email", "test@celeborn.local")
        self._git("config", "user.name", "Celeborn Test")
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self._git("add", relpath)
        self._git("commit", "-qm", "add widget")
        return self._git("rev-parse", "--short", "HEAD").stdout.strip()

    def test_blame_links_git_and_memory(self):
        sha = self._init_git_with_file("src/widget.py", "export const x = 1;\n")
        self.write("decisions.md", (
            "# Decisions\n\n"
            f"## 2026-06-10 — Widget export\n"
            f"- Chose a named export in `src/widget.py` (commit {sha}) because tree-shaking.\n"
        ))
        r = self.cli("blame", "src/widget.py")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Celeborn blame", r.out)
        self.assertIn(sha, r.out)
        self.assertIn("Widget export", r.out)
        self.assertIn("tree-shaking", r.out)

    def test_blame_json_shape(self):
        self._init_git_with_file("lib/a.py", "pass\n")
        data = json.loads(self.cli("blame", "lib/a.py", "--json").out)
        self.assertEqual(data["file"], "lib/a.py")
        self.assertEqual(len(data["commits"]), 1)
        self.assertIn("short", data["commits"][0])

    def test_blame_no_git_is_graceful(self):
        r = self.cli("blame", "missing.py")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("no git history", r.out)


class TestWhy(CelebornTestCase):
    """celeborn why — decision archaeology (decision + date + rationale)."""

    def _seed(self):
        self.write("decisions.md", (
            "# Decisions\n\n"
            "## 2026-06-02 — Billing = GitHub Sponsors; ANY sponsor unlocks sync\n"
            "- GitHub Sponsors IS the billing backend for v1 because it needs zero payment infra.\n\n"
            "## 2026-05-01 — Index is a disposable derived artifact\n"
            "- The SQLite index is regenerable from markdown, so durability is irrelevant.\n"
        ))
        self.write("journal.md", (
            "# Journal\n\n"
            "## 2026-06-03 — Touched the billing webhook in passing\n"
            "- **Did:** noted the billing flow while wiring something else.\n"
        ))

    def test_why_surfaces_decision_date_and_rationale(self):
        self._seed()
        r = self.cli("why", "billing")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Celeborn why", r.out)
        self.assertIn("2026-06-02", r.out)
        self.assertIn("GitHub Sponsors", r.out)        # the rationale (rendered, may wrap)
        # full unwrapped rationale lives in --json
        top = json.loads(self.cli("why", "billing", "--json").out)["hits"][0]
        self.assertIn("zero payment infra", top["rationale"])

    def test_why_ranks_decision_above_journal_mention(self):
        self._seed()
        data = json.loads(self.cli("why", "billing", "--json").out)
        self.assertGreaterEqual(len(data["hits"]), 2)
        self.assertEqual(data["hits"][0]["kind"], "decision")   # locked decision wins over a journal aside

    def test_why_json_shape(self):
        self._seed()
        data = json.loads(self.cli("why", "index disposable", "--json").out)
        self.assertEqual(data["query"], "index disposable")
        top = data["hits"][0]
        for key in ("file", "kind", "title", "date", "score", "matched", "rationale", "anchor"):
            self.assertIn(key, top)
        self.assertEqual(top["date"], "2026-05-01")

    def test_why_limit_caps_results(self):
        self._seed()
        data = json.loads(self.cli("why", "billing", "-n", "1", "--json").out)
        self.assertEqual(len(data["hits"]), 1)

    def test_why_strips_redundant_leading_date_in_render(self):
        self._seed()
        r = self.cli("why", "billing")
        # The chip carries the date; the title line must not repeat "2026-06-02 —".
        self.assertNotIn("] 2026-06-02 —", r.out)

    def test_why_no_match_is_graceful(self):
        self._seed()
        r = self.cli("why", "quantumflux")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("No decision or rationale found", r.out)


class TestTouch(CelebornTestCase):
    """celeborn touch — file-level multi-agent registry."""

    def test_touch_register_list_release(self):
        r = self.cli("touch", "src/widget.py", "--by", "grok", "--task", "t28")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Touch @grok", r.out)
        self.assertIn("src/widget.py", r.out)
        listed = self.cli("touch", "list")
        self.assertIn("@grok", listed.out)
        self.assertIn("t28", listed.out)
        st = self.cli("status")
        self.assertIn("touches.json", st.out)
        self.assertIn("@grok", st.out)
        rel = self.cli("touch", "release", "src/widget.py", "--by", "grok")
        self.assertIsNone(rel.exit_code, rel.all)
        self.assertIn("Released", rel.out)
        self.assertIn("(no active touches)", self.cli("touch", "list").out)

    def test_touch_warns_on_overlap(self):
        self.cli("touch", "lib/a.py", "--by", "claude")
        r = self.cli("touch", "lib/a.py", "--by", "grok")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("claude", r.out)

    def test_touch_stale_pruned_from_orient(self):
        self.cli("touch", "old.py", "--by", "grok")
        data = cb._load_touches(self.ctx)
        data["files"]["old.py"]["at"] = (
            cb._dt.datetime.now() - cb._dt.timedelta(hours=5)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        cb._save_touches(self.ctx, data)
        self.assertNotIn("old.py", self.cli("status").out)

    def test_touch_json_list(self):
        self.cli("touch", "x.py", "--by", "a")
        data = json.loads(self.cli("touch", "list", "--json").out)
        self.assertEqual(len(data["touches"]), 1)
        self.assertEqual(data["touches"][0]["path"], "x.py")

    def test_touch_release_nudges_stale_doing(self):
        self.cli("tasks", "add", "Stale doing", "--state", "doing")
        self.cli("touch", "pkg/a.py", "--by", "grok", "--task", "t1")
        r = self.cli("touch", "release", "pkg/a.py", "--by", "grok")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("celeborn ship t1", r.all)
        self.assertIn("still DOING", r.all)

    def test_touch_release_silent_when_still_touched(self):
        self.cli("tasks", "add", "Two files", "--state", "doing")
        self.cli("touch", "a.py", "--by", "grok", "--task", "t1")
        self.cli("touch", "b.py", "--by", "grok", "--task", "t1")
        r = self.cli("touch", "release", "a.py", "--by", "grok")
        self.assertNotIn("celeborn ship", r.all)


class TestIdentity(CelebornTestCase):
    """celeborn identify — agent family/model registry + its flow into touches/claims/tasks.json."""

    def _agents(self) -> dict:
        return (cb._load_agents(self.ctx).get("agents") or {})

    def test_identify_writes_registry(self):
        r = self.cli("identify", "--family", "Claude", "--model", "Opus 4.8", "--as", "claude")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Claude · Opus 4.8", r.all)
        e = self._agents()["claude"]
        self.assertEqual((e["family"], e["model"]), ("Claude", "Opus 4.8"))

    def test_identify_show(self):
        self.cli("identify", "--family", "Grok", "--model", "Grok 4", "--as", "grok")
        out = self.cli("identify", "--show").out
        self.assertIn("@grok", out)
        self.assertIn("Grok · Grok 4", out)

    def test_identify_partial_merge(self):
        self.cli("identify", "--family", "Claude", "--as", "claude")
        self.cli("identify", "--model", "Sonnet 4.6", "--as", "claude")
        e = self._agents()["claude"]
        self.assertEqual((e["family"], e["model"]), ("Claude", "Sonnet 4.6"))

    def test_identify_requires_something(self):
        self.assertIsNotNone(self.cli("identify", "--as", "x").exit_code)

    def test_touch_inherits_identity_from_registry(self):
        self.cli("identify", "--family", "Claude", "--model", "Opus 4.8", "--by", "claude")
        r = self.cli("touch", "Board.tsx", "--by", "claude", "--why", "adding the Run tab")
        self.assertIn("Claude · Opus 4.8", r.out)
        self.assertIn("adding the Run tab", r.out)
        row = json.loads(self.cli("touch", "list", "--json").out)["touches"][0]
        self.assertEqual(row["family"], "Claude")
        self.assertEqual(row["model"], "Opus 4.8")
        self.assertEqual(row["why"], "adding the Run tab")
        # orient surfaces who + why
        st = self.cli("status").out
        self.assertIn("Opus 4.8", st)
        self.assertIn("adding the Run tab", st)

    def test_touch_without_why_still_registers_with_nudge(self):
        r = self.cli("touch", "x.py", "--by", "claude")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Touch @claude", r.out)
        self.assertIn("--why", r.all)  # nudge present on stderr

    def test_flag_overrides_registry(self):
        self.cli("identify", "--family", "Claude", "--model", "Opus 4.8", "--by", "claude")
        self.cli("touch", "x.py", "--by", "claude", "--model", "Sonnet 4.6")
        row = json.loads(self.cli("touch", "list", "--json").out)["touches"][0]
        self.assertEqual(row["model"], "Sonnet 4.6")

    def test_env_supplies_model(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"CELEBORN_AGENT_FAMILY": "Grok", "CELEBORN_AGENT_MODEL": "Grok 4"}):
            self.cli("touch", "y.py", "--by", "grok")
        row = json.loads(self.cli("touch", "list", "--json").out)["touches"][0]
        self.assertEqual((row["family"], row["model"]), ("Grok", "Grok 4"))
        # env also seeds the registry so the board owner chip can show it
        self.assertEqual(self._agents()["grok"]["model"], "Grok 4")

    def test_tasks_json_enriches_owner_from_registry(self):
        self.cli("identify", "--family", "Claude", "--model", "Opus 4.8", "--by", "claude")
        self.cli("tasks", "add", "Do a thing", "--state", "doing", "--owner", "claude")
        doc = json.loads(self.cli("tasks", "json").out)
        t = doc["tasks"][0]
        self.assertEqual(t["owner_model"], "Opus 4.8")
        self.assertEqual(t["owner_family"], "Claude")


class TestShip(CelebornTestCase):
    """celeborn ship — one-shot card close-out (P0)."""

    def test_ship_releases_touches_and_moves_done(self):
        self.cli("tasks", "add", "Ship me", "--state", "doing", "--owner", "grok")
        self.cli("touch", "src/x.py", "--by", "grok", "--task", "t1")
        self.cli("touch", "src/y.py", "--by", "grok", "--task", "t1")
        r = self.cli("ship", "t1", "--by", "grok")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Shipped", r.out)
        self.assertIn("t1]", r.out)
        self.assertIn("src/x.py", r.out)
        self.assertIn("(no active touches)", self.cli("touch", "list").out)
        show = self.cli("tasks", "show", "t1").out
        self.assertIn("state:      done", show)

    def test_ship_appends_note(self):
        self.cli("tasks", "add", "Noted", "--state", "doing", "--note", "started")
        r = self.cli("ship", "t1", "--note", "SHIPPED: all green")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("SHIPPED: all green", self.cli("tasks", "show", "t1").out)
        self.assertIn("started", self.cli("tasks", "show", "t1").out)


class TestPanicSave(CelebornTestCase):
    """`celeborn panic-save` / `restore` and the PreCompact hook wiring — the t36 felt-survival moment."""

    def hook(self, event, payload=None):
        with mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload or {}))):
            return run_cli("--path", str(self.root), "hook", event)

    def test_panic_save_line_uses_dynamic_counts_and_path(self):
        line = cb._panic_save_line({"stamp": "20260610-124337", "files": ["state.md"] * 8})
        self.assertIn("8 files", line)
        self.assertIn(".context/.panic/20260610-124337/", line)
        self.assertIn("Model context window overflow", line)
        self.assertIn("context compaction", line)
        self.assertIn(cb.PANIC_READ_MORE, line)
        self.assertNotIn("saved your session", line)

    def test_panic_save_snapshots_authored_files_and_prints_felt_line(self):
        self.write("state.md", "FOCUS: ship t36")
        r = self.cli("panic-save")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Model context window overflow", r.out)
        self.assertIn("Celeborn saved you", r.out)
        self.assertIn(".context/.panic/", r.out)
        snaps = cb._panic_snapshots(self.ctx)
        self.assertEqual(len(snaps), 1)
        self.assertIn("FOCUS: ship t36", (snaps[0] / "state.md").read_text())
        meta = json.loads((snaps[0] / "meta.json").read_text())
        self.assertEqual(meta["reason"], "manual")
        self.assertIn("state.md", meta["files"])

    def test_panic_save_increments_metric(self):
        self.cli("panic-save", "--quiet")
        self.cli("panic-save", "--quiet")
        self.assertEqual(cb._load_metrics(self.ctx).get("panic_saves"), 2)

    def test_panic_save_quiet_is_silent(self):
        r = self.cli("panic-save", "--quiet")
        self.assertEqual(r.out.strip(), "")
        self.assertEqual(len(cb._panic_snapshots(self.ctx)), 1)

    def test_panic_save_json(self):
        r = self.cli("panic-save", "--json", "--reason", "alarm")
        info = json.loads(r.out)
        self.assertEqual(info["reason"], "alarm")
        self.assertTrue(info["files"])

    def test_restore_list_empty_and_populated(self):
        self.assertIn("No panic-saves yet", self.cli("restore", "--list").out)
        self.cli("panic-save", "--quiet")
        self.assertIn("manual", self.cli("restore", "--list").out)

    def test_restore_brings_back_prior_content_reversibly(self):
        self.write("state.md", "ORIGINAL")
        self.cli("panic-save", "--quiet")            # snapshot ORIGINAL
        self.write("state.md", "CHANGED")
        r = self.cli("restore")                       # default = most recent
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Restored", r.out)
        self.assertEqual(self.read("state.md"), "ORIGINAL")
        # the pre-restore backup preserved CHANGED, so the restore is itself reversible
        pre = [p for p in cb._panic_snapshots(self.ctx)
               if json.loads((p / "meta.json").read_text()).get("reason") == "pre-restore"]
        self.assertEqual(len(pre), 1)
        self.assertEqual((pre[0] / "state.md").read_text(), "CHANGED")

    def test_restore_from_specific_stamp(self):
        with mock.patch.object(cb, "_panic_stamp", side_effect=["20260101-000001", "20260101-000002"]):
            self.write("state.md", "FIRST")
            self.cli("panic-save", "--quiet")
            self.write("state.md", "SECOND")
            self.cli("panic-save", "--quiet")
        self.write("state.md", "LIVE")
        r = self.cli("restore", "--from", "20260101-000001")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(self.read("state.md"), "FIRST")

    def test_restore_unknown_stamp_errors(self):
        self.cli("panic-save", "--quiet")
        r = self.cli("restore", "--from", "nope")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("no panic-save named", r.all)

    def test_fifo_prune_keeps_only_keep(self):
        stamps = [f"20260101-0000{n:02d}" for n in range(1, 6)]
        with mock.patch.object(cb, "_panic_stamp", side_effect=stamps):
            for _ in stamps:
                self.cli("panic-save", "--quiet", "--keep", "2")
        snaps = [p.name for p in cb._panic_snapshots(self.ctx)]
        self.assertEqual(snaps, stamps[-2:])          # oldest three pruned, newest two kept

    def test_pre_compact_hook_panic_saves_and_leads_with_felt_line(self):
        r = self.hook("pre-compact", {"session_id": "s1"})
        self.assertIsNone(r.exit_code, r.all)
        # the reassuring line comes FIRST, then the model-facing checkpoint nudge
        self.assertLess(r.out.index("context window overflow"), r.out.index("Compaction imminent"))
        self.assertEqual(len(cb._panic_snapshots(self.ctx)), 1)
        m = cb._load_metrics(self.ctx)
        self.assertEqual(m.get("panic_saves"), 1)
        self.assertEqual(m.get("compactions_bridged"), 1)
        meta = json.loads((cb._panic_snapshots(self.ctx)[0] / "meta.json").read_text())
        self.assertEqual(meta["reason"], "compaction")
        self.assertEqual(meta["session"], "s1")

    def test_pre_compact_recurs_silently_no_dialog(self):
        """t62: compaction can recur many times in one session; each one still snapshots and prints the
        felt line, but NO native modal is ever raised (the alert windows were removed)."""
        for _ in range(3):
            self.hook("pre-compact", {"session_id": "s1"})
        self.assertEqual(len(cb._panic_snapshots(self.ctx)), 3)   # every compaction still snapshots
        self.assertEqual(cb._load_metrics(self.ctx).get("panic_saves"), 3)


class TestGrokWire(CelebornTestCase):
    """Grok Build auto-wire on init + per-project rules (orient survives /clear)."""

    def test_grok_sync_rules_writes_managed_block(self):
        cb._ensure_grok_rules(self.root)
        rules = self.root / ".grok" / "rules" / "celeborn.md"
        self.assertTrue(rules.is_file())
        body = rules.read_text()
        self.assertIn(cb.GROK_RULES_BEGIN, body)
        self.assertIn("wire tN", body)
        self.assertIn("grok --cwd", body)
        self.assertIn("celeborn", body)

    def test_grok_sync_rules_cli(self):
        r = self.cli("grok", "sync-rules")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("celeborn.md", r.out)

    def test_init_writes_grok_rules_when_grok_present(self):
        with mock.patch.object(cb, "_wire_grok", return_value=True) as wg:
            tmp = tempfile.mkdtemp()
            try:
                root = Path(tmp)
                r = run_cli("--path", str(root), "init", "--no-scan", "--no-claude-md")
                self.assertIsNone(r.exit_code, r.all)
                wg.assert_called_once()
                self.assertIn("wired Grok Build", r.out)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)


class TestFleet(unittest.TestCase):
    """Live fleet dashboard — multi-project agent liveness (t30)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ctx = self.root / ".context"
        r = run_cli("--path", str(self.root), "init", "--no-scan")
        self.assertIsNone(r.exit_code, r.all)
        self._reg_backup = None
        reg = cb._fleet_registry_path()
        if reg.is_file():
            self._reg_backup = reg.read_text()
            reg.unlink()

    def tearDown(self):
        reg = cb._fleet_registry_path()
        if self._reg_backup is not None:
            reg.parent.mkdir(parents=True, exist_ok=True)
            reg.write_text(self._reg_backup)
        elif reg.is_file():
            reg.unlink()
        self._tmp.cleanup()

    def test_fleet_register_and_json(self):
        r = run_cli("--path", str(self.root), "fleet", "register")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("registered", r.out.lower())
        j = run_cli("--path", str(self.root), "fleet", "--json")
        self.assertIsNone(j.exit_code, j.all)
        data = json.loads(j.out)
        self.assertIn("projects", data)
        self.assertEqual(data["summary"]["projects"], 1)
        p = data["projects"][0]
        self.assertEqual(p["slug"], cb.project_slug(self.ctx))
        self.assertIn("session", p)
        self.assertIn("board", p)

    def test_fleet_unregister(self):
        run_cli("--path", str(self.root), "fleet", "register")
        r = run_cli("--path", str(self.root), "fleet", "unregister", str(self.root))
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("unregistered", r.out.lower())
        self.assertEqual(len(cb._load_fleet_registry()["projects"]), 0)
        # orienting project still appears when `fleet` is run from that repo (even if unregistered)
        j = json.loads(run_cli("--path", str(self.root), "fleet", "--json").out)
        self.assertEqual(j["summary"]["projects"], 1)

    def test_fleet_detects_doing_card(self):
        run_cli("--path", str(self.root), "tasks", "add", "Fleet probe", "--state", "doing",
                 "--owner", "grok")
        run_cli("--path", str(self.root), "fleet", "register")
        data = json.loads(run_cli("--path", str(self.root), "fleet", "--json").out)
        doing = data["projects"][0]["doing"]
        self.assertEqual(len(doing), 1)
        self.assertEqual(doing[0]["owner"], "grok")


class TestIntegrity(unittest.TestCase):
    """Install integrity self-check — detection (not prevention) of in-place edits to core modules."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.man = Path(self._tmp.name) / "integrity.json"
        os.environ["CELEBORN_INTEGRITY_MANIFEST"] = str(self.man)

    def tearDown(self):
        os.environ.pop("CELEBORN_INTEGRITY_MANIFEST", None)
        self._tmp.cleanup()

    def test_unverified_without_manifest(self):
        # No manifest shipped (source/dev/editable install) → silent, never a false "modified".
        st = cb.integrity_status()
        self.assertEqual(st["state"], "unverified")
        self.assertEqual(cb._integrity_notice(), "")

    def test_write_then_ok(self):
        r = run_cli("integrity", "--write")
        self.assertIsNone(r.exit_code, r.all)
        self.assertTrue(self.man.is_file())
        man = json.loads(self.man.read_text())
        self.assertEqual(man["schema"], cb.INTEGRITY_SCHEMA)
        self.assertIn("celeborn.py", man["files"])
        self.assertEqual(cb.integrity_status()["state"], "ok")
        self.assertEqual(cb._integrity_notice(), "")
        self.assertIsNone(run_cli("integrity").exit_code)  # clean check exits 0

    def test_tamper_detected(self):
        run_cli("integrity", "--write")
        man = json.loads(self.man.read_text())
        name = "celeborn.py"
        man["files"][name] = "0" * 64                    # simulate an edited module
        self.man.write_text(json.dumps(man))
        st = cb.integrity_status()
        self.assertEqual(st["state"], "modified")
        self.assertIn(name, st["modified"])
        self.assertIn("modified install", cb._integrity_notice().lower())
        # `celeborn integrity` should exit non-zero and name the file.
        r = run_cli("integrity")
        self.assertEqual(r.exit_code, 1)
        self.assertIn(name, r.all)

    def test_version_mismatch_is_unverified(self):
        # A manifest from a different version can't be trusted → skip, don't false-alarm.
        run_cli("integrity", "--write")
        man = json.loads(self.man.read_text())
        man["version"] = "0.0.0-not-this-build"
        self.man.write_text(json.dumps(man))
        self.assertEqual(cb.integrity_status()["state"], "unverified")


class TestCheckpoint(CelebornTestCase):
    """`celeborn checkpoint` is the only safe writer for session.json — valid JSON, clipping, repair."""

    def _session(self) -> dict:
        return json.loads((self.ctx / "session.json").read_text())

    def test_updates_only_given_fields_and_stamps(self):
        # Seed a known-old stamp so the assertion doesn't hinge on sub-second timing.
        before = self._session()
        before["updated_at"] = "2000-01-01T00:00:00"
        before["next_action"] = "pre-existing next"
        (self.ctx / "session.json").write_text(json.dumps(before))
        r = self.cli("checkpoint", "--focus", "wiring the auth flow", "--status", "green")
        self.assertIsNone(r.exit_code, r.all)
        data = self._session()  # still valid JSON
        self.assertEqual(data["focus"], "wiring the auth flow")
        self.assertEqual(data["status"], "green")
        self.assertNotEqual(data["updated_at"], "2000-01-01T00:00:00")
        # untouched fields survive
        self.assertEqual(data.get("next_action", ""), "pre-existing next")

    def test_repairs_corrupt_session_json(self):
        (self.ctx / "session.json").write_text('{"focus": "half a thought", oops not json')
        r = self.cli("checkpoint", "--next", "ship it")
        self.assertIsNone(r.exit_code, r.all)
        data = self._session()  # parses again
        self.assertEqual(data["next_action"], "ship it")
        self.assertEqual(data.get("schema"), "celeborn/1")
        self.assertIn("invalid", r.all.lower())  # warned about the repair

    def test_clips_overlong_focus(self):
        limit = int(cb.load_config(self.ctx).get("hot_focus_max_chars", 1500))
        r = self.cli("checkpoint", "--focus", "x " * (limit * 2))
        self.assertIsNone(r.exit_code, r.all)
        focus = self._session()["focus"]
        self.assertLessEqual(len(focus), limit + 60)  # clipped + marker
        self.assertIn("clipped", focus)
        self.assertIn("clipped", r.all.lower())  # warned the user

    def test_no_flags_restamps_and_stays_valid(self):
        r = self.cli("checkpoint")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(self._session().get("schema"), "celeborn/1")

    def test_stop_allowed_toggle(self):
        self.cli("checkpoint", "--no-stop-allowed")
        self.assertFalse(self._session()["stop_allowed"])
        self.cli("checkpoint", "--stop-allowed")
        self.assertTrue(self._session()["stop_allowed"])


class TestSyncJsonGuard(CelebornTestCase):
    """Corrupt JSON must not propagate through sync in either direction."""

    def test_invalid_json_helper(self):
        self.assertTrue(cs._json_intact("notes.md", "not json but fine"))
        self.assertTrue(cs._json_intact("session.json", '{"ok": true}'))
        self.assertFalse(cs._json_intact("session.json", "{broken"))

    def test_push_skips_invalid_json(self):
        (self.ctx / "session.json").write_text("{not valid")
        rows, _ = cs.build_push_rows(self.ctx, "pid", [])
        self.assertNotIn("session.json", {r["path"] for r in rows})

    def test_pull_keeps_local_when_remote_invalid(self):
        good = '{"schema": "celeborn/1", "focus": "local good"}'
        (self.ctx / "session.json").write_text(good)
        orig = cs._http
        cs._http = lambda *a, **k: (200, [{"path": "session.json", "content": "{corrupt", "version": "9"}])
        try:
            written = cs._pull(self.ctx, {"url": "https://x", "anon": "a"}, "jwt", "pid")
        finally:
            cs._http = orig
        self.assertEqual(written, 0)
        self.assertEqual((self.ctx / "session.json").read_text(), good)  # local untouched


class TestRunTracker(CelebornTestCase):
    """`celeborn run` — real-time swarm tracker (run.json dir + blackboard)."""

    def _worker(self, wid: str) -> dict:
        return json.loads((self.ctx / "run" / f"w-{wid}.json").read_text())

    def test_start_writes_meta_and_blackboard(self):
        r = self.cli("run", "start", "myrun", "--goal", "do stuff", "--shards", "3", "--units", "75")
        self.assertIsNone(r.exit_code, r.all)
        meta = json.loads((self.ctx / "run" / "meta.json").read_text())
        self.assertEqual(meta["run_id"], "myrun")
        self.assertEqual(meta["totals"], {"shards": 3, "units": 75})
        self.assertTrue((self.ctx / "run" / "blackboard.md").is_file())

    def test_start_clears_prior_workers(self):
        self.cli("run", "start", "r1", "--shards", "1")
        self.cli("run", "beat", "--worker", "w1", "--done", "5", "--total", "10")
        self.assertTrue((self.ctx / "run" / "w-w1.json").is_file())
        self.cli("run", "start", "r2")  # fresh run clears prior workers
        self.assertFalse((self.ctx / "run" / "w-w1.json").is_file())

    def test_start_keep_preserves_workers(self):
        self.cli("run", "start", "r1")
        self.cli("run", "beat", "--worker", "w1", "--done", "1")
        self.cli("run", "start", "r2", "--keep")
        self.assertTrue((self.ctx / "run" / "w-w1.json").is_file())

    def test_beat_upserts_progress_and_sources(self):
        self.cli("run", "start", "r1")
        self.cli("run", "beat", "--worker", "ik_00", "--shard", "s0", "--phase", "Crosswalk",
                 "--item", "erlotinib", "--done", "5", "--total", "25", "--found", "5",
                 "--source-ok", "wikidata")
        w = self._worker("ik_00")
        self.assertEqual(w["current_item"], "erlotinib")
        self.assertEqual(w["progress"], {"done": 5, "total": 25, "found": 5})
        self.assertEqual(w["sources"]["wikidata"]["ok"], 1)
        self.assertEqual(w["status"], "working")
        # a second beat accumulates source counters and keeps first_beat_at
        first_beat = w["first_beat_at"]
        self.cli("run", "beat", "--worker", "ik_00", "--done", "12", "--source-fail", "pubchem",
                 "--source-rl", "gsrs")
        w2 = self._worker("ik_00")
        self.assertEqual(w2["first_beat_at"], first_beat)
        self.assertEqual(w2["progress"]["done"], 12)
        self.assertEqual(w2["sources"]["pubchem"]["fail"], 1)
        self.assertEqual(w2["sources"]["gsrs"]["ratelimited"], 1)

    def test_beat_requires_worker(self):
        self.cli("run", "start", "r1")
        r = self.cli("run", "beat", "--done", "1")  # argparse: --worker required
        self.assertIsNotNone(r.exit_code)

    def test_done_and_fail_set_terminal_status(self):
        self.cli("run", "start", "r1")
        self.cli("run", "beat", "--worker", "a", "--done", "1", "--total", "5")
        self.cli("run", "done", "--worker", "a", "--found", "5", "--missed", "0", "--done", "5")
        self.assertEqual(self._worker("a")["status"], "done")
        self.assertIn("finished_at", self._worker("a"))
        self.cli("run", "beat", "--worker", "b", "--done", "1")
        self.cli("run", "fail", "--worker", "b", "--error", "boom")
        wb = self._worker("b")
        self.assertEqual(wb["status"], "failed")
        self.assertEqual(wb["last_error"], "boom")

    def test_worker_live_status_from_beat_age(self):
        # done/failed are sticky regardless of age
        self.assertEqual(cb._worker_live_status({"status": "done", "last_beat_at": "2000-01-01T00:00:00"}), "done")
        fresh = cb.now_iso()
        self.assertEqual(cb._worker_live_status({"status": "working", "last_beat_at": fresh}), "working")
        old = (cb._dt.datetime.now() - cb._dt.timedelta(seconds=cb._RUN_STUCK_SECONDS + 30)).strftime("%Y-%m-%dT%H:%M:%S")
        self.assertEqual(cb._worker_live_status({"status": "working", "last_beat_at": old}), "stuck")
        mid = (cb._dt.datetime.now() - cb._dt.timedelta(seconds=cb._RUN_WORKING_SECONDS + 10)).strftime("%Y-%m-%dT%H:%M:%S")
        self.assertEqual(cb._worker_live_status({"status": "working", "last_beat_at": mid}), "lagging")

    def test_learn_dedup_and_listing(self):
        self.cli("run", "start", "r1")
        self.cli("run", "learn", "Sleep 0.3s between GSRS calls", "--worker", "ik_01")
        self.cli("run", "learn", "sleep 0.3s   between  GSRS   calls", "--worker", "ik_09")  # dup (norm)
        rows = cb._read_blackboard(self.ctx, limit=50)
        lessons = [r["lesson"] for r in rows]
        self.assertEqual(len(lessons), 1)
        r = self.cli("run", "learnings")
        self.assertIn("Sleep 0.3s between GSRS calls", r.out)

    def test_blackboard_skips_comment_header(self):
        self.cli("run", "start", "r1")
        # the template header is an HTML comment block — it must not parse as lessons
        self.assertEqual(cb._read_blackboard(self.ctx, limit=50), [])

    def test_status_json_rollup(self):
        self.cli("run", "start", "r1", "--shards", "3", "--units", "75")
        self.cli("run", "beat", "--worker", "a", "--done", "10", "--total", "25", "--found", "9",
                 "--missed", "1", "--source-ok", "wikidata")
        self.cli("run", "done", "--worker", "b", "--found", "25", "--missed", "0", "--done", "25", "--total", "25")
        r = self.cli("run", "status", "--json")
        snap = json.loads(r.out)
        self.assertEqual(snap["run_id"], "r1")
        self.assertEqual(snap["workers_total"], 2)
        self.assertEqual(snap["workers_finished"], 1)
        self.assertEqual(snap["resolved"]["found"], 34)
        self.assertEqual(snap["resolved"]["done"], 35)
        self.assertEqual(snap["by_status"]["done"], 1)
        self.assertEqual(snap["sources"]["wikidata"]["ok"], 1)

    def test_status_renders_text(self):
        self.cli("run", "start", "r1", "--goal", "GG")
        self.cli("run", "beat", "--worker", "ik_00", "--item", "erlotinib", "--done", "5", "--total", "25")
        r = self.cli("run", "status")
        self.assertIn("run r1", r.out)
        self.assertIn("ik_00", r.out)
        self.assertIn("erlotinib", r.out)


class TestAdvisorEngine(unittest.TestCase):
    """Harness-neutral advisor seam (t70 Phase 0) + the permission generalization core."""

    def setUp(self):
        os.environ.pop("CELEBORN_HARNESS", None)

    def test_active_adapter_default_is_claude(self):
        self.assertEqual(cb.active_adapter().name, "claude")
        self.assertEqual(cb.active_adapter(name="claude").name, "claude")

    def test_active_adapter_unknown_degrades_to_neutral(self):
        self.assertEqual(cb.active_adapter(name="codex-not-built-yet").name, "neutral")
        self.assertEqual(cb.active_adapter(name="neutral").name, "neutral")

    def test_env_selects_harness(self):
        os.environ["CELEBORN_HARNESS"] = "neutral"
        try:
            self.assertEqual(cb.active_adapter().name, "neutral")
        finally:
            os.environ.pop("CELEBORN_HARNESS", None)

    def test_neutral_render_is_agnostic_fallback(self):
        text, channel = cb.HarnessAdapter().render(
            "reduce-permission-friction", {"signal": "permission-friction", "count": 42})
        self.assertEqual(channel, "instruction")
        self.assertIn("42", text)
        self.assertIn("celeborn permissions --suggest", text)
        self.assertNotIn("/fewer-permission-prompts", text)  # never names a Claude skill

    def test_claude_render_names_the_claude_fix(self):
        text, channel = cb.ClaudeAdapter().render(
            "reduce-permission-friction", {"signal": "permission-friction", "count": 12})
        self.assertEqual(channel, "orient")
        self.assertIn("12", text)
        self.assertIn("celeborn permissions", text)

    def test_match_safe_family(self):
        self.assertEqual(cb._match_safe_family("celeborn tasks"), "Bash(celeborn *)")
        self.assertEqual(cb._match_safe_family("sed -n '1,5p' x.py"), "Bash(sed -n *)")
        self.assertEqual(cb._match_safe_family("python3 -m unittest tests.test_celeborn"),
                         "Bash(python3 -m unittest *)")
        # Outside the read-only/trusted set → never widened.
        self.assertIsNone(cb._match_safe_family("rm -rf build"))
        self.assertIsNone(cb._match_safe_family("git commit -m x"))
        self.assertIsNone(cb._match_safe_family("curl http://x"))
        self.assertIsNone(cb._match_safe_family("sed -i s/a/b/ f"))  # in-place edit is NOT `sed -n`

    def test_count_literal_bash_rules(self):
        allow = ["Bash(celeborn tasks)", "Bash(celeborn *)", "Bash(grep foo)",
                 "mcp__Preview__screenshot", "Bash(git --no-pager diff)"]
        # two literals: `celeborn tasks`, `grep foo`, `git --no-pager diff`  → 3
        self.assertEqual(cb._count_literal_bash_rules(allow), 3)

    def test_generalizable_excludes_unsafe(self):
        # Only literals a safe family can collapse count toward the friction signal — so once the
        # safe rules are applied the advisor goes quiet, even though un-widenable literals remain.
        allow = ["Bash(celeborn tasks)", "Bash(grep foo)",          # 2 generalizable
                 "Bash(rm -rf x)", "Bash(curl http://y)", "Bash(git commit -m z)"]  # 3 bottlenecks
        self.assertEqual(cb._count_literal_bash_rules(allow), 5)
        self.assertEqual(cb._count_generalizable_bash_rules(allow), 2)
        # an all-bottleneck list is NOT actionable friction
        self.assertEqual(cb._count_generalizable_bash_rules(
            ["Bash(rm -rf x)", "Bash(curl z)"]), 0)

    def test_generalize_collapses_safe_keeps_unsafe(self):
        allow = [
            "Bash(python3 scripts/celeborn.py status)",
            "Bash(python3 scripts/celeborn.py tasks)",
            "Bash(grep -n foo bar)",
            "Bash(sed -n '1,4p' x)",
            "Bash(celeborn *)",            # already general → preserved, not double-added
            "Bash(rm -rf node_modules)",   # unsafe literal → kept verbatim, tallied
            "Bash(git commit -m wip)",     # unsafe literal → kept verbatim, tallied
            "mcp__Preview__screenshot",    # non-Bash perm → preserved
        ]
        new_rules, generalized, skipped = cb._generalize_allow(allow)
        self.assertEqual(generalized, 4)  # 2× celeborn.py + grep + sed
        self.assertIn("Bash(python3 scripts/celeborn.py *)", new_rules)
        self.assertIn("Bash(grep *)", new_rules)
        self.assertIn("Bash(sed -n *)", new_rules)
        # unsafe literals survive untouched
        self.assertIn("Bash(rm -rf node_modules)", new_rules)
        self.assertIn("Bash(git commit -m wip)", new_rules)
        self.assertIn("mcp__Preview__screenshot", new_rules)
        # skipped ledger groups by family
        self.assertEqual(skipped.get("rm"), 1)
        self.assertEqual(skipped.get("git commit"), 1)
        self.assertEqual(sum(skipped.values()), 2)
        # the two celeborn.py literals collapse to ONE wildcard (deduped)
        self.assertEqual(new_rules.count("Bash(python3 scripts/celeborn.py *)"), 1)


class TestGrokCodexAdapters(CelebornTestCase):
    """t83 — GrokAdapter/CodexAdapter in core: selection, harness-flavored renders, the Codex
    config.toml permission lever, and the `celeborn harness` pin (rc tier of active_adapter)."""

    def setUp(self):
        super().setUp()
        os.environ.pop("CELEBORN_HARNESS", None)
        os.environ.pop("CODEX_HOME", None)

    def tearDown(self):
        os.environ.pop("CELEBORN_HARNESS", None)
        os.environ.pop("CODEX_HOME", None)
        super().tearDown()

    def test_active_adapter_selects_grok_and_codex(self):
        self.assertEqual(cb.active_adapter(name="grok").name, "grok")
        self.assertEqual(cb.active_adapter(name="codex").name, "codex")
        self.assertIsInstance(cb.active_adapter(name="grok"), cb.GrokAdapter)
        self.assertIsInstance(cb.active_adapter(name="codex"), cb.CodexAdapter)

    def test_env_var_selects_grok_codex(self):
        os.environ["CELEBORN_HARNESS"] = "grok"
        self.assertEqual(cb.active_adapter().name, "grok")
        os.environ["CELEBORN_HARNESS"] = "codex"
        self.assertEqual(cb.active_adapter().name, "codex")

    def test_rc_tier_selects_harness(self):
        cb._update_config(self.ctx, harness="codex")
        self.assertEqual(cb.active_adapter(self.ctx).name, "codex")
        # explicit name and env still outrank rc
        self.assertEqual(cb.active_adapter(self.ctx, name="grok").name, "grok")

    def test_grok_render_has_no_claude_slash_commands(self):
        a = cb.GrokAdapter()
        for intent, sig in (("review-changes", {"count": 3}),
                            ("security-review-changes", {"files": "auth.py"}),
                            ("parallelize-large-changeset", {"count": 14})):
            text, channel = a.render(intent, sig)
            self.assertEqual(channel, "orient")
            self.assertIn("🏹 Celeborn advisor", text)
            self.assertNotIn("/code-review", text)
            self.assertNotIn("/security-review", text)
            self.assertNotIn("/verify", text)
        # on-demand throughput intents render as instructions, still slash-free
        text, channel = a.render("spawn-tangent", None)
        self.assertEqual(channel, "instruction")
        self.assertNotIn("spawn_task", text)  # that's the Claude tool name; grok stays generic

    def test_grok_has_no_permission_lever(self):
        # Grok's rules file isn't an allow-list, so it inherits the neutral (no-target) lever.
        target, how = cb.GrokAdapter().permission_target(self.ctx)
        self.assertIsNone(target)

    def test_codex_permission_target_is_coarse_config_toml(self):
        target, how = cb.CodexAdapter().permission_target(self.ctx)
        self.assertEqual(how, "workspace-trust")
        self.assertTrue(str(target).endswith("config.toml"))

    def test_codex_friction_flags_untrusted_then_clears_on_trust(self):
        home = self.root / "codexhome"
        home.mkdir()
        os.environ["CODEX_HOME"] = str(home)
        a = cb.CodexAdapter()
        # no config at all → interactive on-request, untrusted → friction present
        sigs = a.friction_signals(self.ctx)
        self.assertIn("permission-friction", [s["signal"] for s in sigs])
        # trust THIS project → friction clears
        (home / "config.toml").write_text(
            f'[projects."{self.root.resolve()}"]\ntrust_level = "trusted"\n')
        self.assertNotIn("permission-friction",
                         [s["signal"] for s in a.friction_signals(self.ctx)])
        # approval_policy=never (non-interactive) also clears it
        (home / "config.toml").write_text('approval_policy = "never"\n')
        self.assertNotIn("permission-friction",
                         [s["signal"] for s in a.friction_signals(self.ctx)])

    def test_codex_render_permission_hint_points_at_config(self):
        text, channel = cb.CodexAdapter().render(
            "reduce-permission-friction",
            {"approval_policy": "on-request", "config": "/home/me/.codex/config.toml"})
        self.assertEqual(channel, "orient")
        self.assertIn("trust_level", text)
        self.assertIn("/home/me/.codex/config.toml", text)
        self.assertNotIn("/fewer-permission-prompts", text)  # never a Claude skill

    def test_harness_verb_pins_rc(self):
        r = self.cli("harness", "codex")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual((cb.load_config(self.ctx).get("harness") or "").lower(), "codex")
        self.assertEqual(cb.active_adapter(self.ctx).name, "codex")
        # re-pin is idempotent and switches cleanly
        self.cli("harness", "grok")
        self.assertEqual(cb.active_adapter(self.ctx).name, "grok")

    def test_harness_verb_rejects_unknown(self):
        r = self.cli("harness", "bogus-harness")
        self.assertIsNotNone(r.exit_code)

    def test_init_never_pins_harness(self):
        # Regression: core `init` speculatively wires Grok on any machine with Grok installed (even
        # Claude-primary repos). That wiring must pass --no-harness-pin so it does NOT write
        # harness=grok to .celebornrc and override the default-claude resolution.
        self.assertIsNone(cb.load_config(self.ctx).get("harness"))
        self.assertEqual(cb.active_adapter(self.ctx).name, "claude")


class TestAdvisorNotice(CelebornTestCase):
    """`_advisor_notice` + `celeborn advise` — friction detection, one-nudge-per-session throttle."""

    def setUp(self):
        super().setUp()
        os.environ.pop("CELEBORN_HARNESS", None)

    def _write_settings(self, n_literals: int, shared: bool = False):
        claude = self.root / ".claude"
        claude.mkdir(exist_ok=True)
        allow = [f"Bash(python3 scripts/celeborn.py remind --tokens {i})" for i in range(n_literals)]
        fname = "settings.json" if shared else "settings.local.json"
        (claude / fname).write_text(json.dumps({"permissions": {"allow": allow}}))

    def test_no_notice_when_lean(self):
        self._write_settings(2)  # below the default threshold of 10
        self.assertEqual(cb._advisor_notice(self.ctx, "sess-1"), "")

    def test_notice_fires_on_bloat_then_throttles(self):
        self._write_settings(15)
        first = cb._advisor_notice(self.ctx, "sess-1")
        self.assertIn("permission", first.lower())
        self.assertIn("15", first)
        # Same session → silent (throttled to one nudge/session).
        self.assertEqual(cb._advisor_notice(self.ctx, "sess-1"), "")
        # A new session nudges again.
        self.assertIn("permission", cb._advisor_notice(self.ctx, "sess-2").lower())

    def test_disabled_via_config(self):
        self._write_settings(15)
        (self.ctx / cb.RC_NAME).write_text(json.dumps({"advisor_enabled": False}))
        self.assertEqual(cb._advisor_notice(self.ctx, "sess-1"), "")

    def test_disabled_via_nested_config(self):
        self._write_settings(15)
        (self.ctx / cb.RC_NAME).write_text(json.dumps({"advisor": {"enabled": False}}))
        self.assertEqual(cb._advisor_notice(self.ctx, "sess-1"), "")

    def test_nested_threshold_overrides_default(self):
        self._write_settings(15)  # would fire at the default threshold of 10
        (self.ctx / cb.RC_NAME).write_text(json.dumps({"advisor": {"permission_bloat_min": 100}}))
        self.assertEqual(cb._advisor_notice(self.ctx, "sess-1"), "")

    def test_legacy_flat_threshold_still_honored(self):
        self._write_settings(15)
        (self.ctx / cb.RC_NAME).write_text(json.dumps({"advisor_permission_bloat_min": 100}))
        self.assertEqual(cb._advisor_notice(self.ctx, "sess-1"), "")

    def test_max_per_session_allows_multiple_then_throttles(self):
        self._write_settings(15)
        (self.ctx / cb.RC_NAME).write_text(json.dumps({"advisor": {"max_per_session": 2}}))
        self.assertIn("permission", cb._advisor_notice(self.ctx, "sess-1").lower())
        self.assertIn("permission", cb._advisor_notice(self.ctx, "sess-1").lower())  # 2nd nudge OK
        self.assertEqual(cb._advisor_notice(self.ctx, "sess-1"), "")                 # 3rd → capped

    def test_advisor_config_normalizes_and_deep_fills(self):
        (self.ctx / cb.RC_NAME).write_text(json.dumps({"advisor": {"enabled": False}}))
        cfg = cb._advisor_config(self.ctx)
        self.assertFalse(cfg["enabled"])
        # unspecified sub-keys are still present (deep-filled from DEFAULTS, not dropped)
        self.assertEqual(cfg["permission_bloat_min"], 10)
        self.assertEqual(cfg["max_per_session"], 1)
        self.assertIn("sensitive_globs", cfg)

    def test_advise_cli_reports_friction(self):
        self._write_settings(20)
        r = self.cli("advise")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("recommendation", r.all.lower())
        self.assertIn("permissions", r.all)

    def test_advise_cli_clean_when_lean(self):
        self._write_settings(1)
        r = self.cli("advise")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("no friction", r.all.lower())

    def test_advise_lists_intent_id(self):
        self._write_settings(20)
        r = self.cli("advise")
        self.assertIn("[reduce-permission-friction]", r.all)

    def test_dismiss_silences_notice_and_advise(self):
        self._write_settings(15)
        r = self.cli("advise", "--dismiss", "reduce-permission-friction")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("dismissed", r.all.lower())
        # the SessionStart notice is now silent for that intent
        self.assertEqual(cb._advisor_notice(self.ctx, "fresh-session"), "")
        # and `advise` reports nothing actionable (but counts the dismissal)
        r2 = self.cli("advise")
        self.assertIn("no friction", r2.all.lower())
        self.assertIn("1 dismissed", r2.all)

    def test_restore_reenables_notice(self):
        self._write_settings(15)
        self.cli("advise", "--dismiss", "reduce-permission-friction")
        r = self.cli("advise", "--restore", "reduce-permission-friction")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("restored", r.all.lower())
        self.assertIn("permission", cb._advisor_notice(self.ctx, "fresh-session").lower())

    def test_dismiss_unknown_id_errors(self):
        r = self.cli("advise", "--dismiss", "no-such-intent")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("unknown recommendation id", r.all.lower())


class TestPermissions(CelebornTestCase):
    """`celeborn permissions --suggest|--apply` — generalize safe families, keep + ledger the rest."""

    def setUp(self):
        super().setUp()
        os.environ.pop("CELEBORN_HARNESS", None)
        self.claude = self.root / ".claude"
        self.claude.mkdir(exist_ok=True)

    def _write(self, allow: list, shared: bool = False):
        fname = "settings.json" if shared else "settings.local.json"
        (self.claude / fname).write_text(json.dumps({"permissions": {"allow": allow}}))

    def _read(self, shared: bool = False) -> list:
        fname = "settings.json" if shared else "settings.local.json"
        return json.loads((self.claude / fname).read_text())["permissions"]["allow"]

    def _mixed(self) -> list:
        return ([f"Bash(grep pat{i} f)" for i in range(6)]
                + [f"Bash(python3 scripts/celeborn.py x{i})" for i in range(4)]
                + ["Bash(rm -rf dist)", "Bash(curl http://x)", "Bash(git commit -m y)"])

    def test_suggest_is_read_only(self):
        before = self._mixed()
        self._write(before)
        r = self.cli("permissions", "--suggest")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(self._read(), before)  # file untouched
        self.assertIn("skipped bottlenecks", r.all.lower())

    def test_apply_generalizes_and_keeps_unsafe(self):
        self._write(self._mixed())
        r = self.cli("permissions", "--apply", "--yes")
        self.assertIsNone(r.exit_code, r.all)
        now = self._read()
        self.assertIn("Bash(grep *)", now)
        self.assertIn("Bash(python3 scripts/celeborn.py *)", now)
        # unsafe literals preserved verbatim
        self.assertIn("Bash(rm -rf dist)", now)
        self.assertIn("Bash(curl http://x)", now)
        self.assertIn("Bash(git commit -m y)", now)
        # 10 safe literals collapsed into 2 wildcards → list got shorter
        self.assertLess(len(now), len(self._mixed()))
        # ledger persisted to metrics
        m = json.loads((self.ctx / cb.METRICS_NAME).read_text())
        self.assertEqual(m["advisor"]["permission_rules_generalized"], 10)
        self.assertEqual(m["advisor"]["skipped_bottlenecks_total"], 3)
        self.assertEqual(m["advisor"]["skipped_bottlenecks"].get("rm"), 1)

    def test_apply_surfaces_in_savings_json(self):
        self._write(self._mixed())
        self.cli("permissions", "--apply", "--yes")
        r = self.cli("savings", "--json")
        data = json.loads(r.out)
        self.assertEqual(data["project"]["advisor"]["permission_rules_generalized"], 10)
        self.assertEqual(data["project"]["advisor"]["skipped_bottlenecks_total"], 3)

    def test_apply_shared_targets_committed_settings(self):
        self._write(self._mixed(), shared=True)
        r = self.cli("permissions", "--apply", "--shared", "--yes")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("Bash(grep *)", self._read(shared=True))
        # the personal file was never created
        self.assertFalse((self.claude / "settings.local.json").is_file())

    def test_apply_without_yes_aborts_on_no(self):
        before = self._mixed()
        self._write(before)
        with mock.patch("builtins.input", return_value="n"):
            r = self.cli("permissions", "--apply")
        self.assertIsNone(r.exit_code, r.all)
        self.assertEqual(self._read(), before)  # unchanged
        self.assertIn("aborted", r.all.lower())

    def test_refuses_invalid_json(self):
        (self.claude / "settings.local.json").write_text("{ not json")
        r = self.cli("permissions", "--suggest")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("valid json", r.all.lower())


import celeborn_cmm as cm  # noqa: E402
import celeborn_cmm_provision as cmp  # noqa: E402


class TestCmmMerge(unittest.TestCase):
    """CMM-1: the pure permission merge — the whole feature's safety, de-risked in isolation."""

    def test_exact_tool_id_sets_verbatim(self):
        # A typo'd mcp__codebase-memory-mcp__… id silently fails to clear a prompt — pin the
        # contract from spec §5 verbatim. 11 read-only MCP tools + Grep/Glob in allow; 3 in ask.
        self.assertEqual(cm.CMM_ALLOW_TOOLS, (
            "Grep",
            "Glob",
            "mcp__codebase-memory-mcp__search_graph",
            "mcp__codebase-memory-mcp__query_graph",
            "mcp__codebase-memory-mcp__trace_path",
            "mcp__codebase-memory-mcp__get_architecture",
            "mcp__codebase-memory-mcp__get_graph_schema",
            "mcp__codebase-memory-mcp__get_code_snippet",
            "mcp__codebase-memory-mcp__search_code",
            "mcp__codebase-memory-mcp__detect_changes",
            "mcp__codebase-memory-mcp__list_projects",
            "mcp__codebase-memory-mcp__index_status",
            "mcp__codebase-memory-mcp__index_repository",
        ))
        self.assertEqual(cm.CMM_ASK_TOOLS, (
            "mcp__codebase-memory-mcp__delete_project",
            "mcp__codebase-memory-mcp__manage_adr",
            "mcp__codebase-memory-mcp__ingest_traces",
        ))
        # 14 CMM tools total (11 allow + 3 ask), Grep/Glob excluded from the CMM-tool count.
        self.assertEqual(len(cm.CMM_ALL_TOOLS), 14)

    def test_empty_input_produces_full_block(self):
        merged = cm.merge_cmm_permissions({})
        self.assertEqual(set(merged["allow"]), set(cm.CMM_ALLOW_TOOLS))
        self.assertEqual(set(merged["ask"]), set(cm.CMM_ASK_TOOLS))

    def test_idempotent(self):
        once = cm.merge_cmm_permissions({})
        twice = cm.merge_cmm_permissions(once)
        self.assertEqual(once, twice)

    def test_preserves_user_entries(self):
        existing = {"allow": ["Bash(npm run test:*)", "Read"],
                    "ask": ["WebFetch"], "deny": ["Bash(rm:*)"]}
        merged = cm.merge_cmm_permissions(existing)
        for keep in ("Bash(npm run test:*)", "Read"):
            self.assertIn(keep, merged["allow"])
        self.assertIn("WebFetch", merged["ask"])
        self.assertEqual(merged["deny"], ["Bash(rm:*)"])  # untouched bucket preserved verbatim

    def test_ask_wins_over_stale_allow(self):
        # A user who mistakenly allowed a mutating tool must still be prompted: it moves to ask.
        existing = {"allow": ["mcp__codebase-memory-mcp__delete_project", "Read"]}
        merged = cm.merge_cmm_permissions(existing)
        self.assertNotIn("mcp__codebase-memory-mcp__delete_project", merged["allow"])
        self.assertIn("mcp__codebase-memory-mcp__delete_project", merged["ask"])
        self.assertIn("Read", merged["allow"])

    def test_no_duplicates(self):
        existing = {"allow": ["Grep", "Grep", "mcp__codebase-memory-mcp__search_graph"]}
        merged = cm.merge_cmm_permissions(existing)
        self.assertEqual(len(merged["allow"]), len(set(merged["allow"])))
        self.assertEqual(len(merged["ask"]), len(set(merged["ask"])))

    def test_provenance_and_revert_roundtrip(self):
        before = {"allow": ["Grep", "Read"], "ask": []}  # user already owns Grep
        after = cm.merge_cmm_permissions(before)
        prov = cm.permission_provenance(before, after)
        # Grep was the user's — NOT in provenance, so revert must not strip it.
        self.assertNotIn("Grep", prov["allow_added"])
        self.assertIn("mcp__codebase-memory-mcp__search_graph", prov["allow_added"])
        reverted = cm.revert_cmm_permissions(after, prov)
        self.assertIn("Grep", reverted["allow"])
        self.assertIn("Read", reverted["allow"])
        self.assertNotIn("mcp__codebase-memory-mcp__search_graph", reverted["allow"])
        for t in cm.CMM_ASK_TOOLS:
            self.assertNotIn(t, reverted["ask"])


class TestCmmSchemaGuard(unittest.TestCase):
    """CMM-2: validate before write — never emit a malformed settings file."""

    def test_accepts_well_formed(self):
        ok, why = cm.schema_guard({"permissions": {"allow": ["Read"], "ask": []}})
        self.assertTrue(ok, why)
        self.assertIsNone(why)

    def test_accepts_missing_permissions(self):
        self.assertTrue(cm.schema_guard({})[0])

    def test_rejects_permissions_not_object(self):
        ok, why = cm.schema_guard({"permissions": ["Read"]})
        self.assertFalse(ok)
        self.assertIn("not a JSON object", why)

    def test_rejects_bucket_not_list(self):
        ok, why = cm.schema_guard({"permissions": {"allow": "Read"}})
        self.assertFalse(ok)
        self.assertIn("allow", why)

    def test_rejects_non_string_entry(self):
        ok, why = cm.schema_guard({"permissions": {"allow": [123]}})
        self.assertFalse(ok)
        self.assertIn("non-string", why)

    def test_our_ids_match_identifier_format(self):
        for tid in (*cm.CMM_ALLOW_TOOLS, *cm.CMM_ASK_TOOLS):
            self.assertRegex(tid, cm._TOOL_ID_RE)


class TestCmmEngage(CelebornTestCase):
    """CMM-4/5 end-to-end against a temp project: engage writes the allow-list to the shared
    settings.json, installs the North Star, and `off` reverts only what engage added."""

    def _settings(self) -> Path:
        return self.root / ".claude" / "settings.json"

    def _perms(self) -> dict:
        p = self._settings()
        if not p.is_file():
            return {}
        return (json.loads(p.read_text()).get("permissions") or {})

    def test_engage_writes_allowlist_and_north_star(self):
        r = self.cli("cmm", "engage")
        self.assertIsNone(r.exit_code, r.all)
        perms = self._perms()
        for t in cm.CMM_ALLOW_TOOLS:
            self.assertIn(t, perms.get("allow", []))
        for t in cm.CMM_ASK_TOOLS:
            self.assertIn(t, perms.get("ask", []))
            self.assertNotIn(t, perms.get("allow", []))  # ask wins
        # North Star installed into the agent-facing instructions, not just the planning docs.
        claude_md = (self.root / "CLAUDE.md").read_text()
        self.assertIn(cm.CMM_MD_BEGIN, claude_md)
        self.assertIn("North Star", claude_md)
        self.assertIn("flow", claude_md.lower())
        # MCP server REGISTRATION must NOT leak into the portable settings.json (Decision 3) —
        # no mcpServers block, no binary command path. The tool ids may contain the server name.
        settings = json.loads(self._settings().read_text())
        self.assertNotIn("mcpServers", settings)

    def test_engage_idempotent(self):
        self.cli("cmm", "engage")
        first = self._settings().read_text()
        self.cli("cmm", "engage")
        self.assertEqual(self._settings().read_text(), first)
        # No duplicated managed blocks in CLAUDE.md either.
        self.assertEqual((self.root / "CLAUDE.md").read_text().count(cm.CMM_MD_BEGIN), 1)

    def test_engage_preserves_user_permission(self):
        self._settings().parent.mkdir(parents=True, exist_ok=True)
        self._settings().write_text(json.dumps(
            {"permissions": {"allow": ["Bash(npm run build:*)"]}}, indent=2))
        self.cli("cmm", "engage")
        self.assertIn("Bash(npm run build:*)", self._perms().get("allow", []))

    def test_off_reverts_added_entries_and_sets_opt_out(self):
        self.cli("cmm", "engage")
        r = self.cli("cmm", "off")
        self.assertIsNone(r.exit_code, r.all)
        perms = self._perms()
        for t in cm.CMM_ALLOW_TOOLS:
            if t in cm.CMM_NATIVE_TOOLS:
                continue  # Grep/Glob handled by provenance; engage added them here so they go too
            self.assertNotIn(t, perms.get("allow", []))
        # Sticky opt-out: a plain re-engage is refused.
        r2 = self.cli("cmm", "engage")
        self.assertIn("opted out", r2.all.lower())
        self.assertFalse(any(t in self._perms().get("allow", []) for t in cm.CMM_ALLOW_TOOLS
                             if t not in cm.CMM_NATIVE_TOOLS))
        # North Star block removed.
        self.assertNotIn(cm.CMM_MD_BEGIN, (self.root / "CLAUDE.md").read_text())

    def test_off_does_not_strip_user_owned_grep(self):
        self._settings().parent.mkdir(parents=True, exist_ok=True)
        self._settings().write_text(json.dumps(
            {"permissions": {"allow": ["Grep"]}}, indent=2))  # user owns Grep before engage
        self.cli("cmm", "engage")
        self.cli("cmm", "off")
        self.assertIn("Grep", self._perms().get("allow", []))  # user's Grep survives

    def test_force_reengage_after_opt_out(self):
        self.cli("cmm", "engage")
        self.cli("cmm", "off")
        r = self.cli("cmm", "engage", "--force")
        self.assertIsNone(r.exit_code, r.all)
        self.assertTrue(all(t in self._perms().get("allow", []) for t in cm.CMM_ALLOW_TOOLS))

    def test_status_json(self):
        self.cli("cmm", "engage")
        r = self.cli("cmm", "status", "--json")
        self.assertIsNone(r.exit_code, r.all)
        doc = json.loads(r.out)
        self.assertTrue(doc["engaged"])
        self.assertTrue(doc["allow_list_present"])
        self.assertTrue(doc["ask_list_present"])

    def test_skips_malformed_settings(self):
        self._settings().parent.mkdir(parents=True, exist_ok=True)
        self._settings().write_text(json.dumps({"permissions": {"allow": "not-a-list"}}, indent=2))
        r = self.cli("cmm", "engage")
        self.assertIsNone(r.exit_code, r.all)  # never errors mid-flow
        self.assertIn("skip", r.all.lower())
        # The malformed file was NOT overwritten with our block.
        self.assertEqual(json.loads(self._settings().read_text())["permissions"]["allow"], "not-a-list")


class TestCmmEconomics(CelebornTestCase):
    """The 'prompts auto-allowed' estimate: each capture credits the agent's calls to a CMM-pre-cleared
    tool as a permission prompt avoided, and the economics report sums it (CELE-t92 follow-up)."""

    def _turns(self, *names):
        return [{"events": [{"kind": "tool_use", "name": n} for n in names]}]

    def test_count_auto_allowed_matches_only_allowlist(self):
        allow = {"Grep", "mcp__codebase-memory-mcp__search_graph"}
        turns = self._turns("Grep", "Bash", "mcp__codebase-memory-mcp__search_graph", "Edit", "Grep")
        self.assertEqual(cb._count_auto_allowed(turns, allow), 3)  # 2 Grep + 1 search_graph

    def test_count_auto_allowed_zero_without_allowlist(self):
        turns = self._turns("Grep", "mcp__codebase-memory-mcp__search_graph")
        self.assertEqual(cb._count_auto_allowed(turns, set()), 0)

    def test_credited_tool_names_gated_on_engaged(self):
        # Not engaged → nothing credited (provenance gate).
        self.assertEqual(cm.credited_tool_names(self.ctx), set())
        self.cli("cmm", "engage")
        names = cm.credited_tool_names(self.ctx)
        self.assertIn("mcp__codebase-memory-mcp__search_graph", names)
        self.assertIn("Grep", names)
        # Opted out → nothing credited even though entries were once added.
        self.cli("cmm", "off")
        self.assertEqual(cm.credited_tool_names(self.ctx), set())

    def test_credited_excludes_user_owned_grep(self):
        # User already owns Grep before engage → engage's provenance doesn't include it, so a Grep
        # call is NOT credited as our win.
        s = self.root / ".claude" / "settings.json"
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps({"permissions": {"allow": ["Grep"]}}, indent=2))
        self.cli("cmm", "engage")
        self.assertNotIn("Grep", cm.credited_tool_names(self.ctx))
        self.assertIn("mcp__codebase-memory-mcp__search_graph", cm.credited_tool_names(self.ctx))

    def test_prompts_auto_allowed_combines_advisor_and_cmm(self):
        m = {"advisor": {"permission_rules_generalized": 4},
             "cmm": {"prompts_auto_allowed": 7}}
        self.assertEqual(cb._prompts_auto_allowed(m), 11)
        self.assertEqual(cb._prompts_auto_allowed({}), 0)

    def _transcript(self, *entries) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for e in entries:
            f.write(json.dumps(e) + "\n")
        f.close()
        return f.name

    def test_capture_increments_metric_when_engaged(self):
        self.cli("cmm", "engage")
        tp = self._transcript(
            {"type": "user", "uuid": "u1", "sessionId": "s1",
             "message": {"role": "user", "content": "where is greet defined?"}},
            {"type": "assistant", "uuid": "a1", "sessionId": "s1",
             "message": {"role": "assistant", "content": [
                 {"type": "tool_use", "id": "t1", "name": "mcp__codebase-memory-mcp__search_graph",
                  "input": {"q": "greet"}},
                 {"type": "tool_use", "id": "t2", "name": "Grep", "input": {"pattern": "greet"}},
                 {"type": "tool_use", "id": "t3", "name": "Bash", "input": {"command": "ls"}},
             ]}},
        )
        r = self.cli("capture", "--transcript", tp, "--session", "s1", "--quiet")
        self.assertIsNone(r.exit_code, r.all)
        m = cb._load_metrics(self.ctx)
        # search_graph + Grep credited (both engage-added); Bash is not pre-cleared → not credited.
        self.assertEqual((m.get("cmm") or {}).get("prompts_auto_allowed"), 2)
        os.unlink(tp)

    def test_capture_no_increment_without_engage(self):
        tp = self._transcript(
            {"type": "user", "uuid": "u1", "sessionId": "s2",
             "message": {"role": "user", "content": "look"}},
            {"type": "assistant", "uuid": "a1", "sessionId": "s2",
             "message": {"role": "assistant", "content": [
                 {"type": "tool_use", "id": "t1", "name": "Grep", "input": {"pattern": "x"}}]}},
        )
        self.cli("capture", "--transcript", tp, "--session", "s2", "--quiet")
        self.assertEqual((cb._load_metrics(self.ctx).get("cmm") or {}).get("prompts_auto_allowed", 0), 0)
        os.unlink(tp)

    def test_savings_json_surfaces_prompts_auto_allowed(self):
        self.cli("cmm", "engage")
        m = cb._load_metrics(self.ctx)
        m["cmm"] = {"prompts_auto_allowed": 5}
        m.setdefault("advisor", {})["permission_rules_generalized"] = 2
        cb._save_metrics(self.ctx, m)
        r = self.cli("savings", "--json")
        doc = json.loads(r.out)
        self.assertEqual(doc["project"]["prompts_auto_allowed"], 7)        # 5 cmm + 2 advisor
        self.assertEqual(doc["project"]["cmm_prompts_auto_allowed"], 5)
        self.assertIn("prompts_auto_allowed", doc["fleet"])


class TestAllowlistEconomics(CelebornTestCase):
    """t100 — every tool call the settings.json allow-list auto-allows (the safe baseline + the user's
    own rules) is tallied into the per-project 'prompts auto-allowed' figure at capture time. Disjoint
    from the CMM bucket; surfaced through the unified `_prompts_auto_allowed`."""

    def setUp(self):
        super().setUp()
        # Neutralize the real ~/.claude so global baseline rules can't leak into the count.
        home = tempfile.mkdtemp()
        old = os.environ.get("HOME")
        self.addCleanup(lambda: os.environ.__setitem__("HOME", old) if old is not None
                        else os.environ.pop("HOME", None))
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        os.environ["HOME"] = home

    def _ev(self, name, summary=""):
        return {"kind": "tool_use", "name": name, "summary": summary}

    # ---- _bash_allow_matches: Claude's prefix semantics ----
    def test_bash_prefix_wildcard_matches(self):
        self.assertTrue(cb._bash_allow_matches("git log --oneline", "git log:*"))
        self.assertTrue(cb._bash_allow_matches("grep -n foo x.py", "grep:*"))
        self.assertTrue(cb._bash_allow_matches("curl -sS http://localhost:3000/x", "curl -sS http://localhost:*"))

    def test_bash_advisor_star_form_matches(self):
        self.assertTrue(cb._bash_allow_matches("grep -n foo", "grep *"))   # advisor's `Bash(grep *)`

    def test_bash_exact_rule_requires_exact_command(self):
        self.assertTrue(cb._bash_allow_matches("ls", "ls"))
        self.assertFalse(cb._bash_allow_matches("ls -la", "ls"))           # bare rule ≠ prefix
        self.assertFalse(cb._bash_allow_matches("rm -rf /", "git log:*"))  # unrelated

    # ---- _count_allowlist_auto_allowed: matching + dedup ----
    def test_counts_named_builtins_and_bash_prefixes(self):
        allow = ["Read", "Glob", "Grep", "Bash(grep:*)", "Bash(git log:*)"]
        turns = [{"events": [
            self._ev("Read", "/a.py"),
            self._ev("Bash", "grep -n foo a.py"),
            self._ev("Bash", "git log --oneline"),
            self._ev("Bash", "rm -rf build"),          # not in allow → not counted
            self._ev("Edit", "/a.py"),                  # not an allow-list entry → not counted
        ]}]
        self.assertEqual(cb._count_allowlist_auto_allowed(turns, allow, set()), 3)

    def test_excludes_cmm_credited_names(self):
        allow = ["Grep", "Glob", "Bash(ls:*)"]
        turns = [{"events": [self._ev("Grep", "x"), self._ev("Glob", "*.py"), self._ev("Bash", "ls -la")]}]
        # Grep is CMM-credited here → only Glob (allow-listed, not excluded) + the Bash(ls) call count.
        self.assertEqual(cb._count_allowlist_auto_allowed(turns, allow, {"Grep"}), 2)

    def test_zero_when_allowlist_empty(self):
        turns = [{"events": [self._ev("Read", "/a.py"), self._ev("Bash", "grep foo")]}]
        self.assertEqual(cb._count_allowlist_auto_allowed(turns, [], set()), 0)

    # ---- _effective_allow_rules: global + project union ----
    def test_effective_allow_unions_global_and_project(self):
        ghome = Path(os.environ["HOME"])
        gs = ghome / ".claude" / "settings.json"
        gs.parent.mkdir(parents=True, exist_ok=True)
        gs.write_text(json.dumps({"permissions": {"allow": ["Read", "Bash(grep:*)"]}}))
        ps = self.root / ".claude" / "settings.json"
        ps.parent.mkdir(parents=True, exist_ok=True)
        ps.write_text(json.dumps({"permissions": {"allow": ["Bash(grep:*)", "Bash(my-tool:*)"]}}))
        rules = cb._effective_allow_rules(self.ctx)
        self.assertIn("Read", rules)                    # from global baseline
        self.assertIn("Bash(my-tool:*)", rules)         # from project
        self.assertEqual(rules.count("Bash(grep:*)"), 1)  # de-duped across files

    # ---- end-to-end through capture + savings ----
    def _transcript(self, *entries) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for e in entries:
            f.write(json.dumps(e) + "\n")
        f.close()
        return f.name

    def test_capture_tallies_baseline_auto_allows_to_the_board(self):
        # The project's own settings allow grep + git log (mimicking the baseline reaching this repo).
        ps = self.root / ".claude" / "settings.json"
        ps.parent.mkdir(parents=True, exist_ok=True)
        ps.write_text(json.dumps({"permissions": {"allow": ["Bash(grep:*)", "Bash(git log:*)", "Read"]}}))
        tp = self._transcript(
            {"type": "user", "uuid": "u1", "sessionId": "sa",
             "message": {"role": "user", "content": "look around"}},
            {"type": "assistant", "uuid": "a1", "sessionId": "sa",
             "message": {"role": "assistant", "content": [
                 {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "grep -n foo a.py"}},
                 {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "git log --oneline"}},
                 {"type": "tool_use", "id": "t3", "name": "Read", "input": {"file_path": "/a.py"}},
                 {"type": "tool_use", "id": "t4", "name": "Bash", "input": {"command": "rm -rf build"}},
             ]}},
        )
        r = self.cli("capture", "--transcript", tp, "--session", "sa", "--quiet")
        self.assertIsNone(r.exit_code, r.all)
        m = cb._load_metrics(self.ctx)
        self.assertEqual((m.get("permissions") or {}).get("prompts_auto_allowed"), 3)  # grep+git log+Read; rm not
        # And it flows into the unified board figure.
        doc = json.loads(self.cli("savings", "--json").out)
        self.assertEqual(doc["project"]["prompts_auto_allowed"], 3)
        os.unlink(tp)

    def test_prompts_auto_allowed_sums_all_three_buckets(self):
        m = {"advisor": {"permission_rules_generalized": 4},
             "cmm": {"prompts_auto_allowed": 7},
             "permissions": {"prompts_auto_allowed": 5}}
        self.assertEqual(cb._prompts_auto_allowed(m), 16)


class TestCmmRealInterface(CelebornTestCase):
    """The CLI/MCP contract verified live against the real binary (HEAD e599df1): MCP registration
    uses `args: []` and stays project-scoped; index parses CMM's JSON result line."""

    def test_mcp_registration_is_project_scoped_with_empty_args(self):
        fake_bin = self.root / "fake-cmm"
        fake_bin.write_text("#!/bin/sh\n")
        with mock.patch.dict(os.environ, {"CELEBORN_CMM_BIN": str(fake_bin)}):
            res = cm.ensure_mcp_registration(self.ctx)
        self.assertEqual(res["status"], "registered")
        data = json.loads((self.root / ".mcp.json").read_text())
        entry = data["mcpServers"][cm.CMM_SERVER_NAME]
        self.assertEqual(entry["args"], [])           # bare invocation = MCP server on stdio
        self.assertEqual(entry["command"], str(fake_bin))

    def test_mcp_registration_preserves_existing_servers(self):
        (self.root / ".mcp.json").write_text(json.dumps(
            {"mcpServers": {"other": {"command": "x", "args": []}}}))
        fake_bin = self.root / "fake-cmm"
        fake_bin.write_text("#!/bin/sh\n")
        with mock.patch.dict(os.environ, {"CELEBORN_CMM_BIN": str(fake_bin)}):
            cm.ensure_mcp_registration(self.ctx)
        servers = json.loads((self.root / ".mcp.json").read_text())["mcpServers"]
        self.assertIn("other", servers)              # user's existing entry survives
        self.assertIn(cm.CMM_SERVER_NAME, servers)

    def test_parse_index_result_skips_log_lines(self):
        out = ('level=info msg=mem.init budget_mb=18432\n'
               '{"project":"p","status":"indexed","nodes":3523,"edges":9994}\n')
        parsed = cm._parse_index_result(out)
        self.assertEqual(parsed["status"], "indexed")
        self.assertEqual(parsed["nodes"], 3523)

    def test_parse_index_result_none_when_no_json(self):
        self.assertIsNone(cm._parse_index_result("level=info only\nno json here\n"))


class TestCmmPlatformAndPin(unittest.TestCase):
    """S2 provisioning: platform-key normalization + the shipped pin manifest."""

    def test_platform_key_normalizes_arch(self):
        cases = [
            ("Darwin", "arm64", "darwin-arm64"),
            ("Darwin", "x86_64", "darwin-x86_64"),
            ("Linux", "aarch64", "linux-arm64"),
            ("Linux", "amd64", "linux-x86_64"),
            ("Linux", "x86_64", "linux-x86_64"),
        ]
        for system, machine, expected in cases:
            with mock.patch("platform.system", return_value=system), \
                 mock.patch("platform.machine", return_value=machine):
                self.assertEqual(cmp.platform_key(), expected)

    def test_shipped_pin_is_valid_and_pending(self):
        # The pin-of-record ships and parses; its placeholder checksums are flagged pending so
        # provisioning refuses them until a real upstream sync (CMM-9) finalizes them.
        pin = cmp.load_pin()
        self.assertEqual(pin.get("schema"), cmp.PIN_SCHEMA)
        self.assertTrue(pin.get("artifacts"))
        self.assertTrue(all(a.get("pending") for a in pin["artifacts"].values()))

    def test_resolve_cached_binary_none_for_pending_pin(self):
        self.assertIsNone(cmp.resolve_cached_binary())


class TestCmmProvisionFetch(unittest.TestCase):
    """CMM-6: fetch + checksum-verify + cache the pinned binary; tampered/missing artifact fails safe."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)
        self.blob = b"#!/bin/sh\necho cmm\n"

    def tearDown(self):
        self._tmp.cleanup()

    def _pin(self, *, sha=None, pending=False, version="v1.2.3", url="https://example/cmm"):
        key = cmp.platform_key()
        return {
            "schema": cmp.PIN_SCHEMA, "version": version,
            "source": {"repo": "DeusData/codebase-memory-mcp", "tag": version, "commit": "abc123"},
            "artifacts": {key: {
                "url": url,
                "sha256": sha if sha is not None else cmp._sha256(self.blob),
                "pending": pending,
            }},
        }

    def test_provision_downloads_verifies_and_caches(self):
        calls = []
        def dl(url):
            calls.append(url)
            return self.blob
        res = cmp.provision(self._pin(), downloader=dl, cache_dir=self.cache)
        self.assertEqual(res["status"], "provisioned", res)
        path = Path(res["path"])
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes(), self.blob)
        self.assertTrue(os.access(path, os.X_OK))  # marked executable
        # Idempotent: a second run hits the verified cache, no re-download.
        res2 = cmp.provision(self._pin(), downloader=dl, cache_dir=self.cache)
        self.assertEqual(res2["status"], "cached")
        self.assertEqual(len(calls), 1)

    def test_provision_checksum_mismatch_fails_safe(self):
        bad = self._pin(sha="f" * 64)
        res = cmp.provision(bad, downloader=lambda u: self.blob, cache_dir=self.cache)
        self.assertEqual(res["status"], "error")
        self.assertIn("checksum mismatch", res["reason"])
        # NOTHING written to the cache (fail safe).
        self.assertFalse(cmp.cached_binary_path("v1.2.3", cmp.platform_key(), self.cache).exists())

    def test_provision_pending_skips_without_download(self):
        called = []
        res = cmp.provision(self._pin(pending=True),
                            downloader=lambda u: called.append(u) or self.blob, cache_dir=self.cache)
        self.assertEqual(res["status"], "skipped")
        self.assertEqual(called, [])  # never reached the network

    def test_provision_download_error_degrades(self):
        def boom(url):
            raise OSError("network down")
        res = cmp.provision(self._pin(), downloader=boom, cache_dir=self.cache)
        self.assertEqual(res["status"], "error")
        self.assertIn("download failed", res["reason"])

    def test_provision_unknown_platform_skips(self):
        pin = self._pin()
        pin["artifacts"] = {"plan9-sparc": pin["artifacts"][cmp.platform_key()]}
        res = cmp.provision(pin, downloader=lambda u: self.blob, cache_dir=self.cache)
        self.assertEqual(res["status"], "skipped")

    def test_resolve_cached_binary_roundtrip(self):
        pin = self._pin()
        cmp.provision(pin, downloader=lambda u: self.blob, cache_dir=self.cache)
        self.assertIsNotNone(cmp.resolve_cached_binary(pin, cache_dir=self.cache))
        # A corrupted cache is treated as absent — never hand back a tampered binary.
        cmp.cached_binary_path("v1.2.3", cmp.platform_key(), self.cache).write_bytes(b"tampered")
        self.assertIsNone(cmp.resolve_cached_binary(pin, cache_dir=self.cache))


class TestCmmContract(unittest.TestCase):
    """CMM-8: the interface contract test — 14 tools, clean partition, well-formed ids, drift detection."""

    def test_passes_on_constants_without_binary(self):
        res = cmp.verify_contract(tool_lister=lambda: None)
        self.assertTrue(res["ok"], res["checks"])
        self.assertFalse(res["binary_checked"])

    def test_fourteen_tools_and_clean_partition(self):
        self.assertEqual(len(cmp.expected_tool_names()), 14)
        res = cmp.verify_contract(tool_lister=lambda: None)
        self.assertTrue(any("14 CMM tools" in c for c in res["checks"]))
        self.assertTrue(any("partition" in c for c in res["checks"]))

    def test_passes_with_matching_live_tool_list(self):
        res = cmp.verify_contract(tool_lister=lambda: list(cmp.expected_tool_names()))
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["binary_checked"])

    def test_detects_a_renamed_or_removed_tool(self):
        names = list(cmp.expected_tool_names())
        res = cmp.verify_contract(tool_lister=lambda: names[1:])  # drop one
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["missing"]), 1)

    def test_detects_an_extra_tool(self):
        res = cmp.verify_contract(
            tool_lister=lambda: list(cmp.expected_tool_names()) + ["brand_new_tool"])
        self.assertFalse(res["ok"])
        self.assertIn("brand_new_tool", res["extra"])

    def test_namespaced_and_bare_names_compare_equal(self):
        # A lister that returns fully-namespaced ids still matches the bare expected set.
        res = cmp.verify_contract(tool_lister=lambda: list(cm.CMM_ALL_TOOLS))
        self.assertTrue(res["ok"], res)


class TestCmmSync(unittest.TestCase):
    """CMM-9: gated upstream-sync planning. A newer release passes/fails the contract gate."""

    def _pending_pin(self):
        return {
            "schema": cmp.PIN_SCHEMA, "version": "v0.0.0-pending",
            "source": {"repo": "DeusData/codebase-memory-mcp", "tag": "v0.0.0-pending", "commit": "0" * 40},
            "artifacts": {k: {"url": f"https://x/{k}", "sha256": "0" * 64, "pending": True}
                          for k in ("darwin-arm64", "darwin-x86_64", "linux-x86_64", "linux-arm64")},
        }

    def test_version_compare_pending_is_oldest(self):
        self.assertTrue(cmp._is_newer("v0.1.0", "v0.0.0-pending"))
        self.assertTrue(cmp._is_newer("v2.0.0", "v1.9.9"))
        self.assertFalse(cmp._is_newer("v1.0.0", "v1.0.0"))
        self.assertFalse(cmp._is_newer("v1.0.0", "v1.2.0"))

    def test_up_to_date_when_pin_not_behind(self):
        pin = self._pending_pin()
        pin["version"] = "v2.0.0"
        plan = cmp.plan_sync(pin, release_fetcher=lambda repo: {"tag": "v1.0.0", "assets": {}},
                             tool_lister=lambda: list(cmp.expected_tool_names()))
        self.assertEqual(plan["action"], "up-to-date")

    def test_newer_release_passing_gate_produces_pr(self):
        # THE STOP-CONDITION GATE: a simulated newer upstream release → a gated sync PR plan.
        def fetcher(repo):
            return {"tag": "v1.5.0", "commit": "deadbeef",
                    "assets": {cmp.platform_key(): {"url": "https://x/bin", "sha256": "a" * 64}}}
        plan = cmp.plan_sync(self._pending_pin(), release_fetcher=fetcher,
                             tool_lister=lambda: list(cmp.expected_tool_names()))
        self.assertEqual(plan["action"], "pr")
        self.assertEqual(plan["branch"], "cmm-sync/v1.5.0")
        self.assertEqual(plan["manifest"]["version"], "v1.5.0")
        self.assertTrue(plan["contract"]["ok"])
        self.assertIn("v1.5.0", plan["title"])

    def test_newer_release_failing_gate_is_flagged_not_pr(self):
        def fetcher(repo):
            return {"tag": "v1.5.0", "commit": "x", "assets": {}}
        # A renamed tool upstream → contract fails → flag, never a PR.
        plan = cmp.plan_sync(self._pending_pin(), release_fetcher=fetcher,
                             tool_lister=lambda: list(cmp.expected_tool_names())[:-1])
        self.assertEqual(plan["action"], "flag")
        self.assertFalse(plan["contract"]["ok"])

    def test_unreachable_feed_degrades_to_error(self):
        plan = cmp.plan_sync(self._pending_pin(), release_fetcher=lambda repo: None,
                             tool_lister=lambda: list(cmp.expected_tool_names()))
        self.assertEqual(plan["action"], "error")

    def test_bumped_manifest_marks_unchecksummed_artifacts_pending(self):
        def fetcher(repo):
            return {"tag": "v1.5.0", "commit": "x",
                    "assets": {cmp.platform_key(): {"url": "https://x/bin"}}}  # no sha256
        plan = cmp.plan_sync(self._pending_pin(), release_fetcher=fetcher,
                             tool_lister=lambda: list(cmp.expected_tool_names()))
        self.assertEqual(plan["action"], "pr")
        art = plan["manifest"]["artifacts"][cmp.platform_key()]
        self.assertTrue(art["pending"])  # no checksum yet → still refused by provision


class TestCmmEngageProvision(CelebornTestCase):
    """S2 integration: engage auto-provisions when possible, degrades cleanly when not (CMM-10)."""

    def _perms(self) -> dict:
        p = self.root / ".claude" / "settings.json"
        return (json.loads(p.read_text()).get("permissions") or {}) if p.is_file() else {}

    def test_engage_degrades_when_pin_pending(self):
        # With the shipped (pending) pin and no CMM on PATH, auto-provision is skipped — but engage
        # still completes and the permission pre-clear (the part that must always land) is written.
        r = self.cli("cmm", "engage")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("mcp__codebase-memory-mcp__search_graph", self._perms().get("allow", []))

    def test_engage_no_provision_flag(self):
        r = self.cli("cmm", "engage", "--no-provision")
        self.assertIsNone(r.exit_code, r.all)
        self.assertIn("mcp__codebase-memory-mcp__search_graph", self._perms().get("allow", []))


class TestCmmInitAutoEngage(unittest.TestCase):
    """Celeborn auto-engages CMM on `celeborn init` by default (the 'CMM in every project' design);
    opt-out via --no-cmm / $CELEBORN_NO_CMM; the permission pre-clear lands even without the binary."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ctx = self.root / ".context"

    def tearDown(self):
        self._tmp.cleanup()

    def _init(self, *extra):
        return run_cli("--path", str(self.root), "init", "--no-scan", *extra)

    def _perms(self) -> dict:
        p = self.root / ".claude" / "settings.json"
        return (json.loads(p.read_text()).get("permissions") or {}) if p.is_file() else {}

    def test_init_auto_engages_without_binary(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(cm, "cmm_binary", return_value=None):
            os.environ.pop("CELEBORN_CMM_BIN", None)
            os.environ.pop("CELEBORN_NO_CMM", None)
            r = self._init()
        self.assertIsNone(r.exit_code, r.all)
        # Permission pre-clear lands (the flow win — needs no binary) + North Star installed.
        for t in cm.CMM_ALLOW_TOOLS:
            self.assertIn(t, self._perms().get("allow", []))
        self.assertIn(cm.CMM_MD_BEGIN, (self.root / "CLAUDE.md").read_text())
        # …and the user is pointed at CMM for the structural half.
        self.assertIn("codebase-memory-mcp", r.all)

    def test_no_cmm_flag_skips_engage(self):
        r = self._init("--no-cmm")
        self.assertIsNone(r.exit_code, r.all)
        self.assertFalse(self._perms().get("allow"))
        self.assertNotIn(cm.CMM_MD_BEGIN, (self.root / "CLAUDE.md").read_text())

    def test_env_opt_out_skips_engage(self):
        with mock.patch.dict(os.environ, {"CELEBORN_NO_CMM": "1"}):
            r = self._init()
        self.assertIsNone(r.exit_code, r.all)
        self.assertNotIn(cm.CMM_MD_BEGIN, (self.root / "CLAUDE.md").read_text())

    def test_init_with_binary_registers_mcp_args_empty(self):
        fake = self.root / "fake-cmm"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        with mock.patch.dict(os.environ, {"CELEBORN_CMM_BIN": str(fake)}):
            os.environ.pop("CELEBORN_NO_CMM", None)
            r = self._init()
        self.assertIsNone(r.exit_code, r.all)
        data = json.loads((self.root / ".mcp.json").read_text())
        self.assertEqual(data["mcpServers"][cm.CMM_SERVER_NAME]["args"], [])

    def test_sticky_opt_out_survives_reinit(self):
        # Engage then opt out; a later init must not silently re-engage.
        with mock.patch.object(cm, "cmm_binary", return_value=None):
            self._init()
            run_cli("--path", str(self.root), "cmm", "off")
            (self.root / "CLAUDE.md").write_text("# clean\n")  # prove re-init doesn't re-add the block
            self._init()
        self.assertNotIn(cm.CMM_MD_BEGIN, (self.root / "CLAUDE.md").read_text())


class TestProgressEngine(CelebornTestCase):
    """CELE-t161 — the deterministic progress engine + nudge ladder."""

    SESS = "sess-progress-1"

    def _add(self, title, subs=None):
        self.cli("tasks", "add", title)
        tid = next(t["id"] for t in cb._load_tasks(self.ctx) if t["title"] == title)
        if subs:
            self.cli("tasks", "subtasks", tid, "set", *subs)
        return tid

    def _claim(self, tid):
        return self.cli("claim", tid, "--by", "tester", "--session", self.SESS)

    def _card(self, tid):
        return cb._find_task(cb._load_tasks(self.ctx), tid)

    def _fake_commit(self, n=1):
        return mock.patch.object(cb, "_commits_for_task",
                                 return_value=[{"hash": f"h{i}", "ts": 1700000000 + i,
                                                "subject": "did work", "body": ""} for i in range(n)])

    # --- floor lifecycle -------------------------------------------------------
    def test_claim_sets_floor_5(self):
        tid = self._add("Engine A", subs=["Step one", "Step two"])
        self._claim(tid)
        self.assertEqual(self._card(tid)["progress"], 5)

    def test_first_work_signal_sets_10(self):
        tid = self._add("Engine B", subs=["Step one", "Step two"])  # neutral text → no auto-tick
        self._claim(tid)
        with self._fake_commit():
            card = self._card(tid)
            res = cb._progress_engine_tick(self.ctx, card)
        self.assertTrue(res["rec"]["work_started"])
        self.assertEqual(card["progress"], 10)  # 0/2 milestones, work started → band floor 10

    # --- milestone band, cap, ship --------------------------------------------
    def test_milestone_ratio_floor(self):
        tid = self._add("Engine C", subs=["Alpha", "Beta", "Gamma", "Delta"])
        self._claim(tid)
        self.cli("tasks", "check", tid, "1")
        self.cli("tasks", "check", tid, "2")  # 2/4 done → 10 + round(0.5*89) = 54
        self.assertEqual(self._card(tid)["progress"], 54)

    def test_cap_99_while_doing_then_100_on_ship(self):
        tid = self._add("Engine D", subs=["Alpha", "Beta"])
        self._claim(tid)
        self.cli("tasks", "check", tid, "1")
        self.cli("tasks", "check", tid, "2")  # 2/2 → 10 + 89 = 99 (never 100 while doing)
        self.assertEqual(self._card(tid)["progress"], 99)
        self.cli("ship", tid)
        self.assertEqual(self._card(tid)["progress"], 100)

    # --- monotonic / idempotent ------------------------------------------------
    def test_monotonic_manual_not_lowered(self):
        # No subtasks: the agent's manual crest is the bar (CELE-t106 lets subtasks own % otherwise).
        tid = self._add("Engine E")
        self._claim(tid)
        self.cli("tasks", "edit", tid, "--progress", "60")  # agent crests manually
        with self._fake_commit():
            card = self._card(tid)
            cb._progress_engine_tick(self.ctx, card)  # signal-ramp floor is far below 60 — must not lower it
        self.assertEqual(card["progress"], 60)

    def test_idempotent_same_turn(self):
        tid = self._add("Engine F", subs=["Step one"])
        self._claim(tid)
        with self._fake_commit():
            card = self._card(tid)
            r1 = cb._progress_engine_tick(self.ctx, card)
            p1 = card["progress"]
            r2 = cb._progress_engine_tick(self.ctx, card)
        self.assertTrue(r1["moved"])
        self.assertFalse(r2["moved"])
        self.assertEqual(card["progress"], p1)

    # --- signal-matched auto-tick ---------------------------------------------
    def test_auto_tick_only_with_signal(self):
        tid = self._add("Engine G", subs=["Run the test suite to green", "Polish the UX feel"])
        self._claim(tid)
        # No test signal anywhere → neither ticks (first is signal-gated, second is judgment).
        card = self._card(tid)
        cb._progress_engine_tick(self.ctx, card)
        self.assertFalse(any(s["done"] for s in card["subtasks"]))
        # A test command in activity.md → the tests milestone ticks; the judgment one stays for the agent.
        self.write("activity.md", "## Recent commands\n- python -m unittest tests\n")
        card = self._card(tid)
        cb._progress_engine_tick(self.ctx, card)
        self.assertTrue(card["subtasks"][0]["done"])
        self.assertFalse(card["subtasks"][1]["done"])

    # --- nudge ladder ----------------------------------------------------------
    def test_nudge_ladder_levels(self):
        tid = self._add("Engine H", subs=["Step one", "Step two"])
        self._claim(tid)
        card = self._card(tid)
        sig = {"present": set(), "commits": 0, "touched": False, "corpus": ""}

        def line_at(turns):
            rec = {"turns_since_change": turns, "work_started": True, "engine_floor": 10}
            return cb._progress_nudge_line(self.ctx, card, {"rec": rec, "signals": sig,
                                                            "moved": False, "newly": []})
        self.assertEqual(line_at(0), "")               # silent
        self.assertIn("tick any finished milestones", line_at(2))   # L1
        self.assertIn("hasn't moved", line_at(4))                   # L2
        self.assertIn("auto-advanced", line_at(6))                  # L3 backstop
        # every nudge carries a copy-pasteable command with the real id
        self.assertIn(tid, line_at(2))

    def test_nudge_resets_on_movement(self):
        tid = self._add("Engine I", subs=["Step one", "Step two"])
        self._claim(tid)
        # drive turns up via no-movement hook ticks, then move the bar and confirm the level resets
        for _ in range(6):
            cb._progress_hook(self.ctx, self.SESS)
        rec = cb._progress_rec(cb._load_progress(self.ctx), tid)
        self.assertGreaterEqual(rec["turns_since_change"], 1)
        self.cli("tasks", "check", tid, "1")  # movement
        rec = cb._progress_rec(cb._load_progress(self.ctx), tid)
        self.assertEqual(rec["turns_since_change"], 0)
        self.assertEqual(rec["nudge_level"], 0)

    # --- no-op on todo/done ----------------------------------------------------
    def test_no_op_on_todo(self):
        tid = self._add("Engine J", subs=["Step one"])  # stays todo (not claimed)
        card = self._card(tid)
        res = cb._progress_engine_tick(self.ctx, card)
        self.assertFalse(res["moved"])
        self.assertEqual(card["progress"], 0)

    # --- atomic write ----------------------------------------------------------
    def test_atomic_write_survives_crash(self):
        cb._save_progress(self.ctx, {"cards": {"x": {"engine_floor": 7}}})
        with mock.patch("os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                cb._save_progress(self.ctx, {"cards": {"x": {"engine_floor": 99}}})
        data = cb._load_progress(self.ctx)
        self.assertEqual(data["cards"]["x"]["engine_floor"], 7)  # original intact

    # --- doctor drift + progress verb -----------------------------------------
    def test_doctor_flags_zero_pct_with_commits(self):
        tid = self._add("Engine K")
        self.cli("tasks", "move", tid, "doing")  # move (not claim) → no floor stamp → stuck at 0%
        with self._fake_commit():
            r = self.cli("doctor")
        self.assertIn("despite commits", r.all)

    def test_progress_verb_explains(self):
        tid = self._add("Engine L", subs=["Step one", "Step two"])
        self._claim(tid)
        with self._fake_commit():
            r = self.cli("progress", tid, "--explain")
        self.assertIn("engine floor", r.out)
        self.assertIn("signals present", r.out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
