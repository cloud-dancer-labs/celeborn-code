#!/usr/bin/env python3
"""Celeborn — a long-term context substrate for coding agents.

A single-file, stdlib-only CLI that manages a per-repo `.context/` memory directory:
the deterministic bookkeeping (scaffold, index, search, archive, promote, handoff, health)
that a model does unreliably or expensively. Judgment stays with the agent.

Markdown in `.context/` is the source of truth. The SQLite index (`index.db`) is derived
and disposable — `index` drops and rebuilds it from scratch.

Commands:
  init      Scaffold .context/; gitignore index.db (or all of .context/ with --private)
  status    Print the Hot tier exactly as an agent should load it on Orient
  index     (Re)build the SQLite FTS index from the markdown
  search    Full-text recall -> ranked snippets with file:anchor pointers
  archive   Move journal entries past the threshold into journal-archive/
  promote   Append a formatted entry to a higher tier (learnings / durable)
  handoff   Regenerate handoff.md from state.md + session.json
  doctor    Health check: budgets, index freshness, missing files, memory drift, secret scan
  capture   Mechanically ingest a Claude Code transcript into the local Automatic Context Record (no model)
  login     Sign in with GitHub to unlock hosted sync (premium; Pro subscription)
  sync      Push/pull .context/ to the hosted Supabase backend (secrets redacted out)
  version   Print version; --check looks back at GitHub for a newer Celeborn
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import json
import re
import sys
from pathlib import Path

# NOTE: `sqlite3` is intentionally NOT imported at module level. Only `index` and `search` touch
# the database; importing it lazily there keeps the common hot paths (record/status/handoff/doctor,
# fired by hooks on every session) at bare-interpreter startup cost.

CONTEXT_DIRNAME = ".context"
RC_NAME = ".celebornrc"
INDEX_NAME = "index.db"
METRICS_NAME = "metrics.json"

# The Hot tier — what actually loads on Orient. Everything else is "saved" vs. a naive full load.
HOT_FILES = ["state.md", "session.json", "durable/manifest.md"]

# Pre-compaction panic-save: the authored tiers a compaction threatens to make the model re-derive.
# `panic-save` copies whichever of these exist into .context/.panic/<stamp>/ as a restore point, so
# the survival is a felt artifact ("🏹 Celeborn saved your session"), not an invisible promise. Order
# is restore order. Subpaths (durable/…) are mirrored under the stamp dir.
PANIC_SAVE_FILES = ["state.md", "session.json", "notes.md", "journal.md", "decisions.md",
                    "learnings.md", "handoff.md", "tasks.md", "durable/manifest.md"]
PANIC_DIR = ".panic"        # under .context/ — local-only, gitignored, FIFO-pruned
# Shown in the panic-save user/agent line (t43) — where to read about compaction, /clear, and tiers.
PANIC_READ_MORE = "references/memory-protocol.md"
PANIC_KEEP = 10             # keep the most recent N snapshots; older ones are deleted on each save

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _data_dir() -> Path:
    """Locate the runtime data (schema.sql + init templates).

    Installed builds ship `references/` as the data-only `celeborn_refs` package, so we resolve it
    via importlib.resources (works for pip/uv/wheel installs where the source tree isn't present).
    In a plain source checkout celeborn_refs isn't importable, so fall back to <repo>/references/.
    A frozen PyInstaller binary bundles the tree under `_MEIPASS/celeborn_refs`.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / "celeborn_refs"  # type: ignore[attr-defined]
        if bundled.is_dir():
            return bundled
    try:
        from importlib.resources import files
        packaged = Path(str(files("celeborn_refs")))
        if packaged.is_dir():
            return packaged
    except Exception:
        pass
    return REPO_ROOT / "references"


DATA_DIR = _data_dir()
TEMPLATES_DIR = DATA_DIR / "templates"
SCHEMA_PATH = DATA_DIR / "schema.sql"

DEFAULTS = {
    "journal_keep_entries": 20,
    "done_keep_cards": 30,           # done cards visible on the board; older ones auto-archive
    "done_archive_keep_cards": 100,  # FIFO cap for done-archive.md before oldest entries are dropped
    "state_max_lines": 120,
    "search_default_limit": 8,
    "chars_per_token": 4,        # rough English heuristic for token estimation
    "usd_per_mtok": 3.0,         # blended $/1M input tokens for the "$ saved" flex (standup --tweet, flex)
    "orient_dedupe_seconds": 120,  # hookless fallback: orients within this window = same session
    "project_slug": None,  # short id for project-qualified card markers (⟨celeborn:slug/tN⟩); None = short 4-char prefix derived from the repo folder name
    "qualified_task_ids": True,  # default-on (opt-OUT): display card ids project-qualified (SLUG-tN, driven by project_slug) everywhere — board, CLI, orient, standup. Set False to show bare tN. Fleet (cross-project) always qualifies; resolvers accept qualified ids regardless. Stored ids stay bare tN.
    "board_port": None,  # localhost port for the kanban viewer; None = derive a STABLE per-project port (de-collides repos)
    "board_autostart": True,  # ensure-on-orient: SessionStart starts the viewer (detached) if its port is down. False = never auto-launch.
    "jira_autopush": True,  # after tasks add/move/edit/claim, push linked cards to Jira (best-effort; skips when disconnected)
    "jira_autopush_debounce_seconds": 90,  # per-task minimum gap between Jira transitions (avoids workflow thrash)
    "capture_output_max_chars": 8000,  # per-tool-result cap in the faithful auto record (redact-then-cap)
    # Hot-tier (Orient load) output budgets, in characters. The SessionStart hook injects `status`
    # as additionalContext; a host with a small inline budget persists oversized hook output to a
    # file and feeds the model only a tiny preview — silently killing automatic rehydration. So
    # `status` truncates each variable-length piece with a pointer to the full file (bypass: --full).
    "hot_state_max_chars": 4000,      # state.md body
    "hot_activity_max_chars": 2000,   # activity.md (Automatic Context Record digest)
    "hot_focus_max_chars": 1500,      # each session.json focus/next_action string
    "hot_tasks_max_chars": 1000,      # tasks board summary (counts + in-flight cards)
    "hot_touches_max_chars": 800,     # active file touches (multi-agent editing)
    "touch_ttl_hours": 2,             # stale touches drop out of orient after this many hours
    # Skill advisor (t70) — a quiet throughput/quality nudge layer. Read through `_advisor_config()`,
    # which deep-fills this block from .celebornrc and still honors the legacy flat keys
    # (`advisor_enabled` / `advisor_permission_bloat_min`) older rc files may carry.
    "advisor": {
        "enabled": True,
        "max_per_session": 1,          # at most this many advisor nudges per session (don't nag)
        "permission_bloat_min": 10,    # ≥ this many over-specific Bash allow-rules → friction signal
        "review_min_files": 3,         # ≥ this many changed code files → recommend a code review (Phase 3)
        "parallelize_min_files": 12,   # ≥ this many changed code files → recommend fanning out the review (Phase 4)
        # Sensitive paths (Phase 3 security-review heuristic) — any changed path matching → security pass.
        "sensitive_globs": ["supabase/**", "stripe*", "*billing*", "*auth*", "*sync*"],
    },
    "harness": None,
    "secret_patterns": [
        r"AKIA[0-9A-Z]{16}",
        r"sk-[A-Za-z0-9]{20,}",
        r"ghp_[A-Za-z0-9]{36}",
        r"xox[baprs]-[0-9A-Za-z-]{10,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"AIza[0-9A-Za-z_-]{35}",
    ],
}

# Files the index walks and the tier each maps to. Globs are relative to .context/.
TIER_GLOBS = [
    ("hot", "state.md"),
    ("warm", "notes.md"),            # unbounded working detail — on-demand, not auto-loaded
    ("warm", "tasks.md"),            # task/kanban board — markdown source of truth (Phase 11)
    ("durable", "durable/manifest.md"),
    ("durable", "durable/*.md"),
    ("warm", "journal.md"),
    ("cold", "journal-archive/*.md"),
    ("cold", "done-archive.md"),      # auto-archived done cards (FIFO cap; still searchable)
    ("distilled", "learnings.md"),
    ("distilled", "decisions.md"),
    ("handoff", "handoff.md"),
    ("hot", "activity.md"),          # auto, mechanical — always-current digest
    ("cold", "auto/*.md"),           # auto, mechanical — full per-turn capture
]

# Globs rewritten mechanically by `celeborn capture` on EVERY turn (activity digest + per-turn
# snapshots). They are still INDEXED (searchable via `celeborn search`), but the staleness
# heuristic ignores them: they churn every turn regardless of whether any durable, user-meaningful
# content changed, so counting them would make the index perpetually "stale" inside any live session.
MECHANICAL_GLOBS = {"activity.md", "auto/*.md"}

REQUIRED_FILES = [
    "state.md",
    "notes.md",
    "session.json",
    "journal.md",
    "learnings.md",
    "decisions.md",
    "handoff.md",
    "durable/manifest.md",
]


# --------------------------------------------------------------------------- utils

def now_iso() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def find_context_root(start: Path) -> Path | None:
    """Walk up from `start` looking for a `.context/` directory."""
    start = start.resolve()
    for d in [start, *start.parents]:
        if (d / CONTEXT_DIRNAME).is_dir():
            return d / CONTEXT_DIRNAME
    return None


def require_context(args) -> Path:
    ctx = find_context_root(Path(getattr(args, "path", ".") or "."))
    if ctx is None:
        die("No .context/ found here or in any parent. Run `celeborn init` first.")
    return ctx


def _global_context() -> Path:
    """The home-level capture sink: ~/.context. Used when a session runs outside any repo that has a
    .context/, so no session goes unrecorded (the hybrid model)."""
    return Path.home() / CONTEXT_DIRNAME


def _scaffold_global(gctx: Path) -> Path:
    """Create the MINIMAL global capture sink (not a full `init`): just auto/, a metrics.json for the
    capture cursor, and a .celebornrc giving the record a stable sync identity ("global"). No authored
    tiers — this sink only holds the Automatic Context Record."""
    (gctx / "auto").mkdir(parents=True, exist_ok=True)
    if not (gctx / METRICS_NAME).is_file():
        _save_metrics(gctx, dict(METRICS_TEMPLATE))
    rc = gctx / RC_NAME
    if not rc.is_file():
        rc.write_text(json.dumps({"sync": {"project_name": "global"}}, indent=2) + "\n")
    return gctx


def find_or_create_context(args) -> Path:
    """Capture-only resolution (hybrid): use the repo's .context/ when the cwd is inside one and
    --global wasn't forced; otherwise fall back to the global ~/.context sink, scaffolding on demand.
    Unlike require_context this NEVER dies — every session gets recorded somewhere."""
    if not getattr(args, "global_", False):
        ctx = find_context_root(Path(getattr(args, "path", ".") or "."))
        if ctx is not None:
            return ctx
    return _scaffold_global(_global_context())


def load_config(ctx: Path) -> dict:
    cfg = dict(DEFAULTS)
    rc = ctx / RC_NAME
    if rc.is_file():
        try:
            cfg.update(json.loads(rc.read_text()))
        except json.JSONDecodeError as e:
            warn(f"{RC_NAME} is not valid JSON ({e}); using defaults.")
    return cfg


def _update_config(ctx: Path, **kv) -> None:
    """Merge keys into the project's `.celebornrc` (created if absent), preserving existing keys. Values
    of None are skipped. Best-effort: a malformed rc is replaced rather than crashing the caller."""
    rc = ctx / RC_NAME
    data = {}
    if rc.is_file():
        try:
            data = json.loads(rc.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
    data.update({k: v for k, v in kv.items() if v is not None})
    rc.write_text(json.dumps(data, indent=2) + "\n")


def _advisor_config(ctx: Path) -> dict:
    """Normalized skill-advisor settings (t70). Starts from the DEFAULTS `advisor` block, overlays the
    nested `advisor: {...}` from .celebornrc (a flat `cfg.update` would otherwise drop the unspecified
    sub-keys), then honors the legacy flat keys (`advisor_enabled`, `advisor_permission_bloat_min`) so
    older rc files keep working. Returns {enabled, max_per_session, permission_bloat_min, sensitive_globs}."""
    cfg = load_config(ctx)
    out = dict(DEFAULTS["advisor"])
    block = cfg.get("advisor")
    if isinstance(block, dict):
        for k, v in block.items():
            if v is not None:
                out[k] = v
    if "advisor_enabled" in cfg:                       # legacy flat key (older rc) still wins
        out["enabled"] = bool(cfg["advisor_enabled"])
    if "advisor_permission_bloat_min" in cfg:
        out["permission_bloat_min"] = int(cfg["advisor_permission_bloat_min"])
    out["enabled"] = bool(out.get("enabled", True))
    out["max_per_session"] = max(0, int(out.get("max_per_session", 1) or 0))
    out["permission_bloat_min"] = int(out.get("permission_bloat_min", 10) or 0)
    out["review_min_files"] = max(1, int(out.get("review_min_files", 3) or 3))
    out["parallelize_min_files"] = max(1, int(out.get("parallelize_min_files", 12) or 12))
    if not isinstance(out.get("sensitive_globs"), list):
        out["sensitive_globs"] = list(DEFAULTS["advisor"]["sensitive_globs"])
    return out


def _sanitize_project_slug(raw: str) -> str:
    """Safe token for card markers: letters, digits, underscore, dot, hyphen."""
    s = re.sub(r"[^\w.-]+", "-", (raw or "").strip()).strip("-") or "project"
    return s[:64]


def _short_slug(raw: str, n: int = 4) -> str:
    """Derive a short, readable qualifier from a repo folder name → e.g. `celeborn` → `cele`.
    Keeps the first `n` alphanumerics (so `CELE-t84` is snappy yet traceable); fleet dedup
    (`_dedupe_slug`) resolves any cross-project collisions to `cele`, `cele-2`, … Falls back to the
    full sanitized name when stripping leaves nothing (e.g. an all-symbol folder name)."""
    head = re.sub(r"[^A-Za-z0-9]+", "", (raw or "")).lower()[:n]
    return head or _sanitize_project_slug(raw)


def _repo_root_from_ctx(ctx: Path) -> Path:
    """Parent of a resolved `.context/` dir — the repo folder that owns the board."""
    return ctx.resolve().parent


def project_slug(ctx: Path) -> str:
    """Per-repo qualifier for project-qualified card ids/markers. An explicit project_slug in
    .celebornrc is authority (used verbatim, only sanitized) — the escape hatch for a longer/custom
    prefix. With no explicit value we derive a short 4-char prefix from the repo folder name."""
    explicit = (load_config(ctx).get("project_slug") or "").strip()
    if explicit:
        return _sanitize_project_slug(explicit)
    return _short_slug(_repo_root_from_ctx(ctx).name or "project")


def _slug_matches(a: str, b: str) -> bool:
    """Whether two qualifiers name the same project, tolerant of the short-prefix derivation so a
    legacy long-form ref (`celeborn/tN`) still matches a derived short board (`cele`). Exact
    sanitized match, or equal short prefixes — the latter can rarely coincide for two projects sharing
    a 4-char head, but those are fleet-deduped at register, and this only governs a non-fatal warn."""
    a, b = _sanitize_project_slug(a).lower(), _sanitize_project_slug(b).lower()
    return a == b or _short_slug(a) == _short_slug(b)


def _dedupe_slug(base: str, taken) -> str:
    """Return `base` if its qualifier is free across the fleet, else the first available `base-N` (N≥2).
    Comparison is case-insensitive because the displayed qualifier is upper-cased (SLUG-tN), so `cele`
    and `CELE` collide. `taken` is the slugs already claimed by other registered projects."""
    taken_u = {str(t).strip().upper() for t in taken if str(t).strip()}
    if base.upper() not in taken_u:
        return base
    n = 2
    while f"{base}-{n}".upper() in taken_u:
        n += 1
    return f"{base}-{n}"


# --------------------------------------------------------------------------- board port (de-collide)
#
# The kanban viewer (board/) is one web server per project. A single hard-coded port would collide
# the moment a second Celeborn repo runs its board. So each project gets its OWN stable port: an
# explicit `board_port` in .celebornrc wins; otherwise it's derived deterministically from the
# project path. "Stable" = same project → same port every run, so the URL is bookmarkable and the
# orient line is reliable. We use hashlib (NOT the built-in hash(), which is salted per-process and
# would hand out a different port every run).
BOARD_PORT_BASE = 3141
BOARD_PORT_SPAN = 800   # ports 3141–3940 — a recognizable band, clear of common dev ports (3000/8080/5173…)


def _derive_board_port(project_dir: Path) -> int:
    import hashlib
    key = str(Path(project_dir).resolve()).encode("utf-8", "surrogatepass")
    return BOARD_PORT_BASE + int(hashlib.sha1(key).hexdigest(), 16) % BOARD_PORT_SPAN


# CELE-t170: one shared server serves every project's board on a single port — `/` is the fleet home,
# `/board/<slug>` is a project's board. The per-repo hashed port (`_derive_board_port`) is retired from
# the default path; it stays defined above for the stable-port unit test and any legacy explicit override.
SHARED_BOARD_PORT = BOARD_PORT_BASE  # 3141


def board_port(ctx: Path) -> int:
    """The board's port. Defaults to the shared 3141 — one server for the whole fleet (CELE-t170).
    An explicit, valid `board_port` in .celebornrc still wins as an advanced/legacy override."""
    p = load_config(ctx).get("board_port")
    if isinstance(p, int) and 1 <= p <= 65535:
        return p
    return SHARED_BOARD_PORT


def board_url(ctx: Path) -> str:
    """This project's board on the shared server: http://localhost:<port>/board/<slug>. `/` is the
    fleet home; each project lives under its slug so one server serves them all without data bleed."""
    return f"http://localhost:{board_port(ctx)}/board/{project_slug(ctx)}"


def _board_live(port: int, timeout: float = 0.15) -> bool:
    """True if something is already listening on localhost:<port>. Fast and forgiving (short timeout,
    never raises) — safe to call from a hook / on every orient."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- ensure-on-orient
#
# The board should be effectively always-on: if the SessionStart hook resolves this project's port
# and finds nothing listening, it starts the viewer DETACHED (own session/process group, stdio to a
# local log) and returns immediately — the hook must never block on `next dev`'s multi-second boot.
# `next dev` takes a few seconds to bind, so a naive port probe would relaunch on the next orient
# while the first is still booting. A local-only pidfile (`.context/.board.pid`) records the PID +
# port we launched; while that PID is alive we report `booting`, not `down`, and don't double-launch.
BOARD_PIDFILE = ".board.pid"
BOARD_LOG = ".board.log"


def _board_dir() -> Path:
    """The Next.js kanban viewer that ships with this Celeborn install. One app, launched per-project
    on a de-collided port and pointed at the orienting repo's tasks via CELEBORN_TASKS_JSON."""
    return REPO_ROOT / "board"


def _pid_alive(pid: int) -> bool:
    """True if `pid` names a live process. Signal 0 just probes — it sends nothing."""
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True            # exists but not ours to signal — still alive
    except OSError:
        return False
    return True


# CELE-t170: the pidfile + log are MACHINE-GLOBAL (under ~/.config/celeborn), not per-`.context`,
# because there is now exactly ONE shared board server for the whole machine. This is what makes the
# supervisor a singleton: concurrent orients across repos all consult the same pidfile.
def _board_pidfile_path() -> Path:
    return _config_dir() / BOARD_PIDFILE


def _board_log_path() -> Path:
    return _config_dir() / BOARD_LOG


def _read_board_pidfile() -> dict:
    try:
        d = json.loads(_board_pidfile_path().read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _board_booting(port: int) -> bool:
    """True if the shared board WE launched for this port is still alive (presumably mid-boot, not yet
    bound). Machine-global — any repo's orient sees the same in-flight launch and won't double-spawn."""
    d = _read_board_pidfile()
    pid = d.get("pid")
    return isinstance(pid, int) and d.get("port") == port and _pid_alive(pid)


def _board_runner(board_dir: Path) -> list[str] | None:
    """The argv that starts the viewer, or None if prerequisites are missing (no app dir, deps not
    installed, or no `npm` on PATH) — in which case ensure-on-orient is a quiet no-op."""
    import shutil
    if not (board_dir / "package.json").is_file():
        return None
    if not (board_dir / "node_modules").is_dir():
        return None            # `npm install` in board/ not run yet — nothing to launch
    npm = shutil.which("npm")
    if not npm:
        return None
    return [npm, "run", "dev"]


# --------------------------------------------------------------------------- self-healing supervisor
#
# A bare `next dev` dies for all sorts of reasons mid-session (a route 500, an OOM, a dead file-
# watcher, the historical `.next` build-clobber). When it did, NOTHING relaunched it until the next
# SessionStart — so the user's open tab hit "nothing is listening" for the rest of the session
# (CELE-t99). The fix: we don't detach `next dev` directly. We detach a tiny SUPERVISOR that runs
# `next dev` as a child and relaunches it on every exit with bounded exponential backoff. The
# supervisor is the PID recorded in `.board.pid`; any crash of the dev server self-heals in seconds.
# It gives up loudly only after N restarts that each died near-instantly (a genuinely broken build —
# don't hot-loop). The supervisor re-invokes THIS script as `celeborn board --supervise`.

_BOARD_SUPERVISOR_CHILD = None   # the live `next dev` child, so the signal handler can reap it


def _install_supervisor_signals() -> None:
    """Best-effort: when the supervisor is terminated, take its `next dev` child down with it so we
    don't leak an orphaned dev server. Never raises (signal may be unavailable in odd contexts)."""
    import signal

    def _term(signum, frame):  # noqa: ANN001
        child = _BOARD_SUPERVISOR_CHILD
        try:
            if child is not None:
                child.terminate()
        except Exception:       # noqa: BLE001
            pass
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGTERM, _term)
        signal.signal(signal.SIGINT, _term)
    except Exception:           # noqa: BLE001
        pass


def _board_supervise(runner: list[str], port: int, board_dir: Path, *,
                     spawn=None, sleeper=None, clock=None,
                     max_rapid: int = 5, rapid_window_s: float = 10.0,
                     backoff_cap_s: float = 30.0) -> int:
    """The self-healing core: run the viewer (`next dev`) as a child and relaunch it whenever it
    exits, with bounded exponential backoff. This is the process detached and recorded in the
    machine-global `.board.pid`, so any crash of the dev server self-heals within seconds while the
    tab lives. One shared server on `port` serves every project (CELE-t170) — no per-repo tasks path;
    the route resolves the project from the URL. Gives up (returns) only after `max_rapid` restarts
    that each died inside `rapid_window_s` — a genuinely broken build we shouldn't hot-loop.
    `spawn`/`sleeper`/`clock` are injectable so tests drive the loop without real processes or real
    sleeps. Returns the number of child exits seen."""
    import os, time
    env = {**os.environ, "PORT": str(port)}
    if spawn is None:
        import subprocess

        def spawn():
            return subprocess.Popen(runner, cwd=str(board_dir), env=env, stdin=subprocess.DEVNULL)
    sleeper = sleeper or time.sleep
    clock = clock or time.monotonic
    _install_supervisor_signals()
    global _BOARD_SUPERVISOR_CHILD
    backoff = 1.0
    rapid = 0
    restarts = 0
    while True:
        started = clock()
        try:
            proc = spawn()
        except Exception as e:                  # noqa: BLE001 — launch failed; nothing to supervise
            print(f"🏹 board supervisor: launch failed ({e}) — exiting", flush=True)
            return restarts
        _BOARD_SUPERVISOR_CHILD = proc
        rc = proc.wait()
        ran = clock() - started
        restarts += 1
        print(f"🏹 board supervisor: next dev (rc={rc}) exited after {ran:.0f}s — restarting", flush=True)
        if ran < rapid_window_s:
            rapid += 1
            if rapid >= max_rapid:
                print(f"🏹 board supervisor: {rapid} rapid failures inside {rapid_window_s:.0f}s — "
                      "giving up (fix the build, then `celeborn board --start`)", flush=True)
                return restarts
            backoff = min(backoff * 2, backoff_cap_s)
        else:
            rapid = 0                            # a healthy run resets the rapid-failure budget
            backoff = 1.0
        sleeper(backoff)


def _run_board_supervisor(args) -> None:
    """`celeborn board --supervise` — the detached entrypoint `_spawn_board` launches. Resolves the
    viewer argv itself (no project context needed) and runs the restart loop in the foreground; its
    stdio is the inherited `.board.log`."""
    board_dir = _board_dir()
    runner = _board_runner(board_dir)
    if runner is None:
        print("🏹 board supervisor: viewer unavailable (no app / deps / npm) — exiting", flush=True)
        return
    _board_supervise(runner, int(args.supervise_port), board_dir)


def _spawn_board(board_dir: Path, runner: list[str], port: int) -> int:
    """Start the SUPERVISOR detached and return its PID (the PID we record in the machine-global
    `.board.pid`). Own session (setsid) so it outlives the hook and the spawning session; stdio to the
    machine-global `.board.log`. The supervisor runs `next dev` (one shared server on `port`, CELE-t170)
    as a child and relaunches it on crash. No per-repo tasks path — the route resolves the project from
    the URL. Separated out so tests can stub the actual process launch."""
    import os, subprocess, sys
    env = {**os.environ, "PORT": str(port)}
    _config_dir().mkdir(parents=True, exist_ok=True)
    log = open(_board_log_path(), "ab", buffering=0)
    supervisor = [sys.executable, str(Path(__file__).resolve()), "board",
                  "--supervise", "--supervise-port", str(port)]
    try:
        proc = subprocess.Popen(
            supervisor, cwd=str(board_dir), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True,
        )
    finally:
        log.close()
    return proc.pid


def ensure_board(ctx: Path, *, launch: bool = True) -> dict:
    """Probe the shared board port; if it's down, start the ONE viewer (detached) so it's effectively
    always running on localhost:3141 for the whole fleet (CELE-t170). The seam the SessionStart hook
    calls on every orient — from any repo. Idempotent: once live, every other repo's orient is a no-op.

    Returns a status dict {port, url, live, action, reason?} where `action` is one of:
      live        already listening — nothing to do
      booting     the shared board we launched is still coming up (don't double-launch)
      started     just launched it (pid in the dict)
      off         autostart disabled in .celebornrc (board_autostart=false)
      no-tasks    this project doesn't use the kanban (no tasks.md) — stay quiet
      unavailable can't launch (no board app / deps not installed / no npm)
    Never raises — a hook must never break the user's turn."""
    import os
    port = board_port(ctx)
    url = board_url(ctx)
    base = {"port": port, "url": url}
    if _board_live(port):
        return {**base, "live": True, "action": "live"}
    if not bool(load_config(ctx).get("board_autostart", True)):
        return {**base, "live": False, "action": "off", "reason": "board_autostart=false"}
    if not (ctx / "tasks.md").is_file():
        return {**base, "live": False, "action": "no-tasks", "reason": "no tasks.md — kanban unused"}
    if _board_booting(port):
        return {**base, "live": False, "action": "booting"}
    board_dir = _board_dir()
    runner = _board_runner(board_dir)
    if runner is None:
        return {**base, "live": False, "action": "unavailable",
                "reason": "no board app, deps not installed, or npm missing"}
    if not launch:
        return {**base, "live": False, "action": "down"}
    # Machine-global singleton claim: only ONE concurrent orient (across any repo/thread) may spawn the
    # shared server. O_EXCL makes the check-and-claim atomic; a stale claim (dead pid) is stolen. The
    # fixed port is the ultimate backstop — a losing supervisor can't bind 3141 and backs off.
    try:
        _config_dir().mkdir(parents=True, exist_ok=True)
        pidfile = _board_pidfile_path()
        try:
            fd = os.open(str(pidfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            rec = _read_board_pidfile()
            rp = rec.get("pid")
            if isinstance(rp, int) and _pid_alive(rp):
                return {**base, "live": False, "action": "booting"}
            try:
                os.unlink(str(pidfile))
            except OSError:
                pass
            fd = os.open(str(pidfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            pid = _spawn_board(board_dir, runner, port)
            os.write(fd, (json.dumps({"pid": pid, "port": port, "at": now_iso()}) + "\n").encode())
        finally:
            os.close(fd)
        return {**base, "live": False, "action": "started", "pid": pid}
    except Exception as e:                      # noqa: BLE001 — never let a launch failure break orient
        return {**base, "live": False, "action": "unavailable", "reason": str(e)}


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.strip().lower())
    s = re.sub(r"[\s_]+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-")


def die(msg: str, code: int = 1):
    print(f"celeborn: error: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg: str):
    print(f"  ! {msg}")


def ok(msg: str):
    print(f"  ✓ {msg}")


def info(msg: str):
    """A neutral heads-up — not a warning or a problem (doesn't affect doctor's counts/exit code)."""
    print(f"  · {msg}")


# --------------------------------------------------------------------------- markdown parsing

def strip_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body). Only simple `key: value` pairs are parsed."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip("\n")
    rest = text[end + 4:].lstrip("\n")
    fm: dict = {}
    for line in raw.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm, rest


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
TAG_RE = re.compile(r"(?:^|\s)#([A-Za-z][\w-]+)")


COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def parse_sections(text: str) -> list[dict]:
    """Split markdown into sections by heading. The preamble before the first heading
    becomes a section with an empty title. HTML comments are stripped first so template
    boilerplate and instructional notes never reach the index."""
    fm, body = strip_frontmatter(text)
    body = COMMENT_RE.sub("", body)
    sections: list[dict] = []
    cur = {"title": "", "anchor": "", "level": 0, "lines": []}
    for line in body.splitlines():
        m = HEADING_RE.match(line)
        if m:
            if cur["lines"] or cur["title"]:
                sections.append(cur)
            title = m.group(2)
            cur = {"title": title, "anchor": slugify(title), "level": len(m.group(1)), "lines": []}
        else:
            cur["lines"].append(line)
    if cur["lines"] or cur["title"]:
        sections.append(cur)

    fm_tags = fm.get("tags", "")
    for s in sections:
        s["body"] = "\n".join(s["lines"]).strip()
        inline_tags = set(TAG_RE.findall(s["body"]))
        if fm_tags:
            inline_tags.update(t.strip() for t in re.split(r"[,\s]+", fm_tags) if t.strip())
        s["tags"] = " ".join(sorted(inline_tags))
        s["links"] = WIKILINK_RE.findall(s["body"])
    return sections


# --------------------------------------------------------------------------- journal entries

JOURNAL_ENTRY_RE = re.compile(r"^## ", re.MULTILINE)


def split_journal(text: str) -> tuple[str, list[str]]:
    """Return (header_block, [entry_block, ...]) where each entry starts at a `## ` line.
    `## ` lines inside HTML comments (e.g. the template's format hint) are not entries."""
    spans = [(m.start(), m.end()) for m in COMMENT_RE.finditer(text)]
    matches = [m for m in JOURNAL_ENTRY_RE.finditer(text)
               if not any(a <= m.start() < b for a, b in spans)]
    if not matches:
        return text, []
    header = text[: matches[0].start()]
    entries = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries.append(text[m.start():end])
    return header, entries


# --------------------------------------------------------------------------- metrics

METRICS_TEMPLATE = {
    "schema": "celeborn-metrics/1",
    "tokens_saved_estimate": 0,
    "load_events": 0,
    "orient_events": 0,
    "sessions_resumed": 0,
    "compactions_bridged": 0,
    "panic_saves": 0,
    "handoffs_written": 0,
    "last_session_id": None,
    "last_orient_at": None,
    # Rolling estimate of the live context window, in tokens. Celeborn can't observe the host's
    # window directly, so this is an accumulated proxy: `record turn --tokens N` adds to it, and a
    # new session / `clear` / `compaction` resets it to roughly the Hot-tier load. `remind --auto`
    # reads it. Approximate by design — refined whenever the host supplies real numbers.
    "context_estimate": 0,
    "last_remind_estimate": 0,
    # Cursors for deterministic transcript capture (`celeborn capture`). `captures` is keyed by
    # session id: each Claude session keeps its OWN byte offset, auto file, and running totals, so
    # concurrent or alternating sessions sharing one metrics.json (notably the global ~/.context
    # sink) can't stomp each other's offset and force a full re-read every turn. `capture` mirrors
    # the most-recently-active session — a back-compat slot and the fallback the heartbeat/statusline
    # read when no session id is supplied. tokens_session/idle_streak drive the per-turn `--note`
    # heartbeat (kept unique so Claude Code never suppresses it as a duplicate systemMessage).
    "captures": {},
    "capture": {"session_id": None, "offset": 0, "last_uuid": None, "file": None,
                "tokens_session": 0, "idle_streak": 0, "last_delta": 0},
    # Skill advisor (t70): per-session nudge throttle (`last_notice_session` + `notices_this_session`,
    # capped at advisor.max_per_session), user-dismissed intent ids, and the permission-friction ledger
    # the board surfaces. `permission_rules_generalized` is cumulative; `skipped_bottlenecks` is the last
    # apply's remaining (un-widenable) literals by family, with `_total` the aggregate the economy bar shows.
    "advisor": {"last_notice_session": None, "notices_this_session": 0, "dismissed": [],
                "permission_rules_generalized": 0,
                "skipped_bottlenecks": {}, "skipped_bottlenecks_total": 0, "last_applied_at": None},
    # Quality gates (t70 Phase 2): per-session "a test-relevant file was edited this turn" marker. The
    # post-edit hook sets `dirty_session` when scripts/** or tests/** changes; the quality-stop hook runs
    # the full suite ONCE when it sees its own session here, then clears it — keeping the ~90s suite off
    # the per-edit path.
    "quality": {"dirty_session": None},
    # CMM (codebase-memory-mcp, CELE-t92) economics. `prompts_auto_allowed` is the running ESTIMATE of
    # permission interruptions CMM eliminated: each capture counts the agent's calls to a CMM-pre-cleared
    # tool (the read-only `mcp__codebase-memory-mcp__*` set + the Grep/Glob engage added) — every one is a
    # structural query that flowed without an "Allow"/"Always allow" click, in place of a prompting
    # bash/grep shell-out. Only accrues while CMM is engaged (provenance-gated in celeborn_cmm).
    "cmm": {"prompts_auto_allowed": 0},
    # Permission allow-list economics (t100). `prompts_auto_allowed` is the running tally of prompts the
    # settings.json allow-list eliminated: each capture counts the agent's tool calls that matched an
    # allow rule (the safe baseline `wire --global` ships, plus any rule the user added) and so ran
    # without an "Allow"/"Always allow" click — Bash commands under a `Bash(<prefix>:*)` rule and
    # built-ins (Read/Glob/Grep/…) present verbatim. Excludes CMM's provenance-credited tools so the
    # two buckets never double-count.
    "permissions": {"prompts_auto_allowed": 0},
    # Active-agents bridge (CELE-t131). Maps a Claude session id → {owner, task, at} the moment that
    # session CLAIMS a card (the claim-on-receipt hook passes its session id). `celeborn agents` joins
    # this against the live transcripts to attribute each active context window to a handle + DOING
    # card. Pruned to the most-recent sessions like `captures`; absence just means an unattributed
    # session (shown by its short id, never a raw uuid).
    "agent_sessions": {},
}


def _est_tokens(text: str, cpt: int) -> int:
    """Rough token estimate from character count (~cpt chars/token)."""
    return (len(text) + cpt - 1) // max(1, cpt)


def _measure(ctx: Path, cpt: int) -> tuple[int, int]:
    """Return (hot_tokens, total_memory_tokens). Total = all knowledge an agent would otherwise
    carry if it naively loaded the whole .context/ — excludes derived/config/handoff files."""
    def toks(p: Path) -> int:
        try:
            return _est_tokens(p.read_text(errors="ignore"), cpt)
        except OSError:
            return 0

    hot = sum(toks(ctx / rel) for rel in HOT_FILES if (ctx / rel).is_file())
    skip = {INDEX_NAME, METRICS_NAME, RC_NAME, "handoff.md"}
    total = 0
    for p in ctx.rglob("*"):
        if p.is_file() and p.name not in skip and p.suffix in (".md", ".json"):
            total += toks(p)
    return hot, total


def _load_metrics(ctx: Path) -> dict:
    p = ctx / METRICS_NAME
    m = dict(METRICS_TEMPLATE)
    if p.is_file():
        try:
            m.update(json.loads(p.read_text()))
        except json.JSONDecodeError:
            warn(f"{METRICS_NAME} unreadable; starting fresh.")
    return m


def _save_metrics(ctx: Path, m: dict):
    (ctx / METRICS_NAME).write_text(json.dumps(m, indent=2) + "\n")


def _has_memory(ctx: Path, cpt: int) -> bool:
    hot, total = _measure(ctx, cpt)
    return total > hot


def _credit_savings(ctx: Path, m: dict, cpt: int) -> int:
    """Credit one load event: tokens saved by loading Hot instead of all of .context/."""
    hot, total = _measure(ctx, cpt)
    saved = max(0, total - hot)
    m["tokens_saved_estimate"] += saved
    m["load_events"] += 1
    return saved


def _orient_is_new_session(m: dict, sid: str, cfg: dict) -> bool:
    """Decide whether this orient is a distinct session (vs. a re-orient within the same one)."""
    if sid:
        return sid != (m.get("last_session_id") or "")
    last = m.get("last_orient_at")
    if not last:
        return True
    try:
        age = (_dt.datetime.now() - _dt.datetime.strptime(last, "%Y-%m-%dT%H:%M:%S")).total_seconds()
    except (ValueError, TypeError):
        return True
    return age >= cfg["orient_dedupe_seconds"]


def restarts_avoided(m: dict) -> int:
    return m.get("sessions_resumed", 0) + m.get("compactions_bridged", 0)


def metrics_summary(ctx: Path) -> list[str]:
    m = _load_metrics(ctx)
    saved = m["tokens_saved_estimate"]
    lines = [
        f"tokens saved: ~{saved:,} (est.) — Hot tier vs. loading all of .context/, over {m['load_events']} load event(s)",
        f"restarts avoided: {restarts_avoided(m)}  "
        f"({m['sessions_resumed']} session resume(s) + {m['compactions_bridged']} compaction(s) bridged)",
    ]
    return lines


# --------------------------------------------------------------------------- smart init (read the repo on first run)

# README filenames probed in priority order (plus a case-insensitive fallback).
_README_NAMES = ("README.md", "README.rst", "README.txt", "README", "readme.md")

# (manifest filename, stack label) — first present manifest drives the headline stack + name/desc.
_MANIFESTS = [
    ("package.json", "Node/JS"),
    ("pyproject.toml", "Python"),
    ("setup.py", "Python"),
    ("setup.cfg", "Python"),
    ("requirements.txt", "Python"),
    ("Cargo.toml", "Rust"),
    ("go.mod", "Go"),
    ("pom.xml", "Java/Maven"),
    ("build.gradle", "Java/Gradle"),
    ("Gemfile", "Ruby"),
    ("composer.json", "PHP"),
    ("Package.swift", "Swift"),
    ("pubspec.yaml", "Dart/Flutter"),
    ("CMakeLists.txt", "C/C++"),
]

_LANG_EXT = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".rs": "Rust", ".go": "Go", ".java": "Java",
    ".rb": "Ruby", ".php": "PHP", ".swift": "Swift", ".c": "C", ".h": "C", ".cpp": "C++",
    ".cc": "C++", ".cs": "C#", ".kt": "Kotlin", ".dart": "Dart", ".sh": "Shell",
    ".lua": "Lua", ".scala": "Scala", ".ex": "Elixir", ".exs": "Elixir",
}

_SCAN_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
                   ".next", "target", "vendor", ".context", ".idea", ".mypy_cache"}


def _readme_title_desc(text: str) -> tuple[str, str]:
    """First heading (project title) + first prose paragraph (description), skipping badges/HTML/fences."""
    title, desc_lines = "", []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if desc_lines:
                break  # blank line ends the first paragraph
            continue
        if line.startswith("#"):
            if not title:
                title = line.lstrip("#").strip()
            continue
        if line.startswith(("![", "[![", "<", "---", "===", "```", "|", ">", "- [")):
            continue  # badges, HTML, rules, fences, tables, quotes, ToC links
        desc_lines.append(line)
        if len(" ".join(desc_lines)) > 240:
            break
    desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", " ".join(desc_lines))  # [text](url) -> text
    desc = re.sub(r"[*`_]", "", desc).strip()
    if len(desc) > 240:
        desc = desc[:239].rstrip() + "…"
    return title, desc


def _scan_readme(root: Path) -> tuple[str, str]:
    candidates = [root / n for n in _README_NAMES]
    candidates += [p for p in sorted(root.glob("*")) if p.name.lower().startswith("readme")]
    for p in candidates:
        if p.is_file():
            try:
                return _readme_title_desc(p.read_text(errors="replace"))
            except OSError:
                continue
    return "", ""


def _manifest_name_desc(fname: str, path: Path) -> tuple[str, str]:
    """Best-effort (name, description) from a build manifest. Empty strings when unparseable."""
    try:
        text = path.read_text(errors="replace")
        if fname in ("package.json", "composer.json"):
            d = json.loads(text)
            return str(d.get("name") or ""), str(d.get("description") or "")
        if fname == "go.mod":
            m = re.search(r"^module\s+(\S+)", text, re.MULTILINE)
            return ((m.group(1).rsplit("/", 1)[-1]) if m else ""), ""
        if fname in ("pyproject.toml", "Cargo.toml", "setup.cfg"):
            nm = re.search(r'(?m)^\s*name\s*=\s*["\']?([^"\'\n]+)', text)
            ds = re.search(r'(?m)^\s*description\s*=\s*["\']?([^"\'\n]+)', text)
            return (nm.group(1).strip() if nm else ""), (ds.group(1).strip() if ds else "")
    except (OSError, ValueError):
        return "", ""
    return "", ""


def _scan_manifest(root: Path) -> dict:
    result = {"stack": "", "name": "", "description": "", "manifests": []}
    for fname, label in _MANIFESTS:
        if not (root / fname).is_file():
            continue
        result["manifests"].append(fname)
        if not result["stack"]:
            result["stack"] = label
            result["name"], result["description"] = _manifest_name_desc(fname, root / fname)
    return result


def _scan_git(root: Path) -> dict:
    import subprocess  # lazy: only init's scan pays the import

    def _git(*a):
        try:
            r = subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    if not _git("rev-parse", "--is-inside-work-tree"):
        return {"branch": "", "commit_count": 0, "recent_commits": []}
    count = _git("rev-list", "--count", "HEAD")
    commits = []
    for line in _git("log", "-n5", "--pretty=format:%h\t%s").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            commits.append({"short": parts[0], "subject": parts[1]})
    return {"branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit_count": int(count) if count.isdigit() else 0, "recent_commits": commits}


def _scan_languages(root: Path, cap: int = 4000) -> list[str]:
    """Top ≤3 languages by source-file count — a bounded walk that skips vendor/build dirs."""
    import os
    from collections import Counter
    tally: Counter = Counter()
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SCAN_SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            lang = _LANG_EXT.get(os.path.splitext(fn)[1].lower())
            if lang:
                tally[lang] += 1
            seen += 1
        if seen >= cap:
            break
    return [lang for lang, _ in tally.most_common(3)]


def _smart_scan(root: Path) -> dict:
    """Read-only repo probe for `celeborn init` — README, build manifest, git history, languages.
    Best-effort: every probe degrades to empty, the whole thing never raises."""
    scan = {"name": "", "description": "", "stack": "", "languages": [],
            "manifests": [], "branch": "", "commit_count": 0, "recent_commits": []}
    try:
        title, desc = _scan_readme(root)
        man = _scan_manifest(root)
        git = _scan_git(root)
        scan.update({
            "name": man["name"] or title or root.name,
            "description": desc or man["description"],
            "stack": man["stack"],
            "manifests": man["manifests"],
            "languages": _scan_languages(root),
            "branch": git["branch"],
            "commit_count": git["commit_count"],
            "recent_commits": git["recent_commits"],
        })
    except Exception:
        scan["name"] = scan["name"] or root.name
    return scan


def _scan_stack_label(scan: dict) -> str:
    bits = [scan["stack"]] if scan["stack"] else []
    bits += [l for l in scan["languages"] if l not in bits]
    return ", ".join(b for b in bits[:3] if b)


def _smart_now_block(scan: dict) -> str:
    """The derived `## Now` headline for a freshly-initialized state.md — deliberately tiny."""
    stack = _scan_stack_label(scan) or "—"
    repo = []
    if scan["branch"]:
        repo.append(f"branch `{scan['branch']}`")
    if scan["commit_count"]:
        repo.append(f"{scan['commit_count']} commit(s)")
    repo_line = " · ".join(repo) or "no git history yet"
    desc = scan["description"] or "_(no README description found — add a line on what this is)_"
    return "\n".join([
        "## Now",
        f"- **Project:** {scan['name'] or 'this project'} — {desc}",
        f"- **Stack:** {stack}  ·  **Repo:** {repo_line}",
        "- **Focus:** _Celeborn just initialized here — this is a repo snapshot, not a work focus yet. "
        "Set this to your first task and rewrite this headline._",
        "- **Next action:** _pick your first task; the full repo snapshot (recent commits, manifests) is in `notes.md`._",
        f"- **Branch:** {scan['branch'] or '<git branch>'} · **Status:** in-progress",
        "",
    ])


def _smart_notes_block(scan: dict, stamp: str) -> str:
    """The richer first-run snapshot appended to notes.md (unbounded; on-demand, not auto-loaded)."""
    lines = [f"\n## Repo snapshot — auto-captured by `celeborn init` ({stamp})",
             "_First-run orientation read straight from the repo. Trim or delete once you've set a real focus._"]
    if scan["description"]:
        lines.append(f"- **What it is:** {scan['description']}")
    stack = _scan_stack_label(scan)
    if stack:
        lines.append(f"- **Stack / languages:** {stack}")
    if scan["manifests"]:
        lines.append(f"- **Build manifests:** {', '.join(scan['manifests'])}")
    if scan["branch"] or scan["commit_count"]:
        lines.append(f"- **Git:** branch `{scan['branch'] or '?'}`, {scan['commit_count']} commit(s)")
    if scan["recent_commits"]:
        lines.append("- **Recent commits:**")
        for c in scan["recent_commits"]:
            lines.append(f"  - `{c['short']}` {c['subject']}")
    lines.append("")
    return "\n".join(lines)


def _apply_smart_state(path: Path, scan: dict):
    """Replace the template's `## Now` section (up to `## Pointers`) with the repo-derived headline."""
    text = path.read_text()
    now = _smart_now_block(scan)
    m = re.search(r"## Now\b.*?(?=\n## Pointers)", text, re.DOTALL)
    text = (text[:m.start()] + now + text[m.end():]) if m else (text.rstrip() + "\n\n" + now)
    path.write_text(text)


def _init_is_interactive() -> bool:
    """True only when init is driven by a human at a real terminal (both stdin and stdout are TTYs).
    The single gate for CELE-t121's install-time UX — prompting for a name and popping the board only
    make sense interactively. Headless/CI/test/agent installs return False, so init stays side-effect
    free and the SessionStart ensure-on-orient hook (CELE-t99) brings the board up on the next session."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _resolve_init_name(root: Path, ctx: Path, args) -> str | None:
    """Decide this project's display name on install (CELE-t121). Precedence:
      1. `--name` (explicit; for scripts/CI) — always wins.
      2. an existing `project_name` already in `.celebornrc` — kept, never re-prompted.
      3. an interactive prompt (only on a real TTY) defaulting to the repo folder name.
      4. otherwise None — the caller leaves the rc alone and `_project_name` falls back to the folder.
    Returns the name to persist, or None to leave the config untouched. Never raises."""
    explicit = (getattr(args, "name", None) or "").strip()
    if explicit:
        return explicit
    existing = (load_config(ctx).get("project_name") or "").strip()
    if existing:
        info(f"project name: {existing} (from {RC_NAME})")
        return None
    default = root.name or "this project"
    if not _init_is_interactive():
        return None  # headless/CI install — stay quiet, fall back to the folder name
    try:
        reply = input(f"  Project name for the kanban board [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return reply or default


def _ensure_tasks_md(ctx: Path) -> bool:
    """Seed an empty `tasks.md` so the kanban viewer has a board to serve (CELE-t121 — the board is
    Celeborn's UI and must be launchable right after install). Returns True if it created the file."""
    tp = _tasks_path(ctx)
    if tp.is_file():
        return False
    tp.write_text(TASKS_HEADER)
    return True


def _open_board_on_init(ctx: Path, *, open_browser: bool) -> None:
    """Launch this project's kanban viewer (detached) and, unless `--no-browser`, open it in the
    browser — the install-time half of 'the board is the UI, keep it open' (CELE-t121). Only called on
    an interactive install (the caller gates on `_init_is_interactive`). Best-effort: never raises,
    since failing to launch a viewer or pop a tab must not fail `init`."""
    try:
        st = ensure_board(ctx)
    except Exception:                                   # noqa: BLE001 — init must never die on the board
        return
    url = st.get("url", "")
    action = st.get("action")
    if action in ("started", "live", "booting"):
        verb = {"started": "starting", "live": "already live", "booting": "starting up"}[action]
        ok(f"kanban board {verb} → {url}")
    elif action == "off":
        info("kanban autostart is off (board_autostart=false) — `celeborn board --start` to launch it")
        return
    elif action == "unavailable":
        info(f"kanban board not started ({st.get('reason', 'unavailable')}) — install the board app, "
             "then `celeborn board --start`")
        return
    if not (open_browser and url):
        return  # --no-browser: leave the server running, just don't pop a tab
    import webbrowser
    try:
        webbrowser.open(url)
    except Exception:                                   # noqa: BLE001
        pass


# --------------------------------------------------------------------------- commands

def cmd_init(args):
    root = Path(args.path or ".").resolve()
    ctx = root / CONTEXT_DIRNAME
    if not TEMPLATES_DIR.is_dir():
        die(f"templates not found at {TEMPLATES_DIR} (run from the celeborn repo).")

    (ctx / "durable").mkdir(parents=True, exist_ok=True)
    (ctx / "journal-archive").mkdir(parents=True, exist_ok=True)

    copies = [
        ("state.md", ctx / "state.md"),
        ("notes.md", ctx / "notes.md"),
        ("journal.md", ctx / "journal.md"),
        ("learnings.md", ctx / "learnings.md"),
        ("decisions.md", ctx / "decisions.md"),
        ("handoff.md", ctx / "handoff.md"),
        ("durable/manifest.md", ctx / "durable" / "manifest.md"),
        ("celebornrc", ctx / RC_NAME),
    ]
    print(f"Initializing Celeborn memory at {ctx}")
    created: set[str] = set()
    for tmpl, dest in copies:
        src = TEMPLATES_DIR / tmpl
        if dest.exists():
            warn(f"exists, kept: {dest.relative_to(root)}")
        else:
            dest.write_text(src.read_text())
            created.add(tmpl)
            ok(f"created {dest.relative_to(root)}")

    # Smart init: read the repo so the FIRST orient already knows the project. Best-effort, read-only;
    # only ever seeds files we just created (never clobbers a user's existing state.md / notes.md).
    scan = _smart_scan(root) if getattr(args, "scan", True) else None

    # session.json gets a live timestamp (+ a repo-derived focus when smart init scanned)
    sj = ctx / "session.json"
    if sj.exists():
        warn(f"exists, kept: {sj.relative_to(root)}")
    else:
        data = json.loads((TEMPLATES_DIR / "session.json").read_text())
        data["updated_at"] = now_iso()
        if scan:
            tag = f" ({_scan_stack_label(scan)})" if _scan_stack_label(scan) else ""
            data["focus"] = (f"Fresh Celeborn init on {scan['name'] or root.name}{tag}. Repo snapshot "
                             "(README, recent commits, stack) is in notes.md; no work focus set yet.")
            data["next_action"] = "Pick your first task, then rewrite state.md's headline (Focus / Next action)."
            data["branch"] = scan["branch"] or ""
        _write_session(ctx, data)
        ok(f"created {sj.relative_to(root)}")

    if scan:
        if "state.md" in created:
            _apply_smart_state(ctx / "state.md", scan)
        if "notes.md" in created:
            _append(ctx / "notes.md", _smart_notes_block(scan, now_stamp()))
        if "state.md" in created or "notes.md" in created:
            label = scan["name"] or root.name
            stack = _scan_stack_label(scan)
            ok(f"smart init: read the repo — seeded {label}" + (f" ({stack})" if stack else "")
               + " into state.md + notes.md")

    mp = ctx / METRICS_NAME
    if mp.exists():
        warn(f"exists, kept: {mp.relative_to(root)}")
    else:
        _save_metrics(ctx, dict(METRICS_TEMPLATE))
        ok(f"created {mp.relative_to(root)}")

    private = _decide_private(root, args)
    _ensure_gitignore(root, private=private)
    if getattr(args, "claude_md", True):
        if _ensure_claude_md(root):
            ok("annotated CLAUDE.md (Claude Code auto-loads it → it'll orient via .context/)")
        else:
            warn("CLAUDE.md already annotated, kept")
    if getattr(args, "agents_md", True):
        if _ensure_agents_md(root):
            ok("annotated AGENTS.md (Codex/Grok-style hosts auto-load it → same orient + kanban rules)")
        else:
            warn("AGENTS.md already annotated, kept")
    if private:
        print("\n.context/ is PRIVATE (gitignored): it won't be committed. Carry it across\n"
              "devices with `celeborn sync` instead of git.")
    if _wire_grok(root):
        ok("wired Grok Build hooks + project rules (`.grok/rules/celeborn.md`)")
    # Auto-engage Codebase Memory (CMM) by default — Celeborn installs it into every project so the
    # agent answers structural questions through pre-cleared tools instead of prompting on Bash/Grep.
    # Best-effort + reversible (`cmm off`); opt out with `--no-cmm` / $CELEBORN_NO_CMM. Lazy import so
    # the free core stays dependency-light.
    try:
        __import__("celeborn_cmm").maybe_engage_on_init(args, ctx)
    except Exception:
        pass  # init must never fail on CMM

    # CELE-t121 — name the project, then open its kanban board (Celeborn's UI; keep it live).
    # `--name` persists even headlessly; seeding tasks.md + launching the viewer + popping the browser
    # only happen on an interactive install. A headless/CI/agent install stays side-effect free — the
    # SessionStart ensure-on-orient hook (CELE-t99) brings the board up on the next session instead.
    name = _resolve_init_name(root, ctx, args)
    if name:
        _update_config(ctx, project_name=name)
        ok(f"project name: {name} (saved to {RC_NAME})")
    if getattr(args, "open_board", True) and _init_is_interactive():
        if _ensure_tasks_md(ctx):
            ok("created .context/tasks.md (empty kanban board)")
        _open_board_on_init(ctx, open_browser=getattr(args, "open_browser", True))

    print("\nDone. Next: edit .context/state.md, then `celeborn status` and `celeborn index`.")
    print("Celeborn free is the local CLI you own — offline, no account. Want it on every device? "
          "`celeborn register` (free) then `celeborn upgrade`.")


GROK_RULES_BEGIN = "<!-- BEGIN CELEBORN (managed — regenerated by `celeborn init` / `celeborn grok sync-rules`) -->"
GROK_RULES_END = "<!-- END CELEBORN -->"


def _grok_install_script() -> Path | None:
    """Locate the Grok adapter install script (bundled checkout or installed skill copy)."""
    import shutil
    candidates = [
        REPO_ROOT / "grok" / "scripts" / "install.sh",
        Path.home() / ".grok" / "skills" / "celeborn-grok" / "scripts" / "install.sh",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _grok_rules_block(root: Path) -> str:
    """Per-project Grok rules Grok auto-loads from `.grok/rules/celeborn.md` (see Grok project rules)."""
    ctx = root / CONTEXT_DIRNAME
    slug = project_slug(ctx)
    return (
        f"{GROK_RULES_BEGIN}\n"
        f"# Celeborn — {slug}\n\n"
        f"**Memory:** `.context/` in this repository (orient from `state.md` + `session.json`)\n\n"
        "## Every Grok session — especially after `/clear`\n\n"
        "Grok **does not** inject SessionStart hook output into the model. Orient is **your first\n"
        "action** before replying:\n\n"
        "1. If `.context/.grok-orient-pending.md` exists → read it once, orient from it, **delete it**.\n"
        "2. Else → `celeborn status` (from this repository).\n\n"
        "## Launch Grok on this project (not a parent directory)\n\n"
        "Hooks resolve Celeborn from the session working directory. If you start Grok from `$HOME`,\n"
        "a parent `.context/` can win over this repo. Always launch from **this repository**:\n\n"
        "```bash\ncd <this-repo> && grok --cwd .\n```\n\n"
        "## Kanban shorthand (this project only)\n\n"
        f"| You say | Celeborn does |\n"
        f"|---|---|\n"
        f"| `wire tN` / `claim tN` | `celeborn claim tN --by <you>` then implement |\n"
        f"| `ship tN` | `celeborn ship tN` |\n"
        f"| `hydrate` / `orient` | Read orient-pending or run `celeborn status` |\n\n"
        f"Cards live in `.context/tasks.md`. Qualified marker: `⟨celeborn:{slug}/tN⟩`.\n"
        f"{GROK_RULES_END}\n"
    )


def _ensure_grok_rules(root: Path) -> bool:
    """Write or refresh `.grok/rules/celeborn.md` so Grok loads orient + kanban binding every session."""
    rules_dir = root / ".grok" / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    path = rules_dir / "celeborn.md"
    block = _grok_rules_block(root)
    if path.is_file():
        existing = path.read_text()
        if GROK_RULES_BEGIN in existing and GROK_RULES_END in existing:
            start = existing.index(GROK_RULES_BEGIN)
            stop = existing.index(GROK_RULES_END) + len(GROK_RULES_END)
            new = existing[:start] + block.rstrip("\n") + existing[stop:]
            if new == existing:
                return False
            path.write_text(new)
            return True
    path.write_text(block)
    return True


def _wire_grok(root: Path) -> bool:
    """Install Grok hooks (global, once) + bootstrap orient for this project. Idempotent; best-effort."""
    import shutil
    import subprocess
    if not shutil.which("grok") or not (Path.home() / ".grok").is_dir():
        return False
    install_sh = _grok_install_script()
    if install_sh is None:
        info("Grok Build detected — install the adapter: "
             f"bash {REPO_ROOT / 'grok' / 'scripts' / 'install.sh'} --project {root}")
        _ensure_grok_rules(root)
        return False
    try:
        subprocess.run(
            # --no-harness-pin: this is core's SPECULATIVE wiring (fires on any machine with Grok
            # installed, including Claude-primary repos), so it must not pin harness=grok in
            # .celebornrc and override the default-claude resolution. A deliberate grok install pins.
            ["bash", str(install_sh), "--project", str(root.resolve()), "--no-init", "--no-harness-pin"],
            capture_output=True, text=True, timeout=90, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        warn("Grok wire failed — run manually: "
             f"bash {install_sh} --project {root}")
        _ensure_grok_rules(root)
        return False
    _ensure_grok_rules(root)
    return True


def cmd_grok(args):
    """`celeborn grok wire` — (re)install Grok hooks + project rules. `sync-rules` refreshes
    `.grok/rules/celeborn.md` only (called from the Grok SessionStart hook every session)."""
    ctx = require_context(args)
    root = ctx.parent
    action = (getattr(args, "grok_action", None) or "wire").strip().lower()
    if action == "sync-rules":
        if _ensure_grok_rules(root):
            ok(f"refreshed {root / '.grok' / 'rules' / 'celeborn.md'}")
        else:
            info(f"{root / '.grok' / 'rules' / 'celeborn.md'} already current")
        return
    if action == "wire":
        if _wire_grok(root):
            ok("Grok Build wired for this project")
        else:
            die("Grok wire failed — is `grok` installed and ~/.grok present?")
        return
    die(f"unknown grok subcommand: {action}")


def _decide_private(root: Path, args) -> bool:
    """Whether to keep .context/ out of git. A PUBLIC repo means a public .context/,
    so default to private there; explicit flags always win."""
    if getattr(args, "private", False):
        return True
    if getattr(args, "public", False):
        return False
    vis = _repo_visibility(root)
    if vis == "public":
        warn("detected a PUBLIC repo — keeping .context/ private (gitignored) so your")
        warn("working memory isn't published. Re-run `init --public` to commit it anyway.")
        return True
    if vis is None and (root / ".git").is_dir():
        warn(".context/ will be COMMITTED to git. If this repo is or becomes public, your")
        warn("memory is public too. Use `init --private` to keep it local (and sync across")
        warn("devices with `celeborn sync`).")
    return False


def _repo_visibility(root: Path):
    """'public'/'private' via the gh CLI when available; None if unknown. Only consulted
    for real git repos, and never makes a network call otherwise."""
    if not (root / ".git").is_dir():
        return None
    import shutil
    import subprocess
    if not shutil.which("gh"):
        return None
    try:
        r = subprocess.run(
            ["gh", "repo", "view", "--json", "visibility", "-q", ".visibility"],
            cwd=str(root), capture_output=True, text=True, timeout=8,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    v = r.stdout.strip().lower()
    return v if v in ("public", "private") else None


def _append_gitignore_block(gi: Path, sentinel: str, block: str) -> bool:
    """Append `block` to .gitignore unless `sentinel` already appears. Idempotent. Returns whether
    it wrote anything."""
    existing = gi.read_text() if gi.is_file() else ""
    if sentinel in existing:
        return False
    prefix = "" if (existing.endswith("\n") or not existing) else "\n"
    with gi.open("a") as f:
        f.write(prefix + block)
    return True


def _ensure_gitignore(root: Path, private: bool = False):
    gi = root / ".gitignore"
    # The auto-captured tier is ALWAYS local-only (verbatim prompts + paths) — in every mode.
    _append_gitignore_block(
        gi, ".context/auto/",
        "\n# Celeborn Automatic Context Record is LOCAL-ONLY (verbatim prompts + paths).\n"
        "# It still rides `celeborn sync`.\n.context/auto/\n.context/activity.md\n"
        ".context/touches.json\n")
    _append_gitignore_block(
        gi, ".context/.agents.json",
        "\n# Celeborn agent identity registry (handle -> family/model) — local cache.\n"
        ".context/.agents.json\n")
    _append_gitignore_block(
        gi, ".context/.alerts.json",
        "\n# Celeborn live blocked-progress alerts (CELE-t169) — transient local state.\n"
        ".context/.alerts.json\n")
    if private:
        if _append_gitignore_block(
                gi, "/.context/",
                "\n# Celeborn working memory kept PRIVATE: a public repo means a public\n"
                "# .context/. Carry it across devices with `celeborn sync`, not git.\n/.context/\n"):
            ok("gitignored /.context/ (private — sync across devices with `celeborn sync`)")
        return
    if _append_gitignore_block(
            gi, ".context/index.db",
            "\n# Celeborn derived index (regenerable)\n.context/index.db\n.context/index.db-*\n"):
        ok("added .context/index.db to .gitignore")
    _append_gitignore_block(
        gi, ".context/tasks.json",
        "\n# Celeborn derived task board (regenerable from tasks.md)\n.context/tasks.json\n")
    _append_gitignore_block(
        gi, ".context/.board.pid",
        "\n# Celeborn board viewer runtime (ensure-on-orient) — local-only.\n"
        ".context/.board.pid\n.context/.board.log\n")
    _append_gitignore_block(
        gi, ".context/.panic/",
        "\n# Celeborn pre-compaction panic-saves — local snapshot/restore points.\n"
        ".context/.panic/\n")
    _append_gitignore_block(
        gi, ".context/outbox/",
        "\n# Celeborn prompt outbox — local per-agent hand-off queues, drained into the live session.\n"
        ".context/outbox/\n")
    _append_gitignore_block(
        gi, ".context/.jira-autopush.json",
        "\n# Celeborn Jira auto-push queue (local debounce state)\n.context/.jira-autopush.json\n")
    _append_gitignore_block(
        gi, ".context/.arch-trace.json",
        "\n# Celeborn auto-architecture-trace bookkeeping (CELE-t201) — transient local state.\n"
        ".context/.arch-trace.json\n")
    _append_gitignore_block(
        gi, ".context/progress.json",
        "\n# Celeborn progress-engine bookkeeping (floor/signals/nudge state) — local-only.\n"
        ".context/progress.json\n")
    _append_gitignore_block(
        gi, ".context/product-local.json",
        "\n# Celeborn product federation (CELE-t190): per-machine facet→checkout path bindings.\n"
        "# product.md (the product facts) IS committed; only these local paths stay out of git.\n"
        ".context/product-local.json\n")


# Managed block that announces Celeborn through the file Claude Code auto-loads (CLAUDE.md). This is
# how a fresh agent learns context lives in .context/ even before the skill or hooks are active.
CLAUDE_MD_BEGIN = "<!-- BEGIN CELEBORN (managed block — regenerated by `celeborn init`) -->"
CLAUDE_MD_END = "<!-- END CELEBORN -->"
AGENTS_MD_BEGIN = "<!-- BEGIN CELEBORN (managed block — regenerated by `celeborn init`) -->"
AGENTS_MD_END = "<!-- END CELEBORN -->"


def _rules_rehydration_body() -> str:
    """Orient + kanban guidance written into host rules files (CLAUDE.md, AGENTS.md) at init."""
    return (
        "## Context — maintained by Celeborn\n\n"
        "This project's long-term context lives in `.context/`, managed by Celeborn\n"
        "(<https://github.com/cloud-dancer-labs/celeborn>). **Orient before acting:** read\n"
        "`.context/state.md` (the headline) and `.context/session.json`; then `.context/notes.md`\n"
        "(open threads, constraints, working detail) and `.context/durable/manifest.md`. Run\n"
        "`celeborn search \"<topic>\"` to recall older details, and check `.context/journal.md` so you\n"
        "don't redo finished work. As you make meaningful changes, record them back into the authored\n"
        "tiers (state / journal / decisions / learnings) — that is what keeps the next session cheap.\n\n"
        "**Multi-agent kanban:** When this project uses Celeborn tasks (`.context/tasks.md`), every\n"
        "model sharing the repo sees the same board on orient — who's `doing` what. Celeborn is the\n"
        "live source of truth for work in progress; external issue trackers are downstream reporting\n"
        "only. Before taking a card, read the board (`celeborn tasks`) and choose a TODO that won't\n"
        "interrupt another agent's in-flight work. **One DOING card per agent** — ship (`celeborn ship\n"
        "<id>`) or demote before claiming another; `celeborn claim` blocks while you have other DOING\n"
        "cards (pass `--force` only to override). Prefer `celeborn tasks add \"…\" --claim --by <you>`\n"
        "so the new id is never guessed. Claim with `celeborn claim <id> --by <your name>` — owner and\n"
        "DOING update for everyone. Pasting a copied card (its `⟨celeborn:tN⟩` marker) also claims on\n"
        "receipt.\n\n"
        "**Every card has a Stop condition:** each task carries a logical **Stop condition** (its `stop`\n"
        "field) — a clearly-defined \"this is a clean place to stop\" marker that tells you when the card\n"
        "is at a defensible `/clear` point. `celeborn tasks add` auto-fills a generic default so no card\n"
        "is ever stop-less; when you claim a card, read its Stop condition and, if it still carries the\n"
        "generic default, replace it with a real one for that card: `celeborn tasks edit <id> --stop\n"
        "\"<condition>\"` (or set it up front with `tasks add … --stop \"…\"`). Honor it: reaching the Stop\n"
        "condition is the signal that you may cleanly `/clear`. `celeborn doctor` flags open cards with\n"
        "no Stop condition (or still on the default).\n\n"
        "**Identify yourself (once, on orient):** so other agents see *who* is on a file — not just a\n"
        "bare handle — declare your family + specific model once per session:\n"
        "`celeborn identify --family <Claude|Grok|GPT|Gemini…> --model \"<e.g. Opus 4.8>\"`. After that,\n"
        "every touch/claim/ship carries it automatically. (Two same-model threads? give each a distinct\n"
        "handle via `--by` / `$CELEBORN_AGENT`.)\n\n"
        "**Multi-agent file touches:** Before editing a shared file, register it so other agents see you\n"
        "on orient: `celeborn touch <file> --task <id> --why \"<reason>\"` (your `--by`/identity is\n"
        "inherited), then `celeborn touch release <file>` when done. The `--why` is what lets the next\n"
        "agent tell a deliberate edit from a collision. Never tag every line of code with agent metadata —\n"
        "touches + journal + commit trailers (`Celeborn-Agent`, `Celeborn-Model`, `Celeborn-Task`) are the\n"
        "protocol. Shorthand: `references/multi-agent-editing.md`.\n\n"
        "**Staying current:** Celeborn lives at <https://github.com/cloud-dancer-labs/celeborn>. Run\n"
        "`celeborn version --check` now and then — it looks back at GitHub and tells you if a newer\n"
        "Celeborn is available (and how to update).\n\n"
        "**Surface context-health notices (but NOT the heartbeat):** the Celeborn `UserPromptSubmit`\n"
        "hook injects two distinct channels. The **heartbeat** (`🏹 Celeborn —> M tokens recorded this\n"
        "session · +N last turn`) is tagged *context only, do NOT surface* — it informs you, you do not\n"
        "reprint it. As the context window fills, a **context-health notice** is also injected, tagged\n"
        "*SURFACE THIS TO THE USER*: it first asks you to freshen the Hot tier (rewrite\n"
        "`.context/state.md`, update `.context/session.json`, append `.context/journal.md`) so a `/clear`\n"
        "or compaction loses nothing, then hands you a `🏹 Celeborn —>` line to relay. On surfaces like\n"
        "the Claude desktop/web app the user never sees hook output — only your reply — so when that\n"
        "notice is present: do the checkpoint, then begin your reply with the line(s) it gives you,\n"
        "verbatim on their own lines, before anything else. Don't summarize or skip them.\n\n"
        "**Grok Build — orient survives `/clear`:** Grok does not inject SessionStart hook output into\n"
        "the model. After `/clear` (or any new session), **your first action** is orient: if\n"
        "`.context/.grok-orient-pending.md` exists, read it once and delete it; else run\n"
        "`celeborn status`. Launch Grok from **this repo** so hooks bind here, not a parent\n"
        "`.context/`: `grok --cwd <project-root>`. Kanban shorthand in **this** project:\n"
        "`wire tN` / `claim tN` = take card tN; `ship tN` = close it out (`celeborn ship tN`).\n"
        "Qualified markers `⟨celeborn:slug/tN⟩` on copied cards also claim on paste.\n"
    )


def _claude_md_block() -> str:
    return f"{CLAUDE_MD_BEGIN}\n{_rules_rehydration_body()}{CLAUDE_MD_END}\n"


def _agents_md_block() -> str:
    return f"{AGENTS_MD_BEGIN}\n{_rules_rehydration_body()}{AGENTS_MD_END}\n"


def _ensure_rules_file(root: Path, filename: str, begin: str, end: str, block_fn) -> bool:
    """Annotate a host rules file with a managed Celeborn block. Idempotent: replaces the marked
    block if present, appends it otherwise, creates the file if missing. Returns whether it wrote."""
    path = root / filename
    block = block_fn()
    if not path.exists():
        path.write_text(block)
        return True
    existing = path.read_text()
    if begin in existing and end in existing:
        start = existing.index(begin)
        stop = existing.index(end) + len(end)
        new = existing[:start] + block.rstrip("\n") + existing[stop:]
        if new == existing:
            return False
        path.write_text(new)
        return True
    prefix = "" if (existing.endswith("\n") or not existing) else "\n"
    path.write_text(existing + prefix + "\n" + block)
    return True


def _ensure_claude_md(root: Path) -> bool:
    """Annotate CLAUDE.md so Claude Code — which auto-loads it — knows context lives in .context/."""
    return _ensure_rules_file(root, "CLAUDE.md", CLAUDE_MD_BEGIN, CLAUDE_MD_END, _claude_md_block)


def _ensure_agents_md(root: Path) -> bool:
    """Annotate AGENTS.md so Codex/Grok-style hosts that auto-load it get the same orient guidance."""
    return _ensure_rules_file(root, "AGENTS.md", AGENTS_MD_BEGIN, AGENTS_MD_END, _agents_md_block)


def _clip(text: str, limit: int, pointer: str) -> str:
    """Bound a Hot-tier string to `limit` chars, cutting on a line boundary and appending a pointer.

    The Orient load is injected as SessionStart additionalContext; if it outgrows the host's inline
    budget, the host persists it to a file and the model gets only a preview — automatic rehydration
    silently dies. Clipping keeps the payload small while pointing the agent at the full source.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    if cut < limit // 2:  # no convenient line break near the limit — hard cut
        cut = limit
    dropped = len(text) - cut
    return text[:cut].rstrip("\n") + f"\n\n… [Hot tier clipped — {dropped} more chars in {pointer}]"


def cmd_status(args):
    ctx = require_context(args)
    cfg = load_config(ctx)
    sep = "─" * 72
    full = getattr(args, "full", False)
    state_max = 0 if full else int(cfg.get("hot_state_max_chars", 4000))
    act_max = 0 if full else int(cfg.get("hot_activity_max_chars", 2000))
    focus_max = 0 if full else int(cfg.get("hot_focus_max_chars", 1500))
    tasks_max = 0 if full else int(cfg.get("hot_tasks_max_chars", 1000))
    touches_max = 0 if full else int(cfg.get("hot_touches_max_chars", 800))

    print(sep)
    print("CELEBORN — Orient load (Hot tier)")
    print(sep)

    sj = ctx / "session.json"
    if sj.is_file():
        try:
            data = json.loads(sj.read_text())
            print("session.json:")
            for k in ("focus", "next_action", "branch", "status", "stop_allowed", "updated_at"):
                if k in data:
                    val = data[k]
                    if k in ("focus", "next_action") and isinstance(val, str):
                        val = _clip(val, focus_max, f"session.json:{k}")
                    print(f"  {k}: {val}")
            if data.get("open_threads"):
                print(f"  open_threads: {len(data['open_threads'])}")
        except json.JSONDecodeError:
            warn("session.json is not valid JSON")
    print()

    _print_file(ctx / "state.md", "state.md", clip=(state_max, "state.md"))
    if (ctx / "activity.md").is_file():
        _print_file(ctx / "activity.md", "activity.md (Automatic Context Record — mechanical, always-current)",
                    clip=(act_max, "activity.md"))
    _print_file(ctx / "durable" / "manifest.md", "durable/manifest.md")

    tasks_summary = _tasks_orient_summary(ctx, _load_tasks(ctx))
    if tasks_summary:
        print(sep)
        print("tasks.md (board — in flight; full board: `celeborn tasks`):")
        print(sep)
        print(_clip(tasks_summary, tasks_max, "tasks.md"))
        print()

    touches_summary = _touches_orient_summary(ctx)
    if touches_summary:
        print(sep)
        print("touches.json (active file edits — who is in which file; `celeborn touch list`):")
        print(sep)
        print(_clip(touches_summary, touches_max, "touches.json"))
        print()

    print(sep)
    print("Deeper tiers (on demand — not loaded):")
    notes = ctx / "notes.md"
    if notes.is_file():
        nlines = len(notes.read_text().splitlines())
        print(f"  notes.md: working detail — open threads, constraints, context ({nlines} lines) "
              f"— read it for depth")
    _, entries = split_journal((ctx / "journal.md").read_text()) if (ctx / "journal.md").is_file() else ("", [])
    archive_files = list((ctx / "journal-archive").glob("*.md"))
    learn = _count_sections(ctx / "learnings.md")
    dec = _count_sections(ctx / "decisions.md")
    durable_docs = [p for p in (ctx / "durable").glob("*.md") if p.name != "manifest.md"]
    keep = cfg["journal_keep_entries"]
    flag = "  (over budget — run `celeborn archive`)" if len(entries) > keep else ""
    print(f"  journal.md: {len(entries)} entries (keep {keep}){flag}")
    print(f"  journal-archive/: {len(archive_files)} file(s)")
    print(f"  learnings.md: {learn} · decisions.md: {dec} · durable docs: {len(durable_docs)}")

    idx = ctx / INDEX_NAME
    if idx.is_file():
        stale = _index_is_stale(ctx)
        print(f"  index.db: present{' (STALE — run `celeborn index`)' if stale else ''}")
    else:
        print("  index.db: absent (run `celeborn index` to enable search)")
    print(sep)
    print("Memory economy (estimated):")
    for line in metrics_summary(ctx):
        print(f"  {line}")
    print(sep)


def _print_file(path: Path, label: str, clip: tuple = (0, "")):
    sep = "─" * 72
    print(sep)
    print(f"{label}:")
    print(sep)
    if path.is_file():
        body = path.read_text().rstrip("\n")
        limit, pointer = clip
        if limit:
            body = _clip(body, limit, pointer)
        print(body)
    else:
        warn(f"missing: {label}")
    print()


def _count_sections(path: Path) -> int:
    """Count real entries — level-2 headings — ignoring the file's H1 title and preamble."""
    if not path.is_file():
        return 0
    return sum(1 for s in parse_sections(path.read_text()) if s["level"] == 2)


def cmd_index(args):
    import sqlite3  # lazy: only the DB-touching commands pay the import

    ctx = require_context(args)
    if not SCHEMA_PATH.is_file():
        die(f"schema not found at {SCHEMA_PATH}")
    db_path = ctx / INDEX_NAME
    conn = sqlite3.connect(str(db_path))
    try:
        # The index is derived and regenerable, so durability is irrelevant: skip fsync and
        # disk journaling for a much faster bulk build. A crash just means re-running `index`.
        conn.executescript(
            "PRAGMA journal_mode = MEMORY;"
            "PRAGMA synchronous = OFF;"
            "PRAGMA temp_store = MEMORY;"
        )
        conn.executescript(SCHEMA_PATH.read_text())
        rows = 0
        link_rows = 0
        for tier, glob in TIER_GLOBS:
            for path in sorted(ctx.glob(glob)):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(ctx))
                mtime = _dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%S")
                for s in parse_sections(path.read_text()):
                    if not (s["body"] or s["title"]):
                        continue
                    conn.execute(
                        "INSERT INTO memory_fts (title, body, tags, tier, source_file, anchor, updated_at) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (s["title"], s["body"], s["tags"], tier, rel, s["anchor"], mtime),
                    )
                    rows += 1
                    for tgt in s["links"]:
                        conn.execute(
                            "INSERT INTO links (src_file, src_anchor, target) VALUES (?,?,?)",
                            (rel, s["anchor"], tgt),
                        )
                        link_rows += 1
        conn.execute("INSERT INTO meta (key, value) VALUES ('built_at', ?)", (now_iso(),))
        # FTS5: merge the segments left by the bulk insert into one b-tree -> faster MATCH queries.
        conn.execute("INSERT INTO memory_fts (memory_fts) VALUES ('optimize')")
        conn.commit()
    finally:
        conn.close()
    print(f"Indexed {rows} section(s), {link_rows} link(s) -> {db_path.relative_to(ctx.parent)}")


def cmd_search(args):
    import sqlite3  # lazy: only the DB-touching commands pay the import

    ctx = require_context(args)
    cfg = load_config(ctx)
    db_path = ctx / INDEX_NAME
    if not db_path.is_file():
        die("no index yet. Run `celeborn index` first.")
    if _index_is_stale(ctx):
        warn("index looks stale; results may be behind the markdown. Run `celeborn index`.")
    limit = args.limit or cfg["search_default_limit"]
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        "PRAGMA query_only = ON;"
        "PRAGMA mmap_size = 67108864;"  # 64 MiB memory-mapped reads
    )
    try:
        try:
            cur = conn.execute(
                "SELECT tier, source_file, anchor, title, "
                "snippet(memory_fts, 1, '«', '»', ' … ', 14) AS snip "
                "FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
                (args.query, limit),
            )
            results = cur.fetchall()
        except sqlite3.OperationalError as e:
            die(f"bad FTS query: {e}")
    finally:
        conn.close()

    if not results:
        print(f"No matches for: {args.query}")
        return
    print(f"{len(results)} match(es) for: {args.query}\n")
    for tier, src, anchor, title, snip in results:
        pointer = f"{src}#{anchor}" if anchor else src
        head = title or "(preamble)"
        print(f"[{tier}] {pointer}")
        print(f"    {head}")
        print(f"    {snip.strip()}")
        print()


def cmd_archive(args):
    ctx = require_context(args)
    cfg = load_config(ctx)
    keep = args.keep if args.keep is not None else cfg["journal_keep_entries"]
    jpath = ctx / "journal.md"
    if not jpath.is_file():
        die("no journal.md found.")
    header, entries = split_journal(jpath.read_text())
    if len(entries) <= keep:
        print(f"journal.md has {len(entries)} entries (keep {keep}); nothing to archive.")
        return
    move = entries[: len(entries) - keep]
    kept = entries[len(entries) - keep:]

    arch_dir = ctx / "journal-archive"
    arch_dir.mkdir(exist_ok=True)
    arch_path = arch_dir / "archive.md"
    prefix = arch_path.read_text() if arch_path.is_file() else "# Journal archive\n\n"
    arch_path.write_text(prefix.rstrip("\n") + "\n\n" + "".join(move).rstrip("\n") + "\n")

    jpath.write_text(header.rstrip("\n") + "\n\n" + "".join(kept).rstrip("\n") + "\n")
    print(f"Archived {len(move)} entr(ies) -> journal-archive/archive.md; kept {len(kept)} in journal.md.")
    print("Re-run `celeborn index` to refresh search.")


def cmd_promote(args):
    ctx = require_context(args)
    title = args.title.strip()
    note = (args.note or "").strip()
    stamp = now_stamp()
    if args.to == "learnings":
        path = ctx / "learnings.md"
        block = f"\n## {title}\n- **Lesson:** {note}\n- **Seen in:** promoted {stamp}\n"
        _append(path, block)
        print(f"Promoted to learnings.md: {title}")
    elif args.to == "durable":
        doc = args.doc or "gotchas"
        path = ctx / "durable" / f"{doc}.md"
        if not path.is_file():
            path.write_text(f"# {doc.capitalize()}\n")
            _ensure_manifest_line(ctx, doc)
        block = f"\n## {title}\n{note}\n\n*(promoted {stamp})*\n"
        _append(path, block)
        _ensure_manifest_line(ctx, doc)
        print(f"Promoted to durable/{doc}.md: {title}")
    print("Remember to remove the source note from its old tier, then `celeborn index`.")


def _append(path: Path, block: str):
    existing = path.read_text() if path.is_file() else ""
    path.write_text(existing.rstrip("\n") + "\n" + block)


def _ensure_manifest_line(ctx: Path, doc: str):
    manifest = ctx / "durable" / "manifest.md"
    text = manifest.read_text() if manifest.is_file() else "# Durable docs manifest\n"
    if f"({doc}.md)" in text:
        return
    manifest.write_text(text.rstrip("\n") + f"\n- [{doc}.md]({doc}.md) — promoted durable knowledge\n")


def cmd_handoff(args):
    ctx = require_context(args)
    data = {}
    sj = ctx / "session.json"
    if sj.is_file():
        try:
            data = json.loads(sj.read_text())
        except json.JSONDecodeError:
            warn("session.json unreadable; handoff will be sparse.")
    branch = data.get("branch") or "<branch>"
    status = data.get("status") or "in-progress"
    focus = data.get("focus") or "<focus>"
    nxt = data.get("next_action") or "<next action>"
    threads = data.get("open_threads") or []
    risks = "\n".join(f"- {t}" for t in threads) or "- (none recorded)"

    content = f"""# Handoff

<!-- Regenerated by `celeborn handoff` from state.md + session.json. -->

**Branch:** {branch} · **Status:** {status}
**Focus:** {focus}
**Next required action:** {nxt}

**Open risks / threads:**
{risks}

---

### Resume prompt (paste into a fresh thread)

> Read `.context/state.md` (the headline) and `.context/session.json`, then `.context/notes.md`
> for open threads + constraints and `.context/durable/manifest.md` for repo truths. Continue from
> the Next required action above. Run `celeborn search "<topic>"` for anything older. Do not re-do
> completed work (see `journal.md`).
"""
    (ctx / "handoff.md").write_text(content)
    m = _load_metrics(ctx)
    m["handoffs_written"] += 1
    _save_metrics(ctx, m)
    print("Wrote handoff.md")


def cmd_record(args):
    """Record a memory event for the economy estimate. Called by the hooks; safe to call manually."""
    ctx = require_context(args)
    cfg = load_config(ctx)
    cpt = cfg["chars_per_token"]
    m = _load_metrics(ctx)
    saved = 0
    hot, _ = _measure(ctx, cpt)
    if args.event == "orient":
        m["orient_events"] += 1
        if _orient_is_new_session(m, args.session or "", cfg):
            # A fresh session starts roughly at the Hot-tier load — reset the running estimate.
            m["context_estimate"] = hot
            m["last_remind_estimate"] = 0
            if _has_memory(ctx, cpt):
                m["sessions_resumed"] += 1
                saved = _credit_savings(ctx, m, cpt)
        if args.session:
            m["last_session_id"] = args.session
        m["last_orient_at"] = now_iso()
    elif args.event == "compaction":
        m["compactions_bridged"] += 1
        # Post-compaction the carried context shrinks toward the Hot load — reset the estimate.
        m["context_estimate"] = hot
        m["last_remind_estimate"] = 0
        if _has_memory(ctx, cpt):
            saved = _credit_savings(ctx, m, cpt)
    elif args.event == "clear":
        # After a /clear + rehydrate, the window is back to roughly the Hot-tier load.
        m["context_estimate"] = hot
        m["last_remind_estimate"] = 0
    elif args.event == "turn":
        m["context_estimate"] = m.get("context_estimate", 0) + max(0, args.tokens or 0)
    elif args.event == "handoff":
        m["handoffs_written"] += 1
    _save_metrics(ctx, m)
    note = f" (+~{saved:,} tokens)" if saved else ""
    if args.event in ("turn", "clear"):
        note = f" (context estimate ~{m['context_estimate']:,} tokens)"
    print(f"recorded: {args.event}{note}")


# --------------------------------------------------------------------------- standup / changelog

def dollars_saved(ctx: Path) -> float:
    """Convert the tokens-saved estimate into a $ figure for the build-in-public flex. Rate is
    configurable (`usd_per_mtok` in .celebornrc); the saved tokens are context NOT re-loaded, i.e.
    input tokens, so the input rate is the right basis."""
    m = _load_metrics(ctx)
    rate = float(load_config(ctx).get("usd_per_mtok", 3.0))
    return m.get("tokens_saved_estimate", 0) / 1_000_000 * rate


_JOURNAL_DATE_RE = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}))?\s*(?:—|-|–)?\s*(.*)")


def _parse_dt(s: str):
    """Best-effort parse of an ISO-ish timestamp/date → datetime, or None."""
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(s.strip())
    except ValueError:
        try:
            return _dt.datetime.strptime(s.strip()[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _git_commits_since(repo: Path, since: _dt.datetime) -> list[tuple[str, str]]:
    """[(short_sha, subject), …] committed at/after `since`. Empty if not a git repo / no git."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log", f"--since={since.strftime('%Y-%m-%d %H:%M:%S')}",
             "--pretty=format:%h%x09%s"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.splitlines():
        if "\t" in line:
            sha, subj = line.split("\t", 1)
            rows.append((sha, subj))
    return rows


def _gather_activity(ctx: Path, since: _dt.datetime) -> dict:
    """Mechanical (no-model) aggregation of 'what happened' since `since`: tasks moved to done,
    git commits, and authored journal entries — the three concrete records of progress."""
    # Tasks completed in window (state == done, updated >= since)
    done = []
    for t in _load_tasks(ctx):
        if t["state"] == "done":
            up = _parse_dt(t.get("updated", ""))
            if up and up >= since:
                done.append(t)
    # Git commits in window
    commits = _git_commits_since(ctx.parent, since)
    # Journal entries in window
    jpath = ctx / "journal.md"
    entries = []
    if jpath.is_file():
        _, blocks = split_journal(jpath.read_text())
        for blk in blocks:
            m = _JOURNAL_DATE_RE.match(blk.strip().splitlines()[0])
            if not m:
                continue
            d = _parse_dt(m.group(1))
            if d and d >= _dt.datetime(since.year, since.month, since.day):
                entries.append(m.group(3).strip() or m.group(1))
    return {"done": done, "commits": commits, "journal": entries}


def _project_name(ctx: Path) -> str:
    rc = load_config(ctx)
    return rc.get("project_name") or _repo_root_from_ctx(ctx).name or "this project"


def _render_tweet(ctx: Path, act: dict) -> str:
    """A build-in-public X post (≤280 chars) from the same activity. Mechanical/template — no model,
    no API key — so it works in the free, offline core. Leads with shipped items, closes with the
    Celeborn flex (tokens→$) so every post doubles as soft marketing."""
    name = _project_name(ctx)
    lines = [f"🏹 Building {name} in public — latest:", ""]
    items = [t["title"] for t in act["done"]] or [s for _, s in act["commits"]]
    shown = 0
    for it in items:
        if shown >= 3:
            break
        clipped = it if len(it) <= 60 else it[:57] + "…"
        lines.append(f"✅ {clipped}")
        shown += 1
    extra = []
    if act["commits"]:
        extra.append(f"🔧 {len(act['commits'])} commits")
    usd = dollars_saved(ctx)
    if usd >= 1:
        extra.append(f"💪 ~${usd:,.0f} in tokens saved by Celeborn")
    if extra:
        lines += ["", " · ".join(extra)]
    lines += ["", "#buildinpublic #AI"]
    post = "\n".join(lines)
    # Hard cap at 280; trim trailing item lines if needed.
    while len(post) > 280 and shown > 1:
        # drop the last shown ✅ line
        idx = max(i for i, l in enumerate(lines) if l.startswith("✅"))
        lines.pop(idx)
        shown -= 1
        post = "\n".join(lines)
    return post[:280]


def _render_report(ctx: Path, act: dict, title: str) -> str:
    out = [title, ""]
    if act["done"]:
        out.append("✅ Completed")
        for t in act["done"]:
            out.append(f"  • [{_display_tid(ctx, t['id'])}] {t['title']}")
        out.append("")
    if act["commits"]:
        out.append(f"🔧 Commits ({len(act['commits'])})")
        for sha, subj in act["commits"][:20]:
            out.append(f"  • {sha}  {subj}")
        if len(act["commits"]) > 20:
            out.append(f"  …and {len(act['commits']) - 20} more")
        out.append("")
    if act["journal"]:
        out.append("📓 Journal")
        for j in act["journal"]:
            out.append(f"  • {j}")
        out.append("")
    if not (act["done"] or act["commits"] or act["journal"]):
        out.append("  (nothing recorded in this window)")
    return "\n".join(out).rstrip() + "\n"


def cmd_standup(args):
    """`standup` (default 1 day) and `changelog` (default 7 days) share one engine. `--tweet` emits a
    build-in-public X post instead of the report."""
    ctx = require_context(args)
    kind = getattr(args, "kind", "standup")
    default_days = 7 if kind == "changelog" else 1
    days = getattr(args, "days", None) or default_days
    since = _dt.datetime.now() - _dt.timedelta(days=days)
    act = _gather_activity(ctx, since)

    if getattr(args, "tweet", False):
        print(_render_tweet(ctx, act))
        return
    if getattr(args, "json", False):
        print(json.dumps({"since": since.isoformat(), "days": days, **act}, indent=2, default=str))
        return
    span = "last 24h" if days == 1 else f"last {days} days"
    icon = "📋" if kind == "standup" else "📰"
    print(_render_report(ctx, act, f"{icon} Celeborn {kind} — {_project_name(ctx)} ({span})"))


# --------------------------------------------------------------------------- flex ($ Wrapped card)

def _disp_width(s: str) -> int:
    """Approximate terminal display width of `s`: emoji and East-Asian-wide glyphs take 2 cells;
    variation selectors / ZWJ / combining marks take 0. Good enough to align an ASCII box that may
    carry the 🏹/💪 branding. Pure stdlib (unicodedata) — no wcwidth dependency."""
    import unicodedata
    w = 0
    for ch in s:
        o = ord(ch)
        if o in (0x200D, 0xFE0E, 0xFE0F) or unicodedata.combining(ch):
            continue
        if (o >= 0x1F000 or 0x2600 <= o <= 0x27BF or 0x2B00 <= o <= 0x2BFF
                or unicodedata.east_asian_width(ch) in ("W", "F")):
            w += 2
        else:
            w += 1
    return w


def _fmt_usd(usd: float) -> str:
    """$ figure for the flex: whole dollars once it's meaningful, cents while it's still small — so a
    fresh install flexes a real non-zero number instead of a flat '$0'."""
    return f"${usd:,.0f}" if usd >= 100 else f"${usd:,.2f}"


def _prompts_auto_allowed(m: dict) -> int:
    """Total permission prompts the user never had to click — the unified "prompts auto-allowed"
    figure the economy bar shows. Three complementary sources: the advisor's one-time allow-list rules
    generalized into wildcards (`celeborn permissions --apply`), CMM's per-call pre-clear of
    structural-query tools (CELE-t92), and the settings.json allow-list — the t100 safe baseline plus
    the user's own rules — matched per tool call at capture time. All three are prompts avoided; the
    bar sums them into one honest number (the buckets are disjoint, so nothing double-counts)."""
    generalized = int(((m.get("advisor") or {}).get("permission_rules_generalized")) or 0)
    cmm_calls = int(((m.get("cmm") or {}).get("prompts_auto_allowed")) or 0)
    allowlist = int(((m.get("permissions") or {}).get("prompts_auto_allowed")) or 0)
    return generalized + cmm_calls + allowlist


def _flex_figures(ctx: Path) -> dict:
    """The numbers behind the flex, in one place (shared by the card, the tweet, and --json)."""
    m = _load_metrics(ctx)
    return {
        "dollars_saved": round(dollars_saved(ctx), 2),
        "tokens_saved": m.get("tokens_saved_estimate", 0),
        "restarts_avoided": restarts_avoided(m),
        "sessions_resumed": m.get("sessions_resumed", 0),
        "compactions_bridged": m.get("compactions_bridged", 0),
        "load_events": m.get("load_events", 0),
        "prompts_auto_allowed": _prompts_auto_allowed(m),
        "cmm_prompts_auto_allowed": int(((m.get("cmm") or {}).get("prompts_auto_allowed")) or 0),
        "usd_per_mtok": float(load_config(ctx).get("usd_per_mtok", 3.0)),
        "project": _project_name(ctx),
    }


def _flex_card(ctx: Path) -> str:
    """The 🏹💪 '$ Wrapped' card — a box-drawn, copy-paste-to-X brag built from the memory economy:
    dollars saved (tokens→$), tokens never re-loaded, and restarts avoided. Mechanical, offline, free
    — it's the billboard. Box columns are aligned via _disp_width so the emoji don't skew the border."""
    f = _flex_figures(ctx)
    rows = [
        ("🏹  CELEBORN · $ WRAPPED  💪", "center"),
        ("", "left"),
        (f"{_fmt_usd(f['dollars_saved'])} saved in tokens", "left"),
        (f"{f['tokens_saved']:,} tokens never re-loaded", "left"),
        (f"{f['restarts_avoided']} restarts avoided", "left"),
        (f"  ({f['sessions_resumed']} resume(s) · {f['compactions_bridged']} compaction(s) bridged)", "left"),
    ]
    if f.get("prompts_auto_allowed"):
        rows.append((f"{f['prompts_auto_allowed']} permission prompts auto-allowed", "left"))
    rows += [
        ("", "left"),
        (f"{f['project']} · across {f['load_events']} load event(s)", "left"),
    ]
    inner = max(max(_disp_width(t) for t, _ in rows) + 4, 46)   # 2-space gutter each side, min width
    top, sep, bot = ("╭" + "─" * inner + "╮", "├" + "─" * inner + "┤", "╰" + "─" * inner + "╯")
    out = [top]
    for i, (text, align) in enumerate(rows):
        pad = inner - _disp_width(text)
        if align == "center":
            left = pad // 2
            cell = " " * left + text + " " * (pad - left)
        else:
            cell = "  " + text + " " * (pad - 2)
        out.append("│" + cell + "│")
        if i == 0:
            out.append(sep)
    out.append(bot)
    return "\n".join(out)


def _flex_tweet(ctx: Path) -> str:
    """A ≤280-char build-in-public flex post from the same figures. Leads with the $ + restarts brag,
    then the why (context never reloaded). Trims the explanatory line first if it runs long."""
    f = _flex_figures(ctx)
    full = [
        f"🏹💪 Celeborn has saved me {_fmt_usd(f['dollars_saved'])} and {f['restarts_avoided']} restarts on {f['project']}.",
        "",
        f"{f['tokens_saved']:,} tokens of context I never had to reload — my AI remembers across sessions & compactions, so I just keep building.",
        "",
        "#buildinpublic #AI",
    ]
    post = "\n".join(full)
    if len(post) <= 280:
        return post
    return "\n".join([full[0], "", "#buildinpublic #AI"])[:280]


def cmd_flex(args):
    """`celeborn flex` — the shareable 🏹💪 '$ Wrapped' brag card. Default: a box-drawn terminal card;
    `--tweet`: a ≤280-char build-in-public X post; `--json`: the raw figures."""
    ctx = require_context(args)
    if getattr(args, "json", False):
        print(json.dumps(_flex_figures(ctx), indent=2))
        return
    if getattr(args, "tweet", False):
        print(_flex_tweet(ctx))
        return
    print(_flex_card(ctx))


# --------------------------------------------------------------------------- savings (board surface for flex figures)

def _savings_figures(ctx: Path) -> dict:
    """The running savings totals the board surfaces in place of `flex` updates (t68): this project
    since start, and the same figures summed across every registered Celeborn project (+ this one).
    Each project's $ is computed against its own `usd_per_mtok` rate, then summed — so the fleet total
    is rate-correct even when projects price tokens differently."""
    project = _flex_figures(ctx)
    project["advisor"] = _advisor_figures(ctx)
    fleet = {"projects": 0, "dollars_saved": 0.0, "tokens_saved": 0, "restarts_avoided": 0,
             "sessions_resumed": 0, "compactions_bridged": 0, "load_events": 0,
             "prompts_auto_allowed": 0}
    fleet_adv = {"permission_rules_generalized": 0, "skipped_bottlenecks_total": 0}
    for pdir in _fleet_project_paths(ctx):
        pctx = pdir / CONTEXT_DIRNAME
        if not pctx.is_dir():
            continue
        f = _flex_figures(pctx)
        fleet["projects"] += 1
        fleet["dollars_saved"] += f["dollars_saved"]
        for k in ("tokens_saved", "restarts_avoided", "sessions_resumed",
                  "compactions_bridged", "load_events", "prompts_auto_allowed"):
            fleet[k] += f[k]
        fa = _advisor_figures(pctx)
        for k in fleet_adv:
            fleet_adv[k] += fa[k]
    fleet["dollars_saved"] = round(fleet["dollars_saved"], 2)
    fleet["advisor"] = fleet_adv
    return {"generated_at": now_iso(), "project": project, "fleet": fleet}


def _advisor_figures(ctx: Path) -> dict:
    """The permission-friction ledger the economy bar surfaces: cumulative rules auto-generalized and
    the aggregate bottlenecks (un-widenable literals) still re-prompting. Mirrors `metrics['advisor']`."""
    adv = (_load_metrics(ctx).get("advisor") or {})
    return {
        "permission_rules_generalized": int(adv.get("permission_rules_generalized", 0) or 0),
        "skipped_bottlenecks_total": int(adv.get("skipped_bottlenecks_total", 0) or 0),
    }


def cmd_savings(args):
    """`celeborn savings` — the running savings totals (this project + the whole fleet) the kanban
    board renders as its one-line economy bar, in place of pushed `flex` updates (t68). `--json`
    feeds the board's /api/savings route."""
    ctx = require_context(args)
    data = _savings_figures(ctx)
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2))
        return
    p, fl = data["project"], data["fleet"]
    paa = f" · 🔓 {p['prompts_auto_allowed']} prompts auto-allowed" if p.get("prompts_auto_allowed") else ""
    print(f"💰 {_fmt_usd(p['dollars_saved'])} · 🧠 {p['tokens_saved']:,} tokens · "
          f"♻️ {p['restarts_avoided']} restarts{paa}  —  {p['project']}")
    fpaa = f" · 🔓 {fl['prompts_auto_allowed']} auto-allowed" if fl.get("prompts_auto_allowed") else ""
    print(f"🌐 across {fl['projects']} project(s): {_fmt_usd(fl['dollars_saved'])} · "
          f"🧠 {fl['tokens_saved']:,} tokens · ♻️ {fl['restarts_avoided']} restarts{fpaa}")


# --------------------------------------------------------------------------- blame (git blame for the why)

BLAME_MEMORY_FILES = (
    ("decisions.md", "decision"),
    ("learnings.md", "learning"),
    ("journal.md", "journal"),
    ("notes.md", "note"),
)


def _git_file_history(repo: Path, relpath: str, limit: int = 8) -> list[dict]:
    """Recent commits that touched `relpath`. Empty if not a git repo or the path has no history."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log", f"-n{limit}", "--follow",
             "--pretty=format:%H%x09%h%x09%ad%x09%s", "--date=short", "--", relpath],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split("\t", 3)
        if len(parts) < 4:
            continue
        full, short, date, subject = parts
        rows.append({"full": full, "short": short, "date": date, "subject": subject})
    return rows


def _blame_needles(relpath: str, commits: list[dict]) -> set[str]:
    """Strings to match in memory tiers when surfacing the 'why' behind a file."""
    needles = {relpath, relpath.replace("\\", "/")}
    if "/" in relpath:
        needles.add(relpath.rsplit("/", 1)[-1])
    for c in commits:
        needles.add(c["short"])
        needles.add(c["full"][:12])
        needles.add(c["full"][:7])
    return {n for n in needles if n}


def _blame_memory_hits(ctx: Path, relpath: str, commits: list[dict], limit: int = 5) -> list[dict]:
    """Memory sections that mention the file or its recent commit SHAs — the reasoning, not authorship."""
    needles = _blame_needles(relpath, commits)
    hits: list[dict] = []
    for rel, kind in BLAME_MEMORY_FILES:
        path = ctx / rel
        if not path.is_file():
            continue
        for sec in parse_sections(path.read_text()):
            if sec["level"] != 2 or not sec["title"]:
                continue
            body = "\n".join(sec["lines"]).strip()
            if not body:
                continue
            matched = [n for n in needles if n in body or n in sec["title"]]
            if not matched:
                continue
            excerpt = body if len(body) <= 400 else body[:397].rstrip() + "…"
            hits.append({
                "file": rel,
                "kind": kind,
                "title": sec["title"],
                "score": len(matched),
                "matched": matched[:6],
                "excerpt": excerpt,
            })
    hits.sort(key=lambda h: (-h["score"], h["file"], h["title"]))
    return hits[:limit]


def _render_blame(relpath: str, commits: list[dict], memory: list[dict]) -> str:
    sep = "─" * 72
    lines = [f"🏹 Celeborn blame — {relpath}", sep, "Git history (recent commits on this file):"]
    if commits:
        for c in commits:
            lines.append(f"  {c['short']}  {c['date']}  {c['subject']}")
    else:
        lines.append("  (no git history — not a repo, untracked path, or no commits yet)")
    lines.append(sep)
    lines.append("Memory — the why (decisions / journal / learnings / notes that mention it):")
    if memory:
        for h in memory:
            tags = ", ".join(h["matched"][:3])
            lines.append(f"  [{h['kind']}] {h['title']}  ({h['file']})")
            if tags:
                lines.append(f"    matched: {tags}")
            for el in h["excerpt"].splitlines()[:4]:
                lines.append(f"    {el}")
            if len(h["excerpt"].splitlines()) > 4:
                lines.append("    …")
    else:
        lines.append("  (no linked memory yet — checkpoint decisions/journal entries that cite the file or commit SHA)")
    lines.append(sep)
    return "\n".join(lines)


def cmd_blame(args):
    """`celeborn blame <file>` — git blame for the *why*: recent commits on a file plus Celeborn memory
    (decisions, journal, learnings, notes) that mention the path or its SHAs."""
    ctx = require_context(args)
    repo = ctx.parent
    raw = (getattr(args, "path_arg", None) or "").strip()
    if not raw:
        die("usage: celeborn blame <file>")
    target = Path(raw)
    if target.is_absolute():
        try:
            relpath = str(target.resolve().relative_to(repo.resolve()))
        except ValueError:
            die(f"{raw} is outside the project ({repo})")
    else:
        relpath = raw.lstrip("./")
    limit = getattr(args, "limit", None) or 8
    commits = _git_file_history(repo, relpath, limit=limit)
    memory = _blame_memory_hits(ctx, relpath, commits, limit=getattr(args, "memory", None) or 5)
    if getattr(args, "json", False):
        print(json.dumps({"file": relpath, "commits": commits, "memory": memory}, indent=2))
        return
    print(_render_blame(relpath, commits, memory))


# --------------------------------------------------------------------------- why (decision archaeology)

# Reasoning tiers searched for the "why", richest first. (file, kind).
WHY_MEMORY_FILES = (
    ("decisions.md", "decision"),
    ("learnings.md", "learning"),
    ("journal.md", "journal"),
    ("notes.md", "note"),
)

# Tier richness: a locked decision answers "why" more authoritatively than a passing journal note.
WHY_KIND_WEIGHT = {"decision": 4, "learning": 3, "journal": 2, "note": 1}

_WHY_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_WHY_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _why_terms(query: str) -> list[str]:
    """Lowercased query words for overlap scoring. Keeps short tokens (e.g. 'db', 'ci')."""
    return [w.lower() for w in _WHY_WORD_RE.findall(query)]


def _why_date(title: str, body: str) -> str:
    """Best-effort decision date: the ISO date in the heading (decisions/journal lead with one),
    else the first ISO date in the body, else empty."""
    m = _WHY_DATE_RE.search(title) or _WHY_DATE_RE.search(body)
    return m.group(1) if m else ""


def _why_rationale(body: str, limit: int = 320) -> str:
    """A compact rationale excerpt — the reasoning, not the whole section. Collapses bullet/line
    noise to one flowing snippet, truncated with an ellipsis."""
    lines = [ln.strip().lstrip("-*").strip() for ln in body.splitlines()]
    text = " ".join(ln for ln in lines if ln)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _why_hits(ctx: Path, query: str, limit: int = 5) -> list[dict]:
    """Decision archaeology: rank memory sections by how well they answer 'why <topic>'. A
    self-contained section scan over the reasoning tiers (no FTS index required) — scores by
    distinct query-term overlap (title weighted), then tier richness, then recency."""
    terms = _why_terms(query)
    if not terms:
        return []
    phrase = query.strip().lower()
    hits: list[dict] = []
    for rel, kind in WHY_MEMORY_FILES:
        path = ctx / rel
        if not path.is_file():
            continue
        for sec in parse_sections(path.read_text()):
            if sec["level"] not in (2, 3) or not sec["title"] or not sec["body"]:
                continue
            hay_title = sec["title"].lower()
            hay_body = sec["body"].lower()
            matched = sorted({t for t in terms if t in hay_title or t in hay_body})
            if not matched:
                continue
            score = len(matched) + sum(1 for t in matched if t in hay_title)  # title hits count double
            if len(phrase) >= 4 and (phrase in hay_title or phrase in hay_body):
                score += 3  # exact-phrase match is a strong signal
            hits.append({
                "file": rel,
                "kind": kind,
                "title": sec["title"],
                "date": _why_date(sec["title"], sec["body"]),
                "score": score,
                "matched": matched,
                "rationale": _why_rationale(sec["body"]),
                "anchor": sec["anchor"],
            })
    # Stable multi-pass sort, least-significant first: title asc, date desc, then (score, weight) desc.
    hits.sort(key=lambda h: h["title"])
    hits.sort(key=lambda h: h["date"] or "", reverse=True)
    hits.sort(key=lambda h: (h["score"], WHY_KIND_WEIGHT.get(h["kind"], 0)), reverse=True)
    return hits[:limit]


def _why_display_title(h: dict) -> str:
    """Title with a redundant leading `YYYY-MM-DD —/-/:` stripped (the date rides the chip)."""
    title = h["title"]
    if h["date"] and title.startswith(h["date"]):
        return re.sub(r"^\d{4}-\d{2}-\d{2}\s*[—\-:]\s*", "", title) or title
    return title


def _render_why(query: str, hits: list[dict]) -> str:
    import textwrap  # lazy: only this render path needs it

    sep = "─" * 72
    lines = [f'🏹 Celeborn why — "{query}"', sep]
    if not hits:
        lines.append("  No decision or rationale found in memory for that topic.")
        lines.append("  Try `celeborn search` for broader full-text recall, or widen the topic.")
        lines.append(sep)
        return "\n".join(lines)
    top = hits[0]
    pointer = f"{top['file']}#{top['anchor']}" if top["anchor"] else top["file"]
    lines.append(f"[{top['kind']} · {top['date'] or 'undated'}] {_why_display_title(top)}")
    lines.append(f"  {pointer}")
    for el in textwrap.wrap(top["rationale"], 70) or ["(no rationale recorded)"]:
        lines.append(f"  {el}")
    rest = hits[1:]
    if rest:
        lines.append(sep)
        lines.append("See also:")
        for h in rest:
            lines.append(f"  [{h['kind']} · {h['date'] or 'undated'}] {_why_display_title(h)}  ({h['file']})")
    lines.append(sep)
    return "\n".join(lines)


def cmd_why(args):
    """`celeborn why "<topic>"` — decision archaeology: the decision, its date, and the rationale,
    pulled from the reasoning tiers (decisions, learnings, journal, notes). The 'it remembered why
    from weeks ago' one-liner."""
    ctx = require_context(args)
    query = (getattr(args, "query", None) or "").strip()
    if not query:
        die('usage: celeborn why "<topic>"')
    limit = getattr(args, "limit", None) or 5
    hits = _why_hits(ctx, query, limit=limit)
    if getattr(args, "json", False):
        print(json.dumps({"query": query, "hits": hits}, indent=2))
        return
    print(_render_why(query, hits))


# --------------------------------------------------------------------------- touch (multi-agent file registry)

TOUCHES_NAME = "touches.json"
TOUCHES_SCHEMA = "celeborn-touches/1"


def _touches_path(ctx: Path) -> Path:
    return ctx / TOUCHES_NAME


def _load_touches(ctx: Path) -> dict:
    """Active file-touch registry — who is editing which path right now (design: multi-agent-editing.md)."""
    p = _touches_path(ctx)
    if not p.is_file():
        return {"schema": TOUCHES_SCHEMA, "files": {}}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {"schema": TOUCHES_SCHEMA, "files": {}}
    data.setdefault("schema", TOUCHES_SCHEMA)
    data.setdefault("files", {})
    return data


def _save_touches(ctx: Path, data: dict):
    _touches_path(ctx).write_text(json.dumps(data, indent=2) + "\n")


# --- agent identity registry --------------------------------------------------
# A local, gitignored cache mapping an agent handle -> {family, model} so agents declare their
# model ONCE (`celeborn identify`) instead of on every touch/claim. Never authoritative: the
# resolved family/model are also embedded into each touch record at write time, so records stay
# self-describing even if the cache is wiped. Keyed by handle (the same `by` the board shows).
AGENTS_NAME = ".agents.json"
AGENTS_SCHEMA = "celeborn-agents/1"


def _agents_path(ctx: Path) -> Path:
    return ctx / AGENTS_NAME


def _load_agents(ctx: Path) -> dict:
    p = _agents_path(ctx)
    if not p.is_file():
        return {"schema": AGENTS_SCHEMA, "agents": {}}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {"schema": AGENTS_SCHEMA, "agents": {}}
    data.setdefault("schema", AGENTS_SCHEMA)
    data.setdefault("agents", {})
    return data


def _save_agents(ctx: Path, data: dict):
    _agents_path(ctx).write_text(json.dumps(data, indent=2) + "\n")


# --------------------------------------------------------------------------- alerts (CELE-t169)
# Transient per-card "coding progress is blocked, the user's input is needed" state — a permission
# prompt, a ~60s idle stall, or a stopped turn awaiting direction. Like the /clear-nudge token band
# it is LIVE state that rides the DOING card, never a durable tasks.md field: it lives in a local,
# gitignored `.context/.alerts.json` and is stamped onto the board projection (`_tasks_doc`) + the
# fleet snapshot (`_doing_row`) + the hosted `tasks` push. `celeborn alert` is the reusable service
# (the Notification/Stop hooks are its first callers; any other system can call it too), and it
# clears the moment the user re-engages (a new prompt). No focus-stealing OS dialog (rejected
# t47/t50/t62) — the alert surfaces on the card, locally and on celeborn.thot.ai.
ALERTS_NAME = ".alerts.json"
ALERTS_SCHEMA = "celeborn-alerts/1"
ALERT_KINDS = ("permission", "idle", "stopped")


def _alerts_path(ctx: Path) -> Path:
    return ctx / ALERTS_NAME


def _load_alerts(ctx: Path) -> dict:
    p = _alerts_path(ctx)
    if not p.is_file():
        return {"schema": ALERTS_SCHEMA, "alerts": {}}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {"schema": ALERTS_SCHEMA, "alerts": {}}
    data.setdefault("schema", ALERTS_SCHEMA)
    if not isinstance(data.get("alerts"), dict):
        data["alerts"] = {}
    return data


def _save_alerts(ctx: Path, data: dict):
    import os
    data["schema"] = ALERTS_SCHEMA
    p = _alerts_path(ctx)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, p)


def _alert_for(ctx: Path, task_id: str) -> dict | None:
    """The live alert on a card (bare local id), or None. Read by the projection/snapshot stamps."""
    bare = _split_qualified_tid((task_id or "").strip())[1]
    return (_load_alerts(ctx).get("alerts") or {}).get(bare) if bare else None


def _set_alert(ctx: Path, task_id: str, kind: str, message: str = "", session: str = "") -> dict:
    """Raise (or refresh) the blocked-alert on a card. Idempotent per card — one alert at a time; a
    newer signal overwrites. Returns the stored record."""
    bare = _split_qualified_tid((task_id or "").strip())[1]
    if not bare:
        return {}
    kind = kind if kind in ALERT_KINDS else "idle"
    rec = {"kind": kind, "message": (message or "").strip()[:280], "at": now_iso(),
           "session": (session or "").strip()[:12]}
    data = _load_alerts(ctx)
    data.setdefault("alerts", {})[bare] = rec
    _save_alerts(ctx, data)
    return rec


def _clear_alert(ctx: Path, task_id: str) -> bool:
    """Drop a card's alert (progress resumed / card claimed elsewhere). True if one was present."""
    bare = _split_qualified_tid((task_id or "").strip())[1]
    if not bare:
        return False
    data = _load_alerts(ctx)
    alerts = data.setdefault("alerts", {})
    if bare in alerts:
        del alerts[bare]
        _save_alerts(ctx, data)
        return True
    return False


def _live_alerts(ctx: Path) -> dict:
    """The alerts map with DEAD-session alerts filtered out (CELE-t195). A blocked-alert means "this
    session is awaiting the user" — once that session has ended (`/clear`, logout, exit → tombstoned
    in `ended_sessions`), it is awaiting nothing, so its badge must not keep blinking on the board.
    This is the deterministic catch-all for the stale badge: it holds even when the SessionEnd hook
    never fired to clear the record (the common cause — a window killed without a clean exit). An
    alert with no recorded session is kept (can't prove it's dead). Read by the board projections."""
    alerts = dict(_load_alerts(ctx).get("alerts") or {})
    if not alerts:
        return alerts
    ended = _load_metrics(ctx).get("ended_sessions") or {}
    if not ended:
        return alerts
    live = {}
    for tid, rec in alerts.items():
        sess = ((rec or {}).get("session") or "").strip()   # stored as sid[:12]
        if sess and any(k.startswith(sess) for k in ended):
            continue                                          # owning window is gone — drop the badge
        live[tid] = rec
    return live


def _clear_alert_on_activity(project_dir: str, session: str) -> None:
    """Drop this session's card alert the instant it makes a tool call (CELE-t195). PreToolUse fires
    on every Bash/Edit/Write/NotebookEdit, so a tool call is the earliest observable "work resumed"
    signal — clearing HERE (not only on the next user prompt) is what drops the board's "awaiting you"
    badge within seconds of the user removing the block, including the cases the user-prompt-submit
    clear misses entirely: a permission GRANT or an AskUserQuestion ANSWER both resume the SAME turn
    and never fire a new prompt, so the old code left the badge stale for the rest of the turn.

    Fast-guarded so the ~99% no-alert tool call pays almost nothing: no .context/ → return; no alerts
    file → one stat and return; empty alerts map → one small read and return. Best-effort; a bug here
    must never break a tool call, so everything is wrapped and swallowed."""
    if not session:
        return
    try:
        ctxdir = find_context_root(Path(project_dir))
        if ctxdir is None or not _alerts_path(ctxdir).is_file():
            return
        if not (_load_alerts(ctxdir).get("alerts") or {}):
            return                                    # resting state — nothing to clear
        tid = _session_task_id(ctxdir, session)
        if tid and _clear_alert(ctxdir, tid):
            _refresh_alerted_card(ctxdir, tid)
            __import__("celeborn_sync").schedule_agents_push(ctxdir, min_interval_s=0)
    except Exception:  # noqa: BLE001
        pass


def _refresh_alerted_card(ctx: Path, task_id: str) -> None:
    """After an alert set/clear, regenerate the derived tasks.json (so the local board's next poll
    carries the badge) and live-push the one card to the hosted board. Best-effort — an alert must
    never break a turn or a caller."""
    bare = _split_qualified_tid((task_id or "").strip())[1]
    if not bare:
        return
    try:
        _save_tasks(ctx, _load_tasks(ctx), autopush_ids=[bare])
    except Exception:  # noqa: BLE001
        pass


def _register_agent(ctx: Path, handle: str, family: str = "", model: str = "") -> dict:
    """Upsert a handle's family/model — only non-empty fields overwrite. Returns the merged entry."""
    handle = (handle or "").strip()
    if not handle:
        return {}
    data = _load_agents(ctx)
    agents = data.setdefault("agents", {})
    entry = agents.get(handle) or {}
    if (family or "").strip():
        entry["family"] = family.strip()
    if (model or "").strip():
        entry["model"] = model.strip()
    entry["at"] = now_iso()
    agents[handle] = entry
    _save_agents(ctx, data)
    return entry


def _agent_identity(args, ctx: Path) -> dict:
    """Resolve {handle, family, model} for the calling agent.

    handle: `_claim_identity` (--by -> session short-id -> $CELEBORN_AGENT).
    family/model: explicit flag -> env (CELEBORN_AGENT_FAMILY / CELEBORN_AGENT_MODEL) -> the
    registry entry for the handle. Anything supplied explicitly is upserted so later commands
    in the session inherit it without re-passing the flags."""
    import os
    handle = _claim_identity(args) or "unknown"
    flag_family = (getattr(args, "family", None) or "").strip()
    flag_model = (getattr(args, "model", None) or "").strip()
    env_family = (os.environ.get("CELEBORN_AGENT_FAMILY") or "").strip()
    env_model = (os.environ.get("CELEBORN_AGENT_MODEL") or "").strip()
    reg = (_load_agents(ctx).get("agents") or {}).get(handle) or {}
    family = flag_family or env_family or (reg.get("family") or "")
    model = flag_model or env_model or (reg.get("model") or "")
    # Persist anything from a live source (flag/env) so later commands and the board owner chip
    # (which reads the registry) inherit it; never write back values that only came from the cache.
    if (flag_family or env_family) or (flag_model or env_model):
        _register_agent(ctx, handle, flag_family or env_family, flag_model or env_model)
    return {"handle": handle, "family": family, "model": model}


def _agent_label(family: str, model: str) -> str:
    """'Claude · Opus 4.8' from parts; tolerates either being empty ('' / 'Claude' / 'Opus 4.8')."""
    return " · ".join(p for p in ((family or "").strip(), (model or "").strip()) if p)


def _touch_ttl_hours(ctx: Path) -> float:
    return float(load_config(ctx).get("touch_ttl_hours", 2))


def _parse_touch_at(at: str):
    """Parse a touch timestamp (ISO local from now_iso(), or UTC Z suffix) → datetime or None."""
    if not at:
        return None
    s = at.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return _dt.datetime.fromisoformat(s)
    except ValueError:
        return _parse_dt(s)


def _prune_touches(data: dict, ttl_hours: float) -> bool:
    """Drop touches older than ttl. Returns whether anything was removed."""
    if ttl_hours <= 0:
        return False
    cutoff = _dt.datetime.now() - _dt.timedelta(hours=ttl_hours)
    files = data.get("files") or {}
    stale = []
    for path, meta in files.items():
        at = _parse_touch_at((meta or {}).get("at", ""))
        if at is None or at < cutoff:
            stale.append(path)
    for path in stale:
        del files[path]
    return bool(stale)


def _touch_age_label(at: str) -> str:
    """Human '12m ago' / '2h ago' for orient."""
    ts = _parse_touch_at(at)
    if ts is None:
        return "?"
    # now_iso() is naive local; touches use the same — compare apples to apples.
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    delta = _dt.datetime.now() - ts
    mins = int(delta.total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 48:
        return f"{hours}h ago"
    return f"{delta.days}d ago"


def _resolve_repo_relpath(repo: Path, raw: str) -> str:
    """Normalize a user path to a repo-relative POSIX path."""
    target = Path(raw.strip())
    if target.is_absolute():
        try:
            return str(target.resolve().relative_to(repo.resolve())).replace("\\", "/")
        except ValueError:
            die(f"{raw} is outside the project ({repo})")
    return raw.strip().lstrip("./").replace("\\", "/")


def _active_touches(ctx: Path) -> list[dict]:
    """Non-stale touches, sorted by recency (newest first). Prunes stale entries on read."""
    data = _load_touches(ctx)
    if _prune_touches(data, _touch_ttl_hours(ctx)):
        _save_touches(ctx, data)
    rows = []
    for path, meta in (data.get("files") or {}).items():
        meta = meta or {}
        rows.append({
            "path": path,
            "by": meta.get("by") or "unknown",
            "family": meta.get("family") or "",
            "model": meta.get("model") or "",
            "why": meta.get("why") or "",
            "at": meta.get("at") or "",
            "task": meta.get("task") or "",
            "age": _touch_age_label(meta.get("at", "")),
        })
    rows.sort(key=lambda r: r.get("at") or "", reverse=True)
    return rows


def _task_has_active_touches(ctx: Path, task_id: str) -> bool:
    if not task_id:
        return False
    files = (_load_touches(ctx).get("files") or {})
    return any((m or {}).get("task") == task_id for m in files.values())


def _is_stale_doing(ctx: Path, t: dict) -> bool:
    """DOING card with no active file touches — work is finished or the claim was abandoned."""
    return t.get("state") == "doing" and not _task_has_active_touches(ctx, t["id"])


def _doing_for_owner(tasks: list[dict], owner: str, *, exclude: set[str] | None = None) -> list[dict]:
    who = (owner or "").strip()
    if not who:
        return []
    skip = exclude or set()
    return [t for t in tasks if t["state"] == "doing" and (t.get("owner") or "").strip() == who
            and t["id"] not in skip]


def _claim_preflight(ctx: Path, tasks: list[dict], by: str, claim_ids: list[str], *, force: bool) -> None:
    """One in-flight card per agent. Block new claims while other DOING cards are open unless --force."""
    who = (by or "").strip()
    if not who:
        return
    others = _doing_for_owner(tasks, who, exclude=set(claim_ids))
    if not others:
        return
    lines = []
    stale = []
    for t in others:
        tag = "stale" if _is_stale_doing(ctx, t) else "in flight"
        lines.append(f"  [{_display_tid(ctx, t['id'])}] {t['title']} ({tag})")
        if _is_stale_doing(ctx, t):
            stale.append(t["id"])
    msg = (f"@{who} already has {len(others)} DOING card(s) — ship or demote before claiming another:\n"
           + "\n".join(lines))
    if stale:
        msg += "\n  stale → `celeborn ship <id>` or `celeborn tasks move <id> todo`"
    if not force:
        die(msg + "\n  Pass --force to claim anyway (not recommended).")
    warn(msg + "\n  (--force — proceeding)")


def _touch_release_nudge(ctx: Path, task_id: str) -> str | None:
    """P1: task has no remaining touches but is still DOING — nudge the agent to ship the card."""
    if not task_id:
        return None
    if _task_has_active_touches(ctx, task_id):
        return None
    t = _find_task(_load_tasks(ctx), task_id)
    if not t or t["state"] != "doing":
        return None
    return (f"[{task_id}] has no active touches but is still DOING — "
            f"ship it: `celeborn ship {task_id}`")


def _release_touches_for_task(ctx: Path, task_id: str) -> list[str]:
    """Release every touch tagged with `task_id`. Used by `celeborn ship`."""
    if not task_id:
        return []
    data = _load_touches(ctx)
    files = data.get("files") or {}
    released = [p for p, m in files.items() if (m or {}).get("task") == task_id]
    for p in released:
        del files[p]
    if released:
        _save_touches(ctx, data)
    return released


def _touches_orient_summary(ctx: Path) -> str:
    """Compact active-touch view for Orient — file-level 'who is editing what'."""
    rows = _active_touches(ctx)
    if not rows:
        return ""
    n = len(rows)
    line = f"{n} active touch{'es' if n != 1 else ''}    (`celeborn touch list` · protocol: references/multi-agent-editing.md)"
    out = [line]
    for r in rows:
        label = _agent_label(r.get("family", ""), r.get("model", ""))
        who = f"@{r['by']}" + (f" ({label})" if label else "")
        task = f" [{r['task']}]" if r.get("task") else ""
        why = f" — {r['why']}" if r.get("why") else ""
        out.append(f"  {who} → {r['path']}{task}  ({r['age']}){why}")
    return "\n".join(out)


def cmd_touch(args):
    """`celeborn touch <file>` — register a file edit before other agents collide. `release` / `list` /
    `clear` manage the registry (design: references/multi-agent-editing.md)."""
    ctx = require_context(args)
    words = list(getattr(args, "words", None) or [])
    cmd = words[0] if words and words[0] in ("list", "clear", "release") else None
    path_arg = ""
    if cmd == "release":
        path_arg = words[1] if len(words) > 1 else ""
    elif not cmd and words:
        path_arg = words[0]

    if cmd == "list":
        rows = _active_touches(ctx)
        if getattr(args, "json", False):
            print(json.dumps({"touches": rows}, indent=2))
            return
        if not rows:
            print("(no active touches)")
            return
        for r in rows:
            task = f" [{r['task']}]" if r.get("task") else ""
            label = _agent_label(r.get("family", ""), r.get("model", ""))
            who = f"@{r['by']}" + (f" ({label})" if label else "")
            why = f" — {r['why']}" if r.get("why") else ""
            print(f"{who} → {r['path']}{task}  ({r['age']}){why}")
        return

    if cmd == "clear":
        data = _load_touches(ctx)
        by = (getattr(args, "by", None) or "").strip()
        files = data.get("files") or {}
        if by:
            drop = [p for p, m in files.items() if (m or {}).get("by") == by]
            for p in drop:
                del files[p]
            print(f"Cleared {len(drop)} touch(es) for @{by}")
        else:
            n = len(files)
            data["files"] = {}
            print(f"Cleared {n} touch(es)")
        _save_touches(ctx, data)
        return

    if not path_arg:
        die("usage: celeborn touch <file> [--by <agent>] [--task <id>] [--why <reason>]\n"
            "       celeborn touch release <file> [--by <agent>]\n"
            "       celeborn touch list | clear [--by <agent>]")

    relpath = _resolve_repo_relpath(ctx.parent, path_arg)
    data = _load_touches(ctx)
    files = data.setdefault("files", {})
    ident = _agent_identity(args, ctx)
    who = ident["handle"]

    if cmd == "release":
        meta = files.get(relpath)
        if not meta:
            warn(f"no touch on {relpath}")
            return
        owner = (meta.get("by") or "").strip()
        if owner and owner != who and not getattr(args, "force", False):
            die(f"{relpath} is touched by @{owner} — pass --force to release anyway")
        released_task = (meta.get("task") or "").strip()
        del files[relpath]
        _save_touches(ctx, data)
        print(f"Released {relpath} (@{owner or who})")
        nudge = _touch_release_nudge(ctx, released_task)
        if nudge:
            warn(nudge)
        return

    # register (default)
    prev = files.get(relpath) or {}
    prev_owner = (prev.get("by") or "").strip()
    if prev_owner and prev_owner != who:
        warn(f"@{prev_owner} already touching {relpath} ({_touch_age_label(prev.get('at', ''))}) — registering anyway")
    task = (getattr(args, "task", None) or "").strip()
    why = (getattr(args, "why", None) or "").strip()
    files[relpath] = {
        "by": who,
        "family": ident["family"],
        "model": ident["model"],
        "at": now_iso(),
        "task": task,
        "why": why,
    }
    _save_touches(ctx, data)
    tag = f" [{task}]" if task else ""
    label = _agent_label(ident["family"], ident["model"])
    who_str = f"@{who}" + (f" ({label})" if label else "")
    why_str = f" — {why}" if why else ""
    print(f"Touch {who_str} → {relpath}{tag}{why_str}")
    if not label:
        warn("no model on record — run `celeborn identify --family <Claude|Grok|GPT…> "
             "--model \"<e.g. Opus 4.8>\"` once so touches show who you are.")
    if not why:
        warn("tip: add `--why \"<reason>\"` so other agents see why you're in this file.")


def cmd_board(args):
    """`celeborn board` — print this project's kanban URL (the de-collided per-project port) and
    whether it's live on localhost right now. Port resolution: explicit `board_port` in .celebornrc,
    else a stable hash of the project path. `--port`/`--url` print just that value (for scripts/hooks);
    `--json` emits {port,url,live}. `--start` runs ensure-on-orient: launch the viewer (detached) if
    its port is down — the same thing the SessionStart hook does on every orient."""
    if getattr(args, "supervise", False):
        # Detached restart-loop entrypoint (not a user-facing command) — resolves everything from
        # its own args, so no project context is required.
        _run_board_supervisor(args); return
    ctx = require_context(args)
    port = board_port(ctx)
    url = board_url(ctx)
    if getattr(args, "port_only", False):
        print(port); return
    if getattr(args, "url_only", False):
        print(url); return
    if getattr(args, "start", False):
        st = ensure_board(ctx)
        if getattr(args, "json", False):
            print(json.dumps(st)); return
        verb = {"live": "already live", "started": "started", "booting": "starting up",
                "off": "autostart off", "no-tasks": "no kanban here",
                "unavailable": "can't start"}.get(st["action"], st["action"])
        extra = f" — {st['reason']}" if st.get("reason") else (f" (pid {st['pid']})" if st.get("pid") else "")
        print(f"🏹 {_project_name(ctx)} kanban → {url}  ({verb}{extra})")
        return
    live = _board_live(port)
    if getattr(args, "json", False):
        print(json.dumps({"port": port, "url": url, "live": live})); return
    print(f"🏹 {_project_name(ctx)} kanban → {url}  ({'live' if live else 'not running'})")


# --------------------------------------------------------------------------- run (real-time swarm / Elves tracker)
#
# `fleet` watches multiple *projects* at human cadence (10/30-min staleness). A swarm is the
# opposite problem: ONE run, N short-lived workers (sub-agent "elves") each living seconds-to-
# minutes. `run` tracks that — per-worker heartbeat, current item, progress, yield, and a shared
# blackboard the elves learn from. Concurrency model mirrors the outbox: ONE FILE PER WORKER
# (`run/w-<id>.json`, single writer → no lock needed); the orchestrator writes `run/meta.json`;
# the blackboard is append-only. `run status`/`watch`/the board aggregate by globbing the dir.

RUN_DIRNAME = "run"
RUN_SCHEMA = "celeborn-run/1"
_RUN_WORKING_SECONDS = 45    # last beat newer than this → "working"
_RUN_STUCK_SECONDS = 150     # last beat older than this, not finished → "stuck"


def _safe_worker_slug(wid: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", (wid or "").strip()).strip("-")
    return s or "worker"


def _run_dir(ctx: Path) -> Path:
    return ctx / RUN_DIRNAME


def _run_meta_path(ctx: Path) -> Path:
    return _run_dir(ctx) / "meta.json"


def _worker_path(ctx: Path, wid: str) -> Path:
    return _run_dir(ctx) / f"w-{_safe_worker_slug(wid)}.json"


def _blackboard_path(ctx: Path) -> Path:
    return _run_dir(ctx) / "blackboard.md"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2) + "\n"
    json.loads(text)                          # re-parse: never write something we can't read back
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _load_run_meta(ctx: Path) -> dict:
    p = _run_meta_path(ctx)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_worker(ctx: Path, wid: str) -> dict:
    p = _worker_path(ctx, wid)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _seconds_since_iso(at: str) -> int | None:
    ts = _parse_touch_at(at)
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return max(0, int((_dt.datetime.now() - ts).total_seconds()))


def _fmt_secs(secs: int | None) -> str:
    if secs is None:
        return "?"
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _worker_live_status(w: dict) -> str:
    """done | failed | working | lagging | stuck — explicit terminal state, else heartbeat freshness."""
    st = (w.get("status") or "").strip()
    if st in ("done", "failed"):
        return st
    age = _seconds_since_iso(w.get("last_beat_at", ""))
    if age is None:
        return "stuck"
    if age <= _RUN_WORKING_SECONDS:
        return "working"
    if age > _RUN_STUCK_SECONDS:
        return "stuck"
    return "lagging"


_RUN_STATUS_GLYPH = {
    "working": "●", "lagging": "◐", "stuck": "✗", "done": "✓", "failed": "✗",
}


def _all_workers(ctx: Path) -> list[dict]:
    """Every worker row, status-derived, sorted by id. Mechanical — no model."""
    d = _run_dir(ctx)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("w-*.json")):
        try:
            w = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(w, dict):
            continue
        prog = w.get("progress") or {}
        done = int(prog.get("done") or 0)
        total = int(prog.get("total") or 0)
        elapsed = w.get("elapsed_s")
        if elapsed is None:
            start = w.get("started_at") or w.get("first_beat_at")
            end = w.get("finished_at") or w.get("last_beat_at")
            if start and end:
                a, b = _parse_touch_at(start), _parse_touch_at(end)
                if a and b:
                    elapsed = max(0, int((b - a).total_seconds()))
        rate = None
        if elapsed and done:
            rate = round(done / (elapsed / 60.0), 1)
        out.append({
            "id": w.get("id") or p.stem[2:],
            "shard": w.get("shard") or "",
            "phase": w.get("phase") or "",
            "status": _worker_live_status(w),
            "current_item": w.get("current_item") or "",
            "done": done, "total": total,
            "found": int(prog.get("found") or 0),
            "missed": int(prog.get("missed") or 0),
            "elapsed_s": elapsed,
            "rate_per_min": rate,
            "beat_age_s": _seconds_since_iso(w.get("last_beat_at", "")),
            "last_error": w.get("last_error") or "",
            "sources": w.get("sources") or {},
        })
    return out


def _run_snapshot(ctx: Path) -> dict:
    """Aggregate the whole run: meta + per-worker rows + run-level rollup. The board reads this."""
    meta = _load_run_meta(ctx)
    workers = _all_workers(ctx)
    by_status: dict[str, int] = {}
    sum_done = sum_found = sum_missed = sum_elapsed = 0
    for w in workers:
        by_status[w["status"]] = by_status.get(w["status"], 0) + 1
        sum_done += w["done"]; sum_found += w["found"]; sum_missed += w["missed"]
        sum_elapsed += int(w["elapsed_s"] or 0)
    started = meta.get("started_at") or ""
    wall = _seconds_since_iso(started) if started else None
    totals = meta.get("totals") or {}
    src_roll: dict[str, dict] = {}
    for w in workers:
        for src, v in (w.get("sources") or {}).items():
            agg = src_roll.setdefault(src, {"ok": 0, "fail": 0, "ratelimited": 0})
            for k in ("ok", "fail", "ratelimited"):
                agg[k] += int((v or {}).get(k) or 0)
    finished = by_status.get("done", 0) + by_status.get("failed", 0)
    return {
        "schema": RUN_SCHEMA,
        "run_id": meta.get("run_id") or "",
        "goal": meta.get("goal") or "",
        "started_at": started,
        "updated_at": now_iso(),
        "totals": totals,
        "wall_clock_s": wall,
        "sum_worker_s": sum_elapsed,
        "parallel_efficiency": (round(sum_elapsed / wall, 1) if wall else None),
        "workers_total": len(workers),
        "workers_finished": finished,
        "by_status": by_status,
        "resolved": {"done": sum_done, "found": sum_found, "missed": sum_missed},
        "sources": src_roll,
        "workers": workers,
        "blackboard": _read_blackboard(ctx, limit=200),
    }


def _read_blackboard(ctx: Path, limit: int = 50) -> list[dict]:
    p = _blackboard_path(ctx)
    if not p.is_file():
        return []
    out = []
    try:
        lines = p.read_text().splitlines()
    except OSError:
        return []
    in_comment = False
    for ln in lines:
        ln = ln.strip()
        if in_comment:
            if "-->" in ln:
                in_comment = False
            continue
        if ln.startswith("<!--"):
            if "-->" not in ln:
                in_comment = True
            continue
        if not ln or ln.startswith("#"):
            continue
        # format: - [ts] @worker: lesson
        m = re.match(r"^-\s*\[([^\]]+)\]\s*@(\S+):\s*(.*)$", ln)
        if m:
            out.append({"at": m.group(1), "worker": m.group(2), "lesson": m.group(3)})
        else:
            out.append({"at": "", "worker": "", "lesson": ln.lstrip("- ")})
    return out[-limit:]


def _blackboard_has(ctx: Path, lesson: str) -> bool:
    norm = re.sub(r"\s+", " ", lesson.strip().lower())
    for row in _read_blackboard(ctx, limit=10_000):
        if re.sub(r"\s+", " ", (row.get("lesson") or "").strip().lower()) == norm:
            return True
    return False


def cmd_run(args):
    """`celeborn run` — real-time tracker for ONE multi-agent swarm (the Elves)."""
    ctx = require_context(args)
    action = getattr(args, "run_cmd", None) or "status"

    if action == "start":
        rid = (getattr(args, "run_id", None) or "").strip() or f"run-{now_iso()}"
        d = _run_dir(ctx)
        if d.is_dir() and not getattr(args, "keep", False):
            for p in d.glob("w-*.json"):
                p.unlink()
            bb = _blackboard_path(ctx)
            if bb.is_file():
                bb.unlink()
        meta = {
            "schema": RUN_SCHEMA, "run_id": rid,
            "goal": getattr(args, "goal", None) or "",
            "started_at": now_iso(),
            "totals": {"shards": int(getattr(args, "shards", 0) or 0),
                       "units": int(getattr(args, "units", 0) or 0)},
        }
        _atomic_write_json(_run_meta_path(ctx), meta)
        bb = _blackboard_path(ctx)
        if not bb.is_file():
            bb.write_text(
                f"# Run blackboard · {rid}\n\n"
                "<!-- Shared, append-only, broadcast knowledge channel for the swarm. Each elf reads\n"
                "     `celeborn run learnings` at shard-start and appends discoveries with\n"
                "     `celeborn run learn --worker <id> \"<lesson>\"`. Unlike the outbox, it is never\n"
                "     drained — lesson #1 helps worker #30. -->\n\n")
        ok(f"run started → {rid}  ({meta['totals']['shards']} shards / {meta['totals']['units']} units)")
        return

    if action == "beat":
        wid = (getattr(args, "worker", None) or "").strip()
        if not wid:
            die("run beat needs --worker <id>")
        w = _load_worker(ctx, wid)
        now = now_iso()
        w.setdefault("id", wid)
        w.setdefault("first_beat_at", now)
        w["last_beat_at"] = now
        if w.get("status") in (None, "", "working", "lagging", "stuck"):
            w["status"] = "working"
        for fld in ("shard", "phase"):
            v = getattr(args, fld, None)
            if v is not None:
                w[fld] = v
        if getattr(args, "item", None) is not None:
            w["current_item"] = args.item
        prog = w.setdefault("progress", {})
        for fld in ("done", "total", "found", "missed"):
            v = getattr(args, fld, None)
            if v is not None:
                prog[fld] = int(v)
        # source counters: --source-ok pubchem / --source-fail gsrs / --source-rl gsrs
        srcs = w.setdefault("sources", {})
        for arg_name, key in (("source_ok", "ok"), ("source_fail", "fail"), ("source_rl", "ratelimited")):
            name = getattr(args, arg_name, None)
            if name:
                srcs.setdefault(name, {"ok": 0, "fail": 0, "ratelimited": 0})[key] += 1
        a, b = _parse_touch_at(w["first_beat_at"]), _parse_touch_at(now)
        if a and b:
            w["elapsed_s"] = max(0, int((b - a).total_seconds()))
        _atomic_write_json(_worker_path(ctx, wid), w)
        if not getattr(args, "quiet", False):
            ok(f"beat @{wid} · {prog.get('done',0)}/{prog.get('total',0)} · {w.get('current_item','')}")
        return

    if action in ("done", "fail"):
        wid = (getattr(args, "worker", None) or "").strip()
        if not wid:
            die(f"run {action} needs --worker <id>")
        w = _load_worker(ctx, wid)
        w.setdefault("id", wid)
        now = now_iso()
        w["finished_at"] = now
        w["last_beat_at"] = now
        w["status"] = "done" if action == "done" else "failed"
        prog = w.setdefault("progress", {})
        for fld in ("done", "total", "found", "missed"):
            v = getattr(args, fld, None)
            if v is not None:
                prog[fld] = int(v)
        if action == "fail" and getattr(args, "error", None):
            w["last_error"] = args.error
        a, b = _parse_touch_at(w.get("first_beat_at") or now), _parse_touch_at(now)
        if a and b:
            w["elapsed_s"] = max(0, int((b - a).total_seconds()))
        _atomic_write_json(_worker_path(ctx, wid), w)
        ok(f"{action} @{wid} · found {prog.get('found',0)} / missed {prog.get('missed',0)}")
        return

    if action == "learn":
        wid = _safe_worker_slug(getattr(args, "worker", None) or "anon")
        lesson = (getattr(args, "lesson", None) or "").strip()
        if not lesson:
            die('run learn needs a lesson: celeborn run learn --worker w "..."')
        if _blackboard_has(ctx, lesson):
            if not getattr(args, "quiet", False):
                ok("(already on the blackboard — skipped)")
            return
        bb = _blackboard_path(ctx)
        bb.parent.mkdir(parents=True, exist_ok=True)
        with open(bb, "a") as fh:   # append is atomic for one short line on POSIX
            fh.write(f"- [{now_iso()}] @{wid}: {lesson}\n")
        if not getattr(args, "quiet", False):
            ok(f"📌 blackboard ← @{wid}: {lesson}")
        return

    if action == "learnings":
        rows = _read_blackboard(ctx, limit=int(getattr(args, "limit", None) or 30))
        if getattr(args, "json", False):
            print(json.dumps({"blackboard": rows}, indent=2)); return
        if not rows:
            print("(blackboard empty — no shared learnings yet)"); return
        print(f"🏹 Swarm blackboard — {len(rows)} lesson(s) the elves have shared:")
        for r in rows:
            who = f"@{r['worker']}" if r.get("worker") else ""
            print(f"  • {r['lesson']}  {who}")
        return

    if action in ("status", "watch"):
        if action == "watch":
            _run_watch(ctx, interval=float(getattr(args, "interval", None) or 2.0))
            return
        snap = _run_snapshot(ctx)
        if getattr(args, "json", False):
            print(json.dumps(snap, indent=2)); return
        print(_render_run(snap))
        return

    die(f"unknown run command: {action}")


def _render_run(snap: dict) -> str:
    lines = []
    rid = snap.get("run_id") or "(no run)"
    goal = snap.get("goal") or ""
    bs = snap.get("by_status") or {}
    tot = snap.get("totals") or {}
    res = snap.get("resolved") or {}
    head = (f"🏹 run {rid} — {bs.get('working',0)}● working  {bs.get('lagging',0)}◐ lagging  "
            f"{bs.get('stuck',0)}✗ stuck  {bs.get('done',0)}✓ done  {bs.get('failed',0)} failed")
    lines.append(head)
    if goal:
        lines.append(f"  goal: {goal}")
    wc = _fmt_secs(snap.get("wall_clock_s"))
    eff = snap.get("parallel_efficiency")
    lines.append(f"  {snap.get('workers_finished',0)}/{snap.get('workers_total',0)} workers finished · "
                 f"wall {wc} · sum-worker {_fmt_secs(snap.get('sum_worker_s'))}"
                 f"{f' · {eff}x parallel' if eff else ''}")
    units = tot.get("units")
    lines.append(f"  resolved: {res.get('done',0)} units processed · "
                 f"{res.get('found',0)} found / {res.get('missed',0)} missed"
                 f"{f' (of {units} total)' if units else ''}")
    srcs = snap.get("sources") or {}
    if srcs:
        parts = []
        for name, v in sorted(srcs.items()):
            rl = f" rl{v['ratelimited']}" if v.get("ratelimited") else ""
            parts.append(f"{name}:{v.get('ok',0)}✓/{v.get('fail',0)}✗{rl}")
        lines.append("  sources: " + "  ".join(parts))
    lines.append("")
    workers = snap.get("workers") or []
    # show stuck/working first, then lagging, then done
    order = {"stuck": 0, "failed": 1, "working": 2, "lagging": 3, "done": 4}
    for w in sorted(workers, key=lambda x: (order.get(x["status"], 9), x["id"])):
        g = _RUN_STATUS_GLYPH.get(w["status"], "·")
        prog = f"{w['done']}/{w['total']}" if w["total"] else f"{w['done']}"
        rate = f" {w['rate_per_min']}/min" if w.get("rate_per_min") else ""
        el = _fmt_secs(w.get("elapsed_s"))
        item = w.get("current_item") or ""
        if len(item) > 32:
            item = item[:29] + "…"
        age = w.get("beat_age_s")
        agelbl = ""
        if w["status"] in ("working", "lagging", "stuck") and age is not None:
            agelbl = f" (beat {age}s ago)"
        err = f"  ⚠ {w['last_error']}" if w.get("last_error") else ""
        lines.append(f"  {g} {w['id']:<12} {prog:>7} {el:>6}{rate:<9} {w['phase']:<10} {item}{agelbl}{err}")
    return "\n".join(lines)


def _run_watch(ctx: Path, interval: float = 2.0) -> None:
    import os
    import time
    try:
        while True:
            snap = _run_snapshot(ctx)
            os.system("clear" if os.name != "nt" else "cls")
            print(_render_run(snap))
            print(f"\n  (refresh {interval:g}s · Ctrl-C to stop)")
            bs = snap.get("by_status") or {}
            wt = snap.get("workers_total", 0)
            if wt and (bs.get("done", 0) + bs.get("failed", 0)) >= wt:
                print("\n  ✓ all workers finished.")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  (stopped watching)")


# --------------------------------------------------------------------------- fleet (live multi-project dashboard)

FLEET_REGISTRY = "fleet.json"
FLEET_REGISTRY_SCHEMA = "celeborn-fleet/1"
_ACTIVITY_CAPTURE_RE = re.compile(r"^Last capture:\s*(\S+)", re.M)
_ACTIVITY_CAPTURE_RE = re.compile(r"^Last capture:\s*(\S+)", re.M)
_ACTIVITY_PROMPT_RE = re.compile(r"^Last prompt:\s*(.+)$", re.M)
_FLEET_WORKING_MINUTES = 10   # touches newer than this → agent is "working"
_FLEET_STUCK_MINUTES = 30     # touches older than this with open DOING → "stuck"


def _config_dir() -> Path:
    import os
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "celeborn"


def _fleet_registry_path() -> Path:
    return _config_dir() / FLEET_REGISTRY


def _load_fleet_registry() -> dict:
    p = _fleet_registry_path()
    if not p.is_file():
        return {"schema": FLEET_REGISTRY_SCHEMA, "projects": []}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {"schema": FLEET_REGISTRY_SCHEMA, "projects": []}
    data.setdefault("schema", FLEET_REGISTRY_SCHEMA)
    data.setdefault("projects", [])
    return data


def _save_fleet_registry(data: dict) -> None:
    import os
    d = _config_dir()
    d.mkdir(parents=True, exist_ok=True)
    data["schema"] = FLEET_REGISTRY_SCHEMA
    data["updated_at"] = now_iso()
    p = _fleet_registry_path()
    p.write_text(json.dumps(data, indent=2) + "\n")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def _resolve_project_dir(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def _fleet_register_path(project_dir: Path) -> dict:
    """Add a project to the fleet registry. Idempotent on path."""
    project_dir = _resolve_project_dir(str(project_dir))
    ctx = project_dir / CONTEXT_DIRNAME
    if not ctx.is_dir():
        die(f"No .context/ at {project_dir} — run `celeborn init` there first.")
    key = str(project_dir)
    data = _load_fleet_registry()
    projects = data.get("projects") or []
    for row in projects:
        if _resolve_project_dir(row.get("path", "")) == project_dir:
            return row
    # Cross-fleet dedup: qualified ids (SLUG-tN) must be unambiguous across the machine. Compare this
    # project's qualifier (case-insensitively) against the already-registered ones. An explicit
    # project_slug is the user's authority — keep it, only WARN on a clash; a derived (folder-name) slug
    # is auto-suffixed and the resolved value is persisted to .celebornrc so display + markers agree.
    explicit = bool((load_config(ctx).get("project_slug") or "").strip())
    base = project_slug(ctx)
    taken = [r.get("slug", "") for r in projects]
    if explicit:
        final = base
        if base.upper() in {str(t).upper() for t in taken}:
            warn(f"project_slug {base!r} is already used by another fleet project — qualified ids "
                 f"({base.upper()}-tN) will be ambiguous. Set a distinct project_slug in {RC_NAME}.")
    else:
        final = _dedupe_slug(base, taken)
        if final != base:
            _update_config(ctx, project_slug=final)
            info(f"Fleet dedup: slug '{base}' is taken — this project's qualifier is "
                 f"'{final.upper()}-tN' (saved to {RC_NAME}).")
    row = {"path": key, "slug": final, "name": _project_name(ctx), "added": now_iso()}
    projects.append(row)
    data["projects"] = projects
    _save_fleet_registry(data)
    return row


def _fleet_autoregister(ctxdir: Path) -> None:
    """Best-effort: ensure the orienting project is in the fleet registry (CELE-t124). The board's
    savings bar and `celeborn fleet` only count REGISTERED projects, so a project the user never ran
    `celeborn fleet register` on stayed invisible in the fleet economics even while recording locally — the
    'why are there only 3 projects?' bug. Self-registering on orient closes that gap: every project that
    actually runs Celeborn shows up. Quiet (stdout swallowed) and swallow-all — a registry hiccup must
    never break rehydration. Idempotent. The global ~/.context sink is NOT a project, so skip it."""
    try:
        if ctxdir.resolve() == _global_context().resolve():
            return
        proj = ctxdir.parent.resolve()
        for row in _load_fleet_registry().get("projects") or []:
            if _resolve_project_dir(row.get("path", "")) == proj:
                return                                  # already registered — nothing to do
        with contextlib.redirect_stdout(io.StringIO()):
            _fleet_register_path(proj)
    except Exception:
        pass


def _fleet_repair(apply: bool = True) -> list[dict]:
    """One-shot re-dedup of the whole fleet registry — the repair t85 deferred. t85 only deduped at
    REGISTER time, so projects registered before it (or before t84's short slugs) kept a stale slug
    that never went through dedup; a later project's *short* qualifier could then collide undetected.

    This recomputes each project's CURRENT effective qualifier (explicit `project_slug` is authority and
    kept verbatim — only a clash is flagged; a derived slug is the short folder form), walks them in
    registration order assigning a unique qualifier (case-insensitive, numeric-suffixed on collision),
    and reconciles BOTH the registry row AND the project's `.celebornrc` so display + markers + dedup all
    agree. Returns a list of change records; `apply=False` is a dry run (no writes). Idempotent."""
    data = _load_fleet_registry()
    projects = data.get("projects") or []
    taken: set[str] = set()          # qualifiers (upper-cased) already assigned this pass
    changes: list[dict] = []
    dirty = False
    for row in projects:
        pdir = _resolve_project_dir(row.get("path", ""))
        ctx = pdir / CONTEXT_DIRNAME
        if not ctx.is_dir():
            changes.append({"path": str(pdir), "name": row.get("name", ""), "action": "skip",
                            "reason": "no .context/ (unreachable project)"})
            continue
        explicit = bool((load_config(ctx).get("project_slug") or "").strip())
        base = project_slug(ctx)
        old_slug = row.get("slug", "")
        rc_written = False
        collision = False
        if explicit:
            final = base
            collision = base.upper() in taken     # authority kept; only flagged, never suffixed
        else:
            final = _dedupe_slug(base, taken)
            if final != base and apply:
                _update_config(ctx, project_slug=final)
                rc_written = True
        if row.get("slug") != final:
            if apply:
                row["slug"] = final
            dirty = True
        taken.add(final.upper())
        if old_slug != final or collision:
            changes.append({"path": str(pdir), "name": row.get("name", ""), "old": old_slug,
                            "new": final, "explicit": explicit, "collision": collision,
                            "rc_written": rc_written})
    if apply and dirty:
        data["projects"] = projects
        _save_fleet_registry(data)
    return changes


def _fleet_unregister_path(project_dir: Path) -> bool:
    project_dir = _resolve_project_dir(str(project_dir))
    key = str(project_dir)
    data = _load_fleet_registry()
    before = len(data.get("projects") or [])
    data["projects"] = [r for r in (data.get("projects") or [])
                        if _resolve_project_dir(r.get("path", "")) != project_dir and r.get("path") != key]
    if len(data["projects"]) == before:
        return False
    _save_fleet_registry(data)
    return True


def _fleet_project_paths(ctx: Path | None) -> list[Path]:
    """Registered fleet projects, plus the orienting project if it isn't registered yet."""
    seen: set[str] = set()
    out: list[Path] = []
    for row in _load_fleet_registry().get("projects") or []:
        p = _resolve_project_dir(row.get("path", ""))
        key = str(p)
        if key in seen or not (p / CONTEXT_DIRNAME).is_dir():
            continue
        seen.add(key)
        out.append(p)
    if ctx is not None:
        cur = ctx.parent.resolve()
        key = str(cur)
        if key not in seen and (cur / CONTEXT_DIRNAME).is_dir():
            out.insert(0, cur)
    return out


def _load_session(ctx: Path) -> dict:
    p = ctx / "session.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _clip_store(text: str, limit: int) -> tuple[str, bool]:
    """Bound a stored Hot-tier string to `limit` chars. Unlike `_clip` (which points at a file for
    display), this is for what we persist: long-form detail belongs in state.md/notes.md, not a JSON
    scalar. Returns (text, was_clipped)."""
    if limit <= 0 or not isinstance(text, str) or len(text) <= limit:
        return text, False
    cut = text.rfind(" ", 0, limit)
    if cut < limit // 2:
        cut = limit
    return text[:cut].rstrip() + " …[clipped — keep long-form in state.md / notes.md]", True


def _write_session(ctx: Path, data: dict, cfg: dict | None = None) -> list[str]:
    """The single safe writer for session.json: guarantees a schema, clips the fragile free-text
    fields to `hot_focus_max_chars`, and always emits valid JSON. Returns the list of clipped field
    names (for caller messaging). This is the choke point that hand-editing kept getting wrong."""
    cfg = cfg or load_config(ctx)
    limit = int(cfg.get("hot_focus_max_chars", 1500))
    data.setdefault("schema", "celeborn/1")
    clipped = []
    for field in ("focus", "next_action"):
        if isinstance(data.get(field), str):
            new, hit = _clip_store(data[field], limit)
            data[field] = new
            if hit:
                clipped.append(field)
    (ctx / "session.json").write_text(json.dumps(data, indent=2) + "\n")
    return clipped


def cmd_checkpoint(args):
    """`celeborn checkpoint` — the safe way to update session.json. Loads the current file (repairing
    it from the template if it's missing or unparseable), applies only the flags you pass, stamps
    `updated_at`, clips over-long focus/next_action, and writes valid JSON. Run it with no flags to
    re-stamp and repair in place. This replaces hand-editing the raw JSON (the recurring corruption
    source)."""
    ctx = require_context(args)
    cfg = load_config(ctx)
    sj = ctx / "session.json"

    repaired = False
    data: dict = {}
    if sj.is_file():
        try:
            loaded = json.loads(sj.read_text())
            data = loaded if isinstance(loaded, dict) else {}
            if not isinstance(loaded, dict):
                repaired = True
        except (json.JSONDecodeError, OSError):
            repaired = True
    if not data:
        try:
            data = json.loads((TEMPLATES_DIR / "session.json").read_text())
        except (json.JSONDecodeError, OSError):
            data = {"schema": "celeborn/1", "focus": "", "next_action": "",
                    "branch": "", "status": "in-progress", "stop_allowed": True, "open_threads": []}

    if getattr(args, "focus", None) is not None:
        data["focus"] = args.focus
    if getattr(args, "next", None) is not None:
        data["next_action"] = args.next
    if getattr(args, "branch", None) is not None:
        data["branch"] = args.branch
    if getattr(args, "status", None) is not None:
        data["status"] = args.status
    if getattr(args, "stop_allowed", False):
        data["stop_allowed"] = True
    if getattr(args, "no_stop_allowed", False):
        data["stop_allowed"] = False
    data["updated_at"] = now_iso()

    clipped = _write_session(ctx, data, cfg)
    if repaired:
        warn("session.json was missing or invalid — rebuilt from a clean template.")
    if clipped:
        warn(f"clipped {', '.join(clipped)} to {cfg.get('hot_focus_max_chars', 1500)} chars — "
             "put the long-form detail in state.md / notes.md.")
    fields = [f for f in ("focus", "next_action", "branch", "status") if data.get(f)]
    ok(f"checkpoint written → .context/session.json (updated_at + {', '.join(fields) or 'no fields'})")


def _parse_activity_meta(ctx: Path) -> dict:
    """Mechanical digest fields from activity.md — last capture time + last user prompt."""
    p = ctx / "activity.md"
    if not p.is_file():
        return {}
    try:
        text = p.read_text()
    except OSError:
        return {}
    out: dict = {}
    m = _ACTIVITY_CAPTURE_RE.search(text)
    if m:
        out["last_capture"] = m.group(1).strip()
    m = _ACTIVITY_PROMPT_RE.search(text)
    if m:
        out["last_prompt"] = m.group(1).strip()
    return out


def _minutes_since_iso(at: str) -> int | None:
    ts = _parse_touch_at(at)
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return max(0, int((_dt.datetime.now() - ts).total_seconds() // 60))


def _fleet_agent_status(ctx: Path, owner: str, doing: list[dict], touches: list[dict], *, live: bool = False) -> str:
    """working | stuck | idle — per-agent liveness from touches + DOING cards.

    `live` (CELE-t172): the owning project shows real recent activity — a live transcript window, a
    fresh mechanical capture, or a session heartbeat. File touches are an optional, releasable
    protocol, so a session that released its touches (or merely let them age) while still DOING must
    not read as "stuck" if it's demonstrably alive. When `live`, a touch-derived "stuck" is
    downgraded back to "working"."""
    who = (owner or "").strip() or "_unassigned"
    mine = [t for t in touches if (t.get("by") or "").strip() == who]
    cards = [t for t in doing if ((t.get("owner") or "").strip() or "_unassigned") == who]
    if mine:
        ages = [_minutes_since_iso(t.get("at", "")) for t in mine]
        ages = [a for a in ages if a is not None]
        if ages and min(ages) <= _FLEET_WORKING_MINUTES:
            status = "working"
        elif ages and min(ages) > _FLEET_STUCK_MINUTES and cards:
            status = "stuck"
        elif cards and not _task_has_active_touches(ctx, cards[0]["id"]):
            status = "stuck"
        else:
            status = "working" if ages and min(ages) <= _FLEET_STUCK_MINUTES else "idle"
    elif cards:
        status = "stuck" if any(_is_stale_doing(ctx, t) for t in cards) else "idle"
    else:
        status = "idle"
    if status == "stuck" and live and cards:
        return "working"
    return status


def _fleet_project_snapshot(project_dir: Path) -> dict | None:
    """One project's live fleet row — mechanical, no model."""
    project_dir = _resolve_project_dir(str(project_dir))
    ctx = project_dir / CONTEXT_DIRNAME
    if not ctx.is_dir():
        return None
    tasks = _load_tasks(ctx)
    touches = _active_touches(ctx)
    session = _load_session(ctx)
    activity = _parse_activity_meta(ctx)
    doing = [t for t in tasks if t["state"] == "doing"]

    # Real session liveness (CELE-t172), shared by the stale gate below and the context band. A live
    # transcript window, a fresh mechanical capture, or a recent session heartbeat all mean "someone
    # is actively here" — regardless of whether file touches are present.
    #
    # CELE-t178: context tracking follows the SESSION, not the card — a shipped card must not clear the
    # fleet widget's context band while the owning session is still alive. So we scan transcripts when
    # the project has DOING cards OR shows fresh mechanical activity (capture/session within the stale
    # window). The transcript scan (and its token estimate) stays skipped for genuinely idle projects,
    # preserving the CELE-t170 per-poll perf win — a completed-but-live session keeps a fresh
    # activity.md capture every turn, so it still qualifies.
    last_mins = _minutes_since_iso(activity.get("last_capture", ""))
    sess_mins = _minutes_since_iso(session.get("updated_at", ""))
    _cheap_live = (
        (last_mins is not None and last_mins <= _FLEET_STUCK_MINUTES)
        or (sess_mins is not None and sess_mins <= _FLEET_STUCK_MINUTES)
    )
    agent_rows = _active_agents(ctx, AGENT_ACTIVE_WINDOW_MIN, False) if (doing or _cheap_live) else []
    tokens_by_task: dict[str, int] = {}
    session_by_task: dict[str, str] = {}
    for r in agent_rows:  # sorted fullest-window first, so the loudest session wins per card
        tid = r.get("task_id")
        if tid:
            tokens_by_task[tid] = max(tokens_by_task.get(tid, 0), int(r.get("tokens") or 0))
            sid = (r.get("session") or "")[:6]
            if sid and tid not in session_by_task:
                session_by_task[tid] = sid
    project_live = bool(agent_rows) or _cheap_live

    owners: set[str] = set()
    for t in doing:
        owners.add((t.get("owner") or "").strip() or "_unassigned")
    for t in touches:
        owners.add((t.get("by") or "").strip() or "unknown")
    agents = []
    for who in sorted(owners):
        cards = [{"id": t["id"], "title": t["title"], "stale": _is_stale_doing(ctx, t) and not project_live}
                 for t in doing
                 if ((t.get("owner") or "").strip() or "_unassigned") == who]
        touch_rows = [{"path": t["path"], "task": t.get("task") or "", "age": t.get("age") or ""}
                      for t in touches if (t.get("by") or "").strip() == who]
        if not cards and not touch_rows:
            continue
        agents.append({
            "id": who,
            "status": _fleet_agent_status(ctx, who, doing, touches, live=project_live),
            "doing": cards,
            "touches": touch_rows,
        })
    port = board_port(ctx)
    live = _board_live(port)
    if any(a.get("status") == "working" for a in agents):
        proj_status = "working"
    elif any(a.get("status") == "stuck" for a in agents):
        proj_status = "stuck"
    else:
        proj_status = "idle"
    # Enrichment for the fleet home cards (CELE-t170): the board's owner→model join, a project-
    # qualified display id, and each DOING card's sand-fill progress; plus the top TODO the project
    # should pick up next. The per-card context-window band (`k`) rides `tokens_by_task` above
    # (CELE-t172) — the same live-window value `celeborn agents --json` returns — so the fleet cards
    # carry the /clear-nudge band, matching the tasks board.
    slug = project_slug(ctx)
    reg = _load_agents(ctx).get("agents") or {}
    alerts = _live_alerts(ctx)   # CELE-t195: drop alerts from ended sessions so a dead window can't blink

    def _doing_row(t: dict) -> dict:
        owner = (t.get("owner") or "").strip()
        model = (reg.get(owner) or {}).get("model") or ""
        tokens = tokens_by_task.get(t["id"])
        return {
            "id": t["id"],
            "display_id": _display_tid(ctx, t["id"], slug=slug),
            "title": t["title"],
            # Session / human handle only — a model-derived owner falls back to the live session id
            # (when known) or is suppressed, never rendered as the owner (CELE-t172).
            "owner": _display_owner(owner, model, session_by_task.get(t["id"], "")),
            "owner_model": model,
            "progress": t.get("progress") or 0,
            # Live session (recent transcript/capture/heartbeat) is never stale — touches are an
            # optional, releasable protocol, so a lapsed/released touch alone must not read stale.
            "stale": _is_stale_doing(ctx, t) and not project_live,
            # Live context window in k-tokens for the /clear-nudge band pill; None when no live
            # transcript attributes to this card (the pill degrades gracefully) (CELE-t172).
            "k": (tokens // 1000) if tokens else None,
            # Live blocked-alert (CELE-t169): permission / idle / stopped — None when unblocked.
            "alert": alerts.get(t["id"]),
        }

    # CELE-t178: when a live session has NO doing card (it just shipped one, or hasn't claimed yet)
    # the in-flight block above won't render — so surface the session's live context band separately,
    # keyed on the session id. "session id active = tracked context": a completed card never clears
    # this; the band shows until the session claims another card (then the in-flight block takes the
    # slot) or the session goes idle / clears (agent_rows drops it). agent_rows is fullest-window
    # first, so the loudest session wins the card's row.
    active_session = None
    if agent_rows and not doing:
        top = agent_rows[0]
        toks = int(top.get("tokens") or 0)
        sid = (top.get("session") or "")[:6]
        # _active_agents sets `agent` to a real handle (@claim / CELEBORN_AGENT) or, absent one, the
        # session short id itself — so a real owner is one that isn't just that session-id fallback.
        agent_h = (top.get("agent") or "").strip()
        active_session = {
            "session": sid,
            "owner": agent_h if (agent_h and agent_h != sid) else "",
            "k": (toks // 1000) if toks else None,
        }

    todo_top = next((t for t in tasks if t["state"] == "todo"), None)
    suggested_todo = (
        {"id": todo_top["id"], "display_id": _display_tid(ctx, todo_top["id"], slug=slug),
         "title": todo_top["title"]}
        if todo_top else None
    )
    return {
        "path": str(project_dir),
        "slug": slug,
        "name": _project_name(ctx),
        "status": proj_status,
        "session": {
            "focus": (session.get("focus") or "")[:280],
            "next_action": (session.get("next_action") or "")[:280],
            "branch": session.get("branch") or "",
            "status": session.get("status") or "",
            "updated_at": session.get("updated_at") or "",
        },
        "counts": {s: sum(1 for t in tasks if t["state"] == s) for s in TASK_STATES},
        "doing": [_doing_row(t) for t in doing],
        # Live context of the owning session when there is no in-flight card to carry the band (t178).
        "active_session": active_session,
        "suggested_todo": suggested_todo,
        "agents": agents,
        "activity": {**activity, "minutes_since_capture": last_mins},
        "board": {"port": port, "url": board_url(ctx), "live": live},
    }


def _fleet_snapshot(ctx: Path | None) -> dict:
    projects = []
    for pdir in _fleet_project_paths(ctx):
        row = _fleet_project_snapshot(pdir)
        if row:
            projects.append(row)
    working = sum(1 for p in projects if p["status"] == "working")
    stuck = sum(1 for p in projects if p["status"] == "stuck")
    return {
        "generated_at": now_iso(),
        "registry": str(_fleet_registry_path()),
        "summary": {
            "projects": len(projects),
            "working": working,
            "stuck": stuck,
            "idle": len(projects) - working - stuck,
        },
        "projects": projects,
    }


def _render_fleet(snapshot: dict) -> str:
    lines = ["🏹 Celeborn fleet — live agent dashboard", ""]
    s = snapshot.get("summary") or {}
    lines.append(f"  {s.get('projects', 0)} project(s) · "
                 f"{s.get('working', 0)} working · {s.get('stuck', 0)} stuck · "
                 f"{s.get('idle', 0)} idle")
    lines.append(f"  registry: {snapshot.get('registry', '')}")
    lines.append("")
    for p in snapshot.get("projects") or []:
        icon = {"working": "🟢", "stuck": "🟡", "idle": "⚪"}.get(p["status"], "⚪")
        board = p.get("board") or {}
        bl = "live" if board.get("live") else "down"
        lines.append(f"{icon} {p['name']} ({p['slug']}) — {p['status']} · board {bl}")
        if p.get("doing"):
            for t in p["doing"]:
                owner = f" @{t['owner']}" if t.get("owner") else ""
                stale = " ⚠ stale" if t.get("stale") else ""
                # Fleet is inherently cross-project — always qualify with the project's own slug so the
                # overseer can reference a card unambiguously (t79 driver).
                # No per-project ctx in the cross-project fleet view; the explicit slug is authority.
                disp = _display_tid(None, t["id"], slug=p.get("slug") or "")
                lines.append(f"    doing → [{disp}] {t['title']}{owner}{stale}")
        for a in p.get("agents") or []:
            if a.get("status") == "idle" and not a.get("touches"):
                continue
            touch = ""
            if a.get("touches"):
                t0 = a["touches"][0]
                touch = f" · {t0['path']}"
                if t0.get("task"):
                    touch += f" [{t0['task']}]"
            lines.append(f"    @{a['id']} — {a['status']}{touch}")
        act = p.get("activity") or {}
        if act.get("last_prompt"):
            prompt = act["last_prompt"]
            if len(prompt) > 72:
                prompt = prompt[:69] + "…"
            lines.append(f"    last prompt: {prompt}")
        if board.get("url"):
            lines.append(f"    → {board['url']}")
        lines.append("")
    if not snapshot.get("projects"):
        lines.append("  (no projects — run `celeborn fleet register` from each repo)")
    return "\n".join(lines).rstrip() + "\n"


def cmd_fleet(args):
    """`celeborn fleet` — live multi-project dashboard: who's working, stuck, or idle across every
    registered Celeborn project on this machine. `register` / `unregister` manage the fleet registry
    at ~/.config/celeborn/fleet.json. The orienting project is always included when run from a repo.
    `--json` feeds the board viewer's Fleet tab; hosted sync (Pro) extends this across devices."""
    ctx = find_context_root(Path(getattr(args, "path", ".") or "."))
    action = (getattr(args, "fleet_action", None) or "").strip().lower()
    if action == "register":
        raw = getattr(args, "fleet_path", None) or getattr(args, "fleet_target", None)
        pdir = _resolve_project_dir(raw) if raw else (ctx.parent if ctx else None)
        if pdir is None:
            die("usage: celeborn fleet register [--path <project-dir>]  (or run inside a Celeborn repo)")
        row = _fleet_register_path(pdir)
        ok(f"registered {row['name']} → {row['path']}")
        return
    if action == "unregister":
        raw = getattr(args, "fleet_target", None) or getattr(args, "fleet_path", None)
        if not raw:
            die("usage: celeborn fleet unregister <project-dir>")
        if _fleet_unregister_path(raw):
            ok(f"unregistered {raw}")
        else:
            warn(f"not in fleet registry: {raw}")
        return
    if action == "repair":
        dry = getattr(args, "dry_run", False)
        changes = _fleet_repair(apply=not dry)
        if not changes:
            ok("Fleet registry already consistent — every project's qualifier is unique. Nothing to repair.")
            return
        head = "Would repair" if dry else "Repaired"
        ok(f"{head} {len(changes)} fleet slug(s):")
        for c in changes:
            if c.get("action") == "skip":
                warn(f"  skip {c['name'] or c['path']} — {c['reason']}")
            elif c.get("collision"):
                warn(f"  ⚠ {c['name']}: explicit project_slug {c['new']!r} clashes with another project — "
                     f"qualified ids ({c['new'].upper()}-tN) stay ambiguous; set a distinct project_slug in {RC_NAME}")
            else:
                rc = " (+.celebornrc)" if c.get("rc_written") else ""
                print(f"    {c['old'] or '∅'} → {c['new']}{rc}  [{c['name'] or c['path']}]")
        if dry:
            info("Dry run — re-run `celeborn fleet repair` (without --dry-run) to apply.")
        return
    snap = _fleet_snapshot(ctx)
    if getattr(args, "json", False):
        print(json.dumps(snap, indent=2))
        return
    print(_render_fleet(snap))


def cmd_metrics(args):
    ctx = require_context(args)
    m = _load_metrics(ctx)
    if args.json:
        print(json.dumps(m, indent=2))
        return
    print("Celeborn memory economy (estimated)")
    for line in metrics_summary(ctx):
        print(f"  {line}")
    print(f"  handoffs written: {m['handoffs_written']} · orient events: {m['orient_events']}"
          f" · panic-saves: {m.get('panic_saves', 0)}")
    cpt = load_config(ctx)["chars_per_token"]
    print(f"\n  Estimate basis: ~{cpt} chars/token; 'saved' = tokens(all .context/) − tokens(Hot tier) per load event")


def _iter_transcript(path: Path, start_offset: int = 0):
    """Yield (byte_offset_after_line, parsed_obj) for each JSONL line from `start_offset`.

    Reads in binary and tracks `tell()` so the offset is an exact cursor for the next run.
    Skips blank lines and JSON-decode failures — the latter covers a truncated trailing line a
    Stop hook may catch mid-flush; the offset is only advanced past lines that fully parsed."""
    try:
        f = path.open("rb")
    except OSError:
        return
    with f:
        if start_offset:
            try:
                f.seek(start_offset)
            except OSError:
                f.seek(0)
        for raw in f:
            off = f.tell()
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8", "ignore"))
            except json.JSONDecodeError:
                continue
            yield off, obj


def _estimate_transcript_tokens(path: Path, cpt: int) -> int:
    """Best-effort *current context size* from a Claude Code transcript (JSONL).

    Claude Code records each assistant turn's `usage`; the most recent one's
    input + cache + output ≈ the live window the model just saw — the real number, not a proxy.
    Falls back to a char/token estimate over message text if no usage is present."""
    latest_usage = 0
    char_total = 0
    for _off, obj in _iter_transcript(path):
        msg = obj.get("message") or {}
        usage = msg.get("usage") or obj.get("usage") or {}
        if usage:
            total = (
                (usage.get("input_tokens") or 0)
                + (usage.get("cache_read_input_tokens") or 0)
                + (usage.get("cache_creation_input_tokens") or 0)
                + (usage.get("output_tokens") or 0)
            )
            if total:
                latest_usage = total  # last wins ≈ most recent turn's window
        content = msg.get("content")
        if isinstance(content, str):
            char_total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    char_total += len(str(part.get("text", "")))
    return latest_usage or (char_total // max(1, cpt))


# --- active agents (live per-session context windows) --------------------------------------------
# `celeborn agents` answers "who is working right now, and how full is each one's context window?"
# The board renders it as the per-session /clear-nudge chips. It JOINS two real signals:
#   1. live transcripts  — every Claude session writes a JSONL transcript under ~/.claude/projects/<enc>.
#      Its mtime is the truth of "active recently"; `_estimate_transcript_tokens` reads the latest
#      `usage` for the live window — the real number, not the cumulative `tokens_session` proxy.
#   2. agent_sessions     — the session→{owner,task} link `cmd_claim` records (the claim hook passes the
#      session id). Lets us attribute each live window to a handle + the DOING card it owns.
# An active session with no link is shown by its short id (never a raw uuid) with no card.

AGENT_ACTIVE_WINDOW_MIN = 30   # a transcript touched within this many minutes = a live agent
ENDED_SESSIONS_KEEP = 50       # how many ended-session tombstones to retain (bounded FIFO)
_SESSION_ID_RE = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)


def _looks_like_session_id(s: str) -> bool:
    return bool(_SESSION_ID_RE.fullmatch((s or "").strip()))


# Tokens that occur only in model names — never in a hex session short-id (hex letters are a–f, so
# 'opus'/'sonnet'/'gemini'/… can't appear) nor in a bare family/human handle ('grok', 'scotch').
# Used to keep model strings out of the owner chip: a card is owned by its session, not its model
# (CELE-t131/t172).
_MODEL_TOKEN_RE = re.compile(r"opus|sonnet|haiku|fable|gpt[0-9]|gemini[0-9]|claude[0-9]")

# Bare model-family words that _MODEL_TOKEN_RE (which needs a digit for gpt/gemini/claude) misses —
# used only to sharpen the "record your model with `identify`" nudge when a superseded --by was a
# generic family name (e.g. `--by claude`). NOT used to change ownership: outside a Claude window a
# human may legitimately attribute `--by claude`, so this never rejects, it only advises.
_GENERIC_MODEL_FAMILIES = {"claude", "gpt", "chatgpt", "gemini", "opus", "sonnet", "haiku", "fable",
                           "llama", "mistral", "anthropic", "openai"}


def _looks_like_model_handle(owner: str, model: str = "") -> bool:
    """True when a claim handle embeds a model name rather than an identity — e.g. 'claude-opus48',
    'Claude/Opus 4.8', 'Opus 4.8'. A session short-id, a bare family, or a human handle is not.
    `model` (the handle's registered model, when known) catches custom names the token list misses."""
    norm = re.sub(r"[^a-z0-9]", "", (owner or "").lower())
    if not norm:
        return False
    m = re.sub(r"[^a-z0-9]", "", (model or "").lower())
    if len(m) >= 4 and m in norm:
        return True
    return bool(_MODEL_TOKEN_RE.search(norm))


def _display_owner(owner: str, model: str = "", session_id: str = "") -> str:
    """The owner chip shows a session / human handle, never a model (CELE-t172). When a card's
    recorded owner is model-derived, prefer the live session short-id if known, else show nothing
    rather than leak model text onto the board."""
    owner = (owner or "").strip()
    if owner and _looks_like_model_handle(owner, model):
        return (session_id or "").strip()
    return owner


def _cc_project_dir(repo: Path) -> Path:
    """Where Claude Code stores `<session>.jsonl` transcripts for `repo`: ~/.claude/projects/<enc>,
    with <enc> the repo's absolute path and every non-alphanumeric char replaced by '-' (CC's rule)."""
    enc = re.sub(r"[^A-Za-z0-9]", "-", str(repo))
    return Path.home() / ".claude" / "projects" / enc


def _record_agent_session(ctx: Path, session: str | None, owner: str, task_ids: list[str]) -> None:
    """Remember that `session` (a Claude session id) is owned by `owner` and now holds `task_ids` — the
    bridge `celeborn agents` joins against the live transcripts. No-op without a real session id or a
    handle that's just the session-id fallback. Pruned like `captures` so the map stays bounded."""
    sid = (session or "").strip()
    owner = (owner or "").strip()
    if not sid or not task_ids or not owner or _looks_like_session_id(owner):
        return
    m = _load_metrics(ctx)
    sess = m.get("agent_sessions")
    if not isinstance(sess, dict):
        sess = {}
    sess.pop(sid, None)                       # reinsert at the end so order tracks recency
    sess[sid] = {"owner": owner, "task": task_ids[-1], "at": now_iso()}
    while len(sess) > CAPTURE_KEEP_SESSIONS:
        sess.pop(next(iter(sess)))
    m["agent_sessions"] = sess
    _save_metrics(ctx, m)


def _mark_session_ended(ctx: Path, session: str | None) -> bool:
    """Tombstone a Claude session as ENDED (`/clear`, logout, exit) so it drops off the active-agents
    board immediately instead of lingering for the 30-min mtime window (CELE-t131). A `/clear` starts a
    fresh session id, so the old window's transcript keeps a recent mtime and would otherwise show as a
    ghost chip. Keyed by full session id; bounded FIFO. Returns True if a new tombstone was recorded."""
    sid = (session or "").strip()
    if not sid:
        return False
    m = _load_metrics(ctx)
    ended = m.get("ended_sessions")
    if not isinstance(ended, dict):
        ended = {}
    ended.pop(sid, None)                      # reinsert at the end so FIFO eviction tracks recency
    ended[sid] = now_iso()
    while len(ended) > ENDED_SESSIONS_KEEP:
        ended.pop(next(iter(ended)))
    m["ended_sessions"] = ended
    # A cleared session no longer owns its card on the board — drop its session→card link too.
    sess = m.get("agent_sessions")
    if isinstance(sess, dict):
        sess.pop(sid, None)
    _save_metrics(ctx, m)
    return True


def _stash_clear_carryover(ctx: Path, session: str | None) -> None:
    """A `/clear` mints a NEW session id, orphaning the `agent_sessions` link (owner+card) the cleared
    session held — so the continuation shows as an unowned chip even though it's the same agent on the
    same DOING card (CELE-t131). Called from SessionEnd(reason="clear"): stash the ending session's
    attribution so SessionStart(source="clear") can hand it to the new session. Precise to THIS cleared
    session (multi-agent-safe), one-shot, and time-boxed on consume. No-op for an unattributed session."""
    sid = (session or "").strip()
    if not sid:
        return
    m = _load_metrics(ctx)
    link = (m.get("agent_sessions") or {}).get(sid) or {}
    owner = (link.get("owner") or "").strip()
    task = (link.get("task") or "").strip()
    if not owner or not task or _looks_like_session_id(owner):
        return
    m["clear_carryover"] = {"owner": owner, "task": task, "from": sid, "at": now_iso()}
    _save_metrics(ctx, m)


def _consume_clear_carryover(ctx: Path, new_session: str | None) -> bool:
    """SessionStart(source="clear"): inherit the cleared session's owner+card (stashed by
    `_stash_clear_carryover`) so the continuation keeps showing the same agent on the same card instead
    of a fresh unowned chip (CELE-t131). One-shot (always cleared), fail-safe: ignored if stale (>15m,
    e.g. hook ordering left it unconsumed) or if the carried card is no longer in flight. Returns True
    if an attribution was inherited."""
    sid = (new_session or "").strip()
    m = _load_metrics(ctx)
    co = m.get("clear_carryover") or {}
    if "clear_carryover" in m:
        m.pop("clear_carryover", None)            # one-shot regardless of outcome
        _save_metrics(ctx, m)
    owner = (co.get("owner") or "").strip()
    task = (co.get("task") or "").strip()
    if not sid or not owner or not task or sid == co.get("from"):
        return False
    try:                                          # only inherit a RECENT clear — the continuation is seconds later
        if (_dt.datetime.now() - _dt.datetime.fromisoformat(co["at"])).total_seconds() > 900:
            return False
    except Exception:
        pass
    card = next((t for t in _load_tasks(ctx) if t["id"] == task), None)
    if card is None or card.get("state") not in ("doing",):
        return False                              # card shipped/abandoned since the clear — don't inherit
    _record_agent_session(ctx, sid, owner, [task])
    return True


def _active_agents(ctx: Path, window_min: float, show_all: bool) -> list[dict]:
    """One row per live context window for this repo (see the section header). Sorted fullest-first."""
    cfg = load_config(ctx)
    cpt = int(cfg.get("chars_per_token", 4)) or 4
    repo = ctx.parent
    slug = project_slug(ctx)
    _m = _load_metrics(ctx)
    sessions_map = _m.get("agent_sessions") or {}
    ended = set(_m.get("ended_sessions") or {})   # sessions /cleared or ended — never show as live (t131)
    by_id = {t["id"]: t for t in _load_tasks(ctx)}
    now = _dt.datetime.now().timestamp()

    rows: list[dict] = []
    proj_dir = _cc_project_dir(repo)
    if proj_dir.is_dir():
        for tp in proj_dir.glob("*.jsonl"):
            try:
                mtime = tp.stat().st_mtime
            except OSError:
                continue
            age_min = (now - mtime) / 60.0
            if not show_all and age_min > window_min:
                continue
            sid = tp.stem
            if sid in ended:
                continue                       # session was /cleared or ended — not a live window (t131)
            link = sessions_map.get(sid) or {}
            owner = (link.get("owner") or "").strip()
            if _looks_like_session_id(owner):
                owner = ""
            card = by_id.get((link.get("task") or "").strip())
            if card is not None and card.get("state") not in ("doing",):
                card = None                   # the card was shipped/moved — don't keep claiming it
            rows.append({
                # The session id IS the agent's name (CELE-t131): show its short head ("d0c13a"), not
                # "session d0c13a". A real handle (CELEBORN_AGENT / claim) still wins and renders "@handle".
                "agent": owner or sid[:6],
                "task": _display_tid(ctx, card["id"], cfg=cfg, slug=slug) if card else None,
                "task_id": card["id"] if card else None,
                "tokens": _estimate_transcript_tokens(tp, cpt),
                "session": sid[:8],
                "last_active": _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%S"),
                "age_min": round(age_min, 1),
                "owned": bool(owner),
                "project": slug,
            })
    rows.sort(key=lambda r: r["tokens"], reverse=True)
    return rows


def cmd_agents(args):
    """`celeborn agents [--json] [--window-min N] [--all]` — the live per-session context windows the
    board renders as /clear-nudge chips. Active = a Claude transcript touched within the window."""
    ctx = require_context(args)
    # `celeborn agents forget <session>` — manually wipe a ghost chip (a session that ended without a
    # clean SessionEnd hook). Accepts a full session id or the 8-char short id the board shows. Matches
    # against live transcripts so the short id resolves to the real session id (CELE-t131).
    forget = (getattr(args, "session", None) or "").strip() if getattr(args, "action", None) == "forget" else ""
    if getattr(args, "action", None) == "forget" and not forget:
        die("usage: celeborn agents forget <session-id>")
    if forget:
        proj_dir = _cc_project_dir(ctx.parent)
        full = forget
        if proj_dir.is_dir():
            hit = next((tp.stem for tp in proj_dir.glob("*.jsonl") if tp.stem == forget or tp.stem.startswith(forget)), None)
            full = hit or forget
        _mark_session_ended(ctx, full)
        try:
            __import__("celeborn_sync").schedule_agents_push(ctx, min_interval_s=0)
        except Exception:
            pass
        ok(f"Forgot session {full[:8]} — wiped from the active-agents board (local + hosted).")
        return
    window_min = float(getattr(args, "window_min", None) or AGENT_ACTIVE_WINDOW_MIN)
    show_all = bool(getattr(args, "all", False))
    rows = _active_agents(ctx, window_min, show_all)
    out = {
        "generated_at": now_iso(),
        "project": project_slug(ctx),
        "window_min": window_min,
        "count": len(rows),
        "agents": rows,
    }
    if getattr(args, "json", False):
        print(json.dumps(out, indent=2))
        return
    if not rows:
        print(f"No active agents — no transcript touched in the last {int(window_min)}m "
              f"(`celeborn agents --all` to include idle sessions).")
        return
    print(f"🏹 Active agents — {out['project']} ({len(rows)} live in the last {int(window_min)}m)")
    for r in rows:
        k = r["tokens"] // 1000
        task = f" · {r['task']}" if r["task"] else ""
        print(f"  @{r['agent']}{task} · ~{k}k ctx · {r['age_min']}m ago · {r['session']}")


# --- automatic capture (deterministic; no model) -------------------------------------------------
# Mechanically ingests the Claude Code transcript into a local-only, searchable auto tier + an
# always-fresh activity digest. Writes ONLY .context/auto/* and .context/activity.md — never the
# judgment-authored tiers (state.md/journal.md/session.json).

_FILE_TOOLS = {"Edit", "Write", "NotebookEdit"}
_SKIP_TYPES = {"file-history-snapshot", "last-prompt", "system", "ai-title", "queue-operation", "attachment"}
_GIT_COMMIT_CMD_RE = re.compile(r"\bgit\s+commit\b")
_COMMIT_OUT_RE = re.compile(r"\[[\w./-]+\s+([0-9a-f]{7,40})\]\s*(.*)")
_TEST_RE = re.compile(r"(\d+ passed|\d+ failed|\bFAILED\b|Ran \d+ tests?|failures=\d+|\d+ error)", re.I)


def _user_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text" and b.get("text"))
    return ""


def _is_tool_result_carrier(entry: dict, msg: dict) -> bool:
    if entry.get("toolUseResult") is not None:
        return True
    c = msg.get("content")
    return isinstance(c, list) and bool(c) and isinstance(c[0], dict) and c[0].get("type") == "tool_result"


def _result_text(entry: dict, msg: dict) -> str:
    tur = entry.get("toolUseResult")
    if isinstance(tur, dict):
        return (str(tur.get("stdout") or "") + "\n" + str(tur.get("stderr") or "")).strip()
    c = msg.get("content")
    if isinstance(c, list) and c and isinstance(c[0], dict):
        rc = c[0].get("content")
        if isinstance(rc, str):
            return rc
        if isinstance(rc, list):
            return "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in rc)
    return ""


def _tool_summary(name: str, inp: dict) -> str:
    """A faithful summary of a tool call's meaningful input — best-effort per known tool, compact
    JSON fallback for the rest (incl. mcp__*). Bash keeps its FULL command body (the cold record is
    faithful); the bounded digest separately keeps just the first line."""
    if not isinstance(inp, dict):
        return ""
    if name in _FILE_TOOLS:
        return str(inp.get("file_path") or inp.get("notebook_path") or "")
    if name == "Bash":
        return str(inp.get("command") or "").strip()
    if name in ("Read", "Glob"):
        return str(inp.get("file_path") or inp.get("path") or inp.get("pattern") or "")
    if name == "Grep":
        pat, path = str(inp.get("pattern") or ""), inp.get("path")
        return f"{pat} in {path}" if path else pat
    if name in ("Task", "Agent"):
        return str(inp.get("description") or inp.get("subagent_type") or "")
    if name == "WebFetch":
        return str(inp.get("url") or "")
    if name == "WebSearch":
        return str(inp.get("query") or "")
    try:
        return json.dumps(inp, sort_keys=True)[:300]
    except (TypeError, ValueError):
        return str(inp)[:300]


def _extract_turns(path: Path, start_offset: int):
    """Walk new transcript entries; return (turns, last_offset, last_uuid, first_sid).
    A turn = {ts, prompt, events[], files[], commands[], commits[], tests[]}. `events` is the
    FAITHFUL, ordered render stream (assistant text, every tool call, every tool result) that goes to
    the cold auto file; files/commands/commits/tests are the derived, bounded digest facts that feed
    activity.md. One pass yields both; pure structural extraction, no model."""
    turns, cur = [], None
    last_offset, last_uuid, first_sid = start_offset, None, None
    for off, obj in _iter_transcript(path, start_offset):
        last_offset = off
        if obj.get("uuid"):
            last_uuid = obj["uuid"]
        if first_sid is None:
            first_sid = obj.get("sessionId")
        t = obj.get("type")
        if t in _SKIP_TYPES or obj.get("isMeta") or obj.get("isSidechain"):
            continue
        msg = obj.get("message") or {}
        if t == "user":
            if _is_tool_result_carrier(obj, msg):
                if cur is None:
                    continue
                c = msg.get("content")
                head = c[0] if isinstance(c, list) and c and isinstance(c[0], dict) else {}
                out = _result_text(obj, msg)
                cur["events"].append({"kind": "tool_result", "tool_use_id": head.get("tool_use_id"),
                                      "text": out, "is_error": bool(head.get("is_error"))})
                cm = _COMMIT_OUT_RE.search(out)
                if cm:
                    cur["commits"].append(f"{cm.group(1)[:7]} {cm.group(2).strip()}".strip())
                tm = _TEST_RE.search(out)
                if tm:
                    verdict = "fail" if re.search(r"fail|error", out, re.I) else "pass"
                    cur["tests"].append(f"{tm.group(0)} ({verdict})")
                continue
            text = _user_text(msg.get("content")).strip()
            if not text:
                continue
            cur = {"ts": obj.get("timestamp", ""), "prompt": text, "events": [],
                   "files": [], "commands": [], "commits": [], "tests": []}
            turns.append(cur)
        elif t == "assistant" and cur is not None:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "text":
                    txt = (b.get("text") or "").strip()
                    if txt:
                        cur["events"].append({"kind": "assistant_text", "text": txt})
                elif btype == "tool_use":
                    name, inp = b.get("name"), (b.get("input") or {})
                    cur["events"].append({"kind": "tool_use", "name": name or "",
                                          "summary": _tool_summary(name, inp),
                                          "tool_use_id": b.get("id")})
                    if name in _FILE_TOOLS:
                        fp = inp.get("file_path")
                        if fp and fp not in cur["files"] and len(cur["files"]) < 50:
                            cur["files"].append(fp)
                    elif name == "Bash":
                        cmd = (inp.get("command") or "").strip()
                        if cmd and len(cur["commands"]) < 50:
                            cur["commands"].append(cmd.splitlines()[0][:200])
    return turns, last_offset, last_uuid, first_sid


def _redact_turn(turn: dict, patterns: list, output_max: int = 8000) -> dict:
    """Return a redacted copy of a turn — EVERY persisted field scrubbed of secrets (defense in
    depth; the auto tier is local-only but rides `celeborn sync`). Tool-result bodies are redacted
    FIRST, then size-capped, so a secret straddling the cap boundary can never survive."""
    from celeborn_sync import redact

    def red(s):
        return redact(s or "", patterns)[0]

    events = []
    for e in turn.get("events", []):
        k = e.get("kind")
        if k == "assistant_text":
            events.append({"kind": k, "text": red(e.get("text", ""))})
        elif k == "tool_use":
            events.append({"kind": k, "name": e.get("name", ""), "summary": red(e.get("summary", ""))})
        elif k == "tool_result":
            body = red(e.get("text", ""))
            if len(body) > output_max:
                body = body[:output_max] + f"\n…[truncated {len(body) - output_max} chars]"
            events.append({"kind": k, "text": body, "is_error": bool(e.get("is_error"))})
    return {
        "ts": turn["ts"],
        "prompt": red(turn["prompt"]),
        "events": events,
        "files": list(turn["files"]),
        "commands": [red(c) for c in turn["commands"]],
        "commits": list(turn["commits"]),
        "tests": list(turn["tests"]),
    }


def _digest_facts(rt: dict) -> dict:
    """The bounded, facts-only projection of a (redacted) turn for window.json + activity.md.
    Excludes the faithful `events` stream, which is large and lives ONLY in the cold auto file —
    this is what keeps the Hot tier (activity.md, loaded on Orient) small."""
    return {
        "ts": rt.get("ts", ""),
        "prompt": (rt.get("prompt") or "").replace("\n", " ").strip()[:200],
        "files": list(rt.get("files", [])),
        "commands": list(rt.get("commands", [])),
        "commits": list(rt.get("commits", [])),
        "tests": list(rt.get("tests", [])),
    }


def _format_turn_block(rt: dict) -> str:
    """Render one redacted turn as the faithful cold-tier block: the prompt, then assistant text and
    tool calls/results interleaved in transcript order. Keeps the `## turn <ts>` heading so each turn
    stays one indexed section."""
    lines = [f"## turn {rt['ts']}", "", f"**prompt:** {rt['prompt']}"]
    for e in rt.get("events", []):
        k = e.get("kind")
        if k == "assistant_text":
            lines += ["", f"**assistant:** {e['text']}"]
        elif k == "tool_use":
            lines += ["", f"- tool `{e['name']}`: {e['summary']}"]
        elif k == "tool_result" and e.get("text"):
            lines += [f"  result{' (error)' if e.get('is_error') else ''}:", "~~~", e["text"], "~~~"]
    sig = []
    if rt["commits"]:
        sig.append("commits: " + "; ".join(rt["commits"]))
    if rt["tests"]:
        sig.append("tests: " + ", ".join(rt["tests"]))
    if sig:
        lines += ["", "**signals:** " + " · ".join(sig)]
    return "\n".join(lines) + "\n"


def _write_activity_digest(ctx: Path, window: list, sid8: str, max_lines: int = 40) -> None:
    """Overwrite .context/activity.md from the rolling window of (already-redacted) turn facts.
    Bounded by construction so it can load on Orient without bloating it."""
    files: dict = {}
    commands, commits, last_prompt = [], [], ""
    for rt in window:
        last_prompt = rt.get("prompt") or last_prompt
        for f in rt.get("files", []):
            files[f] = files.get(f, 0) + 1
        commands += rt.get("commands", [])
        commits += rt.get("commits", [])
    out = ["# Automatic Context Record — current activity (mechanical)", "",
           "<!-- Regenerated by `celeborn capture` every turn. Local-only, gitignored.",
           "     Always-current 'what actually happened' — backstops a stale state.md. -->", "",
           f"Last capture: {now_iso()}  ·  session {sid8}"]
    if last_prompt:
        out.append(f"Last prompt: {last_prompt[:200]}")
    if files:
        out += ["", "## Recently touched files"]
        out += [f"- {f}" + (f" (×{n})" if n > 1 else "")
                for f, n in sorted(files.items(), key=lambda kv: -kv[1])[:12]]
    if commands:
        out += ["", "## Recent commands"] + [f"- `{c}`" for c in commands[-10:]]
    if commits:
        out += ["", "## Recent commits"] + [f"- {c}" for c in commits[-8:]]
    (ctx / "activity.md").write_text("\n".join(out[:max_lines]) + "\n")


def _prune_auto(ctx: Path, keep: int) -> None:
    autod = ctx / "auto"
    files = sorted(autod.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


CAPTURE_KEEP_SESSIONS = 200   # bound the per-session cursor map; the oldest sessions age out


def _write_capture(m: dict, caps: dict, sid: str, entry: dict,
                   keep: int = CAPTURE_KEEP_SESSIONS) -> None:
    """Persist `entry` as session `sid`'s cursor: store it under `captures[sid]` (reinserted at the
    end so insertion order tracks recency), prune the oldest sessions past `keep`, and mirror it to
    the flat `capture` slot (back-compat + the no-session-id heartbeat/statusline fallback)."""
    caps.pop(sid, None)
    caps[sid] = entry
    while len(caps) > keep:
        caps.pop(next(iter(caps)))
    m["captures"] = caps
    m["capture"] = entry


def _capture_cursor(m: dict, sid: str | None) -> dict:
    """Read a capture cursor for display (heartbeat/statusline): the entry for `sid` from the
    per-session map, else the most-recently-active session (the flat `capture` mirror)."""
    caps = m.get("captures")
    if isinstance(caps, dict) and sid and sid in caps:
        return caps[sid]
    return m.get("capture") or {}


def _capture_note(delta: int, session_total: int, idle_streak: int) -> str:
    """The Stop hook's per-turn line, as a Claude Code `systemMessage` JSON object.

    Every note is deliberately UNIQUE turn-to-turn. Claude Code suppresses a Stop-hook
    `systemMessage` that is identical to the one it just showed, so a constant string (the old
    "Nothing material happened this turn.") rendered once and then silently vanished — which is
    exactly why the per-turn note seemed never to fire. An active turn varies by the running
    session total (which only grows); an idle turn varies by a consecutive-idle counter. So the
    heartbeat stays visible on every single turn."""
    if delta > 0:
        msg = f"🏹 Celeborn —> +{delta:,} tokens this turn · {session_total:,} this session"
    else:
        msg = f"🏹 Celeborn —> idle ×{idle_streak} · {session_total:,} this session"
    return json.dumps({"systemMessage": msg})


def _count_auto_allowed(turns: list, allow_names: set) -> int:
    """Estimate the permission prompts a CMM pre-clear avoided in these turns: one per agent call to
    a pre-cleared tool (`allow_names`). Each such `tool_use` is a structural query that ran without an
    Allow/Always-allow click — what it replaced (a Bash `grep`/`rg`/`find` shell-out) would have
    prompted. Zero when nothing's pre-cleared, so non-engaged projects never accrue."""
    if not allow_names:
        return 0
    n = 0
    for t in turns:
        for e in t.get("events", []):
            if e.get("kind") == "tool_use" and e.get("name") in allow_names:
                n += 1
    return n


def _bash_allow_matches(command: str, inner: str) -> bool:
    """Does Bash `command` fall under the allow-rule body `inner` (the text inside `Bash(...)`)?
    Mirrors Claude Code's prefix semantics: `grep:*` (or the advisor's `grep *`) auto-allows any
    command starting with `grep`; a bare `ls` auto-allows only the exact command `ls`."""
    inner = inner.strip()
    if inner.endswith(":*"):
        prefix = inner[:-2]
    elif inner.endswith("*"):
        prefix = inner[:-1].rstrip()
    else:
        return command == inner
    return bool(prefix) and (command == prefix or command.startswith(prefix))


def _effective_allow_rules(ctx: Path) -> list:
    """The permission allow-rules in force for this project: the global baseline
    (`~/.claude/settings.json`, where `wire --global` writes the t100 baseline) plus the project's own
    shared + local settings. Deduped, order-stable; best-effort — a missing or malformed file just
    contributes nothing."""
    out: list = []
    seen: set = set()
    for p in (Path.home() / ".claude" / "settings.json",
              ctx.parent / ".claude" / "settings.json",
              ctx.parent / ".claude" / "settings.local.json"):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for r in (data.get("permissions") or {}).get("allow") or []:
            if isinstance(r, str) and r not in seen:
                seen.add(r)
                out.append(r)
    return out


def _count_allowlist_auto_allowed(turns: list, allow: list, exclude_names: set) -> int:
    """Estimate the permission prompts the settings.json allow-LIST avoided in these turns — the t100
    safe baseline plus any rule the user added. Each matching `tool_use` ran without an Allow click: a
    built-in present verbatim in `allow` (Read/Glob/Grep/…), or a Bash call whose command matches a
    `Bash(<prefix>:*)` rule. `exclude_names` (CMM's provenance-credited tools) are skipped so the
    figure never double-counts what the CMM bucket already claims."""
    named = {r for r in allow if isinstance(r, str) and "(" not in r}
    bash_rules = [inner for r in allow if isinstance(r, str)
                  for inner in (_parse_bash_rule(r),) if inner is not None]
    if not named and not bash_rules:
        return 0
    n = 0
    for t in turns:
        for e in t.get("events", []):
            if e.get("kind") != "tool_use":
                continue
            name = e.get("name") or ""
            if name in exclude_names:
                continue                                 # CMM already credited this call
            if name == "Bash":
                cmd = (e.get("summary") or "").strip()   # _tool_summary keeps Bash's full command
                if cmd and any(_bash_allow_matches(cmd, inner) for inner in bash_rules):
                    n += 1
            elif name in named:
                n += 1
    return n


def cmd_capture(args):
    ctx = find_or_create_context(args)
    cfg = load_config(ctx)
    patterns = cfg.get("secret_patterns", [])
    path = Path(args.transcript)
    if not path.is_file():
        die(f"transcript not found: {path}")
    m = _load_metrics(ctx)
    caps = m.get("captures")
    if not isinstance(caps, dict):
        caps = {}
    legacy = m.get("capture") or {}
    if not caps and legacy.get("session_id"):       # migrate the old single-slot layout
        caps[legacy["session_id"]] = dict(legacy)
    run_sid = args.session or None

    # Per-session cursor: each Claude session advances its OWN offset/total, so alternating sessions
    # sharing this metrics.json (esp. the global ~/.context sink) can't invalidate each other and
    # force a full re-read every turn. With no session id, fall back to the most-recent session.
    sid_key = run_sid or legacy.get("session_id")
    cur = dict(caps.get(sid_key) or {})

    # Decide where to start reading + which session file to write.
    new_session = bool(run_sid) and run_sid not in caps
    start_offset = 0 if new_session else int(cur.get("offset") or 0)
    if start_offset > path.stat().st_size:   # file shrank (compaction rewrote it) → reset
        start_offset, new_session = 0, True

    turns, last_offset, last_uuid, first_sid = _extract_turns(path, start_offset)
    sid = run_sid or first_sid or cur.get("session_id") or "session"
    sid8 = sid[:8]

    # Per-turn heartbeat state — reset when a new session begins.
    sess_tokens = 0 if new_session else int(cur.get("tokens_session") or 0)
    idle_streak = 0 if new_session else int(cur.get("idle_streak") or 0)

    if not turns:
        # Nothing to record. Advance the cursor past any consumed (meta/snapshot) lines so we don't
        # re-scan them, and bump the idle counter so the heartbeat note stays unique; never create
        # files. On a new session, drop the stale file pointer.
        idle_streak += 1
        _write_capture(m, caps, sid, {"session_id": sid, "offset": last_offset,
                       "last_uuid": last_uuid or cur.get("last_uuid"),
                       "file": (None if new_session else cur.get("file")),
                       "tokens_session": sess_tokens, "idle_streak": idle_streak,
                       "last_delta": 0})
        _save_metrics(ctx, m)
        if not getattr(args, "quiet", False):
            print("capture: no new entries")
        if getattr(args, "note", False):
            print(_capture_note(0, sess_tokens, idle_streak))
        return

    autod = ctx / "auto"
    autod.mkdir(parents=True, exist_ok=True)
    sess_file = cur.get("file") if not new_session and cur.get("file") else f"auto/{now_iso()[:10]}-{sid8}.md"
    sp = ctx / sess_file
    if not sp.is_file():
        sp.write_text(f"# Automatic Context Record — session {sid8}\n\n"
                      "<!-- Mechanical capture by `celeborn capture`. Local-only, gitignored. "
                      "Do not edit by hand. -->\n")

    omax = int(cfg.get("capture_output_max_chars", 8000))
    redacted = [_redact_turn(t, patterns, omax) for t in turns]
    recorded_chars = 0
    for rt in redacted:
        block = _format_turn_block(rt)
        _append(sp, "\n" + block)
        recorded_chars += len(block)

    # rolling window holds only the bounded digest FACTS (not the faithful `events`), so activity.md
    # stays small; the complete record lives in the cold auto file above.
    win_path = autod / "window.json"
    window = []
    if win_path.is_file():
        try:
            window = json.loads(win_path.read_text())
        except (OSError, json.JSONDecodeError):
            window = []
    if new_session:
        window = []
    window = (window + [_digest_facts(rt) for rt in redacted])[-int(cfg.get("activity_window_turns", 15)):]
    win_path.write_text(json.dumps(window, indent=2) + "\n")
    _write_activity_digest(ctx, window, sid8, int(cfg.get("activity_max_lines", 40)))

    _prune_auto(ctx, int(cfg.get("auto_keep_files", 30)))

    cpt = int(cfg.get("chars_per_token", 4)) or 4
    delta = (recorded_chars + cpt - 1) // cpt   # mirrors _est_tokens
    sess_tokens += delta

    # CMM economics (CELE-t92): credit the permission prompts the pre-clear avoided this capture —
    # the agent's calls to CMM-pre-cleared tools that ran without an Allow click. Provenance-gated in
    # celeborn_cmm so only engaged projects accrue; best-effort (never block capture on it).
    try:
        allow_names = __import__("celeborn_cmm").credited_tool_names(ctx)
    except Exception:
        allow_names = set()
    auto_allowed = _count_auto_allowed(turns, allow_names)
    if auto_allowed:
        cmm_m = dict(m.get("cmm") or {})
        cmm_m["prompts_auto_allowed"] = int(cmm_m.get("prompts_auto_allowed", 0) or 0) + auto_allowed
        m["cmm"] = cmm_m

    # Permission allow-list economics (t100): also credit the prompts the settings.json allow-list
    # avoided this capture — the safe baseline `wire --global` ships plus the user's own rules. Counts
    # Bash commands under a `Bash(<prefix>:*)` rule and built-ins (Read/Glob/Grep/…) that ran without
    # an Allow click; excludes CMM's credited tools so the buckets stay disjoint. Best-effort.
    try:
        perm_allowed = _count_allowlist_auto_allowed(turns, _effective_allow_rules(ctx), allow_names)
    except Exception:
        perm_allowed = 0
    if perm_allowed:
        perm_m = dict(m.get("permissions") or {})
        perm_m["prompts_auto_allowed"] = int(perm_m.get("prompts_auto_allowed", 0) or 0) + perm_allowed
        m["permissions"] = perm_m

    _write_capture(m, caps, sid, {"session_id": sid, "offset": last_offset,
                   "last_uuid": last_uuid or cur.get("last_uuid"), "file": sess_file,
                   "tokens_session": sess_tokens, "idle_streak": 0, "last_delta": delta})
    _save_metrics(ctx, m)

    if not getattr(args, "quiet", False):
        nf = sum(len(t["files"]) for t in turns)
        nc = sum(len(t["commands"]) for t in turns)
        nk = sum(len(t["commits"]) for t in turns)
        print(f"captured: {len(turns)} turn(s), {nf} file(s), {nc} command(s), {nk} commit(s) -> {sess_file}")
    if getattr(args, "note", False):
        print(_capture_note(delta, sess_tokens, 0))
    try:
        __import__("celeborn_jira").flush_auto_push(ctx, quiet=True)
    except Exception:
        pass
    # Keep the hosted active-agents token chips tracking this live window between card mutations
    # (CELE-t131) — throttled + detached, a no-op when hosted sync isn't configured / signed in.
    try:
        __import__("celeborn_sync").schedule_agents_push(ctx)
    except Exception:
        pass  # hosted liveness is best-effort — never break capture


def cmd_heartbeat(args):
    """Print the per-turn capture heartbeat to PLAIN stdout — for the UserPromptSubmit hook.

    Why a second channel: the Stop hook's `systemMessage` is shown inline in a terminal but is NOT
    surfaced by the Claude desktop/web app (it lands there as a hidden `hook_system_message`
    transcript attachment). UserPromptSubmit-hook stdout, by contrast, is reliably user-visible on
    BOTH surfaces — it's the same channel the context reminder rides. So this is how app users
    actually see the heartbeat. UserPromptSubmit fires at the START of a turn, so it reports what was
    banked as of the PREVIOUS turn's capture (read from the metrics cursor; no transcript needed)."""
    ctx = find_context_root(Path(getattr(args, "path", ".") or "."))
    if ctx is None:
        return                                   # outside a .context/ repo — stay silent
    cap = _capture_cursor(_load_metrics(ctx), getattr(args, "session", None))
    if not cap.get("session_id"):
        return                                   # nothing captured yet this machine — stay silent
    total = int(cap.get("tokens_session") or 0)
    delta = int(cap.get("last_delta") or 0)
    line = f"🏹 Celeborn —> {total:,} tokens recorded this session"
    if delta > 0:
        line += f" · +{delta:,} last turn"
    print(line)


def cmd_statusline(args):
    """Render Celeborn's status line (the Claude Code `statusLine` command).

    A statusLine is painted persistently in the host's UI chrome and — unlike a hook `systemMessage`,
    which some surfaces (the Claude app) deliver to the model but never show the user — it can't be
    suppressed. So it's the deterministic way to keep the per-turn capture visible. Compact:
    banked-this-session from the capture cursor, plus the live context size when a transcript is
    passed. Always prints one line (statusLine output replaces the default status line)."""
    ctx = find_context_root(Path(getattr(args, "path", ".") or "."))
    if ctx is None and _global_context().exists():
        ctx = _global_context()
    parts = []
    if ctx is not None:
        cap = _capture_cursor(_load_metrics(ctx), getattr(args, "session", None))
        recorded = int(cap.get("tokens_session") or 0)
        if recorded:
            parts.append(f"{recorded:,} tokens recorded")
    tp = getattr(args, "transcript", None)
    if tp and Path(tp).is_file():
        cpt = int((load_config(ctx) if ctx else {}).get("chars_per_token", 4)) or 4
        live = _estimate_transcript_tokens(Path(tp), cpt)
        if live:
            parts.append(f"ctx ~{live:,}")
    print(f"🏹 Celeborn —> {' · '.join(parts)}".rstrip())


# --------------------------------------------------------------------------- hook dispatch (collapse)
#
# Phase 1 of the executable-app plan (references/executable-app.md §3, §9.1): one in-process entry
# point for every Claude Code hook event. `celeborn hook <event>` reads the host's JSON payload from
# stdin and runs the per-turn work HERE — no bash control flow, no inline `python3 -c` JSON parsing,
# no `$CELEBORN_HOME` resolver. dispatch_hook() is the importable "one logic" module §3 calls for:
# today the thin client runs it cold against disk; the daemon (phase 2) will run the same dispatch
# against warm state. It never raises — a hook must never break the user's turn.

# event token (CLI arg) -> the Claude Code hook event name it serves.
HOOK_EVENTS = {
    "session-start": "SessionStart",
    "user-prompt-submit": "UserPromptSubmit",
    "stop": "Stop",
    "pre-compact": "PreCompact",
    "session-end": "SessionEnd",
    "statusline": "statusLine",
    # Quality gates (t70 Phase 2) — installed only by `celeborn wire-quality`, never by `wire`.
    "post-edit": "PostToolUse",      # cheap per-edit check (py_compile / `tsc --noEmit`)
    "quality-stop": "Stop",          # full suite once per turn when test-relevant files changed
    # Safety guard (t101) — installed by `wire`. Blocks an un-approvable `cd … > rel/file` compound.
    "pre-tool-use": "PreToolUse",    # steer shell redirection → the Write/Edit tool
}

# The checkpoint reminder the PreCompact hook prints (was hooks/pre-compact.sh's heredoc).
PRECOMPACT_MSG = (
    "[celeborn] Compaction imminent. CHECKPOINT now before context is summarized:\n"
    "  1. Rewrite .context/state.md in place (Now / Next action / Open threads).\n"
    "  2. Append one entry to the bottom of .context/journal.md (what + evidence + next).\n"
    "  3. Run: celeborn checkpoint --focus \"...\" --next \"...\" --status \"...\"  "
    "(safely updates session.json — valid JSON, auto-clips; never hand-edit the raw file).\n"
    "Anything not written to .context/ will be lost on compaction."
)


def _read_hook_payload(raw: str | None = None) -> dict:
    """Parse the Claude Code hook JSON from stdin into a dict. Returns {} on anything unexpected
    (no stdin, empty, non-JSON, non-object) — the hooks must degrade to a no-op, never crash."""
    if raw is None:
        try:
            raw = sys.stdin.read()
        except Exception:
            return {}
    if not raw or not raw.strip():
        return {}
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


def _hook_run(fn, **ns) -> str:
    """Run an existing cmd_* function with a synthesized Namespace and return its stdout as a string.

    This is what makes the collapse a *reuse*, not a rewrite: every hook drives the same command
    implementations the CLI exposes. die()/SystemExit and any unexpected error are swallowed (the bash
    hooks `|| true`'d everything) so one bad turn degrades to silence, never a broken turn."""
    buf = io.StringIO()
    args = argparse.Namespace(**ns)
    with contextlib.redirect_stdout(buf):
        try:
            fn(args)
        except (Exception, SystemExit):
            pass
    return buf.getvalue()


def _compose_user_prompt_envelope(heartbeat: str, nudge: str, handoff: str = "", claim: str = "",
                                  directive: str = "", progress_nudge: str = "", arch_notice: str = "") -> str:
    """Build the UserPromptSubmit JSON envelope (was the python3 tail of hooks/context-watch.sh).

    `additionalContext` is delivered to the MODEL only — never painted for the user on any surface
    (cc#50542). The heartbeat rides as quiet do-not-surface context; the nudge is phrased as an
    imperative so the model relays it into its reply, where every surface renders it. A handoff is a
    card the user sent from the board to be worked on NOW — injected as an actual work instruction. A
    directive is the card-less-work gate (t131) — a top-priority instruction, so it leads the envelope.
    Returns "" when there is nothing to inject (so the hook emits no output at all)."""
    parts = []
    if directive:
        parts.append(
            "[Celeborn card gate — TOP PRIORITY, resolve this before doing the work this turn:]\n"
            + directive)
    if handoff:
        parts.append(
            "[Celeborn task hand-off — the user sent the following card from the board to be worked on "
            "NOW. Treat it as the user's prompt for this turn and act on it (alongside anything they "
            "also typed):]\n" + handoff)
    if claim:
        parts.append(
            "[Celeborn card claim — you now own the following board card(s); they have been moved to "
            "DOING under your name. Every other agent sharing this .context/ will see this on their next "
            "orient. Identify yourself in your reply (who claimed this). Before future autonomous "
            "claims, read the board and avoid cards that would interrupt in-flight work. Work these "
            "cards as the user's request for this turn:]\n" + claim)
    if heartbeat:
        parts.append("[Celeborn heartbeat — context only, do NOT surface this to the user]\n" + heartbeat)
    if progress_nudge:
        parts.append(
            "[Celeborn progress nudge — SURFACE THIS TO THE USER. Your doing card's bar hasn't moved; "
            "relay the line below verbatim (on its own line) and run the copy-pasteable command it "
            "names if the milestone is genuinely done:]\n" + progress_nudge)
    if arch_notice:
        parts.append(
            "[Celeborn architecture trace — SURFACE THIS TO THE USER. The auto-trace (CELE-t201) found a "
            "new piece in the stack and remapped the hosted Stack; relay the line(s) below verbatim, on "
            "their own line(s):]\n" + arch_notice)
    if nudge:
        parts.append(
            "[Celeborn context-health notice — FRESHEN THE HOT TIER FIRST, THEN SURFACE THIS TO THE USER] "
            "A checkpoint is due: Celeborn is about to tell the user it's safe to /clear \"without need to "
            "rehydrate\". That promise only holds if the Hot tier is fresh RIGHT NOW — so before you relay "
            "the line below, checkpoint the authored Hot tier so a /clear or compaction loses nothing:\n"
            "  1. Rewrite .context/state.md in place (Now / Next action / Open threads) to reflect this turn.\n"
            "  2. Run: celeborn checkpoint --focus \"...\" --next \"...\" --status \"...\"  "
            "(safely updates session.json — valid JSON, auto-clips; never hand-edit the raw file).\n"
            "  3. If meaningful work landed since the last entry, append one line to .context/journal.md.\n"
            "Skip a step only if it is already current. THEN, because the user is likely on a surface (e.g. "
            "the Claude desktop app) where hook output is invisible to them and only your reply is shown, "
            "begin your reply with the following line(s) verbatim, on their own lines, before anything else, "
            "so the notice reaches them — then answer normally:\n"
            + nudge)
    if not parts:
        return ""
    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": "\n\n".join(parts),
    }})


# --------------------------------------------------------------------------- PreToolUse safety guard
#
# (t101) Two levers steer the model away from a `cd … && … > relative/file` compound. Such a command
# can't be statically approved: once the cwd changes, a relative redirect target resolves against the
# *new* dir, so the permission matcher can't bound where the write lands (the path-resolution bypass
# the gate exists to stop) — Claude Code falls back to a manual prompt on every run.
#   • Lever 1 (soft, cheap): SHELL_HYGIENE_RULE rides the SessionStart orient channel — a standing
#     directive to prefer the Write/Edit tool and tool-native dir flags, so the pattern rarely forms.
#   • Lever 2 (hard, deterministic): the `pre-tool-use` hook turns the recurring prompt into a *deny*
#     with a corrective message. The model keeps an explicit escape hatch — a trailing
#     `# celeborn:allow-redirect[: why]` comment makes the guard AUTO-ALLOW the command with no prompt
#     (the operator accepted the path-resolution risk for marked writes), for the rare case a shell
#     redirect is genuinely the only way.
SHELL_HYGIENE_RULE = (
    "🏹 Celeborn shell rule —> Prefer the Write/Edit tool over shell output redirection (`>`/`>>`), and "
    "reach a directory with a tool's own flag (`git -C`, `npm --prefix`, `make -C`) or an absolute path "
    "rather than `cd … && …`. A `cd` plus a relative-path redirect can't be auto-approved — Celeborn's "
    "PreToolUse guard will block it (override with a trailing `# celeborn:allow-redirect: <why>`)."
)
REDIRECT_BYPASS_MARKER = "celeborn:allow-redirect"
# `cd <arg>` beginning a command segment (start of string, or after a newline / ; / & / | separator —
# `&&` and `||` end in the single-char class, so they match too). Requires an argument, so a bare
# `cd` (go-home) doesn't trip it.
_CD_SEGMENT_RE = re.compile(r"(?:^|[\n;&|])\s*cd\s+\S")
# A `>`/`>>` redirect to a FILE. Negative lookbehind drops fd-numbered/chained forms (`2>`, `>>`'s
# inner `>`); negative lookahead drops fd duplication (`>&2`). The target is captured for the
# absolute-vs-relative test below.
_REDIRECT_TARGET_RE = re.compile(r"(?<![\d&>])>>?(?!&)\s*(\"[^\"]+\"|'[^']+'|[^\s|;&<>]+)")


def _has_relative_write_redirect(cmd: str) -> bool:
    """True if `cmd` writes to a RELATIVE-path file via `>`/`>>` — the only redirect a `cd` can move.
    Absolute targets and `/dev/null` resolve the same regardless of cwd, and fd dups (`2>&1`, `>&2`)
    aren't file writes — none are the bypass risk, so none count."""
    for m in _REDIRECT_TARGET_RE.finditer(cmd):
        tgt = m.group(1).strip("\"'")
        if not tgt or tgt.startswith("&"):
            continue                                   # fd duplication, not a file write
        if tgt == "/dev/null" or tgt.startswith("/"):
            continue                                   # absolute / null target — cwd-independent
        return True
    return False


def _is_cd_redirect_pattern(cmd: str) -> bool:
    """The gated shape: a `cd` into a new dir AND a relative-path write redirect in the same command —
    exactly what forces a manual approval on every run. Independent of the bypass marker (the decision
    function inspects that separately to choose deny vs auto-allow)."""
    return bool(cmd) and bool(_CD_SEGMENT_RE.search(cmd)) and _has_relative_write_redirect(cmd)


_CD_REDIRECT_DENY = (
    "🏹 Celeborn blocked `cd … > file`: a directory change plus a relative-path redirect can't be "
    "statically approved — the write target resolves against the cd'd dir (the path-resolution bypass the "
    "permission gate guards), so it would prompt on every run. Use one of, in order:\n"
    "  • the Write/Edit tool to create or modify the file (no shell redirect at all — preferred)\n"
    "  • a tool's own directory flag instead of `cd`: `git -C <abs>`, `npm --prefix <abs>`, `make -C <abs>`\n"
    "  • an ABSOLUTE redirect target (`> /abs/path/out`) so the destination is unambiguous\n"
    "If a shell redirect is genuinely the only option, re-run with a trailing "
    "`# celeborn:allow-redirect: <why>` comment — Celeborn will auto-allow it with no prompt (the operator "
    "opted into the path-resolution risk for marked writes)."
)
_CD_REDIRECT_ALLOW = (
    "🏹 Celeborn auto-allowed `cd … > file` on an explicit `# celeborn:allow-redirect` marker — the "
    "operator accepted the path-resolution risk for this write. The default posture denies the pattern "
    "and steers to the Write/Edit tool; this command opted out by name."
)


def _pre_tool_use_decision(payload: dict) -> str:
    """PreToolUse guard (lever 2). For the gated `cd … > relative/file` Bash compound: DENY with a
    corrective message by default, or AUTO-ALLOW (no prompt) when the command carries an explicit
    `# celeborn:allow-redirect` marker — the operator's accepted-risk escape hatch. Emits nothing for
    anything else, so every other tool call flows through untouched. Harness-independent on purpose —
    a universal shell-hygiene rule, not a `.context/` concern."""
    if (payload.get("tool_name") or "") != "Bash":
        return ""
    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not _is_cd_redirect_pattern(cmd):
        return ""                                      # not the gated shape — flow through untouched
    if REDIRECT_BYPASS_MARKER in cmd:
        decision, reason = "allow", _CD_REDIRECT_ALLOW
    else:
        decision, reason = "deny", _CD_REDIRECT_DENY
    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason,
    }})


# --------------------------------------------------------------------------- PreToolUse publish guard (t191)
#
# Layer B of the product federation (CELE-t188 §6). A publish/release action — twine/flit/poetry/hatch/
# maturin publish, npm/pnpm/yarn/bun publish, cargo publish, `gh release create`, or a tag push — that
# targets a facet whose role is `server:private` or any `oss:*` is a policy violation: server:private
# never publishes (full rights reserved), and oss:* is stewarded code we contribute back via fork→PR, never
# publish as ours. Only `client:public` publishes (still honoring the CELE-t168 BUSL gate elsewhere).
# Same rail + hard-DENY vocabulary as the redirect guard above, and the same accepted-risk escape hatch:
# a trailing `# celeborn:allow-publish: <why>` marker auto-ALLOWS the command (mirrors allow-redirect).
# Silent for single-repo projects (no product.md) and for any command that isn't a publish action — the
# cheap regex runs first, so only publish-shaped Bash commands ever pay the registry lookup.
PUBLISH_BYPASS_MARKER = "celeborn:allow-publish"
_PUBLISH_ACTION_RE = re.compile(
    r"\btwine\s+upload\b"
    r"|\bpython\d?\s+-m\s+twine\b"
    r"|\b(?:flit|poetry|hatch|maturin)\s+publish\b"
    r"|\b(?:npm|pnpm|yarn|bun)\s+publish\b"
    r"|\bcargo\s+publish\b"
    r"|\bgh\s+release\s+create\b"
    r"|\bgit\s+push\b[^\n|;&]*--(?:tags|follow-tags)\b",
    re.I,
)


def _is_publish_action(cmd: str) -> bool:
    """True if `cmd` is a package-registry publish or a release/tag push — the actions the publish guard
    (and cmd_push's in-command check) enforce role policy on. A plain branch `git push` is NOT one."""
    return bool(cmd) and bool(_PUBLISH_ACTION_RE.search(cmd))


def _role_forbids_publish(role: str) -> bool:
    """Publish policy from the role vocabulary (t188 §3): server:private never publishes; every oss:*
    contributes via fork→PR, never publish-as-ours. Only client:public may publish."""
    role = (role or "").strip()
    return role == "server:private" or role.startswith("oss:")


def _publish_policy_reason(key: str, role: str, action: str = "this publish/release action") -> str:
    """The hard-DENY / refusal message for a publish targeting a forbidden facet — shared by the
    PreToolUse guard and cmd_push's in-command check so the wording is identical wherever it fires."""
    if (role or "").startswith("oss:"):
        why = f"role {role} — stewarded OSS; contribute via fork → PR, never publish as ours"
    else:
        why = f"role {role} — private, full rights reserved; it never publishes"
    return (f"🏹 Celeborn publish guard: {action} targeting facet '{key}' is refused ({why}). If this is "
            f"genuinely intended, the operator can override the raw command with a trailing "
            f"`# {PUBLISH_BYPASS_MARKER}: <why>` comment (accepted-risk, exactly like `# celeborn:allow-redirect`).")


def _publish_guard_decision(payload: dict, project_dir: str) -> str:
    """PreToolUse publish guard (t191). Hard-DENY a Bash publish/release action that targets a
    server:private/oss:* facet — resolved from the product registry either by a bound checkout path
    appearing in the command or by the project the command runs in. AUTO-ALLOW on an explicit
    `# celeborn:allow-publish` marker. Emits nothing for anything else (no product.md, not a publish
    action, or targeting a client:public facet), so every other Bash call flows through untouched."""
    if (payload.get("tool_name") or "") != "Bash":
        return ""
    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not _is_publish_action(cmd):
        return ""                                      # not a publish action — flow through (cheap fast path)
    ctxdir = find_context_root(Path(project_dir))
    if ctxdir is None:
        return ""                                      # not a Celeborn project — never guard
    targets = _publish_guard_targets(ctxdir, cmd, project_dir)
    if not targets:
        return ""                                      # no forbidden facet in scope (e.g. client:public) — allow
    key, role = targets[0]
    if PUBLISH_BYPASS_MARKER in cmd:
        return json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": (
                f"🏹 Celeborn auto-allowed a publish action on facet '{key}' ({role}) via an explicit "
                f"`# {PUBLISH_BYPASS_MARKER}` marker — the operator accepted the policy risk. The default "
                f"posture hard-DENYs publishing a server:private/oss:* facet."),
        }})
    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": _publish_policy_reason(key, role),
    }})


# --------------------------------------------------------------------------- card-less-work gate (t131)
#
# Celeborn is the live source of truth for who's doing what — work done off-board is invisible to the
# other agents sharing the repo. So a session that owns no board card (and was handed none) gets
# steered onto one before it acts, with the same two-lever shape as the redirect guard above:
#   • Lever 1 (soft): a top-priority UserPromptSubmit directive — "no task: ask the user or claim the
#     obvious open card" — injected each turn the session is card-less.
#   • Lever 2 (hard, deterministic): the PreToolUse hook soft-DENIES Edit/Write/NotebookEdit until a
#     card is linked, with a corrective message. The `celeborn` CLI (over Bash) and read-only tools are
#     never gated. Two exemptions keep it from ever being a dead-end: the board being EMPTY (no open
#     card to claim — nothing to gate on, "add one") and the operator's accepted-risk escape hatch
#     CELEBORN_ALLOW_NO_CARD=1 (analogous to the `# celeborn:allow-redirect` marker).
CARDLESS_BYPASS_ENV = "CELEBORN_ALLOW_NO_CARD"
# Tools the PreToolUse gate hard-denies for a card-less session (CELE-t134 hardens CELE-t131): not just
# file edits but the tools substantive *research* and delegation run through — web access and subagent
# spawn. Deliberately EXCLUDES Bash and Read so a gated session can always orient and claim (the
# `celeborn` CLI is Bash; reading the board is Read). `Task`/`Agent` cover the subagent tool across
# harnesses (Claude Code names it `Task`; this harness names it `Agent`).
_CARD_GATED_TOOLS = ("Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Task", "Agent")


def _cardless_bypass() -> bool:
    """The operator's accepted-risk escape hatch for the card-less-work gate (t131) — set
    CELEBORN_ALLOW_NO_CARD=1 in the launch env to let a session edit files without owning a board card
    (both the directive and the Edit/Write deny fall silent). Mirrors the `# celeborn:allow-redirect`
    marker: a deliberate, named opt-out, not a default."""
    import os
    return (os.environ.get(CARDLESS_BYPASS_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def _session_owns_live_card(ctx: Path, sid: str) -> bool:
    """True if this session is attributable to a live (doing) card — by the recorded
    session→card link (`_session_has_task`) or, as a fallback, because a live card's owner is this
    session's short id (the t131 part-B identity). The fallback means a `claim --by <sid[:6]>` that
    omitted `--session` still clears the gate."""
    if _session_has_task(ctx, sid):
        return True
    short = (sid or "").strip()[:6]
    if not short:
        return False
    return any((t.get("owner") or "").strip() == short
               for t in _load_tasks(ctx)
               if (t.get("state") or "") in ("doing",))


def _card_gate_status(ctx: Path, sid: str) -> str:
    """Classify a session for the card-less-work gate (t131):
      'ok'    — owns a live card, or the bypass env is armed → never gate.
      'empty' — the board has no open card to claim → exempt ('add one'); never hard-block.
      'gated' — open cards exist but this session owns none → nudge + soft-deny edits.
    """
    if _cardless_bypass() or _session_owns_live_card(ctx, sid):
        return "ok"
    has_open = any((t.get("state") or "") in ("todo", "doing") for t in _load_tasks(ctx))
    return "gated" if has_open else "empty"


def _cardless_claim_hint(sid: str) -> str:
    """`celeborn claim` invocation that records THIS session as the card's owner — so claiming it
    clears the gate in the same turn (`--session` writes the session→card link `_session_has_task`
    reads). The full session id is embedded because the agent has no other way to know it."""
    return "celeborn claim <id>" + (f" --session {sid}" if sid else "")


def _cardless_directive(sid: str) -> str:
    """Top-priority UserPromptSubmit directive (t131 lever 1): injected when a session owns no card and
    named none but the board has open cards. Steers the session onto a card before it acts."""
    return (
        "⛔ NO TASK CLAIMED — this session owns no Celeborn board card and you (the user) named none. "
        "Celeborn is the live source of truth for who's doing what; work done off-board is invisible to "
        "the other agents sharing this repo. A card is MANDATORY before ANY work this turn — answering a "
        "question, research, design, or edits all count equally. Do NOT rationalize that you are 'just "
        "answering' or that the work 'touches no files': if the turn produces work, it needs a card "
        "first. Web access (WebFetch/WebSearch), subagents (Task/Agent), and file edits "
        "(Edit/Write/NotebookEdit) are hard-blocked until a card is linked. Before doing the work, do "
        "ONE of:\n"
        "  • Ask the user which card this belongs to (best when it's ambiguous), or\n"
        f"  • Claim the obvious open card yourself: `{_cardless_claim_hint(sid)}`, or\n"
        "  • If no open card fits, add one and claim it: `celeborn tasks add \"<title>\"`.\n"
        "Read the board first (`celeborn tasks`) and don't grab a card another agent is mid-flight on. "
        "The `celeborn` CLI and board-reading (Bash/Read) are never blocked, so you can always orient "
        "and claim. Only a launch with CELEBORN_ALLOW_NO_CARD=1, or an empty board (nothing to claim), "
        "lifts this gate."
    )


def _cardless_deny(sid: str) -> str:
    """Corrective message for the PreToolUse hard-deny (CELE-t131 lever 2, widened in CELE-t134). Fires
    on edits AND research/subagent tools, so the wording is tool-agnostic."""
    return (
        "🏹 Celeborn blocked this action: a card is MANDATORY and this session owns none. Celeborn is the "
        "source of truth for who's doing what, so untracked work — edits, research, or subagents — is "
        "invisible to the other agents sharing this repo. Link a card first:\n"
        f"  • Claim the card you're working, then re-try: `{_cardless_claim_hint(sid)}`.\n"
        "  • Unsure which? Read the board (`celeborn tasks`) and ask the user, or "
        "`celeborn tasks add \"<title>\"` then claim it.\n"
        "If you must work without a card, the operator can launch with CELEBORN_ALLOW_NO_CARD=1 to lift "
        "this gate (accepted-risk, like `# celeborn:allow-redirect`). The `celeborn` CLI and board-reading "
        "(Bash/Read) are never blocked, so you can always orient and claim."
    )


def _card_gate_pre_tool_use(payload: dict, project_dir: str) -> str:
    """PreToolUse card-less-work gate (CELE-t131 lever 2, widened in CELE-t134). Hard-DENY any tool in
    `_CARD_GATED_TOOLS` (edits + web research + subagent spawn) when this session owns no live board card
    and the board has an open one to claim. No-op outside a Celeborn project, when the bypass env is
    armed, or when the board is empty (nothing to claim). Unlike the redirect guard this needs the
    `.context/` — but only after the cheap tool-name filter, so ungated calls still return in
    microseconds."""
    if (payload.get("tool_name") or "") not in _CARD_GATED_TOOLS:
        return ""
    ctxdir = find_context_root(Path(project_dir))
    if ctxdir is None:
        return ""                                      # not a Celeborn project — never gate
    sid = payload.get("session_id") or ""
    if _card_gate_status(ctxdir, sid) != "gated":
        return ""
    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": _cardless_deny(sid),
    }})


def dispatch_hook(event: str, payload: dict, project_dir: str) -> str:
    """Run one hook event in-process and return the text to write to stdout ("" = emit nothing).

    The single per-turn logic module. `project_dir` is where to start the `.context/` search (the
    host's CLAUDE_PROJECT_DIR / cwd). Resolution mirrors the old bash hooks exactly, including the
    hybrid sink (capture/statusline fall through to the global ~/.context when outside a repo) and the
    no-op-outside-.context safety property that makes the hooks safe to enable globally."""
    payload = payload or {}
    # PreToolUse fires on EVERY tool call. The Bash redirect guard (t101) runs first — cheap and
    # `.context/`-free, so the overwhelmingly common case returns in microseconds. If it has no opinion,
    # fall through to the card-less-work gate (CELE-t131, widened in CELE-t134) on the gated tool set
    # (edits + web research + subagents), which does need the `.context/` (gated behind its own tool-name
    # filter so ungated calls — Bash/Read included — never pay for the lookup).
    if event == "pre-tool-use":
        redirect = _pre_tool_use_decision(payload)
        if redirect:
            return redirect
        # Publish guard (CELE-t191): after the redirect guard has no opinion, refuse a publish/release
        # action targeting a server:private/oss:* facet (§6). Its own cheap publish-action regex gates the
        # registry lookup, so a non-publish Bash call returns in microseconds, same as the redirect guard.
        publish = _publish_guard_decision(payload, project_dir)
        if publish:
            return publish
        # Work resumed (CELE-t195): a tool call means this session is actively working again, so any
        # "awaiting you" alert on its card (permission / idle / stopped) is stale. Clear it here — the
        # earliest resume signal — so the badge drops within seconds even when the user unblocks
        # WITHOUT a new prompt (permission grant, AskUserQuestion answer). Fast-guarded; best-effort.
        _clear_alert_on_activity(project_dir, payload.get("session_id") or "")
        return _card_gate_pre_tool_use(payload, project_dir)
    sid = payload.get("session_id") or ""
    tp = payload.get("transcript_path") or ""
    ctxdir = find_context_root(Path(project_dir))      # the .context/ dir, or None
    proj = str(ctxdir.parent) if ctxdir is not None else str(Path(project_dir))

    if event == "session-start":
        if ctxdir is None:
            return ""                                  # not a Celeborn project — do nothing
        # A `/clear` mints a new session id; inherit the cleared session's card attribution so the
        # active-agents chip keeps showing the same agent on the same DOING card (CELE-t131) instead of
        # a fresh unowned chip. Gated to source="clear"; best-effort + fail-safe.
        if (payload.get("source") or "") == "clear":
            try:
                if _consume_clear_carryover(ctxdir, sid):
                    __import__("celeborn_sync").schedule_agents_push(ctxdir, min_interval_s=0)
            except Exception:
                pass
        _hook_run(cmd_record, path=proj, event="orient", session=sid, tokens=None)
        # Self-register into the fleet (CELE-t124): an active project must show up in the fleet
        # economics even if it was never `fleet add`ed by hand. Best-effort, quiet, idempotent.
        _fleet_autoregister(ctxdir)
        # Ensure-on-orient: bring the kanban viewer up on its resolved port if it's down. Detached and
        # best-effort — swallow anything so a launch hiccup never breaks rehydration.
        try:
            ensure_board(ctxdir)
        except Exception:
            pass
        # Keep the Matt Pocock skills current: a detached, throttled (weekly) background refresh when due.
        # Claude-only + best-effort — never blocks or breaks orient (t116).
        try:
            _ensure_skills_fresh(ctxdir)
        except Exception:
            pass
        # Cheap install-integrity check: if the installed core modules were edited in place, lead the
        # Orient load with a one-line self-diagnosing notice. Best-effort (never raises); silent on a
        # clean or source/dev install.
        # Lead the Orient load with at most two one-line self-diagnosing notices: the install-integrity
        # check, then the skill advisor (t70) — friction/quality recommendations the agent can act on,
        # throttled to one per session. Both are best-effort and silent when there's nothing to say.
        notices = [_integrity_notice(), _advisor_notice(ctxdir, sid), _product_banner(ctxdir), SHELL_HYGIENE_RULE]
        head = "\n\n".join(n for n in notices if n)
        head = (head + "\n\n") if head else ""
        return head + "## Celeborn memory (Orient load)\n\n" + _hook_run(cmd_status, path=proj, full=False)

    if event == "pre-compact":
        if ctxdir is None:
            return ""
        _hook_run(cmd_record, path=proj, event="compaction", session=None, tokens=None)
        # Pre-compaction panic-save (t36): snapshot the authored tiers to a restore point NOW —
        # deterministically, before the window is summarized — so survival is a felt, recoverable
        # artifact and not just a nag. Best-effort: a hiccup here must never break compaction.
        saved_line = ""
        try:
            info = _do_panic_save(ctxdir, reason="compaction", session=sid)
            m = _load_metrics(ctxdir)
            m["panic_saves"] = int(m.get("panic_saves", 0) or 0) + 1
            saved_line = _panic_save_line(info)
            # The snapshot + stdout line below happen every time. We do NOT raise a native OS dialog
            # here: focus-stealing modal alert windows were repeatedly flagged as annoying (t47/t50/t62)
            # and have been removed — the reassurance rides the returned stdout line instead.
            _save_metrics(ctxdir, m)
        except Exception:
            pass
        return (saved_line + "\n\n" + PRECOMPACT_MSG) if saved_line else PRECOMPACT_MSG

    if event == "session-end":
        if ctxdir is None:
            return ""
        # A session ending — `/clear` (reason="clear"), logout, or exit — should drop its active-agents
        # chip from the board NOW, not linger for the 30-min mtime window (CELE-t131). `/clear` opens a
        # fresh session id, so the old transcript keeps a recent mtime and would otherwise ghost. We
        # tombstone the ending session and force a hosted refresh so celeborn.thot.ai prunes it too.
        try:
            # On a `/clear`, stash this session's card attribution FIRST (before the tombstone drops the
            # link) so its continuation can inherit it (CELE-t131). Gate to "clear" or a missing reason
            # (robust if the host omits the field) — but NOT an explicit non-clear end (logout/exit), so
            # a closed terminal can't bleed its card onto someone else's later /clear.
            if (payload.get("reason") or "") in ("", "clear"):
                _stash_clear_carryover(ctxdir, sid)
            # A stopped/idle/permission alert dies with its session (CELE-t195): the window is gone,
            # so it awaits nothing. Clear the record now so .alerts.json doesn't accumulate the stale
            # badges that _live_alerts otherwise has to filter (belt-and-suspenders with that guard).
            _tid = _session_task_id(ctxdir, sid)
            if _tid:
                _clear_alert(ctxdir, _tid)
            if _mark_session_ended(ctxdir, sid):
                __import__("celeborn_sync").schedule_agents_push(ctxdir, min_interval_s=0)
        except Exception:
            pass
        return _hook_run(cmd_handoff, path=proj)

    if event == "stop":
        # Hybrid sink: a repo's own .context/ when inside one, else the global ~/.context — so no
        # session goes unrecorded. Needs a transcript to read; without one there's nothing to capture.
        if not tp:
            return ""
        # Idle-Stop alert (CELE-t169): the turn ended and the session's DOING card is unfinished, so
        # coding progress has paused awaiting the user's next direction. Raise a low-severity "stopped"
        # alert on the card — it clears the instant the user replies (user-prompt-submit below). A card
        # that was just shipped is no longer doing, so `_session_task_id` returns "" and no alert fires.
        if ctxdir is not None:
            try:
                tid = _session_task_id(ctxdir, sid)
                if tid:
                    _set_alert(ctxdir, tid, "stopped", "Turn ended — awaiting your direction.", sid)
                    _refresh_alerted_card(ctxdir, tid)
                    __import__("celeborn_sync").schedule_agents_push(ctxdir, min_interval_s=0)
            except Exception:  # noqa: BLE001
                pass
        return _hook_run(cmd_capture, path=proj, transcript=tp, session=sid,
                         quiet=True, note=True, global_=(ctxdir is None))

    if event == "notification":
        # Claude Code fires Notification when it needs tool-use permission or the prompt has been idle
        # ~60s — exactly "agentic progress is blocked, the user's input is needed" (CELE-t169). Raise the
        # matching alert on the session's DOING card so it surfaces on the board (locally + hosted). The
        # message classifies the kind; it clears when the user next replies. Best-effort — never break.
        if ctxdir is None:
            return ""
        try:
            tid = _session_task_id(ctxdir, sid)
            if tid:
                msg = (payload.get("message") or "").strip()
                low = msg.lower()
                kind = "permission" if ("permission" in low or "approve" in low or "waiting for your" in low) else "idle"
                _set_alert(ctxdir, tid, kind, msg or ("Needs permission to proceed." if kind == "permission"
                                                      else "Waiting for your input."), sid)
                _refresh_alerted_card(ctxdir, tid)
                __import__("celeborn_sync").schedule_agents_push(ctxdir, min_interval_s=0)
        except Exception:  # noqa: BLE001
            pass
        return ""

    if event == "statusline":
        # cmd_statusline handles its own global fallback; pass the start dir and a transcript if any.
        return _hook_run(cmd_statusline, path=proj, transcript=(tp or None), session=sid)

    if event == "user-prompt-submit":
        if ctxdir is None or not tp:
            return ""                                  # need a transcript to read the live ctx size
        # Resume clears the block (CELE-t169): the user has replied, so any permission/idle/stopped
        # alert on this session's DOING card is stale — drop it and refresh the card. Best-effort.
        try:
            _tid = _session_task_id(ctxdir, sid)
            if _tid and _clear_alert(ctxdir, _tid):
                _refresh_alerted_card(ctxdir, _tid)
                __import__("celeborn_sync").schedule_agents_push(ctxdir, min_interval_s=0)
        except Exception:  # noqa: BLE001
            pass
        # Per-turn board re-ensure (CELE-t99 safety net): the user taking a turn is a good proxy for
        # "the board tab is open", so revive a downed board here too — covers the gap if even the
        # supervisor was killed. Cheap (~150ms probe; relaunch is detached) and strictly best-effort:
        # never delay or break a turn.
        try:
            ensure_board(ctxdir)
        except Exception:                              # noqa: BLE001
            pass
        nudge = _hook_run(cmd_remind, path=proj, transcript=tp, tokens=None, every=50_000,
                          last=None, auto=False, force=False, soft_limit=150_000,
                          clear_cmd="/clear").strip()
        heartbeat = _hook_run(cmd_heartbeat, path=proj, session=sid).strip()
        handoff = _hook_run(cmd_outbox, path=proj, outbox_cmd="drain").strip()
        # Claim-on-receipt: if the user pasted a card (its marker rides in the prompt text), this
        # session claims it — owner ← me, TODO → DOING. The act of receiving the card is the
        # assignment; the human chose *which* model by choosing *which* window to paste into.
        slug = project_slug(ctxdir) if ctxdir is not None else ""
        prompt_text = payload.get("prompt") or ""
        refs, rejects = _find_card_refs(prompt_text, expected_slug=slug or None)
        claim = _hook_run(cmd_claim, path=proj, ids=refs, by=None, session=sid).strip() if refs else ""
        # Prose claim-on-mention (CELE-t131): no pasted marker, but the human named a project-qualified
        # card in prose ("work on CELE-t131"). An explicit opening mention is a strong, intentional
        # signal — treat it like a paste and CLAIM the card: owner ← this session's short id (the
        # session IS the agent's name), TODO → DOING. The session-id owner holds no other cards, so the
        # one-in-flight preflight never blocks it. Vacuum-fill — only when this session has no live task
        # yet — so a later casual mention of another card can't thrash the board off your current work.
        if not refs and not rejects and not _session_has_task(ctxdir, sid):
            open_ids = {t["id"] for t in _load_tasks(ctxdir)
                        if (t.get("state") or "") in ("todo", "doing")}
            prose = _find_prose_card_refs(prompt_text, expected_slug=slug or None, claimable_ids=open_ids)
            if prose:
                prose_claim = _hook_run(cmd_claim, path=proj, ids=prose[:1], by=None, session=sid).strip()
                if prose_claim:
                    claim = f"{claim}\n\n{prose_claim}".strip() if claim else prose_claim
                try:
                    __import__("celeborn_sync").schedule_agents_push(ctxdir, min_interval_s=0)
                except Exception:
                    pass
        if rejects:
            reject_blk = "[Celeborn card markers — project mismatch, not claimed:]\n" + "\n".join(rejects)
            claim = f"{claim}\n\n{reject_blk}".strip() if claim else reject_blk
        # Card-less-work gate (t131 lever 1): with no card claimed this turn and none already owned, but
        # open cards on the board, lead the envelope with a top-priority directive to claim one. Computed
        # AFTER the claim block so a paste/prose claim this turn (which now owns a card) suppresses it; a
        # hand-off is the user explicitly sending work, so it's exempt too.
        directive = (_cardless_directive(sid)
                     if not handoff and _card_gate_status(ctxdir, sid) == "gated" else "")
        # Progress engine (CELE-t161): tick THIS session's doing card off observable signals and, if the
        # bar is lagging, return a copy-pasteable nudge. Best-effort — never delay or break a turn.
        try:
            progress_nudge = _progress_hook(ctxdir, sid)
        except Exception:  # noqa: BLE001
            progress_nudge = ""
        # Auto-architecture-trace (CELE-t201): every few turns (and draining any manifest-edit trace note),
        # re-detect the stack and remap the hosted Stack when a new piece appears. Best-effort — never breaks.
        try:
            arch_notice = _maybe_arch_trace_on_turn(ctxdir)
        except Exception:  # noqa: BLE001
            arch_notice = ""
        return _compose_user_prompt_envelope(heartbeat, nudge, handoff, claim, directive, progress_nudge, arch_notice)

    if event == "post-edit":
        # Quality gate (t70 Phase 2), PostToolUse after Edit/Write. CHEAP check only: byte-compile an
        # edited .py for a syntax error / type-check the board — and, for a test-relevant edit, mark the
        # turn dirty so the full suite runs once on quality-stop. Surfaces failures, never blocks.
        if ctxdir is None:
            return ""
        ti = payload.get("tool_input") or {}
        fp = ti.get("file_path") or ti.get("path") or ""
        if not fp:
            return ""
        proj_root = Path(proj)
        try:
            rel = str(Path(fp).resolve().relative_to(proj_root.resolve()))
        except (ValueError, OSError):
            rel = fp
        # Auto-architecture-trace (CELE-t201): a dependency-manifest edit means a piece may have entered
        # the stack — trace NOW (bypassing the cadence) and remap. The note is stashed for the next
        # user-prompt-submit to surface (PostToolUse additionalContext reaches the model only). Best-effort.
        try:
            _maybe_arch_trace_on_edit(ctxdir, rel)
        except Exception:  # noqa: BLE001
            pass
        qcfg = _quality_config(ctxdir)
        gate = _quality_gate_for(rel, qcfg)
        if gate is None:
            return ""
        notice = None
        if gate == "test":
            m = _load_metrics(ctxdir)
            m["quality"] = dict(m.get("quality") or {}, dirty_session=(sid or "session"))
            _save_metrics(ctxdir, m)
            notice = _run_quality_cmd([sys.executable, "-m", "py_compile", str(proj_root / rel)], cwd=proj)
        elif gate == "type":
            notice = _run_quality_cmd(qcfg["type_cmd"], cwd=str(proj_root / qcfg["type_dir"]))
        if not notice:
            return ""
        return json.dumps({"hookSpecificOutput": {
            "hookEventName": "PostToolUse", "additionalContext": notice}})

    if event == "quality-stop":
        # The heavy half of the gate: run the full suite ONCE per turn, but only when a scripts/** or
        # tests/** edit marked the session dirty. A separate Stop group from `capture` so it never
        # collides with capture's systemMessage envelope. Surfaces failures, never blocks.
        if ctxdir is None:
            return ""
        suite = _maybe_run_suite_on_stop(ctxdir, sid)
        nudge = _review_nudge_on_stop(ctxdir, sid)   # Phase 3: review/security nudge when tree is dirty
        return "\n\n".join(p for p in (suite, nudge) if p)

    return ""


def _quality_config(ctx: Path) -> dict:
    """Quality-gate settings (t70 Phase 2). Defaults match this project (a unittest suite under tests/
    + a Next.js board type-checked with `tsc --noEmit`); override via a `quality: {...}` block in
    .celebornrc for a different layout. NEVER `next build` — it clobbers the live dev `.next`."""
    out = {
        "test_cmd": ["python3", "-m", "unittest", "discover", "-s", "tests"],
        "test_globs": ["scripts/", "tests/"],
        "test_suffixes": [".py"],
        "type_cmd": ["npx", "tsc", "--noEmit"],
        "type_dir": "board",
        "type_globs": ["board/"],
        "type_suffixes": [".tsx", ".ts"],
    }
    try:
        block = load_config(ctx).get("quality")
    except Exception:
        block = None
    if isinstance(block, dict):
        for k, v in block.items():
            if v is not None and k in out:
                out[k] = v
    return out


def _quality_gate_for(rel: str, qcfg: dict) -> str | None:
    """Which gate applies to an edited repo-relative path: 'test', 'type', or None."""
    rel = rel.replace("\\", "/").lstrip("./")
    if rel.endswith(tuple(qcfg["test_suffixes"])) and any(rel.startswith(g) for g in qcfg["test_globs"]):
        return "test"
    if rel.endswith(tuple(qcfg["type_suffixes"])) and any(rel.startswith(g) for g in qcfg["type_globs"]):
        return "type"
    return None


def _run_quality_cmd(cmd: list, cwd: str | None = None, timeout: int = 180) -> str | None:
    """Run a quality-gate command. Returns None when it PASSES (exit 0) — or when the tool is simply
    absent (e.g. no `npx`/`tsc`), so an unconfigured machine stays silent rather than nagging. Returns
    a one-block failure notice (the tail of its output) when it actually fails. Never raises."""
    import subprocess
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None                                  # tool not installed — skip quietly
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode == 0:
        return None
    out = ((r.stdout or "") + ("\n" if r.stdout and r.stderr else "") + (r.stderr or "")).strip()
    tail = "\n".join(out.splitlines()[-20:]) if out else "(no output)"
    return (f"🏹 Celeborn quality gate FAILED — `{' '.join(cmd)}`:\n{tail}\n"
            f"(surfaced, not blocking — fix before you call this change done.)")


def _maybe_run_suite_on_stop(ctx: Path, session: str | None) -> str:
    """If a test-relevant file was edited this turn (post-edit marked the session dirty), run the full
    suite ONCE and clear the marker. Returns a surfaced notice on failure, else "". Best-effort: clears
    the marker BEFORE running so a crash can't loop, and never raises into the hook."""
    try:
        m = _load_metrics(ctx)
        q = m.get("quality") or {}
        sess = session or "session"
        if (q.get("dirty_session") or None) != sess:
            return ""
        m["quality"] = dict(q, dirty_session=None)   # clear first — a mid-run crash must not re-trigger
        _save_metrics(ctx, m)
        return _run_quality_cmd(_quality_config(ctx)["test_cmd"], cwd=str(ctx.parent), timeout=600) or ""
    except Exception:
        return ""


# Stop-time quality recommendation (t70 Phase 3) — surfaced through the quality Stop group, so it
# reaches the model exactly when a turn of edits has finished. Only the change-derived quality signals
# fire here (permission friction is an orient concern); security review outranks code review.
_QUALITY_TRIGGERS = ("sensitive-changes", "uncommitted-changes")


def _review_nudge_on_stop(ctx: Path, session: str | None) -> str:
    """When the working tree has review-worthy changes at Stop, surface the top quality recommendation
    (security > code review) ONCE per session. Honors the advisor enable flag + dismissed intents.
    Best-effort and read-only (the dedupe marker aside); never raises into the hook."""
    try:
        if not _advisor_config(ctx)["enabled"]:
            return ""
        m = _load_metrics(ctx)
        q = m.get("quality") or {}
        sess = session or "session"
        if (q.get("review_nudged_session") or None) == sess:
            return ""                                  # already nudged this session — don't nag
        dismissed = set((m.get("advisor") or {}).get("dismissed") or [])
        adapter = active_adapter(ctx)
        sigs = [s for s in adapter.friction_signals(ctx, session)
                if s.get("signal") in _QUALITY_TRIGGERS]
        if not sigs:
            return ""
        intent = _signal_to_intent(sigs[0])
        if not intent or intent in dismissed:
            return ""
        text, _ch = adapter.render(intent, sigs[0])
        if not text:
            return ""
        m["quality"] = dict(q, review_nudged_session=sess)
        _save_metrics(ctx, m)
        return text
    except Exception:
        return ""


def cmd_hook(args):
    """`celeborn hook <event>` — the collapsed, in-process hook entry point (executable-app §3).

    Resolves the project dir (explicit --path wins, else $CLAUDE_PROJECT_DIR, else cwd), reads the
    host's JSON payload from stdin, and relays dispatch_hook()'s output to stdout. Never raises."""
    import os
    explicit = getattr(args, "path", None)
    if explicit and explicit != ".":
        project_dir = explicit
    else:
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    payload = _read_hook_payload(getattr(args, "_stdin", None))
    out = dispatch_hook(args.event, payload, project_dir)
    if out:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")


# The hook wiring Celeborn installs, in event order. Post-collapse (executable-app §3) every command
# is a single in-process `celeborn hook <event>` — no bash wrapper, no inline python3, no
# $CELEBORN_HOME. `wire` injects these (plus the statusLine) into settings.json. The legacy bash
# script names are kept only so re-running `wire` can detect and MIGRATE an old install in place.
WIRE_HOOKS = [
    ("SessionStart", "session-start", "session-start.sh"),
    ("UserPromptSubmit", "user-prompt-submit", "context-watch.sh"),
    ("PreCompact", "pre-compact", "pre-compact.sh"),
    ("SessionEnd", "session-end", "session-end.sh"),
    ("Stop", "stop", "capture.sh"),
    # Notification fires on a permission prompt / ~60s idle input — the blocked-progress alert
    # (CELE-t169). No legacy bash form ever existed; the third field is an inert migration label.
    ("Notification", "notification", "notification.sh"),
]
WIRE_STATUSLINE = {"type": "command", "command": "celeborn hook statusline"}
# Safety hooks `wire` installs alongside the five above (t101). Unlike WIRE_HOOKS these carry a
# Claude `matcher` (only run on the named tools) and have no legacy bash form. The single PreToolUse
# group fans out inside dispatch_hook: Bash → the cd+redirect guard (t101); Edit/Write/NotebookEdit →
# the card-less-work gate (t131). The matcher lists exactly those four so the hook never fires on
# Read/Grep/etc. (An older install wired with the Bash-only matcher is migrated up in `cmd_wire`.)
SAFETY_HOOKS = [
    ("PreToolUse", "pre-tool-use", "Bash|Edit|Write|NotebookEdit"),
]
# A legacy hooks/*.sh path in a wired command — what marks a group as ours but in the OLD bash form.
_LEGACY_HOOK_NAMES = ("statusline.sh", *(s for _, _, s in WIRE_HOOKS))


def _is_celeborn_command(text: str, event_token: str, legacy_script: str) -> bool:
    """True if `text` (a command string / serialized group) is a Celeborn hook for this event —
    either the new `celeborn hook <event>` form or the legacy `…/hooks/<script>.sh` form."""
    return f"hook {event_token}" in text or legacy_script in text


# t100 — the SAFE "big three" permission baseline merged into the GLOBAL Claude settings.json on
# `wire --global`. PROACTIVE + universal: it sets a safe floor *before* any approval history exists,
# so every user stops re-approving the same read-only commands out of the box. Complementary to
# `celeborn permissions --suggest|--apply` (which REACTIVELY learns wildcards from a user's own
# history). Read-only / trivially-reversible commands ONLY — NEVER sed/awk/redirection/rm or any
# non-localhost network reach. Emitted as Claude Code prefix wildcards: `Bash(<prefix>:*)`.
BASELINE_ALLOW_TOOLS = ["Read", "Glob", "Grep"]
BASELINE_BASH_PREFIXES = [
    # file / search (read-only)
    "grep", "rg", "find", "cat", "ls", "head", "tail", "wc", "tree", "file", "stat",
    "which", "pwd", "realpath", "dirname", "basename", "diff", "sort", "uniq",
    # git reads
    "git log", "git diff", "git show", "git status", "git branch", "git remote -v",
    # github reads
    "gh pr view", "gh pr list", "gh issue list", "gh repo view", "gh run list",
    # tool info
    "npm ls", "node --version", "python3 --version", "pip show", "jq", "env",
    # process / port (localhost only)
    "lsof", "ps", "curl -sS http://localhost",
]
BASELINE_DEFAULT_MODE = "acceptEdits"


def _baseline_allow_rules() -> list:
    """The full SAFE allow-list t100 ships: the read-only built-in tools, then each safe Bash prefix
    as a `Bash(<prefix>:*)` wildcard. One flat, order-stable list — the merge dedupes against
    whatever the user already has."""
    return list(BASELINE_ALLOW_TOOLS) + [f"Bash({pre}:*)" for pre in BASELINE_BASH_PREFIXES]


def _merge_permission_baseline(data: dict) -> dict:
    """Merge the t100 baseline into an already-loaded settings dict, IN PLACE and ask-wins. Returns a
    report {'added': [...], 'default_mode_set': bool}.

    Iron rules: never replace or reorder an existing entry; only APPEND allow-rules absent from BOTH
    `permissions.allow` and `permissions.deny` (deny wins); only set `defaultMode` when the user has
    not set it to anything. Re-running is a no-op (dedupe by exact string)."""
    perms = data.setdefault("permissions", {})
    allow = perms.setdefault("allow", [])
    existing = set(allow)
    denied = set(perms.get("deny") or [])
    added = []
    for rule in _baseline_allow_rules():
        if rule in existing or rule in denied:
            continue                          # already allowed, or the user denied it → leave it
        allow.append(rule)
        existing.add(rule)
        added.append(rule)
    default_mode_set = False
    if "defaultMode" not in perms:            # ask-wins: never override a user's chosen mode
        perms["defaultMode"] = BASELINE_DEFAULT_MODE
        default_mode_set = True
    return {"added": added, "default_mode_set": default_mode_set}


# --------------------------------------------------------------------------- Danger Zone (t115)
# The FULL, intentionally-unsafe auto-allow spectrum, surfaced (and gated behind a typed confirmation)
# on the board Settings page. Arming this lets the agent run ANY command, read/write ANY file, reach
# ANY network host, and use every MCP tool — and `bypassPermissions` stops Claude asking about anything.
# Listed explicitly so the UI can enumerate exactly what gets turned on. NEVER applied without --yes.
DANGER_SPECTRUM = [
    "Bash(*)",                 # any shell command — incl. rm, git push, curl to any host
    "Read", "Edit", "Write",   # unrestricted file read + write
    "WebFetch", "WebSearch",   # arbitrary network access
    "mcp__*",                  # every MCP tool, including mutating ones
]
DANGER_DEFAULT_MODE = "bypassPermissions"   # Claude stops asking permission for ANYTHING
DANGER_CONFIRM_PHRASE = "DISABLE ALL SAFETY"


# --------------------------------------------------------------------------- skill catalog (t115)
# The three groups the board Settings page renders: Celeborn's own bundled verbs (SKILL.md), the Claude
# slash-commands the t70 advisor points at, and the Matt Pocock skill suite that Celeborn installs
# default-on (https://github.com/mattpocock/skills).
CELEBORN_CORE_SKILLS = [
    {"name": "Orient", "command": "celeborn status",
     "description": "Cheap rehydration — prints the Hot tier (state headline, session focus, board, "
                    "recent activity) so a fresh thread knows where things stand without re-reading everything."},
    {"name": "Checkpoint", "command": "celeborn checkpoint",
     "description": "Safe writer for session.json — records focus/next-action/branch/status, stamps "
                    "updated_at, clips over-long fields, repairs a corrupt file. Use instead of hand-editing the JSON."},
    {"name": "Forget", "command": "celeborn archive",
     "description": "Moves old journal entries to cold storage so the Hot tier stays small and cheap to load every turn."},
    {"name": "Promote", "command": "celeborn promote --to learnings|durable",
     "description": "Distills knowledge up a tier (journal -> learnings -> durable docs) so hard-won facts survive and stop being rediscovered."},
    {"name": "Handoff", "command": "celeborn handoff",
     "description": "Writes a tiny resume prompt so a brand-new thread can pick up exactly where the last one died."},
]
RECOMMENDED_SKILLS = [
    {"name": "/code-review", "description": "Reviews the working diff for correctness bugs + cleanups; "
        "the advisor surfaces it when there are substantial uncommitted code changes."},
    {"name": "/verify", "description": "Runs the app and confirms a change actually behaves as intended — paired with /code-review before calling work done."},
    {"name": "/security-review", "description": "Security pass over pending changes (authn/authz, secret handling, input validation, injection/SSRF); advised when changes touch sensitive paths."},
    {"name": "/fewer-permission-prompts", "description": "Learns wildcard allow-rules from your own approval history so you stop re-approving the same commands. Pairs with `celeborn permissions`."},
    {"name": "/loop", "description": "Repeats a prompt/step on an interval for polling or recurring tasks; checkpoints via Celeborn so a restart resumes."},
    {"name": "/elves", "description": "Multi-batch autonomous development for long unattended runs — implements a plan in sprint-sized batches with PR-based review."},
]
# Matt Pocock's suite. Names match the skill directories the `skills` CLI installs under .claude/skills/.
MATTPOCOCK_SKILLS = [
    {"name": "ask-matt", "description": "Routes you to the right skill for your current situation."},
    {"name": "grill-with-docs", "description": "Detailed discovery interview that builds a project domain model and updates CONTEXT.md."},
    {"name": "triage", "description": "Moves issues through a state-machine triage workflow."},
    {"name": "improve-codebase-architecture", "description": "Identifies architectural improvements and presents a visual HTML report."},
    {"name": "to-issues", "description": "Breaks a plan into independently-grabbable issues using vertical slices."},
    {"name": "to-prd", "description": "Synthesizes a conversation into a publishable PRD."},
    {"name": "prototype", "description": "Builds throwaway prototypes to explore designs."},
    {"name": "diagnosing-bugs", "description": "Systematic debugging loop: reproduce, minimize, hypothesize, instrument, fix, test."},
    {"name": "tdd", "description": "Red-green-refactor loop for features and bug fixes."},
    {"name": "domain-modeling", "description": "Actively builds and sharpens domain models with terminology validation."},
    {"name": "codebase-design", "description": "Establishes shared vocabulary for designing deep, maintainable modules."},
    {"name": "grill-me", "description": "Comprehensive interview that resolves all decision branches before building."},
    {"name": "handoff", "description": "Creates compact handoff documents for agent-to-agent continuation."},
    {"name": "teach", "description": "Multi-session skill instruction using a directory as a stateful workspace."},
    {"name": "writing-great-skills", "description": "Reference guide for skill vocabulary and authoring principles."},
    {"name": "grilling", "description": "The reusable interview loop underlying grill-me and grill-with-docs."},
    {"name": "git-guardrails-claude-code", "description": "Blocks dangerous git operations via pre-execution hooks."},
    {"name": "migrate-to-shoehorn", "description": "Converts type assertions to @total-typescript/shoehorn."},
    {"name": "scaffold-exercises", "description": "Creates structured exercise directories."},
    {"name": "setup-pre-commit", "description": "Configures Husky hooks with linting and testing."},
    {"name": "setup-matt-pocock-skills", "description": "Configures the suite for your repo — run once after install."},
]
MATTPOCOCK_INSTALL_CMD = ["npx", "--yes", "skills@latest", "add", "mattpocock/skills"]
MATTPOCOCK_SOURCE = "https://github.com/mattpocock/skills"

# t116 — "stay updated": the suite is refreshed (re-pull @latest) on a weekly cadence. State lives GLOBAL
# (the skills install to ~/.claude/skills, so the throttle must be fleet-wide, not per-project) in
# ~/.config/celeborn/skills.json. The SessionStart hook fires a DETACHED, non-blocking refresh when due
# — never delaying orient. Opt out with autoupdate:false there, the CELEBORN_NO_SKILLS env, or
# `wire --no-skills`. Claude-only (the skills only exist under .claude/skills).
SKILLS_STATE_FILE = "skills.json"
SKILLS_REFRESH_DAYS = 7


def _skills_state_path() -> Path:
    return _config_dir() / SKILLS_STATE_FILE


def _load_skills_state() -> dict:
    p = _skills_state_path()
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_skills_state(data: dict) -> None:
    p = _skills_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")


def _skills_autoupdate_due(state: dict | None = None) -> bool:
    """True when a weekly Matt Pocock refresh is due. False when opted out, or refreshed within the
    window. A missing/garbled timestamp counts as due (first run)."""
    import os
    if os.environ.get("CELEBORN_NO_SKILLS"):
        return False
    state = _load_skills_state() if state is None else state
    if state.get("autoupdate") is False:
        return False
    last = state.get("last_refresh")
    if not last:
        return True
    try:
        then = _dt.datetime.fromisoformat(str(last))
    except ValueError:
        return True
    return (_dt.datetime.now() - then).days >= SKILLS_REFRESH_DAYS


def _spawn_skills_refresh() -> bool:
    """Fire a DETACHED `celeborn skills update --global` (own session, stdio discarded) so the weekly
    refresh never blocks orient. Stamps `last_refresh` optimistically BEFORE spawning so a slow/failed
    background run doesn't re-trigger every session — the next window retries. Returns False (no-op) when
    npx is absent. Best-effort: never raises."""
    import os
    import shutil
    import subprocess
    import sys
    if shutil.which("npx") is None:
        return False
    state = _load_skills_state()
    state["last_refresh"] = now_iso()
    try:
        _save_skills_state(state)
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "skills", "update", "--global"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:                                  # noqa: BLE001 — a refresh hiccup never breaks orient
        return False


def _ensure_skills_fresh(ctx: Path) -> None:
    """SessionStart seam: when due (weekly) and the active harness is Claude, kick off a detached
    background refresh of the Matt Pocock suite. Claude-only — the skills only exist under .claude/skills;
    Grok/Codex read .grok//.codex/ and get the advisor's guidance as prose instead. Never raises."""
    try:
        if active_adapter(ctx).name != "claude":
            return
        if _skills_autoupdate_due():
            _spawn_skills_refresh()
    except Exception:                                  # noqa: BLE001
        pass


def _settings_path_for_scope(ctx: Path, scope: str) -> Path:
    """Resolve the Claude settings file for a permission scope. 'global' -> ~/.claude/settings.json;
    'shared' -> project .claude/settings.json; 'local' (default) -> project .claude/settings.local.json."""
    if scope == "global":
        return Path.home() / ".claude" / "settings.json"
    base = ctx.parent / ".claude"
    return base / ("settings.json" if scope == "shared" else "settings.local.json")


def _read_settings(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _resolved_permissions(ctx: Path) -> dict:
    """Effective allow set + defaultMode across all Claude settings scopes. `allow` is the union;
    `effective_mode` follows local > project(shared) > global precedence."""
    scopes = [("global", _settings_path_for_scope(ctx, "global")),
              ("shared", _settings_path_for_scope(ctx, "shared")),
              ("local", _settings_path_for_scope(ctx, "local"))]
    union, seen, per_file, mode = [], set(), {}, {}
    for name, p in scopes:
        data = _read_settings(p)
        perms = data.get("permissions") or {}
        allow = list(perms.get("allow") or [])
        per_file[name] = {"path": str(p), "exists": p.is_file(), "allow": allow,
                          "defaultMode": perms.get("defaultMode")}
        mode[name] = perms.get("defaultMode")
        for r in allow:
            if r not in seen:
                seen.add(r); union.append(r)
    effective_mode = mode["local"] or mode["shared"] or mode["global"]
    return {"allow": union, "allow_set": seen, "effective_mode": effective_mode, "per_file": per_file}


def _permissions_state_json(ctx: Path) -> dict:
    """The full read-only state the board Settings page renders: which baseline rules are active, the
    Danger Zone spectrum + whether armed, the resolved allow-list, and per-scope file breakdown."""
    res = _resolved_permissions(ctx)
    allow_set, eff_mode = res["allow_set"], res["effective_mode"]
    tools = [{"rule": t, "active": t in allow_set} for t in BASELINE_ALLOW_TOOLS]
    bash = [{"rule": f"Bash({pre}:*)", "prefix": pre, "active": f"Bash({pre}:*)" in allow_set}
            for pre in BASELINE_BASH_PREFIXES]
    danger = [{"rule": r, "active": r in allow_set} for r in DANGER_SPECTRUM]
    return {
        "effective_default_mode": eff_mode,
        "baseline": {
            "tools": tools,
            "bash_prefixes": bash,
            "default_mode": {"value": BASELINE_DEFAULT_MODE, "active": eff_mode == BASELINE_DEFAULT_MODE},
            "all_active": all(x["active"] for x in tools + bash) and eff_mode == BASELINE_DEFAULT_MODE,
        },
        "danger": {
            "spectrum": danger,
            "default_mode": {"value": DANGER_DEFAULT_MODE, "active": eff_mode == DANGER_DEFAULT_MODE},
            # "Armed" keys ONLY on the unambiguous danger signals — blanket shell (`Bash(*)`) or
            # bypassPermissions. Benign spectrum members (Read/Edit/Write) being individually allowed is
            # NOT the Danger Zone; per-rule `active` flags above still show exactly what is present.
            "armed": eff_mode == DANGER_DEFAULT_MODE or ("Bash(*)" in allow_set),
            "confirm_phrase": DANGER_CONFIRM_PHRASE,
        },
        "current_allow": res["allow"],
        "scopes": res["per_file"],
    }


def _backup_and_load_settings(path: Path) -> dict:
    """Load a settings file for writing, keeping a .celeborn-bak backup (mirrors cmd_wire). Refuses to
    proceed on invalid JSON so a bad write can never clobber a good file."""
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            die(f"{path} is not valid JSON — refusing to rewrite it. Fix the file by hand first.")
        (path.parent / (path.name + ".celeborn-bak")).write_text(json.dumps(data, indent=2) + "\n")
    return data


def _remove_permission_baseline(data: dict) -> dict:
    """Strip exactly the t100 baseline rules; revert defaultMode only if it is still the baseline value."""
    perms = data.setdefault("permissions", {})
    allow = perms.get("allow") or []
    baseline = set(_baseline_allow_rules())
    removed = [r for r in allow if r in baseline]
    perms["allow"] = [r for r in allow if r not in baseline]
    reverted = False
    if perms.get("defaultMode") == BASELINE_DEFAULT_MODE:
        perms.pop("defaultMode", None)
        reverted = True
    return {"removed": removed, "default_mode_reverted": reverted}


def _arm_danger_zone(data: dict) -> dict:
    perms = data.setdefault("permissions", {})
    allow = perms.setdefault("allow", [])
    existing, added = set(allow), []
    for r in DANGER_SPECTRUM:
        if r not in existing:
            allow.append(r); existing.add(r); added.append(r)
    prev = perms.get("defaultMode")
    perms["defaultMode"] = DANGER_DEFAULT_MODE
    return {"added": added, "prev_mode": prev}


def _disarm_danger_zone(data: dict) -> dict:
    perms = data.setdefault("permissions", {})
    allow = perms.get("allow") or []
    danger = set(DANGER_SPECTRUM)
    removed = [r for r in allow if r in danger]
    perms["allow"] = [r for r in allow if r not in danger]
    perms["defaultMode"] = BASELINE_DEFAULT_MODE
    return {"removed": removed}


def _skills_dirs(ctx: Path) -> list:
    return [ctx.parent / ".claude" / "skills", Path.home() / ".claude" / "skills"]


def _mattpocock_installed_names(ctx: Path) -> set:
    known = {s["name"] for s in MATTPOCOCK_SKILLS}
    found = set()
    for d in _skills_dirs(ctx):
        if d.is_dir():
            for child in d.iterdir():
                if child.is_dir() and child.name in known:
                    found.add(child.name)
    return found


def _skills_state_json(ctx: Path) -> dict:
    installed = _mattpocock_installed_names(ctx)
    mp = [{**s, "installed": s["name"] in installed} for s in MATTPOCOCK_SKILLS]
    sstate = _load_skills_state()
    return {
        # Harness scope (t116): the recommended slash-commands AND the Matt Pocock suite are Claude-only.
        # Grok/Codex receive the SAME advisor recommendations as branded prose (no slash commands), and
        # don't read .claude/skills. The board surfaces this so "Claude skills" isn't misread.
        "harness": "claude",
        "recommended_note": "Claude Code slash-commands. On Grok/Codex the advisor surfaces the same "
                            "recommendations as prose, not installable skills.",
        "core": CELEBORN_CORE_SKILLS,
        "recommended": RECOMMENDED_SKILLS,
        "mattpocock": {
            "source": MATTPOCOCK_SOURCE,
            "install_cmd": " ".join(MATTPOCOCK_INSTALL_CMD),
            "setup_hint": "/setup-matt-pocock-skills",
            "installed_count": len(installed),
            "total": len(MATTPOCOCK_SKILLS),
            "claude_only": True,
            "last_refresh": sstate.get("last_refresh"),
            "autoupdate": sstate.get("autoupdate", True),
            "refresh_days": SKILLS_REFRESH_DAYS,
            "skills": mp,
        },
    }


def _install_mattpocock(ctx: Path, scope: str = "local") -> dict:
    """Run the community `skills` CLI to add the Matt Pocock suite. Network + Node required — dies with a
    clear message if npx is missing. Idempotent (re-detects what is now installed)."""
    import shutil
    import subprocess
    if shutil.which("npx") is None:
        die("npx not found — install Node.js to add the Matt Pocock skills "
            f"(`{' '.join(MATTPOCOCK_INSTALL_CMD)}`).")
    cwd = Path.home() if scope == "global" else ctx.parent
    try:
        # stdin closed so any installer prompt fails fast (EOF) instead of hanging a non-interactive run.
        proc = subprocess.run(MATTPOCOCK_INSTALL_CMD, cwd=str(cwd),
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        die(f"failed to run the skills installer: {e}")
    if proc.returncode != 0:
        die(f"skills installer failed (rc={proc.returncode}): {(proc.stderr or proc.stdout or '').strip()[:500]}")
    installed = sorted(_mattpocock_installed_names(ctx))
    return {"ok": True, "cwd": str(cwd), "installed": installed, "count": len(installed)}


def cmd_skills(args):
    """`celeborn skills [list|install-mattpocock]` — list Celeborn/recommended/Matt-Pocock skills (the
    board Settings page consumes `--json`), or install the Matt Pocock suite into .claude/skills/."""
    ctx = require_context(args)
    action = getattr(args, "skills_cmd", None) or "list"
    if action in ("install-mattpocock", "update"):
        scope = "global" if getattr(args, "global_", False) else "local"
        rep = _install_mattpocock(ctx, scope)
        # Record the refresh so the weekly auto-update throttle (t116) counts a manual run too.
        st = _load_skills_state()
        st["last_refresh"] = now_iso()
        _save_skills_state(st)
        if getattr(args, "json", False):
            print(json.dumps({**rep, "last_refresh": st["last_refresh"]}, indent=2))
            return
        verb = "updated" if action == "update" else "installed"
        ok(f"{verb} Matt Pocock skills (latest) — {rep['count']} present under .claude/skills/.")
        if action != "update":
            info("finish setup: run `/setup-matt-pocock-skills` in a Claude Code session.")
        return
    state = _skills_state_json(ctx)
    if getattr(args, "json", False):
        print(json.dumps(state, indent=2))
        return
    print("Celeborn skills")
    print("  Core (bundled — the five verbs):")
    for s in state["core"]:
        print(f"    {s['name']:<11} {s['command']}")
    print("  Recommended (the advisor points at these Claude skills):")
    for s in state["recommended"]:
        print(f"    {s['name']}")
    mp = state["mattpocock"]
    print(f"  Matt Pocock ({mp['installed_count']}/{mp['total']} installed) — {mp['source']}:")
    for s in mp["skills"]:
        print(f"    [{'x' if s['installed'] else ' '}] {s['name']}")


def cmd_wire(args):
    """Merge Celeborn's `statusLine` and five hook groups into a Claude Code settings.json —
    idempotently. The programmatic alternative to hand-merging hooks/settings.snippet.json. Preserves
    everything already in the file: existing keys, unrelated hooks, and (unless `--force`) a
    non-Celeborn statusLine. Re-running never duplicates a hook group, and MIGRATES a legacy
    bash-based install to the collapsed `celeborn hook <event>` form. Backs the file up before writing."""
    if getattr(args, "global_", False):
        settings = Path.home() / ".claude" / "settings.json"
        scope = "global"
    else:
        settings = Path(getattr(args, "path", ".") or ".").resolve() / ".claude" / "settings.json"
        scope = "project"
    settings.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if settings.is_file():
        try:
            data = json.loads(settings.read_text())
        except json.JSONDecodeError:
            die(f"{settings} is not valid JSON; refusing to overwrite. Fix or remove it first.")
        (settings.parent / (settings.name + ".celeborn-bak")).write_text(json.dumps(data, indent=2) + "\n")

    added, migrated, skipped = [], [], []

    sl = data.get("statusLine")
    if sl == WIRE_STATUSLINE:
        pass
    elif sl and "statusline.sh" in json.dumps(sl):
        data["statusLine"] = dict(WIRE_STATUSLINE)        # migrate a legacy Celeborn statusLine
        migrated.append("statusLine")
    elif sl and not getattr(args, "force", False):
        skipped.append("statusLine — a non-Celeborn statusLine is already set; rerun with --force to replace it")
    else:
        data["statusLine"] = dict(WIRE_STATUSLINE)
        added.append("statusLine")

    hooks = data.setdefault("hooks", {})
    for event, token, legacy in WIRE_HOOKS:
        groups = hooks.setdefault(event, [])
        new_cmd = f"celeborn hook {token}"
        mine = [g for g in groups if _is_celeborn_command(json.dumps(g), token, legacy)]
        if not mine:
            groups.append({"hooks": [{"type": "command", "command": new_cmd}]})
            added.append(f"hooks.{event}")
            continue
        # Already wired — migrate any legacy bash command in our group(s) to the collapsed form.
        for g in mine:
            for h in g.get("hooks", []):
                if h.get("command") != new_cmd:
                    h["command"] = new_cmd
                    migrated.append(f"hooks.{event}")

    # t101 — safety hooks carry a tool matcher and have no legacy bash form, so they're wired in a
    # separate matcher-aware pass (idempotent: detect our group by the `hook <token>` command string).
    for event, token, matcher in SAFETY_HOOKS:
        groups = hooks.setdefault(event, [])
        existing = next((g for g in groups if f"hook {token}" in json.dumps(g)), None)
        if existing is not None:
            # Already wired — but migrate an outdated matcher in place (t131 widened PreToolUse from
            # "Bash" to also cover Edit/Write/NotebookEdit for the card-less-work gate).
            if matcher and existing.get("matcher") != matcher:
                existing["matcher"] = matcher
                migrated.append(f"hooks.{event} matcher")
            continue
        group = {"hooks": [{"type": "command", "command": f"celeborn hook {token}"}]}
        if matcher:
            group["matcher"] = matcher
        groups.append(group)
        added.append(f"hooks.{event}")

    # t100 — merge the SAFE permission baseline, but ONLY on a global Claude wire (the "big three" are
    # Claude Code concepts; project settings + other harnesses are out of scope). Opt-out via
    # --no-permission-baseline. Ask-wins + idempotent + scoped, all inside _merge_permission_baseline.
    baseline = None
    if (scope == "global" and active_adapter(None).name == "claude"
            and not getattr(args, "no_permission_baseline", False)):
        baseline = _merge_permission_baseline(data)

    _atomic_write_json(settings, data)

    if added or migrated:
        bits = []
        if added:
            bits.append("added " + ", ".join(added))
        if migrated:
            bits.append("migrated " + ", ".join(sorted(set(migrated))))
        ok(f"wired Celeborn into {settings} ({scope}): {'; '.join(bits)}")
    else:
        info(f"{settings} already wired — nothing to add")
    for s in skipped:
        warn(s)
    if baseline is not None:
        n, dm = len(baseline["added"]), baseline["default_mode_set"]
        if n or dm:
            bits = []
            if n:
                bits.append(f"+{n} allow rule(s)")
            if dm:
                bits.append(f"defaultMode={BASELINE_DEFAULT_MODE}")
            ok(f"safe permission baseline: {', '.join(bits)} → {settings}")
            info("revert any time: run `/permissions`, or edit ~/.claude/settings.json "
                 "(a settings.json.celeborn-bak backup was kept if the file already existed).")
            info("⚠ applies to NEW sessions — `Shift+Tab` toggles acceptEdits in the current one; "
                 "`/permissions` reloads config.")
        else:
            info("safe permission baseline already present — nothing added.")
    # t115 — install the Matt Pocock skill suite default-on on a GLOBAL Claude wire (opt out via
    # --no-skills). Best-effort: a missing npx / failed install must NEVER fail the wire.
    if (scope == "global" and active_adapter(None).name == "claude"
            and not getattr(args, "no_skills", False)):
        import shutil
        if shutil.which("npx") is None:
            warn("skipped Matt Pocock skills: npx not found (install Node) — add later with "
                 "`celeborn skills install-mattpocock`.")
        else:
            ctx0 = find_context_root(Path(getattr(args, "path", ".") or "."))
            try:
                rep = _install_mattpocock(ctx0 or Path(getattr(args, "path", ".") or "."), "global")
                ok(f"Matt Pocock skills installed default-on — {rep['count']} present. "
                   f"Run `/setup-matt-pocock-skills` to finish.")
            except SystemExit:
                warn("skipped Matt Pocock skills (installer failed) — add later with "
                     "`celeborn skills install-mattpocock`. Opt out permanently with `wire --no-skills`.")
    info("commands are in-process `celeborn hook <event>` — `celeborn` must be on PATH "
         "(pip/uv install). No $CELEBORN_HOME or hooks/ dir needed.")
    if scope == "project":
        info("project-scoped — pass --global to wire ~/.claude/settings.json for every session.")
    consent = _load_consent()
    if consent.get("agreed"):
        n = len(consent.get("opted_out") or [])
        info(f"consent on record for {consent.get('name')}"
             + (f" ({n} opt-out{'s' if n != 1 else ''})." if n else " — all click-reducers enabled."))
    else:
        info("review what Celeborn automates for you (all opt-out) + accept the User Agreement: "
             "run `celeborn consent`")
        info(f"User Agreement: {AGREEMENT_URL}")
    info(_legal_docs_line())
    if getattr(args, "grok", False):
        ctx = find_context_root(Path(getattr(args, "path", ".") or "."))
        if ctx is None:
            warn("--grok: no .context/ here — run `celeborn init` first")
        elif _wire_grok(ctx.parent):
            ok("also wired Grok Build for this project")


# ------------------------------------------------------------------------------- setup (t120)
#
# "Install like Modal" (CELE-t120). Modal's whole onboarding is `pip install modal` + `python3 -m
# modal setup` (a browser auth) — two commands and you're running code. `celeborn setup` is the
# post-package-install half of that: ONE guided command that wires Claude Code, scaffolds the current
# project, and signs you in (browser PKCE — required, Modal parity), then prints a "you're ready"
# next-step. It is a THIN, idempotent, resumable orchestrator over the existing first-class verbs
# (`wire`/`init`/`login`) — never a reimplementation, so the manual path stays intact. Re-running it
# resumes: `wire` is idempotent, `init` is skipped when `.context/` already exists, and `login` is
# skipped when a session is already on record — so a setup interrupted at any step finishes on a re-run.
# Design + rationale: references/setup-onboarding-plan.md.


def _setup_step_init(args, path: str) -> None:
    """Scaffold the current project unless it's already a Celeborn project (idempotent/resumable)."""
    if getattr(args, "no_init", False):
        info("scaffold: skipped (--no-init).")
        return
    root = Path(path or ".").resolve()
    if (root / CONTEXT_DIRNAME).is_dir():
        info(f"scaffold: already a Celeborn project (.context/ present at {root.name}/) — skipped.")
        return
    init_args = argparse.Namespace(
        path=path, private=False, public=False, claude_md=True, agents_md=True, scan=True,
        no_cmm=getattr(args, "no_cmm", False), name=getattr(args, "name", None),
        open_board=not getattr(args, "no_open", False),
        open_browser=not getattr(args, "no_browser", False))
    cmd_init(init_args)


def _setup_step_login(args) -> bool:
    """Sign in (browser PKCE by default; email+password when --email is given). Required by default —
    Modal parity — but skippable with --no-login, and impossible to force on a non-TTY shell (CI/headless
    can't open a browser). Returns True if a session is on record afterward. A failed interactive sign-in
    WARNS and lets setup finish (wire+init already succeeded and are usable locally) rather than aborting."""
    sync = __import__("celeborn_sync")
    creds = sync.load_creds()
    if creds.get("access_token") or creds.get("refresh_token"):
        who = creds.get("email") or creds.get("username") or "your account"
        info(f"sign-in: already signed in as {who} — skipped (run `celeborn logout` to switch).")
        return True
    if getattr(args, "no_login", False):
        info("sign-in: skipped (--no-login). Run `celeborn login --github` later to enable the hosted board.")
        return False
    if not _init_is_interactive():
        warn("sign-in: non-interactive shell — can't open a browser. Run `celeborn login --github` from a "
             "terminal to enable the hosted board. (Local Celeborn works fully without an account.)")
        return False
    use_email = bool(getattr(args, "email", None))
    info("opening your browser to sign in with GitHub…" if not use_email
         else "signing in with email + password…")
    login_args = argparse.Namespace(github=not use_email, email=getattr(args, "email", None), password=None)
    try:
        sync.cmd_login(login_args)
        return True
    except SystemExit:
        warn("sign-in didn't complete — finish it later with `celeborn login --github` (or re-run "
             "`celeborn setup`). Continuing — local Celeborn works without an account.")
        return False


def _setup_ready(path: str, signed_in: bool) -> None:
    """Modal-style closing: where the board is + the single next thing to do."""
    print("\n✅ Celeborn is ready.\n")
    ctx = find_context_root(Path(path or "."))
    if ctx is not None:
        info(f"Your kanban board (Celeborn's UI): {board_url(ctx)}")
    print("\n  Next:")
    print("    • Open Claude Code in this project — Celeborn orients automatically every session.")
    print("    • Inspect what an agent loads on orient:  celeborn status")
    if not signed_in:
        print("    • Enable the hosted board (optional):  celeborn login --github")
    print()


def cmd_setup(args):
    """Modal-clean first run: one guided command = wire Claude Code + scaffold this project + sign in.
    A thin orchestrator over `wire`/`init`/`login`, each idempotent so re-running resumes (CELE-t120).
    Order is wire → init → login so the local-first project is fully set up even if browser auth is the
    one step that doesn't complete; login is the final, gated step."""
    path = getattr(args, "path", ".") or "."
    print("\n🏹 Celeborn setup — wiring Claude Code, scaffolding this project, and signing you in.\n")

    print("[1/3] Wiring Claude Code (hooks + statusLine + safe baseline + skills)…")
    wire_args = argparse.Namespace(
        global_=not getattr(args, "project", False), force=getattr(args, "force", False),
        no_permission_baseline=getattr(args, "no_permission_baseline", False),
        no_skills=getattr(args, "no_skills", False), grok=False, path=path)
    cmd_wire(wire_args)

    print("\n[2/3] Scaffolding this project…")
    _setup_step_init(args, path)

    print("\n[3/3] Signing you in (browser)…")
    signed_in = _setup_step_login(args)

    _setup_ready(path, signed_in)


# --------------------------------------------------------------------------- consent / opt-out (t102)
#
# Celeborn is a TOOL: it has no will of its own — it performs exactly the click-reducing automations the
# operator turns on by installing and wiring it. `celeborn consent` makes that explicit: it shows every
# behavior that removes an approval click (all ON by default — opt-out, not opt-in), links the User
# Agreement, and records the operator's name + timestamp + any opt-outs to ~/.context/consent.json.
# `wire` prints a one-line pointer to it but NEVER blocks — install must stay non-interactive for CI/hooks.
AGREEMENT_URL = "https://celeborn.thot.ai/agreement"
AGREEMENT_VERSION = "2026-06-18"

# Standard published legal documents (CELE-t158), hosted on the thot.ai apex site. These are the
# best-practices Privacy / Cookie / User-Agreement instruments (US + EU/GDPR), distinct from the
# automation-consent disclosure at AGREEMENT_URL above. Surfaced on the consent screen and linked from
# both the local and hosted board footers, the way a standard web app footers its required agreements.
PRIVACY_URL = "https://thot.ai/privacy"
COOKIE_URL = "https://thot.ai/cookies"
USER_AGREEMENT_URL = "https://thot.ai/user-agreement"
LEGAL_DOCS = (
    ("Privacy Policy", PRIVACY_URL),
    ("Cookie Policy", COOKIE_URL),
    ("User Agreement", USER_AGREEMENT_URL),
)


def _legal_docs_line() -> str:
    """One-line pointer to the published legal documents, for CLI surfaces (consent screen, etc.)."""
    return "Legal & policies: " + "  ·  ".join(f"{name} {url}" for name, url in LEGAL_DOCS)

# (key, what it does, why it saves clicks, safety note or "") — the single source the CLI checklist
# renders from; the web User Agreement mirrors it in prose. Keep the two in sync when this changes.
CONSENT_ITEMS = [
    ("permission-baseline",
     "Pre-approve safe read-only commands and enable acceptEdits mode",
     "so you stop clicking “Allow” for ls / grep / git-status and routine file edits",
     "file edits and the safe-listed commands run without a per-action prompt"),
    ("permission-learn",
     "Generalize your repeated approvals into reusable allow-rules (`celeborn permissions`)",
     "a command you approve more than once becomes a wildcard rule you never re-approve again",
     ""),
    ("session-hooks",
     "Capture and restore project context automatically across sessions",
     "orient load, per-turn capture and checkpoint reminders — so a /clear never makes you re-explain",
     ""),
    ("cd-redirect-guard",
     "Steer an un-approvable `cd … > file` write to the Write tool (PreToolUse guard)",
     "turns a recurring manual approval into an invisible, statically-safe file write",
     ""),
    ("cd-redirect-autoallow",
     "Auto-allow a marked `cd … > file` write with no prompt",
     "a command you tag `# celeborn:allow-redirect` runs without asking",
     "a write whose target the permission system cannot statically verify then runs with no human check"),
    ("board-autostart",
     "Auto-launch your local kanban board on orient",
     "the board comes up without a manual command",
     ""),
    ("claim-on-paste",
     "Auto-claim a task card when you paste its marker",
     "pasting a card assigns it to you without a separate claim step",
     ""),
    ("quality-gates",
     "Run tests / typecheck automatically after edits (opt-in via `wire-quality`)",
     "surfaces failures without you remembering to run the suite",
     ""),
]
CONSENT_KEYS = [k for k, *_ in CONSENT_ITEMS]


def _consent_path() -> Path:
    return _global_context() / "consent.json"


def _load_consent() -> dict:
    """The recorded agreement (name, timestamp, opt-outs), or {} if none / unreadable. Best-effort."""
    p = _consent_path()
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _render_consent_checklist(opted_out: set) -> str:
    """The opt-out list: every click-reducer with a checkbox — [x] enabled (default), [ ] opted out."""
    lines = []
    for i, (key, what, why, risk) in enumerate(CONSENT_ITEMS, 1):
        box = "[ ]" if key in opted_out else "[x]"
        lines.append(f"  {box} {i}. {what}{'   ⚠' if risk else ''}")
        lines.append(f"        ↳ {why}")
        if risk:
            lines.append(f"        ⚠ safety: {risk}")
    return "\n".join(lines)


def _parse_optouts(tokens, warn_unknown: bool = True) -> set:
    """Map a comma-list of item numbers or keys to the canonical opt-out keys."""
    out: set = set()
    for tok in str(tokens or "").replace(" ", "").split(","):
        if not tok:
            continue
        if tok in CONSENT_KEYS:
            out.add(tok)
        elif tok.isdigit() and 1 <= int(tok) <= len(CONSENT_ITEMS):
            out.add(CONSENT_KEYS[int(tok) - 1])
        elif warn_unknown:
            warn(f"unknown item '{tok}' — ignored (valid: 1–{len(CONSENT_ITEMS)} or a key)")
    return out


def cmd_consent(args):
    """`celeborn consent` — the install-time opt-out screen. Shows every behavior that removes an
    approval click (all ON by default), links the User Agreement, and records the operator's name +
    any opt-outs to ~/.context/consent.json. Non-interactive via --name / --opt-out / --yes (CI +
    tests); --show prints the recorded consent and exits."""
    existing = _load_consent()
    if getattr(args, "show", False):
        if not existing.get("agreed"):
            info("No consent on record. Run `celeborn consent` to review the automations and agree.")
            return
        print(json.dumps(existing, indent=2))
        return

    flag = getattr(args, "opt_out", None)
    name = getattr(args, "name", None)
    interactive = sys.stdin.isatty() and not getattr(args, "yes", False) and flag is None and not name
    opted_out = _parse_optouts(flag) if flag is not None else set()

    print("🏹  Celeborn — what it automates for you (opt-out)\n")
    print("Celeborn is a tool. It has no mind of its own; it performs only the actions you turn on by")
    print("using it. Every item below removes an approval click and is ENABLED by default — uncheck any")
    print(f"you don't want. Full detail + safety notes: {AGREEMENT_URL}")
    print(_legal_docs_line() + "\n")
    print(_render_consent_checklist(opted_out) + "\n")

    if interactive:
        try:
            raw = input("Numbers to OPT OUT of (comma-separated), or Enter to keep all enabled: ").strip()
        except EOFError:
            raw = ""
        opted_out |= _parse_optouts(raw)
        if opted_out:
            print("\nUpdated:\n" + _render_consent_checklist(opted_out))
        print(f"\nI agree to the Celeborn User Agreement ({AGREEMENT_URL}).")
        try:
            name = input("Type your full name to agree (blank to cancel): ").strip()
        except EOFError:
            name = ""

    if not name:
        die("Agreement not recorded — no name provided. Re-run `celeborn consent` to agree.")

    record = {
        "agreed": True,
        "name": name,
        "agreed_at": now_iso(),
        "agreement_url": AGREEMENT_URL,
        "agreement_version": AGREEMENT_VERSION,
        "enabled": [k for k in CONSENT_KEYS if k not in opted_out],
        "opted_out": sorted(opted_out),
    }
    _scaffold_global(_global_context())
    _consent_path().write_text(json.dumps(record, indent=2) + "\n")
    ok(f"Agreement recorded for {name} ({record['agreed_at']}).")
    if opted_out:
        info("Opted out of: " + ", ".join(sorted(opted_out)) +
             " — recorded to consent.json (behaviors that honor these flags read it on run).")
    else:
        info("All click-reducers enabled. Change any time with `celeborn consent` (or --opt-out).")
    info(f"User Agreement: {AGREEMENT_URL}")
    info(_legal_docs_line())


# Quality-gate hook groups (t70 Phase 2), installed ONLY by `celeborn wire-quality` (opt-in) — never by
# `wire`. PostToolUse runs the cheap per-edit check; Stop runs the deferred full suite once per turn.
QUALITY_HOOKS = [
    ("PostToolUse", "post-edit", "Edit|Write|MultiEdit"),
    ("Stop", "quality-stop", None),
]
QUALITY_MD_BEGIN = "<!-- BEGIN CELEBORN QUALITY (managed by `celeborn wire-quality`) -->"
QUALITY_MD_END = "<!-- END CELEBORN QUALITY -->"


def _quality_instruction_text() -> str:
    """The harness-neutral quality rule — used as the AGENTS.md fallback where a host has no hooks."""
    return ("After editing files under `scripts/**` or `tests/**`, run the test suite "
            "(`python3 -m unittest discover -s tests`). After editing `board/**/*.tsx`, run "
            "`npx tsc --noEmit` in `board/` — NEVER `next build` (it clobbers the live dev server). "
            "Surface any failure and fix it before calling the change done.")


def _wire_quality_hooks_json(settings: Path, scope: str):
    """Merge the PostToolUse + Stop quality groups into a Claude settings.json — idempotently, mirroring
    cmd_wire. Preserves everything already there; re-running never duplicates a group; backs up first."""
    settings.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if settings.is_file():
        try:
            data = json.loads(settings.read_text())
        except json.JSONDecodeError:
            die(f"{settings} is not valid JSON; refusing to overwrite. Fix or remove it first.")
        (settings.parent / (settings.name + ".celeborn-bak")).write_text(json.dumps(data, indent=2) + "\n")
    hooks = data.setdefault("hooks", {})
    added = []
    for event, token, matcher in QUALITY_HOOKS:
        groups = hooks.setdefault(event, [])
        if any(f"hook {token}" in json.dumps(g) for g in groups):
            continue                                 # already wired — leave it
        group = {"hooks": [{"type": "command", "command": f"celeborn hook {token}"}]}
        if matcher:
            group["matcher"] = matcher
        groups.append(group)
        added.append(f"hooks.{event}")
    settings.write_text(json.dumps(data, indent=2) + "\n")
    if added:
        ok(f"wired quality gates into {settings} ({scope}): added {', '.join(added)}")
    else:
        info(f"{settings} already has the quality gates — nothing to add")
    info("on edit: py_compile (scripts/tests) · `tsc --noEmit` (board, never `next build`); "
         "on stop: full suite once per turn when scripts/** or tests/** changed. Failures surface, never block.")


def _wire_quality_agents_md(path: Path):
    """AGENTS.md fallback for a harness with no structured hooks: write/refresh a managed instruction
    block so a Codex/Grok-style host that auto-loads AGENTS.md still runs the gates by hand."""
    block = (f"{QUALITY_MD_BEGIN}\n## Quality gates (Celeborn)\n\n"
             f"{_quality_instruction_text()}\n{QUALITY_MD_END}\n")
    existing = path.read_text() if path.is_file() else ""
    if QUALITY_MD_BEGIN in existing and QUALITY_MD_END in existing:
        start = existing.index(QUALITY_MD_BEGIN)
        stop = existing.index(QUALITY_MD_END) + len(QUALITY_MD_END)
        new = existing[:start] + block.rstrip("\n") + existing[stop:]
        if new == existing:
            info(f"{path} quality block already current")
            return
        path.write_text(new)
        ok(f"refreshed the quality block in {path}")
        return
    sep = "\n" if (existing and not existing.endswith("\n")) else ""
    path.write_text(existing + sep + ("\n" if existing else "") + block)
    ok(f"appended quality gates to {path} (no structured hooks on this harness)")


def cmd_wire_quality(args):
    """`celeborn wire-quality` — opt-in deterministic quality gates (t70 Phase 2), routed through the
    active adapter. Claude: merge a PostToolUse + a Stop hook group into settings.json (SHARED by
    default — they help every contributor; `--local` → personal settings.local.json). A harness without
    structured hooks: append an AGENTS.md instruction instead. Surfaces failures, never blocks "done"."""
    ctx = require_context(args)
    shared = not getattr(args, "local", False)
    adapter = active_adapter(ctx)
    kind, target = adapter.quality_hook_target(ctx, shared=shared)
    if kind == "hooks-json":
        _wire_quality_hooks_json(target, scope=("shared" if shared else "personal"))
    elif kind == "agents-md":
        _wire_quality_agents_md(target)
    else:
        info("This harness exposes no quality-hook target. Add this to your agent instructions:")
        print("  " + _quality_instruction_text())


# The /clear nudge ends with a one-word (or one-line) sign-off that ALTERNATES every time the nudge
# fires, so a line a user sees often never reads stale. The workhorse pool is "flow" synonyms; roughly
# one firing in REMIND_WELLNESS_EVERY swaps in a wellness tidbit instead — a small wink that looks
# after the human in the chair, not just the context window. Rotation is deterministic (keyed off a
# persisted fire counter), never a clock or RNG — the codebase forbids both (they break resume/replay).
REMIND_FLOW_CLOSERS = (
    "Flow.", "Momentum.", "Cadence.", "Cruise.", "Glide.", "Coast.", "Roll.", "Stride.",
    "Rhythm.", "Tempo.", "Sail.", "Onward.", "Tessellate.", "Groove.", "Smooth.", "Continue.",
    "Next.", "Zone.", "Velocity.", "Resume.",
)
REMIND_WELLNESS_CLOSERS = (
    "Remember to hydrate your body.", "Unclench your jaw.", "Drop your shoulders.",
    "Blink — look 20 feet away.", "Stretch your wrists.", "Stand up, quick stretch.",
    "Take one slow breath.", "Rest your eyes a moment.", "Refill your water.",
    "Be kind to future-you — leave a note.",
)
REMIND_WELLNESS_EVERY = 10  # one wellness tidbit per this many firings; all the rest are flow words


def _remind_closer(n: int) -> str:
    """The rotating sign-off for the n-th (0-based) /clear nudge firing. Every REMIND_WELLNESS_EVERY-th
    firing is a wellness tidbit; all others cycle the flow pool. A pure function of n — deterministic,
    no clock/RNG — so it's reproducible and unit-testable. flow_idx = n minus the wellness firings
    already consumed, which keeps the flow cycle gap-free and even."""
    if n % REMIND_WELLNESS_EVERY == REMIND_WELLNESS_EVERY - 1:
        return REMIND_WELLNESS_CLOSERS[(n // REMIND_WELLNESS_EVERY) % len(REMIND_WELLNESS_CLOSERS)]
    flow_idx = n - (n // REMIND_WELLNESS_EVERY)
    return REMIND_FLOW_CLOSERS[flow_idx % len(REMIND_FLOW_CLOSERS)]


def _next_remind_closer(ctx) -> str:
    """Read-increment-persist the per-project nudge fire counter and return the closer for it, so the
    sign-off advances once per firing across sessions. Degrades to the first flow word if metrics
    can't be read/written (a nudge must never crash on a bad metrics file)."""
    try:
        m = _load_metrics(ctx)
        n = int(m.get("remind_fire_count", 0) or 0)
        closer = _remind_closer(n)
        m["remind_fire_count"] = n + 1
        _save_metrics(ctx, m)
        return closer
    except Exception:
        return REMIND_FLOW_CLOSERS[0]


def _remind_line(tokens, clear_cmd: str, closer: str) -> str:
    """The single, uniform /clear nudge line — used on every channel (stdout + the GUI modal). Names
    the stale-token weight (the precise live count; omitted when no count is known), the safe action,
    the no-rehydrate guarantee, and the rotating sign-off. The ~ marks it as an estimate, but the
    full digits are shown — never rounded."""
    weight = f"Carrying ~{tokens:,} stale tokens. " if tokens else ""
    return (f"🏹 Celeborn —> {weight}Safe to {clear_cmd} — state is saved, "
            f"nothing to re-explain. {closer}")


def cmd_remind(args):
    """Print a reassuring, Tolkien-voiced checkpoint-and-renew reminder.

    Portable across coding systems: the host supplies the live context size via `--tokens` (it is
    the only part the CLI cannot observe itself). With `--last`, the reminder stays silent unless a
    new `--every`-sized milestone has been crossed, so a host can call it on every render/turn and
    only surface it once per increment. The "button" is whatever clear action the host shows
    alongside it; `--clear-cmd` sets the wording.
    """
    ctx = require_context(args)
    cfg = load_config(ctx)
    every = args.every if args.every and args.every > 0 else 100_000
    tokens = args.tokens
    last = args.last

    # --transcript: read the live context size straight from the Claude Code transcript (the real
    # number). --auto: use Celeborn's own rolling estimate. Both persist to metrics so the
    # host hook can stay stateless and `status`/`metrics` reflect the latest reading.
    metrics = None
    track = args.auto or bool(getattr(args, "transcript", None))
    if getattr(args, "transcript", None):
        metrics = _load_metrics(ctx)
        tokens = _estimate_transcript_tokens(Path(args.transcript), cfg["chars_per_token"])
        metrics["context_estimate"] = tokens
        last = metrics.get("last_remind_estimate", 0)
        _save_metrics(ctx, metrics)  # record the reading even if we stay silent
    elif args.auto:
        metrics = _load_metrics(ctx)
        tokens = metrics.get("context_estimate", 0)
        last = metrics.get("last_remind_estimate", 0)

    # Silence unless a fresh milestone was crossed (vs. the last-reminded token count).
    if tokens is not None and last is not None and not args.force:
        if tokens // every == max(0, last) // every:
            return

    # We're going to speak — remember where, so we stay silent until the next band.
    if track and metrics is not None:
        metrics["last_remind_estimate"] = tokens or 0
        _save_metrics(ctx, metrics)

    clear_cmd = args.clear_cmd or "/clear"

    # One rotating sign-off per firing. Advancing the counter here means it ticks once per nudge.
    line = _remind_line(tokens, clear_cmd, _next_remind_closer(ctx))

    print(line)


# --------------------------------------------------------------------------- pre-compaction panic-save


def _panic_stamp() -> str:
    """Filesystem-safe local timestamp for a panic-save dir name. Lexical order == chronological."""
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _panic_snapshots(ctx: Path) -> list:
    """Existing panic-save dirs under .context/.panic/, oldest first (names sort chronologically)."""
    base = ctx / PANIC_DIR
    if not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name)


def _do_panic_save(ctx: Path, reason: str = "manual", session=None, keep: int = PANIC_KEEP) -> dict:
    """Copy whichever PANIC_SAVE_FILES exist into .context/.panic/<stamp>/ (subpaths mirrored), write a
    meta.json, FIFO-prune to `keep`, and return {stamp, dir, files, reason}. This is the deterministic
    safety net behind the "🏹 Celeborn saved your session" moment: a restore point that survives a
    compaction regardless of whether the model freshened the Hot tier. Best-effort per file — an
    unreadable file is skipped, never fatal."""
    import shutil
    stamp = _panic_stamp()
    dest = ctx / PANIC_DIR / stamp
    # De-collide: two panic-saves in the same second (e.g. a burst of compaction events) would
    # otherwise share a stamp dir and silently clobber each other's restore point. Suffix -2, -3, …
    # which still sorts chronologically (the bare stamp is a prefix, so it sorts first).
    if dest.exists():
        n = 2
        while (ctx / PANIC_DIR / f"{stamp}-{n}").exists():
            n += 1
        stamp = f"{stamp}-{n}"
        dest = ctx / PANIC_DIR / stamp
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for rel in PANIC_SAVE_FILES:
        src = ctx / rel
        if not src.is_file():
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, out)
            saved.append(rel)
        except OSError:
            pass
    meta = {"schema": "celeborn-panic/1", "stamp": stamp, "reason": reason,
            "session": session, "at": now_iso(), "files": saved}
    try:
        (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    except OSError:
        pass
    if keep and keep > 0:                       # FIFO: keep the most recent `keep` (this one included)
        for old in _panic_snapshots(ctx)[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
    return {"stamp": stamp, "dir": str(dest), "files": saved, "reason": reason}


def _panic_save_line(info: dict) -> str:
    """User/agent-visible panic-save confirmation (t36 felt moment, t43 copy). All counts/paths from `info`."""
    files = info.get("files") or []
    n = len(files)
    stamp = info.get("stamp") or ""
    snap_path = f".context/{PANIC_DIR}/{stamp}/" if stamp else f".context/{PANIC_DIR}/"
    file_word = "file" if n == 1 else "files"
    return (
        f"Model context window overflow. Celeborn saved you — {n} {file_word} snapshotted to "
        f"{snap_path} (restore: `celeborn restore`). Nothing lost to context compaction. "
        f"To avoid last-minute saves, `/clear` before the context window limit. "
        f"[read more: {PANIC_READ_MORE}]"
    )


def cmd_panic_save(args):
    """`celeborn panic-save` — snapshot the authored tiers to a restore point and print a visible
    "🏹 Celeborn saved your session" line. Runs automatically from the PreCompact hook (compaction
    imminent) and is callable by hand. The continuous Stop-hook `capture` already salvages live
    transcript work every turn; this adds the deterministic, restorable snapshot + the felt moment."""
    ctx = require_context(args)
    info = _do_panic_save(ctx, reason=getattr(args, "reason", None) or "manual",
                          session=getattr(args, "session", None),
                          keep=getattr(args, "keep", None) or PANIC_KEEP)
    m = _load_metrics(ctx)
    m["panic_saves"] = int(m.get("panic_saves", 0) or 0) + 1
    _save_metrics(ctx, m)
    if getattr(args, "json", False):
        print(json.dumps(info, indent=2))
        return
    if not getattr(args, "quiet", False):
        print(_panic_save_line(info))


def cmd_restore(args):
    """`celeborn restore` — bring back a panic-save snapshot. Default restores the most recent; --from
    <stamp> picks one; --list shows what's available. The current files are themselves panic-saved
    first (reason "pre-restore"), so a restore is always reversible."""
    ctx = require_context(args)
    snaps = _panic_snapshots(ctx)
    if getattr(args, "list", False):
        if not snaps:
            print("No panic-saves yet.")
            return
        for p in reversed(snaps):               # newest first
            meta = {}
            try:
                meta = json.loads((p / "meta.json").read_text())
            except (OSError, ValueError):
                pass
            print(f"  {p.name}  ({meta.get('reason', '?')}, {len(meta.get('files', []))} files, "
                  f"{meta.get('at', '?')})")
        return
    if not snaps:
        die("no panic-saves to restore from.")
    want = getattr(args, "from_", None)
    if want:
        chosen = next((p for p in snaps if p.name == want), None)
        if chosen is None:
            die(f"no panic-save named {want!r}. Try `celeborn restore --list`.")
    else:
        chosen = snaps[-1]                       # most recent
    # Read the chosen snapshot into memory BEFORE backing up current state — the pre-restore save
    # FIFO-prunes, and could otherwise delete `chosen` out from under us.
    payload = {}
    for rel in PANIC_SAVE_FILES:
        src = chosen / rel
        if src.is_file():
            try:
                payload[rel] = src.read_bytes()
            except OSError:
                pass
    _do_panic_save(ctx, reason="pre-restore", keep=getattr(args, "keep", None) or PANIC_KEEP)
    restored = []
    for rel, data in payload.items():
        dst = ctx / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.write_bytes(data)
            restored.append(rel)
        except OSError:
            pass
    if getattr(args, "json", False):
        print(json.dumps({"restored_from": chosen.name, "files": restored}, indent=2))
        return
    print(f"🏹 Restored {len(restored)} file(s) from .context/{PANIC_DIR}/{chosen.name}/ "
          f"(current state backed up first — `celeborn restore --list`).")


# --------------------------------------------------------------------------- version / update check

GITHUB_REPO = "cloud-dancer-labs/celeborn"  # where Celeborn looks back to for updates


def _local_version() -> str:
    """Celeborn's version. Prefers installed package metadata (the only source that exists for a
    pip/uv install); falls back to the repo's pyproject.toml in a source checkout (regex — no toml dep)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("celeborn")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    # Frozen binaries have no package metadata or source tree; the build bakes a VERSION file
    # alongside the bundled data (celeborn_refs/VERSION).
    try:
        baked = (DATA_DIR / "VERSION").read_text().strip()
        if baked:
            return baked
    except OSError:
        pass
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', (REPO_ROOT / "pyproject.toml").read_text(), re.M)
        return m.group(1) if m else "unknown"
    except OSError:
        return "unknown"


def _git_head(root: Path):
    """Short+full HEAD sha if `root` is a git checkout, else None. Never raises."""
    if not (root / ".git").exists():
        return None
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def _fetch_url(url: str, accept: str = "application/vnd.github+json") -> str:
    """GET a URL and return the body text. Lazy-imports urllib so the core stays import-light.
    Tests monkeypatch this. Raises urllib/OS errors on failure (caller handles offline)."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "celeborn-update-check", "Accept": accept})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode()


def cmd_version(args):
    """Print Celeborn's version (and git HEAD). With --check, look back at GitHub to see whether a
    newer Celeborn is available. The plain form is offline; only --check touches the network."""
    local_v = _local_version()
    head = _git_head(REPO_ROOT)
    line = f"Celeborn {local_v}"
    if head:
        # Source checkout: surface the repo path (used by the `git pull` update hint below).
        line += f" (git {head[:8]})  ·  {REPO_ROOT}"
    print(line)
    if not getattr(args, "check", False):
        return

    import json as _json
    import urllib.error
    try:
        if head:
            # git checkout: compare local HEAD against origin/main via the GitHub API.
            latest = _json.loads(_fetch_url(f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"))
            remote = latest.get("sha", "")
            if not remote:
                warn("update check: couldn't read latest commit from GitHub."); return
            if remote == head:
                ok("up to date with origin/main."); return
            behind = None
            try:
                cmp = _json.loads(_fetch_url(
                    f"https://api.github.com/repos/{GITHUB_REPO}/compare/{head}...main"))
                if cmp.get("status") == "identical":
                    ok("up to date with origin/main."); return
                behind = cmp.get("ahead_by")  # commits on main not in local HEAD
            except (urllib.error.URLError, OSError, ValueError):
                pass
            n = f"{behind} commit(s) behind" if isinstance(behind, int) else "behind"
            warn(f"a newer Celeborn is available — {n} origin/main (latest {remote[:8]}).")
            print(f"  Update:  git -C \"{REPO_ROOT}\" pull")
        else:
            # non-git install (e.g. pip): compare the version in main's pyproject.toml.
            pp = _fetch_url(f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/pyproject.toml",
                            accept="text/plain")
            m = re.search(r'^version\s*=\s*"([^"]+)"', pp, re.M)
            remote_v = m.group(1) if m else None
            if not remote_v:
                warn("update check: couldn't read latest version from GitHub."); return
            if remote_v == local_v:
                ok(f"up to date (latest {remote_v})."); return
            warn(f"a newer Celeborn may be available: {remote_v} (you have {local_v}).")
            print(f"  Update:  reinstall from https://github.com/{GITHUB_REPO}")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        warn(f"update check skipped — couldn't reach GitHub ({e}).")


def cmd_about(args):
    """Print Celeborn Code's identity + canonical links, and disambiguate from the same-named
    projects (Apache Celeborn; the frkngksl/Celeborn Windows tool). This exists so an agent that
    installed 'celeborn' mid-conversation can run `celeborn about` and confirm it grabbed the
    coding-agent context substrate — not one of the namesakes. Offline; no network."""
    local_v = _local_version()
    print(f"Celeborn Code {local_v} — a long-term context substrate for coding agents.")
    print("  Memory on disk for AI coding agents (Claude Code, Codex, Grok): survives compaction,")
    print("  keeps its place across sessions, saves tokens.")
    print()
    print("  Install:  uv tool install celeborn   (or: pip install celeborn)")
    print("  PyPI:     https://pypi.org/project/celeborn/")
    print(f"  Source:   https://github.com/{GITHUB_REPO}")
    print("  Home:     https://celeborn.thot.ai")
    print()
    print("  This is NOT the other projects named 'Celeborn':")
    print("    · Apache Celeborn  — a Spark/Flink remote shuffle service (big-data infra). Unrelated.")
    print("    · frkngksl/Celeborn — a Windows API-unhooking research tool. Unrelated.")
    print("  If you were told to install 'Celeborn' for coding-agent memory, this is the right one.")


# --------------------------------------------------------------------------- install integrity (detection)

INTEGRITY_MANIFEST = "integrity.json"     # shipped inside DATA_DIR (celeborn_refs) by the release build
INTEGRITY_SCHEMA = "celeborn-integrity/1"
# The core modules whose bytes define behavior. If a user edits these in place, the install no longer
# matches the published release — that is what we DETECT (we cannot, and do not try to, prevent it), so
# a "I edited celeborn.py and it broke" situation self-reports instead of becoming a confused bug report.
INTEGRITY_MODULES = ("celeborn.py", "celeborn_sync.py", "celeborn_jira.py")


def _integrity_manifest_path() -> Path:
    """Where the per-version checksum manifest lives. The default ships inside the data package so
    pip/uv/wheel installs carry it; CELEBORN_INTEGRITY_MANIFEST overrides it (used by the release build
    that generates it, and by tests)."""
    import os
    override = os.environ.get("CELEBORN_INTEGRITY_MANIFEST")
    return Path(override) if override else (DATA_DIR / INTEGRITY_MANIFEST)


def _sha256_file(p: Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _compute_integrity(modules: tuple = INTEGRITY_MODULES) -> dict:
    """sha256 of each shipped core module that exists beside this file, keyed by filename."""
    out = {}
    for name in modules:
        digest = _sha256_file(SCRIPT_DIR / name)
        if digest:
            out[name] = digest
    return out


def _load_integrity_manifest() -> dict | None:
    p = _integrity_manifest_path()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def integrity_status() -> dict:
    """Detection, not prevention. Returns
        {'state': 'ok'|'modified'|'unverified', 'modified': [filenames], 'reason': str}.
    'unverified' = no manifest shipped (source/dev/editable checkout) or a version mismatch — we stay
    SILENT in that case so a contributor editing the tree is never nagged. Only a released install
    (manifest present AND version match) can ever report 'modified'."""
    man = _load_integrity_manifest()
    files = man.get("files") if isinstance(man, dict) else None
    if not isinstance(files, dict) or not files:
        return {"state": "unverified", "modified": [], "reason": "no integrity manifest (source/dev install)"}
    if man.get("version") and man["version"] != _local_version():
        return {"state": "unverified", "modified": [], "reason": "manifest is for a different version"}
    current = _compute_integrity(tuple(files.keys()))
    modified = sorted(name for name, digest in files.items() if current.get(name) != digest)
    if modified:
        return {"state": "modified", "modified": modified, "reason": ""}
    return {"state": "ok", "modified": [], "reason": ""}


def _integrity_notice() -> str:
    """One-line notice for the SessionStart Orient load when the install has been modified. Empty
    string when ok/unverified. Best-effort + never raises (the hook must degrade to silence)."""
    try:
        st = integrity_status()
    except Exception:
        return ""
    if st["state"] != "modified":
        return ""
    return ("⚠ Celeborn integrity: modified install detected (" + ", ".join(st["modified"]) + ") — run "
            "`celeborn doctor` for details. Reinstall to reset; local edits are unsupported (submit a PR).")


def cmd_integrity(args):
    """`celeborn integrity` — verify the installed core modules match the published per-version
    checksum manifest. `--write` (re)generates the manifest from the current files — a release/build
    step, not for end users. Detection only: a mismatch means the install was edited in place."""
    if getattr(args, "write", False):
        man = {
            "schema": INTEGRITY_SCHEMA,
            "version": _local_version(),
            "generated_at": now_iso(),
            "files": _compute_integrity(),
        }
        p = _integrity_manifest_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(man, indent=2) + "\n")
        ok(f"wrote integrity manifest ({len(man['files'])} file(s)) → {p}")
        return
    st = integrity_status()
    if st["state"] == "ok":
        ok("install integrity verified — core modules match the published release.")
    elif st["state"] == "unverified":
        info(f"integrity check skipped — {st['reason']} (nothing to verify).")
    else:
        warn("modified install detected — these core module(s) differ from the published release:")
        for name in st["modified"]:
            print(f"      {name}")
        print("  Fix:  reinstall to reset (`uv tool install --force celeborn` / `pipx reinstall celeborn`).")
        print("        Local edits to the installed CLI are unsupported — submit a PR instead.")
        sys.exit(1)


# --------------------------------------------------------------------------- skill advisor (t70)
#
# A throughput + quality layer (sibling of `_integrity_notice`). It detects FRICTION (the human is a
# bottleneck) and recommends the harness's fix. The engine is HARNESS-NEUTRAL: it speaks in canonical
# signals + neutral "intents" and never names a slash command or `.claude/` path itself. Each harness
# is a thin HarnessAdapter that (a) normalizes its raw friction into canonical signals — the same way
# `_TOOL_MAP` normalizes tool names — and (b) renders an intent into that harness's idiom + channel.
# Claude is the implicit default adapter that used to be hard-coded; an unknown harness degrades to the
# NeutralAdapter (plain instruction + the literal `celeborn` command), never an error. Grok/Codex
# adapters are future subclasses living in their own bridges (grok/, codex/) — core stays untouched.

# A neutral recommendation: a canonical `trigger` signal → an intent, with a harness-agnostic fallback
# string so an unknown harness still gets the advice. Phase 1 ships exactly one.
ADVISOR_INTENTS = {
    "reduce-permission-friction": {
        "trigger": "permission-friction",
        "summary": "Repeated permission approvals are interrupting the loop.",
        "auto_actionable": True,
        "neutral": ("Permission friction: {count} over-specific allow-rules that never re-match. "
                    "Run `celeborn permissions --suggest` to collapse them into reusable wildcard rules."),
    },
    # Phase 3 — portable quality recommendations. Celeborn can't RUN a review skill, so these
    # *recommend* one: the `neutral` text is a self-contained checklist (the portable "prompt pack")
    # for harnesses without the skill; the ClaudeAdapter render points at the matching slash command.
    "security-review-changes": {
        "trigger": "sensitive-changes",
        "summary": "Uncommitted changes touch security-sensitive paths — do a security pass.",
        "auto_actionable": False,
        "neutral": ("Security review: uncommitted changes touch sensitive paths ({files}). Before "
                    "finishing, review them for authn/authz gaps, secret handling, input validation, and "
                    "injection/SSRF/path-traversal."),
    },
    "review-changes": {
        "trigger": "uncommitted-changes",
        "summary": "Substantial uncommitted changes — review before calling it done.",
        "auto_actionable": False,
        "neutral": ("Code review: {count} changed code file(s) uncommitted. Before finishing, read each "
                    "hunk for correctness, edge cases, error paths, and leftover debug/TODOs — then verify "
                    "the behavior end to end, don't just trust that it compiles."),
    },
    # Phase 4 — throughput / autonomy. #1 auto-fires on a large changeset; #2/#3 are on-demand
    # (they need conversation judgment the CLI can't see) — surfaced only via `celeborn advise
    # --throughput`, never auto-nagged. Renders map to each harness's equivalent, else stay generic.
    "parallelize-large-changeset": {
        "trigger": "large-changeset",
        "summary": "Large changeset — parallelize the review instead of one linear pass.",
        "auto_actionable": False,
        "neutral": ("Large changeset ({count} code files): split the review across independent chunks / "
                    "parallel workers rather than one linear pass — it's faster and catches more."),
    },
    "spawn-tangent": {
        "trigger": None,
        "on_demand": True,
        "summary": "Drifted onto unrelated work? Peel it into its own task.",
        "auto_actionable": False,
        "neutral": ("Working on something unrelated to the current task? Split it into its own "
                    "task/session so this thread stays focused and reviewable."),
    },
    "unattended-run": {
        "trigger": None,
        "on_demand": True,
        "summary": "Long unattended run? Drive it in checkpointed batches.",
        "auto_actionable": False,
        "neutral": ("For a long unattended run, drive it in batches and checkpoint state between them "
                    "so a restart resumes instead of redoing work."),
    },
}


def _signal_to_intent(sig: dict) -> str | None:
    """Map a canonical friction signal to the highest-value intent it triggers."""
    trig = sig.get("signal")
    for iid, spec in ADVISOR_INTENTS.items():
        if spec["trigger"] == trig:
            return iid
    return None


# The ONLY families a permission rule is auto-widened into: read-only inspection commands + the
# project's own trusted Celeborn CLI + the test runners. Each entry is a prefix to PRESERVE; the
# generalized rule is `Bash(<prefix>*)`. A literal whose command doesn't start with one of these is
# NEVER widened (it stays verbatim and is tallied as a "skipped bottleneck"). This deliberately
# mirrors the hand-written .claude/settings.json the user produced via `/fewer-permission-prompts`.
_SAFE_BASH_PREFIXES = (
    "python3 scripts/celeborn.py ",
    "python scripts/celeborn.py ",
    "scripts/celeborn.py ",
    "celeborn ",
    "sed -n ",
    "grep ",
    "git --no-pager diff ",
    "git --no-pager log ",
    "git --no-pager show ",
    "git diff ",
    "git log ",
    "git show ",
    "PYTHONPATH=scripts python3 -m unittest ",
    "python3 -m unittest ",
    "python -m unittest ",
    "python3 -m pytest ",
    "python -m pytest ",
    "python3 -m py_compile ",
)


def _parse_bash_rule(rule: str) -> str | None:
    """`'Bash(grep -n foo)'` → `'grep -n foo'`; a non-Bash permission (MCP/tool name) → None."""
    if rule.startswith("Bash(") and rule.endswith(")"):
        return rule[5:-1]
    return None


def _match_safe_family(inner: str) -> str | None:
    """The generalized wildcard rule that SAFELY subsumes this literal command, or None when the
    command falls outside the read-only/trusted allow-set. Longest prefix wins so a specific runner
    (`python3 -m unittest `) beats a shorter accidental prefix."""
    for pre in sorted(_SAFE_BASH_PREFIXES, key=len, reverse=True):
        if inner.startswith(pre):
            return f"Bash({pre}*)"
    return None


def _bottleneck_key(inner: str) -> str:
    """A short family label for a skipped (un-widenable) literal — the leading command, plus the
    subcommand when the head is a multiplexer (git/python/npm/…). Used to tally remaining friction."""
    toks = inner.split()
    if not toks:
        return "?"
    head = toks[0]
    if head in ("git", "python3", "python", "npm", "npx", "uv", "pip", "pip3",
                "cargo", "go", "docker", "make") and len(toks) > 1:
        return f"{head} {toks[1]}"
    return head


def _count_literal_bash_rules(allow: list) -> int:
    """How many allow-rules are over-specific Bash literals (a `Bash(cmd args)` that does NOT end in
    `*`). Wildcard rules and non-Bash permissions don't count."""
    n = 0
    for rule in allow or []:
        inner = _parse_bash_rule(rule) if isinstance(rule, str) else None
        if inner is not None and not inner.rstrip().endswith("*"):
            n += 1
    return n


def _count_generalizable_bash_rules(allow: list) -> int:
    """The ACTIONABLE friction signal: literals this tool can SAFELY collapse into a wildcard. The
    un-widenable bottlenecks (curl/rm/git-commit/…) are deliberately excluded — so once the safe rules
    are applied this drops to 0 and the advisor goes quiet, even though raw literals still remain."""
    n = 0
    for rule in allow or []:
        inner = _parse_bash_rule(rule) if isinstance(rule, str) else None
        if inner is not None and not inner.rstrip().endswith("*") and _match_safe_family(inner):
            n += 1
    return n


def _generalize_allow(rules: list) -> tuple[list, int, dict]:
    """Collapse an allow-list. Returns (new_rules, generalized_count, skipped_ledger).

    Safe-family literals are replaced by one shared wildcard each; already-general rules and non-Bash
    permissions pass through untouched; literals outside the safe set are KEPT VERBATIM and tallied by
    family in `skipped` (the bottlenecks Celeborn can't safely remove). Order: the new generalizations
    first, then everything preserved — deduped, original order otherwise kept."""
    family_rules: list = []      # the wildcards we synthesize, in first-seen order
    preserved: list = []         # general rules + non-Bash perms + skipped literals
    seen_family: set = set()
    generalized = 0
    skipped: dict = {}
    for rule in rules or []:
        inner = _parse_bash_rule(rule) if isinstance(rule, str) else None
        if inner is None or inner.rstrip().endswith("*"):
            preserved.append(rule)               # non-Bash perm or already a wildcard → keep
            continue
        fam = _match_safe_family(inner)
        if fam:
            generalized += 1
            if fam not in seen_family:
                seen_family.add(fam)
                family_rules.append(fam)
        else:
            skipped[_bottleneck_key(inner)] = skipped.get(_bottleneck_key(inner), 0) + 1
            preserved.append(rule)               # un-widenable literal → keep verbatim
    new_rules: list = []
    for r in family_rules + preserved:
        if r not in new_rules:
            new_rules.append(r)
    return new_rules, generalized, skipped


# Phase 3 — harness-agnostic quality signals derived from the working-tree diff.
_CODE_SUFFIXES = (".py", ".ts", ".tsx", ".js", ".jsx", ".sh", ".sql", ".go", ".rs", ".rb", ".java",
                  ".kt", ".c", ".cc", ".cpp", ".h", ".hpp", ".php", ".swift", ".scala", ".cs")


def _changed_files(root: Path) -> list:
    """Repo-relative paths with uncommitted changes (staged + unstaged + untracked) via
    `git status --porcelain`. Returns [] outside a git checkout or on any error — a non-git project
    is never nagged. Rename records resolve to the new path; git-quoted paths are de-quoted best-effort."""
    if not (root / ".git").exists():
        return []
    import subprocess
    try:
        # --untracked-files=all expands wholly-new directories to individual files (git otherwise
        # collapses them to `dir/`, under-counting the review heuristic); .gitignore is still honored.
        r = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
                           capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:                  # rename/copy record: keep the destination path
            path = path.split(" -> ", 1)[1]
        out.append(path.strip().strip('"'))
    return out


def _is_code_file(rel: str) -> bool:
    return rel.lower().endswith(_CODE_SUFFIXES)


def _is_sensitive(rel: str, globs: list) -> bool:
    """True if a repo-relative path matches any sensitive glob — tested against both the full path and
    the bare filename so `*auth*` catches `lib/auth.ts` and `supabase/**` catches a nested file."""
    import fnmatch
    rl = (rel or "").replace("\\", "/")
    base = rl.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(rl, g) or fnmatch.fnmatch(base, g) for g in (globs or []))


def _change_review_signals(ctx: Path) -> list:
    """Harness-agnostic Phase-3 signals from the working tree: changes touching sensitive paths
    (→ security review) and a substantial count of changed code files (→ code review + verify).
    Best-effort and read-only; [] outside a git repo. Sensitive is listed first so it wins the single
    orient-nudge slot when both fire."""
    try:
        acfg = _advisor_config(ctx)
    except Exception:
        return []
    files = _changed_files(ctx.parent)
    if not files:
        return []
    sensitive = sorted({f for f in files if _is_sensitive(f, acfg.get("sensitive_globs") or [])})
    code = sorted({f for f in files if _is_code_file(f)})
    sigs = []
    if sensitive:
        shown = ", ".join(sensitive[:4]) + ("…" if len(sensitive) > 4 else "")
        sigs.append({"signal": "sensitive-changes", "count": len(sensitive), "files": shown})
    if len(code) >= int(acfg.get("review_min_files", 3)):
        sigs.append({"signal": "uncommitted-changes", "count": len(code)})
    if len(code) >= int(acfg.get("parallelize_min_files", 12)):
        sigs.append({"signal": "large-changeset", "count": len(code)})  # Phase 4: fan out the review
    return sigs


class HarnessAdapter:
    """The harness seam. The base class is also the NEUTRAL fallback: it produces the harness-agnostic
    quality signals (Phase 3), exposes no permission target, and renders an intent as the neutral
    instruction. Concrete adapters override for their host. Core calls only these — never a
    harness-specific path/name."""

    name = "neutral"

    def friction_signals(self, ctx: Path, session: str | None = None) -> list:
        # Quality signals are host-independent (a git diff is a git diff), so every harness — including
        # the neutral fallback Grok/Codex ride — gets them. Permission friction is added per-adapter.
        return _change_review_signals(ctx)

    def permission_target(self, ctx: Path, shared: bool = False) -> tuple:
        return (None, None)

    def render(self, intent: str, signal: dict | None = None) -> tuple:
        spec = ADVISOR_INTENTS.get(intent, {})
        sig = signal or {}
        text = (spec.get("neutral") or spec.get("summary") or "").format(
            count=sig.get("count", "several"), files=sig.get("files", "several files"))
        return (text, "instruction")

    def inject(self, text: str, channel: str) -> str:
        # Neutral hosts have no structured channel — the text is returned for the caller to place.
        return text

    def quality_hook_target(self, ctx: Path, shared: bool = True) -> tuple:
        # No structured PostToolUse/Stop hooks on a neutral host — fall back to an AGENTS.md instruction
        # so a Codex/Grok-style harness that auto-loads it still gets the "run tests after editing" rule.
        return ("agents-md", ctx.parent / "AGENTS.md")


class ClaudeAdapter(HarnessAdapter):
    """Claude Code: per-command `permissions.allow` in `.claude/settings*.json`, slash-command skills,
    and the SessionStart `hookSpecificOutput` orient channel. This is the path that was implicit in
    core before t70 — now an explicit adapter so the engine no longer hard-codes Claude."""

    name = "claude"

    def permission_target(self, ctx: Path, shared: bool = False) -> tuple:
        base = ctx.parent / ".claude"
        fname = "settings.json" if shared else "settings.local.json"
        return (base / fname, "per-command-allow")

    def quality_hook_target(self, ctx: Path, shared: bool = True) -> tuple:
        # Quality gates ride PostToolUse + Stop in settings.json — shared by default (they help every
        # contributor on the checkout); `--local` writes the personal settings.local.json instead.
        base = ctx.parent / ".claude"
        fname = "settings.json" if shared else "settings.local.json"
        return ("hooks-json", base / fname)

    def _allow(self, ctx: Path) -> list:
        out: list = []
        for shared in (False, True):
            target, _how = self.permission_target(ctx, shared=shared)
            if target and target.is_file():
                try:
                    data = json.loads(target.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                out.extend((data.get("permissions") or {}).get("allow") or [])
        return out

    def friction_signals(self, ctx: Path, session: str | None = None) -> list:
        # Host-independent quality signals (Phase 3) first, then Claude's own permission signal. Count
        # only the literals this tool can collapse — the moment the safe rules apply the signal clears,
        # even though un-widenable bottlenecks (curl/rm/…) still remain.
        sigs = list(super().friction_signals(ctx, session))
        n = _count_generalizable_bash_rules(self._allow(ctx))
        thresh = _advisor_config(ctx)["permission_bloat_min"]
        if n >= thresh:
            target, _how = self.permission_target(ctx)
            sigs.append({"signal": "permission-friction", "count": n, "file": str(target)})
        return sigs

    def render(self, intent: str, signal: dict | None = None) -> tuple:
        sig = signal or {}
        if intent == "reduce-permission-friction":
            n = sig.get("count")
            cnt = f"{n} " if n else ""
            text = (f"🏹 Celeborn advisor —> {cnt}repeated approvals can be auto-generalized into reusable "
                    f"wildcard rules. Fix: run `celeborn permissions --suggest` then `--apply` (or the "
                    f"`/fewer-permission-prompts` skill).")
            return (text, "orient")
        if intent == "security-review-changes":
            files = sig.get("files") or "sensitive paths"
            text = (f"🏹 Celeborn advisor —> uncommitted changes touch sensitive paths ({files}). Run the "
                    f"`/security-review` skill before finishing — check authn/authz, secret handling, "
                    f"input validation, and injection/SSRF.")
            return (text, "orient")
        if intent == "review-changes":
            n = sig.get("count")
            cnt = f"{n} " if n else ""
            text = (f"🏹 Celeborn advisor —> {cnt}uncommitted code file(s) — run `/code-review` then "
                    f"`/verify` before calling it done (correctness + edge cases, then confirm behavior).")
            return (text, "orient")
        if intent == "parallelize-large-changeset":
            n = sig.get("count")
            cnt = f"{n} " if n else ""
            text = (f"🏹 Celeborn advisor —> large changeset ({cnt}files) — fan the review out: spawn "
                    f"subagents (the Task tool) per area, or a Workflow for staged review, instead of one "
                    f"linear pass.")
            return (text, "orient")
        if intent == "spawn-tangent":
            text = ("🏹 Celeborn advisor —> off on a tangent? Use spawn_task (or a fresh session) to peel "
                    "the unrelated fix into its own thread — keeps this change focused and reviewable.")
            return (text, "instruction")
        if intent == "unattended-run":
            text = ("🏹 Celeborn advisor —> long unattended run ahead? `/loop` repeats a step on an "
                    "interval; `/elves` runs multi-batch autonomous development — both checkpoint via "
                    "Celeborn so a restart resumes.")
            return (text, "instruction")
        return super().render(intent, signal)


# --------------------------------------------------------------------------- Grok / Codex adapters
#
# Grok Build and the OpenAI Codex CLI both ride the NEUTRAL surface for quality (a git diff is a git
# diff) and orient via a pending file rather than a structured SessionStart channel — so neither has
# Claude's slash commands. `_branded_quality_render` is the shared advisor voice for both: the same
# branded `🏹 Celeborn advisor —>` text Claude emits, minus the `/code-review`-style command pointers.

def _branded_quality_render(intent: str, signal: dict | None) -> tuple | None:
    """The advisor render shared by harnesses with no slash-command surface (Grok, Codex). Returns
    (text, channel) for the quality + throughput intents, else None so the caller can fall through to
    a harness-specific intent (e.g. Codex's permission hint) or the neutral base render."""
    sig = signal or {}
    if intent == "security-review-changes":
        files = sig.get("files") or "sensitive paths"
        return ((f"🏹 Celeborn advisor —> uncommitted changes touch sensitive paths ({files}). Before "
                 f"finishing, do a security pass — authn/authz gaps, secret handling, input validation, "
                 f"and injection/SSRF/path-traversal."), "orient")
    if intent == "review-changes":
        n = sig.get("count")
        cnt = f"{n} " if n else ""
        return ((f"🏹 Celeborn advisor —> {cnt}uncommitted code file(s) — review each hunk for correctness, "
                 f"edge cases, and error paths, then verify the behavior end to end before calling it done."),
                "orient")
    if intent == "parallelize-large-changeset":
        n = sig.get("count")
        cnt = f"{n} " if n else ""
        return ((f"🏹 Celeborn advisor —> large changeset ({cnt}files) — split the review across independent "
                 f"chunks (a separate session/agent per area) instead of one linear pass."), "orient")
    if intent == "spawn-tangent":
        return (("🏹 Celeborn advisor —> off on a tangent? Peel the unrelated fix into its own Celeborn "
                 "task/session so this change stays focused and reviewable."), "instruction")
    if intent == "unattended-run":
        return (("🏹 Celeborn advisor —> long unattended run ahead? Drive it in checkpointed batches — "
                 "update `.context` between them so a restart resumes instead of redoing work."), "instruction")
    return None


def _codex_home() -> Path:
    import os
    return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))


def _codex_permission_status(root: Path) -> dict:
    """Read Codex's coarse permission lever from ~/.codex/config.toml WITHOUT a TOML parser (stdlib
    regex, tolerant of any layout): global `approval_policy`/`sandbox_mode`, plus whether THIS project
    is trusted via a `[projects."<root>"]` table with `trust_level = "trusted"`. Mirrors the codex/
    bridge so core's advisor sees the same friction the bridge does."""
    cfg = _codex_home() / "config.toml"
    status = {"config": str(cfg), "exists": cfg.is_file(),
              "approval_policy": None, "sandbox_mode": None, "project_trusted": False}
    if not cfg.is_file():
        return status
    try:
        text = cfg.read_text(encoding="utf-8")
    except OSError:
        return status
    m = re.search(r'(?m)^\s*approval_policy\s*=\s*"([^"]+)"', text)
    if m:
        status["approval_policy"] = m.group(1)
    m = re.search(r'(?m)^\s*sandbox_mode\s*=\s*"([^"]+)"', text)
    if m:
        status["sandbox_mode"] = m.group(1)
    rp = str(root.resolve())
    hdr = re.search(r'(?m)^\s*\[projects\.\"' + re.escape(rp) + r'\"\]\s*$', text)
    if hdr:
        rest = text[hdr.end():]
        nxt = re.search(r'(?m)^\s*\[', rest)
        block = rest[: nxt.start()] if nxt else rest
        if re.search(r'(?m)^\s*trust_level\s*=\s*"trusted"', block):
            status["project_trusted"] = True
    return status


def _codex_interactive(status: dict) -> bool:
    # Default (no config / unset) is interactive `on-request`; `never` is the only non-prompting mode.
    pol = status.get("approval_policy")
    if pol is None:
        return True
    return pol in ("untrusted", "on-request", "on-failure")


class GrokAdapter(HarnessAdapter):
    """Grok Build: project rules auto-load from `.grok/rules/celeborn.md` and orient rides a pending
    file (no structured SessionStart channel). Grok has NO per-command permission allow-list — its
    rules file isn't a lever — so permission_target/friction_signals stay neutral (quality only). The
    advisor renders Grok-flavored guidance (no Claude slash commands)."""

    name = "grok"

    def render(self, intent: str, signal: dict | None = None) -> tuple:
        r = _branded_quality_render(intent, signal)
        return r if r is not None else super().render(intent, signal)


class CodexAdapter(HarnessAdapter):
    """OpenAI Codex CLI: orient rides AGENTS.md + a pending file. The permission lever is COARSE —
    ~/.codex/config.toml (`approval_policy`/`sandbox_mode` + a per-project `[projects."<root>"]`
    trust_level), NOT a per-command allow-list, so `celeborn permissions` declines to generalize it.
    friction_signals flags an interactive, untrusted workspace; render emits the config.toml trust
    hint. Mirrors the codex/ bridge so core's `celeborn advise` is Codex-aware under CELEBORN_HARNESS=codex."""

    name = "codex"

    def permission_target(self, ctx: Path, shared: bool = False) -> tuple:
        # Coarse workspace-trust lever — deliberately NOT "per-command-allow", so cmd_permissions
        # reports it as a coarse lever rather than trying to widen it like Claude's allow-list.
        return (_codex_home() / "config.toml", "workspace-trust")

    def friction_signals(self, ctx: Path, session: str | None = None) -> list:
        sigs = list(super().friction_signals(ctx, session))
        status = _codex_permission_status(ctx.parent)
        if not status["project_trusted"] and _codex_interactive(status):
            sigs.append({"signal": "permission-friction",
                         "approval_policy": status["approval_policy"] or "on-request",
                         "config": status["config"]})
        return sigs

    def render(self, intent: str, signal: dict | None = None) -> tuple:
        sig = signal or {}
        if intent == "reduce-permission-friction":
            cfg = sig.get("config") or str(_codex_home() / "config.toml")
            pol = sig.get("approval_policy") or "on-request"
            text = (f"🏹 Celeborn advisor —> Codex still pauses for approval in this workspace "
                    f"(approval_policy={pol}). Trust it once so Codex stops re-prompting — add a "
                    f'`[projects."<root>"]` table with `trust_level = "trusted"` to {cfg} (or set '
                    f'`approval_policy = "never"` + `sandbox_mode = "workspace-write"` globally). The '
                    f"Codex lever is coarse — a workspace trust flag, not per-command rules like Claude.")
            return (text, "orient")
        r = _branded_quality_render(intent, sig)
        return r if r is not None else super().render(intent, sig)


def active_adapter(ctx: Path | None = None, name: str | None = None) -> HarnessAdapter:
    """Resolve the active harness adapter: explicit `name` > $CELEBORN_HARNESS > rc `harness` >
    'claude'. An unknown name degrades to the neutral base adapter (never raises)."""
    import os
    chosen = (name or os.environ.get("CELEBORN_HARNESS") or "").strip().lower()
    if not chosen and ctx is not None:
        try:
            chosen = str(load_config(ctx).get("harness") or "").strip().lower()
        except Exception:
            chosen = ""
    if not chosen:
        chosen = "claude"
    return {"claude": ClaudeAdapter, "grok": GrokAdapter, "codex": CodexAdapter,
            "neutral": HarnessAdapter}.get(chosen, HarnessAdapter)()


def _advisor_notice(ctx: Path, session: str | None = None) -> str:
    """One-line SessionStart recommendation (sibling of `_integrity_notice`). Emits at most
    `advisor.max_per_session` nudges per session (default 1) — throttled via the advisor metrics block
    so the same nudge never repeats turn after turn — and skips any intent the user has `--dismiss`ed.
    Best-effort + never raises (the hook must degrade to silence)."""
    try:
        acfg = _advisor_config(ctx)
        if not acfg["enabled"]:
            return ""
        m = _load_metrics(ctx)
        adv = m.get("advisor") or {}
        sess = session or ""
        count = int(adv.get("notices_this_session", 0) or 0) if sess == (adv.get("last_notice_session") or "") else 0
        if count >= acfg["max_per_session"]:
            return ""                                # session nudge budget already spent
        dismissed = set(adv.get("dismissed") or [])
        adapter = active_adapter(ctx)
        sigs = adapter.friction_signals(ctx, session)
        if not sigs:
            return ""
        intent = _signal_to_intent(sigs[0])
        if not intent or intent in dismissed:
            return ""
        text, _channel = adapter.render(intent, sigs[0])
        if not text:
            return ""
        new_adv = dict(adv)                          # reassign (don't mutate the shared template dict)
        new_adv["last_notice_session"] = sess
        new_adv["notices_this_session"] = count + 1
        m["advisor"] = new_adv
        _save_metrics(ctx, m)
        return text
    except Exception:
        return ""


def cmd_advise(args):
    """`celeborn advise` — print the throughput/quality recommendations that apply RIGHT NOW given the
    detected friction signals. Read-only; also the engine `_advisor_notice` calls on every orient.
    `--dismiss <id>` permanently silences one intent (and `--restore <id>` un-silences it)."""
    ctx = require_context(args)

    dismiss = getattr(args, "dismiss", None)
    restore = getattr(args, "restore", None)
    if dismiss or restore:
        iid = (dismiss or restore)
        if iid not in ADVISOR_INTENTS:
            die(f"unknown recommendation id: {iid}\n  known ids: {', '.join(sorted(ADVISOR_INTENTS))}")
        m = _load_metrics(ctx)
        adv = dict(m.get("advisor") or {})
        dismissed = [d for d in (adv.get("dismissed") or []) if d in ADVISOR_INTENTS]
        if dismiss:
            if iid not in dismissed:
                dismissed.append(iid)
            ok(f"Dismissed '{iid}' — the advisor will no longer recommend it. Restore: "
               f"`celeborn advise --restore {iid}`")
        else:
            dismissed = [d for d in dismissed if d != iid]
            ok(f"Restored '{iid}' — the advisor may recommend it again.")
        adv["dismissed"] = dismissed
        m["advisor"] = adv
        _save_metrics(ctx, m)
        return

    adapter = active_adapter(ctx, getattr(args, "harness", None))
    dismissed = set((_load_metrics(ctx).get("advisor") or {}).get("dismissed") or [])
    sigs = adapter.friction_signals(ctx, None)
    recs, suppressed = [], 0
    for sig in sigs:
        intent = _signal_to_intent(sig)
        if not intent:
            continue
        if intent in dismissed:
            suppressed += 1
            continue
        text, channel = adapter.render(intent, sig)
        recs.append({"intent": intent, "signal": sig, "text": text, "channel": channel})
    if getattr(args, "throughput", False):
        # On-demand throughput recommendations (Phase 4): not signal-triggered (they need judgment the
        # CLI can't make), so they surface only when explicitly asked. Skip dismissed + unrenderable.
        for iid, spec in ADVISOR_INTENTS.items():
            if not spec.get("on_demand") or iid in dismissed:
                continue
            text, channel = adapter.render(iid, None)
            if text:
                recs.append({"intent": iid, "signal": None, "text": text, "channel": channel})
    if getattr(args, "json", False):
        print(json.dumps({"harness": adapter.name, "recommendations": recs,
                          "dismissed": sorted(dismissed)}, indent=2))
        return
    if not recs:
        msg = "No friction detected — nothing to recommend right now."
        if suppressed:
            msg += f" ({suppressed} dismissed)"
        ok(msg)
        return
    print(f"🏹 Celeborn advisor ({adapter.name}) — {len(recs)} recommendation(s):")
    for r in recs:
        print(f"  • [{r['intent']}] {r['text']}")
    print(f"\n  Silence one: celeborn advise --dismiss <id>")


_KNOWN_HARNESSES = ("claude", "grok", "codex", "neutral")


def cmd_harness(args):
    """`celeborn harness [<name>]` — read or pin the active harness in `.celebornrc`. With no name it
    prints the resolved adapter (env $CELEBORN_HARNESS > rc `harness` > default 'claude'). With a name
    it persists `harness: <name>` to the project rc — the durable half of harness selection the Grok/
    Codex bridges call at bootstrap (the env var covers per-call resolution; the rc covers any direct
    `celeborn` invocation in the repo)."""
    ctx = require_context(args)
    name = (getattr(args, "name", None) or "").strip().lower()
    if not name:
        adapter = active_adapter(ctx)
        rc = (load_config(ctx).get("harness") or "").strip().lower() or "(unset)"
        ok(f"Active harness: {adapter.name}  (rc harness: {rc})")
        return
    if name not in _KNOWN_HARNESSES:
        die(f"unknown harness: {name}\n  known: {', '.join(_KNOWN_HARNESSES)}")
    _update_config(ctx, harness=name)
    ok(f"Pinned harness '{name}' in {ctx / RC_NAME} — `active_adapter` now resolves '{name}' for this repo.")


def cmd_permissions(args):
    """`celeborn permissions --suggest|--apply` — productize the manual `/fewer-permission-prompts`
    fix. Scans the harness's allow-list, proposes generalized wildcard rules for the safe families
    (read-only inspection + the trusted Celeborn CLI + test runners), and leaves every un-widenable
    literal verbatim while tallying it as a remaining bottleneck. `--apply` writes; default target is
    the personal `settings.local.json` (`--shared` → the committed `settings.json`)."""
    ctx = require_context(args)

    # t115 — read-only JSON state for the board Settings page.
    if getattr(args, "json", False):
        print(json.dumps(_permissions_state_json(ctx), indent=2))
        return

    scope = ("global" if getattr(args, "global_", False)
             else "shared" if getattr(args, "shared", False) else "local")

    # t115 — apply / remove the SAFE t100 baseline. Default target is GLOBAL (where `wire --global` puts
    # it), unless the caller explicitly chose --shared/local.
    if getattr(args, "baseline", False):
        if not getattr(args, "global_", False) and not getattr(args, "shared", False):
            scope = "global"
        target = _settings_path_for_scope(ctx, scope)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _backup_and_load_settings(target)
        if getattr(args, "remove", False):
            rep = _remove_permission_baseline(data)
            _atomic_write_json(target, data)
            ok(f"removed {len(rep['removed'])} safe-baseline rule(s) from {target}"
               + (" (defaultMode reverted)" if rep["default_mode_reverted"] else "") + ".")
        else:
            rep = _merge_permission_baseline(data)
            _atomic_write_json(target, data)
            ok(f"safe baseline applied -> {target}: +{len(rep['added'])} rule(s)"
               + (f", defaultMode={BASELINE_DEFAULT_MODE}" if rep["default_mode_set"] else "")
               + ". Applies to NEW sessions.")
        return

    # t115 — Danger Zone arm/disarm (the FULL unsafe spectrum). Default target is LOCAL (least blast
    # radius). Arming requires --yes; the board passes it only after a typed confirmation.
    if getattr(args, "danger_zone", False):
        target = _settings_path_for_scope(ctx, scope)
        target.parent.mkdir(parents=True, exist_ok=True)
        if getattr(args, "disarm", False):
            data = _backup_and_load_settings(target)
            rep = _disarm_danger_zone(data)
            _atomic_write_json(target, data)
            ok(f"Danger Zone DISARMED -> {target}: removed {len(rep['removed'])} rule(s), "
               f"defaultMode restored to {BASELINE_DEFAULT_MODE}.")
            return
        if not getattr(args, "yes", False):
            die("refusing to ARM the Danger Zone without --yes — this enables the FULL unsafe "
                "auto-allow spectrum + bypassPermissions.")
        data = _backup_and_load_settings(target)
        rep = _arm_danger_zone(data)
        _atomic_write_json(target, data)
        warn("DANGER ZONE ARMED — the agent may now run ANY command, read/write ANY file, reach ANY "
             "network host, and use every MCP tool; Claude will NOT ask permission (bypassPermissions).")
        ok(f"wrote {target}: +{len(rep['added'])} rule(s), defaultMode={DANGER_DEFAULT_MODE}. Disarm: "
           f"`celeborn permissions --danger-zone --disarm{' --global' if scope == 'global' else ''}`.")
        return

    adapter = active_adapter(ctx, getattr(args, "harness", None))
    target, how = adapter.permission_target(ctx, shared=getattr(args, "shared", False))
    if target is None or how != "per-command-allow":
        die(f"permissions: the '{adapter.name}' harness has no per-command allow-list to generalize "
            f"(its lever is coarse approval/sandbox config, not yet productized).")

    data = {}
    if target.is_file():
        try:
            data = json.loads(target.read_text())
        except json.JSONDecodeError:
            die(f"{target} is not valid JSON — refusing to rewrite it. Fix the file by hand first.")
    allow = (data.get("permissions") or {}).get("allow") or []
    new_rules, generalized, skipped = _generalize_allow(allow)
    skipped_total = sum(skipped.values())
    removed = len(allow) - len(new_rules)

    info(f"Permission target: {target}")
    print(f"  current rules:     {len(allow)}")
    print(f"  after generalize:  {len(new_rules)}  ({generalized} literal(s) collapsed, {removed} fewer)")
    if generalized == 0 and not skipped:
        ok("Nothing to generalize — the allow-list is already lean.")
        return
    if skipped:
        print(f"  ⚠ skipped bottlenecks (kept verbatim — can't be widened safely): {skipped_total}")
        for key, cnt in sorted(skipped.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"      {cnt:>3}×  {key}")

    if not getattr(args, "apply", False):
        print("\n  Proposed allow-list (run again with --apply to write it):")
        for r in new_rules:
            print(f"      {r}")
        print(f"\n  Apply:  celeborn permissions --apply"
              f"{' --shared' if getattr(args, 'shared', False) else ''}")
        return

    if not getattr(args, "yes", False):
        try:
            resp = input(f"\nRewrite {target.name} with {len(new_rules)} rule(s)? [y/N] ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            warn("Aborted — nothing written.")
            return

    data.setdefault("permissions", {})["allow"] = new_rules
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n")

    m = _load_metrics(ctx)
    adv = dict(m.get("advisor") or {})
    adv["permission_rules_generalized"] = int(adv.get("permission_rules_generalized", 0) or 0) + generalized
    adv["skipped_bottlenecks"] = skipped
    adv["skipped_bottlenecks_total"] = skipped_total
    adv["last_applied_at"] = now_iso()
    m["advisor"] = adv
    _save_metrics(ctx, m)
    ok(f"Wrote {target} — {generalized} literal rule(s) generalized, {skipped_total} bottleneck(s) remain.")


# A path-shaped token: optional ./ or ../ lead, then a name and at least one slash. The trailing
# `.ext` is enforced by the caller (so we only flag things that look like real files, not URLs/dirs).
_DRIFT_PATH_RE = re.compile(r"(?:\.{0,2}/)?[\w.\-]+(?:/[\w.\-]+)+")


def _extract_memory_paths(text: str) -> list[str]:
    """Repo-relative file paths referenced inside inline-code spans of authored memory.

    Only backtick-wrapped tokens that look like a path (a slash plus a real file extension) are
    returned — prose words, command examples, and bare identifiers are ignored. This is a trust
    feature: a wrong drift flag costs more than a missed one, so precision beats recall here."""
    out: list[str] = []
    for span in re.findall(r"`([^`]+)`", text):
        if "://" in span:  # a URL span, not a file reference
            continue
        for tok in _DRIFT_PATH_RE.findall(span):
            tok = tok.strip().rstrip(".,;:)")
            if tok.startswith("/") or "<" in tok or ">" in tok:  # absolute/host frag, <placeholder>
                continue
            base = tok.rsplit("/", 1)[-1]
            if "." not in base or base.startswith("."):  # need name.ext, not a dotfile/dir
                continue
            ext = base.rsplit(".", 1)[-1]
            if not (1 <= len(ext) <= 5 and ext.isalnum()):  # plausible extension
                continue
            out.append(tok)
    return out


def _repo_tracked_files(repo: Path) -> set[str]:
    """Git-tracked files (repo-relative POSIX) — the authoritative 'what exists' set. Empty on any
    git failure, which makes the caller fall back to plain filesystem existence."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(repo), "ls-files"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return {ln for ln in r.stdout.split("\n") if ln}
    except (OSError, subprocess.SubprocessError):
        pass
    return set()


def _memory_drift(ctx: Path) -> list[tuple[str, str]]:
    """File paths referenced in the LIVE memory tiers (state.md, notes.md) that the repo no longer
    has — a deleted or renamed file/module the authored memory still points at.

    journal/decisions/learnings are append-only HISTORY, so a since-deleted file named there is
    correct, not drift; they're exempt. Returns (tier, path) pairs, deduped per tier."""
    repo = ctx.parent
    tracked = _repo_tracked_files(repo)

    def present(p: str) -> bool:
        if (repo / p).exists() or (ctx / p).exists():  # filesystem (handles ../ and untracked)
            return True
        if p in tracked:
            return True
        suffix = "/" + p
        return any(f.endswith(suffix) for f in tracked)  # path given relative to a subdir

    drift: list[tuple[str, str]] = []
    for tier in ("state.md", "notes.md"):
        fp = ctx / tier
        if not fp.is_file():
            continue
        seen: set[str] = set()
        for p in _extract_memory_paths(fp.read_text()):
            if p in seen:
                continue
            seen.add(p)
            if not present(p):
                drift.append((tier, p))
    return drift


def cmd_doctor(args):
    ctx = require_context(args)
    cfg = load_config(ctx)
    problems = 0
    warnings = 0
    print("celeborn doctor")

    # required files
    for rel in REQUIRED_FILES:
        if (ctx / rel).is_file():
            ok(rel)
        else:
            warn(f"MISSING required file: {rel}")
            problems += 1

    # state.md budget
    sp = ctx / "state.md"
    if sp.is_file():
        n = len(sp.read_text().splitlines())
        if n > cfg["state_max_lines"]:
            warn(f"state.md is {n} lines (budget {cfg['state_max_lines']}) — condense it")
            warnings += 1
        else:
            ok(f"state.md within budget ({n}/{cfg['state_max_lines']} lines)")

    # Hot-tier char budget — the Orient load is injected as SessionStart additionalContext; if a
    # piece outgrows its char budget, `status` clips it (with a pointer), so authored detail stops
    # reaching the model on rehydration. Flag the clip so it's visible, not silent.
    state_max = int(cfg.get("hot_state_max_chars", 4000))
    act_max = int(cfg.get("hot_activity_max_chars", 2000))
    hot_over = []
    if sp.is_file() and len(sp.read_text()) > state_max:
        hot_over.append(f"state.md {len(sp.read_text())}/{state_max} chars")
    ap = ctx / "activity.md"
    if ap.is_file() and len(ap.read_text()) > act_max:
        hot_over.append(f"activity.md {len(ap.read_text())}/{act_max} chars")
    if hot_over:
        warn("Hot tier over char budget — clipped on Orient load, so detail won't fully rehydrate: "
             + "; ".join(hot_over))
        print("  Fix:  condense state.md (history → journal.md), or raise hot_*_max_chars in .celebornrc")
        warnings += 1
    else:
        ok("Hot tier within char budget (full Orient load reaches the model)")

    # journal budget
    jp = ctx / "journal.md"
    if jp.is_file():
        _, entries = split_journal(jp.read_text())
        if len(entries) > cfg["journal_keep_entries"]:
            warn(f"journal.md has {len(entries)} entries (keep {cfg['journal_keep_entries']}) — run `celeborn archive`")
            warnings += 1
        else:
            ok(f"journal.md within budget ({len(entries)}/{cfg['journal_keep_entries']} entries)")

    # done-column budget (auto-archives on the next `celeborn tasks` save)
    if (ctx / TASKS_FILE).is_file():
        all_tasks = _load_tasks(ctx)
        done_n = len(_done_tasks_ordered(all_tasks))
        keep_done = cfg["done_keep_cards"]
        if done_n > keep_done:
            warn(f"tasks.md has {done_n} done card(s) (keep {keep_done}) — run `celeborn tasks archive`")
            warnings += 1
        else:
            ok(f"done column within budget ({done_n}/{keep_done} cards)")

        # Stop-condition contract (CELE-t81): every open card should carry a logical Stop condition,
        # and ideally a real one rather than the generic auto-filled default. Advisory only — flag
        # open cards (not done) that are missing or still carry the default so an owner can sharpen it.
        open_tasks = [t for t in all_tasks if t["state"] != "done"]
        missing_stop = [t for t in open_tasks if not (t.get("stop") or "").strip()]
        default_stop = [t for t in open_tasks if (t.get("stop") or "").strip() == DEFAULT_STOP]
        if missing_stop:
            warn(f"{len(missing_stop)} open card(s) have no Stop condition: "
                 + ", ".join(_display_tid(ctx, t["id"]) for t in missing_stop))
            print("  Fix:  set one — `celeborn tasks edit <id> --stop \"<clean /clear point>\"`")
            warnings += len(missing_stop)
        elif default_stop:
            info(f"{len(default_stop)} open card(s) still carry the generic default Stop condition: "
                 + ", ".join(_display_tid(ctx, t["id"]) for t in default_stop)
                 + " — replace with a card-specific one when you pick them up.")
        else:
            ok("every open card carries a Stop condition")
        # Progress-engine drift (CELE-t161): a doing card stuck at 0% that already has commits carrying
        # its trailer means the engine never moved it — flag so it's visible (complements the Stop check).
        drifted = [t for t in all_tasks if t.get("state") == "doing"
                   and int(t.get("progress", 0) or 0) == 0 and _commits_for_task(ctx, t["id"], limit=50)]
        if drifted:
            warn(f"{len(drifted)} doing card(s) at 0% despite commits with their trailer: "
                 + ", ".join(_display_tid(ctx, t["id"]) for t in drifted))
            print("  Fix:  `celeborn progress <id> --explain`  (runs the engine + shows the derivation)")
            warnings += len(drifted)
        else:
            ok("no progress-engine drift on doing cards")
        # Owner-attribution contract (CELE-t194): a DOING card is owned by its SESSION short-id (the
        # code grabs it), never by a model name and never left "unknown". A card in either bad state
        # was claimed by an unfixed binary or a hand-run `--by claude` — the fleet then shows @claude /
        # @unknown and can't attach a context-token chip. Re-claiming from the owning window now grabs
        # CLAUDE_CODE_SESSION_ID automatically and repairs both. Backstop for the in-path guard.
        doing = [t for t in all_tasks if t.get("state") == "doing"]
        mis_owned = [t for t in doing
                     if (o := (t.get("owner") or "").strip()).lower() in ("", "unknown")
                     or _looks_like_model_handle(o)]
        if mis_owned:
            warn(f"{len(mis_owned)} doing card(s) not owned by a session short-id (shows @unknown/"
                 f"model, no context chip): " + ", ".join(_display_tid(ctx, t["id"]) for t in mis_owned))
            print("  Fix:  re-claim from the owning window — `celeborn claim <id>` "
                  "(auto-grabs CLAUDE_CODE_SESSION_ID; no --by/--session needed)")
            warnings += len(mis_owned)
        else:
            ok("every doing card is owned by a session short-id")
        arch_path = ctx / DONE_ARCHIVE_FILE
        if arch_path.is_file():
            arch_n = len(_parse_tasks(arch_path.read_text()))
            arch_cap = cfg["done_archive_keep_cards"]
            ok(f"done-archive.md: {arch_n}/{arch_cap} card(s)")

    # session.json valid
    sj = ctx / "session.json"
    if sj.is_file():
        try:
            json.loads(sj.read_text())
            ok("session.json is valid JSON")
        except json.JSONDecodeError as e:
            warn(f"session.json INVALID JSON: {e}")
            info("    repair it with: celeborn checkpoint  (rebuilds valid JSON from the template)")
            problems += 1

    # index freshness
    if (ctx / INDEX_NAME).is_file():
        if _index_is_stale(ctx):
            warn("index.db is stale — run `celeborn index`")
            warnings += 1
        else:
            ok("index.db is fresh")
    else:
        warn("index.db absent — run `celeborn index` to enable search")
        warnings += 1

    # memory drift — live tiers (state.md/notes.md) pointing at files the repo no longer has.
    # This is the honesty check: if memory references a deleted or renamed file, the next session
    # rehydrates a lie. History tiers (journal/decisions/learnings) are exempt — see _memory_drift.
    drift = _memory_drift(ctx)
    if drift:
        warn(f"memory drift — {len(drift)} stale file reference(s) in live memory (deleted/renamed):")
        for tier, p in drift:
            print(f"      {tier} → {p}")
        print("  Fix:  correct the path in state.md/notes.md (or restore the file) so memory matches the repo")
        warnings += len(drift)
    else:
        ok("memory matches repo (no stale file references in state.md/notes.md)")

    # secret scan
    hits = _secret_scan(ctx, cfg["secret_patterns"])
    if hits:
        for h in hits:
            warn(f"POSSIBLE SECRET in {h}")
        problems += len(hits)
    else:
        ok("no obvious secrets in committed memory")

    # install integrity — shipped core modules vs the published per-version checksum manifest.
    # Detection, not prevention: an in-place edit means the install no longer matches the release, so a
    # "works after I hacked celeborn.py" break self-reports here instead of becoming a confused bug
    # report. Source/editable checkouts ship no manifest → 'unverified' → we stay silent (no dev nag).
    ist = integrity_status()
    if ist["state"] == "modified":
        warn("modified install detected — core module(s) differ from the published release: "
             + ", ".join(ist["modified"]))
        print("  Fix:  reinstall to reset (uv tool install --force / pipx reinstall); local edits are "
              "unsupported — submit a PR.")
        warnings += 1
    elif ist["state"] == "ok":
        ok("install integrity verified (core modules match the published release)")
    # 'unverified' (source/dev install): say nothing — never nag a contributor editing the tree.

    # Fleet economics integrity (CELE-t124): the board's savings bar and `celeborn fleet` only count
    # REGISTERED projects, and savings from any session that runs outside a project .context/ (e.g. from
    # your home dir) land in the global ~/.context sink — attributed to no project. Surface a large sink
    # so a project's economics never silently goes missing. Orient now self-registers (so the sink stops
    # growing for projects you actually open), but historical sink savings stay until redistributed.
    gctx = _global_context()
    if gctx.is_dir() and (gctx / METRICS_NAME).is_file():
        sink_tokens = _load_metrics(gctx).get("tokens_saved_estimate", 0)
        proj_tokens = _load_metrics(ctx).get("tokens_saved_estimate", 0)
        if sink_tokens > max(proj_tokens, 5_000_000):
            warn(f"global ~/.context sink holds ~{sink_tokens:,} tokens of savings not attributed to any "
                 f"project — work that ran outside a project .context/.")
            print("  Fix:  open each project from inside its own dir so orient records (and self-registers) "
                  "there; `celeborn fleet register --path <dir>` counts a project in the board economics.")
            warnings += 1

    # Informational heads-up (NOT a Celeborn problem): Claude Code's own PR-status panel shells out
    # to `gh`. If gh is installed but not logged in, Claude Code shows a "GitHub CLI authentication
    # expired" banner. Most people run Celeborn inside Claude Code, so surface a friendly, actionable
    # note rather than leaving them to wonder whether Celeborn caused it. Doesn't touch the counts.
    if _gh_unauthenticated():
        info("Claude Code's PR-status panel may show a `gh` auth banner — that's Claude Code, not "
             "Celeborn. Clear it with `gh auth login` (git keeps working via your keychain regardless).")

    # Grok Build — hooks + per-project rules (orient survives /clear when cwd is this repo).
    import shutil
    root = ctx.parent
    grok_rules = root / ".grok" / "rules" / "celeborn.md"
    if shutil.which("grok") and (Path.home() / ".grok").is_dir():
        hooks = Path.home() / ".grok" / "hooks" / "celeborn.json"
        if hooks.is_file():
            ok("Grok hooks installed (~/.grok/hooks/celeborn.json)")
        else:
            warn("Grok detected but Celeborn hooks missing — run `celeborn grok wire`")
            warnings += 1
        if grok_rules.is_file():
            ok(".grok/rules/celeborn.md present (Grok auto-loads orient + kanban binding)")
        else:
            warn("missing .grok/rules/celeborn.md — run `celeborn grok sync-rules`")
            warnings += 1

    print(f"\n{warnings} warning(s), {problems} problem(s).")
    if problems:
        print("Doctor found problems that should be fixed.")
        sys.exit(1)


def _gh_unauthenticated() -> bool:
    """True iff the GitHub CLI is installed but not logged in — the exact state that makes Claude
    Code's PR-status panel show its `gh` auth banner. Returns False if gh is absent or the check
    can't run, so this never invents a problem. Used only for an informational note in `doctor`."""
    import shutil
    if not shutil.which("gh"):
        return False
    import subprocess
    try:
        # `gh auth status` exits non-zero when no host is logged in; it's a local, network-free read.
        r = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, timeout=5)
        return r.returncode != 0
    except Exception:
        return False


def _secret_scan(ctx: Path, patterns: list[str]) -> list[str]:
    regexes = [re.compile(p) for p in patterns]
    hits: list[str] = []
    for path in ctx.rglob("*"):
        if not path.is_file() or path.name == INDEX_NAME or path.name == RC_NAME:
            continue
        if path.suffix not in (".md", ".json", ".txt", ""):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for rx in regexes:
            if rx.search(text):
                hits.append(str(path.relative_to(ctx)))
                break
    return hits


def _index_is_stale(ctx: Path) -> bool:
    """Stale if any indexed source file is newer than the index file. Uses filesystem mtime
    only — no DB open — so status/doctor stay sqlite-free. (`built_at` still lives in the DB's
    meta table for informational inspection; it just isn't needed here.)

    MECHANICAL_GLOBS are skipped: `celeborn capture` rewrites them every turn, so counting them
    would report the index stale within one turn of any live session even when nothing a user
    would re-index for has changed. They remain indexed/searchable — only this heuristic ignores
    them."""
    db = ctx / INDEX_NAME
    if not db.is_file():
        return True
    built = db.stat().st_mtime
    for tier, glob in TIER_GLOBS:
        if glob in MECHANICAL_GLOBS:
            continue
        for path in ctx.glob(glob):
            if path.is_file() and path.stat().st_mtime > built + 1:
                return True
    return False


# --------------------------------------------------------------------------- tasks (Phase 11)
#
# A lightweight, agent-native task/kanban board. `tasks.md` is the markdown source of truth
# (`celeborn tasks` edits it); `tasks.json` is a derived projection the board viewer reads,
# regenerated on every command and gitignored — the same markdown-truth / disposable-derived
# split the SQLite index follows. Offline, stdlib-only, no board UI in the core.

TASKS_FILE = "tasks.md"
TASKS_JSON = "tasks.json"
# The "blocked" state was retired (CELE-t135): kanban discipline discourages a Blocked column, so
# DOING reclaims its space. Cards still record dependencies in their `blocked_by` list — that lives on
# independently of any column. Legacy cards stored as `blocked` load as `todo` (see _load_tasks).
TASK_STATES = ["todo", "doing", "done"]
TASK_STATE_LABELS = {"todo": "TODO", "doing": "DOING", "done": "DONE"}

# Every task carries a logical Stop condition (CELE-t81): the "this is a clean place to stop" marker
# that tells the model when the card is at a defensible `/clear` point. `tasks add` auto-fills this
# generic default when no `--stop` is supplied so no card is ever stop-less — the agent protocol then
# nudges the owner to replace it with a real, card-specific condition. Deterministic, stdlib-only (the
# core makes no LLM calls); the "intelligent query" lives in the agent loop + CLAUDE.md contract, not here.
DEFAULT_STOP = "Acceptance criteria met, tests green, change committed"

TASK_HEADING_RE = re.compile(r"^\[(?P<id>[A-Za-z0-9_-]+)\]\s+(?P<title>.*)$")
# Only `## [id] title` lines open a new card — `##` headings inside a card's notes must stay in notes.
TASK_BLOCK_SPLIT_RE = re.compile(r"(?m)^##[ \t]+(?=\[[A-Za-z0-9_-]+\])")
TASK_META_RE = re.compile(r"^-\s+(?P<key>[a-z-]+):\s*(?P<val>.*)$")
# Subtask checklist (CELE-t106): markdown checkbox lines under a `### Subtasks` heading in a card body.
# Optional trailing `×N` weight (default 1). Checking items auto-derives the card's `progress`.
SUBTASKS_HEADING = "### Subtasks"
SUBTASK_RE = re.compile(r"^-\s+\[(?P<done>[ xX])\]\s+(?P<text>.*?)(?:\s+×(?P<weight>\d+))?\s*$")

TASKS_HEADER = (
    "# Tasks\n\n"
    "<!-- Celeborn task board (Phase 11). Markdown is the source of truth; `celeborn tasks`\n"
    "     edits it. `.context/tasks.json` is the derived projection the board viewer reads —\n"
    "     regenerated on every `celeborn tasks` command, gitignored, disposable.\n"
    "     One task per `## [id]` block; states: todo | doing | done. -->\n"
)

# Machine-readable block injected into every card's agent view (tasks.json, copy prompt, board DOM).
# Omitted from tasks.md and from the board's visible UI — agents must read it to work the card.
AGENT_PROTOCOL_MARKER = "⟨celeborn:protocol⟩"


def _agent_card_protocol(task_id: str) -> str:
    """Required agent instructions bundled with every kanban card. Not shown to humans on the board."""
    return (
        f"{AGENT_PROTOCOL_MARKER}\n"
        f"AGENT PROTOCOL for [{task_id}] — required before any file edit:\n"
        f"ALIGNMENT GATE — before doing the work: this card was likely typed quickly, so its title "
        f"and description may be rough or under-specified. Do NOT start editing on assumptions. First "
        f"confirm you understand it: ask the user 1–3 short clarifying questions about intent, scope, "
        f"and the right Stop condition, and wait for their answers. Use the project's permanent context "
        f"(you already have it on orient) to ask sharp questions. Once aligned, sharpen the card itself "
        f"(`celeborn tasks edit {task_id} --title \"…\" --note \"…\" --stop \"…\"`) so the next reader "
        f"inherits the clarity — then proceed. Skip the questions only if the card is already unambiguous.\n"
        f"0. Identify once per session: `celeborn identify --family <Claude|Grok|GPT…> "
        f"--model \"<e.g. Opus 4.8>\"` so your touches show who you are.\n"
        f"1. Move this card to DOING FIRST: `celeborn claim {task_id} --by <you>` "
        f"(or `celeborn tasks move {task_id} doing --owner <you>`). The board must show DOING "
        f"before you touch files — session focus alone is not enough. If Celeborn is blocking your "
        f"edits with a NO-TASK gate, add `--session <your-session-id>` to the claim (it hands you the "
        f"exact command) so the claim records the session→card link that lifts the gate this turn.\n"
        f"2. Then register each shared file: `celeborn touch <file> --by <you> --task {task_id} "
        f"--why \"<reason>\"`.\n"
        f"3. CLOSE OUT: as you START writing your ready-to-ship message, crest the sand-fill bar with "
        f"`celeborn tasks edit {task_id} --progress 99` (an unshipped card tops out at 99 — 100% is "
        f"reserved for Done); THEN, at the end, `celeborn ship {task_id}` (releases touches, moves to "
        f"Done, and fills the bar to 100%). 100% means shipped — never set it by hand on a DOING card. "
        f"DOING with zero touches is stale.\n"
        f"4. Honor this card's Stop condition (the `stop` field): it marks the clean `/clear` point. "
        f"If it still carries the generic default, replace it with a real one: "
        f"`celeborn tasks edit {task_id} --stop \"<condition>\"`.\n"
        f"See references/multi-agent-editing.md."
    )


def _tasks_path(ctx: Path) -> Path:
    return ctx / TASKS_FILE


def _tasks_json_path(ctx: Path) -> Path:
    return ctx / TASKS_JSON


def _csv(v) -> list[str]:
    """Split a comma/space-separated string into a clean list."""
    return [x.strip() for x in re.split(r"[,\s]+", v or "") if x.strip()]


def _clamp_pct(v) -> int:
    """Coerce a value to an int percent in [0, 100] (0 on anything unparseable). The progress field
    that drives the In-Progress card's sand-fill bar (CELE-t106)."""
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, n))


def _parse_subtask_spec(spec: str) -> dict:
    """A `set`/`add` item like 'Wire the CLI *2' → {text, weight, done}. Trailing `*N` sets weight."""
    m = re.match(r"^(?P<t>.*?)(?:\s*\*(?P<w>\d+))?\s*$", spec or "")
    text = (m.group("t") or "").strip()
    weight = int(m.group("w")) if m and m.group("w") else 1
    return {"text": text, "weight": max(1, weight), "done": False}


def _split_subtasks(notes_lines: list[str]) -> tuple[list[dict], list[str]]:
    """Pull the `### Subtasks` checkbox block out of a card's body lines. Returns (subtasks, remaining
    note lines). Only the contiguous checkbox lines immediately under the heading are consumed."""
    idx = next((i for i, l in enumerate(notes_lines) if l.strip().lower() == SUBTASKS_HEADING.lower()), None)
    if idx is None:
        return [], notes_lines
    subs: list[dict] = []
    j = idx + 1
    while j < len(notes_lines):
        m = SUBTASK_RE.match(notes_lines[j].strip())
        if not m:
            break
        subs.append({
            "text": m.group("text").strip(),
            "weight": int(m.group("weight")) if m.group("weight") else 1,
            "done": m.group("done").lower() == "x",
        })
        j += 1
    return subs, notes_lines[:idx] + notes_lines[j:]


def _normalize_progress(t: dict) -> None:
    """Enforce the project rule: 100% means shipped to Done — and ONLY Done (CELE-t131). A card that
    has not yet shipped is capped at 99 no matter what was set manually (`--progress 100`) or derived
    from all-checked subtasks; moving the card to `done` fills it to 100. So the In-Progress sand-fill
    bar can crest to 99 ("ship it") but never reads 'complete' until the card actually leaves for Done."""
    pct = _clamp_pct(t.get("progress", 0))
    t["progress"] = 100 if t.get("state") == "done" else min(99, pct)


def _recompute_progress(t: dict) -> None:
    """When a card has subtasks, its `progress` is DERIVED: the weighted fraction of checked items
    (CELE-t106). No subtasks → progress is left as-is (explicit/manual). Either way the 100%=Done
    invariant is then enforced (CELE-t131): an unshipped card tops out at 99, never 100."""
    subs = t.get("subtasks") or []
    if subs:
        total = sum(max(1, int(s.get("weight", 1))) for s in subs)
        done = sum(max(1, int(s.get("weight", 1))) for s in subs if s.get("done"))
        if total:
            t["progress"] = _clamp_pct(round(100 * done / total))
    # CELE-t161: never below the persisted engine floor (claim/work/band). 0 on non-engine cards → the
    # pure-ratio CELE-t106 behavior is exactly preserved (incl. uncheck lowering the bar).
    floor = _clamp_pct(int(t.get("engine_floor", 0) or 0))
    if floor:
        t["progress"] = max(_clamp_pct(t.get("progress", 0)), floor)
    _normalize_progress(t)


# CELE-t176 — the "mandatory complete" gate. A card may not leave DOING for Done until its progress
# bar is crested to 99% (the ceiling an unshipped card can reach; see _normalize_progress). This makes
# the final "it's complete" step deliberate, so a card never silently disappears from DOING (in Fleet
# or board/) at partial progress. The operator crests (`--progress 99`, or all subtasks checked), THEN
# ships — ship fills the last 1% to 100. Enforced on every path a card can reach `done` by: `ship`,
# `tasks move … done`, and `tasks edit … --state done`.
CREST_PCT = 99


def _require_crest_for_done(ctx: Path, t: dict) -> None:
    """Refuse a transition into `done` unless the card's progress bar is crested to CREST_PCT (99).
    A no-op at/above the crest; otherwise `die`s with the exact commands to crest and ship."""
    pct = _clamp_pct(t.get("progress", 0))
    if pct >= CREST_PCT:
        return
    disp = _display_tid(ctx, t["id"])
    die(
        f"[{disp}] is at {pct}% — a card must be crested to {CREST_PCT}% before it can leave DOING for "
        f"Done (CELE-t176). Finish the work, then crest and ship:\n"
        f"  celeborn tasks edit {t['id']} --progress {CREST_PCT}\n"
        f"  celeborn ship {t['id']}\n"
        f"(100% is reserved for shipped cards.)"
    )


def _render_subtasks(subs: list[dict]) -> list[str]:
    out = [SUBTASKS_HEADING]
    for s in subs:
        box = "x" if s.get("done") else " "
        w = f" ×{int(s.get('weight', 1))}" if int(s.get("weight", 1)) != 1 else ""
        out.append(f"- [{box}] {s.get('text', '').strip()}{w}")
    return out


def _valid_task_id(tid: str) -> bool:
    """True when `tid` is a non-empty stored card key (e.g. t132). Id-less blocks are never cards."""
    return bool(tid and re.fullmatch(r"[A-Za-z0-9_-]+", tid))


def _parse_tasks(text: str) -> list[dict]:
    """Parse tasks.md into a list of task dicts. Each `## [id] title` block carries `- key: value`
    metadata lines; any remaining lines become the task's freeform notes."""
    tasks: list[dict] = []
    for blk in TASK_BLOCK_SPLIT_RE.split(text)[1:]:
        lines = blk.splitlines()
        head = lines[0].strip() if lines else ""
        m = TASK_HEADING_RE.match(head)
        if not m:
            continue  # malformed heading — never mint an id-less card
        tid, title = m.group("id"), m.group("title").strip()
        if not _valid_task_id(tid):
            continue
        meta: dict = {}
        notes_lines: list[str] = []
        in_meta = True  # only the `- key: value` run directly under the heading is card metadata
        for ln in lines[1:]:
            if in_meta:
                if not ln.strip():
                    continue
                mm = TASK_META_RE.match(ln)
                if mm:
                    meta[mm.group("key")] = mm.group("val").strip()
                    continue
                in_meta = False
            notes_lines.append(ln)
        subtasks, notes_lines = _split_subtasks(notes_lines)
        # Any state not in TASK_STATES (e.g. a legacy `blocked` card from before CELE-t135 retired
        # that column) falls back to `todo` so it stays visible rather than vanishing off the board.
        raw_state = (meta.get("state") or "todo").lower()
        t = {
            "id": tid,
            "title": title,
            "state": raw_state if raw_state in TASK_STATES else "todo",
            "owner": meta.get("owner", ""),
            "tags": _csv(meta.get("tags", "")),
            "blocked_by": _csv(meta.get("blocked-by", "")),
            "phase": meta.get("phase", ""),  # which plan phase card this task drills down from
            # Logical Stop condition: a clearly-defined "this is a clean place to stop" marker, so the
            # model knows when the card is at a defensible `/clear` point (CELE-t81). Free text;
            # "" on legacy cards that predate the field — auto-filled with a default on `tasks add`.
            "stop": meta.get("stop", ""),
            # Percent complete (0-100) — drives the In-Progress card's sand-fill bar (CELE-t106).
            # Absent on legacy cards → 0. Explicit for now (`tasks edit --progress`); a context-derived
            # auto-estimate is the planned follow-up.
            "progress": _clamp_pct(meta.get("progress", 0)),
            # CELE-t161 progress-engine floor: a deterministic minimum the bar is held at (claim 5,
            # first-work 10, then the milestone band). 0/absent on non-engine cards → no effect.
            "engine_floor": _clamp_pct(meta.get("engine-floor", 0)),
            "jira": meta.get("jira", ""),    # linked Jira issue key (e.g. SCRUM-2); set by `jira pull`
            "created": meta.get("created", ""),
            "updated": meta.get("updated", ""),
            "subtasks": subtasks,  # checklist (CELE-t106); derives progress when present
            "notes": "\n".join(notes_lines).strip(),
        }
        _recompute_progress(t)  # subtasks (incl. hand-edited checkboxes) are the source of truth for %
        tasks.append(t)
    return tasks


DONE_ARCHIVE_FILE = "done-archive.md"
DONE_ARCHIVE_HEADER = (
    "# Done archive\n\n"
    "<!-- Celeborn auto-archives done cards that fall off the bottom of the Done column.\n"
    "     Cap: done_archive_keep_cards (default 100); oldest entries are dropped FIFO.\n"
    "     Still searchable via `celeborn search`. Regenerated by `celeborn tasks`. -->\n"
)


def _render_tasks(tasks: list[dict], *, header: str = TASKS_HEADER) -> str:
    out = [header]
    for t in tasks:
        out.append(f"## [{t['id']}] {t['title']}")
        out.append(f"- state: {t['state']}")
        out.append(f"- owner: {t['owner']}")
        out.append(f"- tags: {', '.join(t['tags'])}")
        out.append(f"- blocked-by: {', '.join(t['blocked_by'])}")
        out.append(f"- phase: {t['phase']}")
        out.append(f"- stop: {t.get('stop', '')}")  # logical Stop condition (CELE-t81); always rendered so every card advertises the slot
        if t.get("progress"):  # only when >0, so legacy cards stay byte-identical (CELE-t106)
            out.append(f"- progress: {_clamp_pct(t['progress'])}")
        if t.get("engine_floor"):  # CELE-t161; only when >0, so non-engine cards stay byte-identical
            out.append(f"- engine-floor: {_clamp_pct(t['engine_floor'])}")
        if t.get("jira"):
            out.append(f"- jira: {t['jira']}")
        out.append(f"- created: {t['created']}")
        out.append(f"- updated: {t['updated']}")
        if t.get("subtasks"):  # checklist block (CELE-t106) — rendered between metadata and notes
            out.append("")
            out.extend(_render_subtasks(t["subtasks"]))
        if t["notes"]:
            out.append("")
            out.append(t["notes"])
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def _tasks_doc(ctx: Path, tasks: list[dict]) -> dict:
    """JSON projection for the board viewer. Adds per-card agent_protocol (not stored in tasks.md)
    and joins the local agent registry so the owner chip can show family/model — tasks.md itself
    stays handle-only (the committed public contract is unchanged)."""
    agents = (_load_agents(ctx).get("agents") or {})
    alerts = _live_alerts(ctx)   # CELE-t195: an ended session's alert must not surface on the board
    cfg = load_config(ctx)
    slug = project_slug(ctx)
    qualified = bool(cfg.get("qualified_task_ids"))

    def _enrich(t: dict) -> dict:
        owner = (t.get("owner") or "").strip()
        reg = agents.get(owner) or {}
        model = reg.get("model") or ""
        return {
            **t,
            # `id` stays the canonical bare key the viewer's claim/move calls use; `display_id` is the
            # presentation form (qualified when the project opts in) the board chip can show instead.
            "display_id": _display_tid(ctx, t["id"], cfg=cfg),
            "agent_protocol": _agent_card_protocol(t["id"]),
            # Owner chip shows a session / human handle only — a model-derived owner is suppressed so
            # model text never lands on the board (CELE-t172). tasks.md itself is untouched.
            "owner": _display_owner(owner, model),
            "owner_family": reg.get("family") or "",
            "owner_model": model,
            # Live blocked-alert (CELE-t169) — only doing cards can be blocked; None means clear. Rides
            # the projection so the local board + hosted push both carry the badge; not a tasks.md field.
            "alert": alerts.get(t["id"]) if t.get("state") == "doing" else None,
        }

    enriched = [_enrich(t) for t in tasks]
    return {
        "generated_at": now_iso(),
        "project_slug": slug,
        "project_name": _project_name(ctx),
        "qualified_task_ids": qualified,  # board hint: render display_id (SLUG-tN) instead of id
        "id_prefix": slug.upper() if qualified else "",
        "states": TASK_STATES,
        "tasks": enriched,
    }


def _load_tasks(ctx: Path) -> list[dict]:
    p = _tasks_path(ctx)
    return _parse_tasks(p.read_text()) if p.is_file() else []


def _tasks_orient_summary(ctx: Path, tasks: list[dict]) -> str:
    """Compact task-board view for the Hot tier (Orient load): one count line, then the cards an
    agent resuming actually needs to see — what's in flight (doing), with any blocked_by deps flagged.
    Read-only; never touches tasks.md or the derived JSON. Returns "" when there are no tasks."""
    if not tasks:
        return ""
    counts = {s: sum(1 for t in tasks if t["state"] == s) for s in TASK_STATES}
    line = " · ".join(f"{counts[s]} {s}" for s in TASK_STATES)
    out = [f"{line}    (board: `celeborn tasks` · viewer: board/)"]
    actionable = [t for t in tasks if t["state"] == "doing"]
    cfg = load_config(ctx)
    for t in actionable:
        owner = f"  @{t['owner']}" if t["owner"] else ""
        blocked = f"  ⛔ {', '.join(t['blocked_by'])}" if t["blocked_by"] else ""
        disp = _display_tid(ctx, t["id"], cfg=cfg)
        stale = ""
        if t["state"] == "doing" and not _task_has_active_touches(ctx, t["id"]):
            stale = "  ⚠ stale (no touches) — `celeborn ship " + disp + "`"
        out.append(f"  {t['state']} → [{disp}] {t['title']}{owner}{blocked}{stale}")
    return "\n".join(out)


def _write_tasks_json(ctx: Path, tasks: list[dict]):
    _tasks_json_path(ctx).write_text(json.dumps(_tasks_doc(ctx, tasks), indent=2) + "\n")


def _save_tasks(ctx: Path, tasks: list[dict], *, autopush_ids: list[str] | None = None):
    """Persist the markdown source of truth, then refresh the derived JSON projection."""
    ctx = ctx.resolve()
    bad = [t for t in tasks if not _valid_task_id(t.get("id", ""))]
    if bad:
        sample = ", ".join(repr((t.get("title") or "")[:48]) for t in bad[:3])
        die(
            f"refusing to save {len(bad)} task(s) without valid id ({sample}). "
            "Use `###` for section headings inside card notes — only `## [tN] title` opens a card."
        )
    cfg = load_config(ctx)
    for t in tasks:
        _normalize_progress(t)  # 100%=Done invariant on every write (CELE-t131): ship→100, doing≤99
    tasks, archived = _archive_overflow_done(ctx, tasks, cfg)
    _tasks_path(ctx).write_text(_render_tasks(tasks))
    _write_tasks_json(ctx, tasks)
    if archived:
        info(f"Archived {len(archived)} done card(s) → {DONE_ARCHIVE_FILE} "
             f"(kept {cfg['done_keep_cards']} on board)")
    if autopush_ids:
        try:
            __import__("celeborn_jira").schedule_auto_push(ctx, tasks, autopush_ids)
        except Exception:
            pass  # auto-push is best-effort — never break tasks save
        try:
            # Live-push the changed cards to the hosted board so celeborn.thot.ai updates in ~realtime
            # (detached + gated; a no-op when hosted sync isn't configured / signed in).
            __import__("celeborn_sync").schedule_hosted_push(ctx, autopush_ids)
        except Exception:
            pass  # hosted liveness is best-effort — never break tasks save


def _next_task_id(tasks: list[dict]) -> str:
    mx = 0
    for t in tasks:
        m = re.fullmatch(r"t(\d+)", t["id"])
        if m:
            mx = max(mx, int(m.group(1)))
    return f"t{mx + 1}"


# Project-qualified card ids. Stored ids stay bare `tN` (canonical key in tasks.md/.json + markers);
# qualification is presentation (display SLUG-tN) + input-acceptance (resolvers strip the qualifier).
# The qualifier is whatever precedes the final `-tN` / `/tN` — slugs may contain hyphens, so we anchor
# on the trailing t-number rather than splitting on the first separator.
_QUALIFIED_TID_RE = re.compile(r"^\s*(?:(?P<slug>.+)[-/])?(?P<tid>t\d+)\s*$", re.I)


def _split_qualified_tid(raw: str) -> tuple[str | None, str]:
    """Parse a (possibly) project-qualified card id → (slug_or_None, bare_tN). Accepts the displayed
    `SLUG-tN`, the marker form `slug/tN`, and bare `tN`. Returns (None, stripped_raw) when `raw` isn't
    a recognizable id, so callers fall back to an exact match. The bare id is lower-cased to match the
    stored `tN` key (display may upper-case the slug but never the t-number)."""
    m = _QUALIFIED_TID_RE.match(raw or "")
    if not m:
        return None, (raw or "").strip()
    return (m.group("slug") or None), m.group("tid").lower()


def _display_tid(ctx: Path | None, tid: str, *, cfg: dict | None = None, slug: str | None = None) -> str:
    """Render a card id for human output. Qualified → `SLUG-tN`; otherwise the bare `tN`. Qualify when
    an explicit `slug` is passed (cross-project views like fleet, where ambiguity is real) or when the
    local `qualified_task_ids` config is on. Never changes the stored id. `ctx` may be None ONLY when an
    explicit `slug` is supplied (the fleet path) — the local-config lookup is then skipped entirely."""
    if slug is None:
        cfg = cfg if cfg is not None else load_config(ctx)
        if not cfg.get("qualified_task_ids"):
            return tid
        slug = project_slug(ctx)
    return f"{slug.upper()}-{tid}" if slug else tid


def _resolve_task_arg(ctx: Path, raw: str) -> str:
    """Accept a project-qualified id (`SLUG-tN`, `slug/tN`) or bare `tN` from the CLI → bare `tN`.
    Warns (never fails) when the qualifier names a different project than this board — the local board
    only holds its own cards, so we resolve the bare id locally and let the caller's lookup decide."""
    slug, bare = _split_qualified_tid(raw)
    if slug:
        local = project_slug(ctx)
        if not _slug_matches(slug, local):
            warn(f"{raw!r}: project qualifier {slug!r} ≠ this board ({local!r}); resolving {bare} locally.")
    return bare


def _find_task(tasks: list[dict], tid: str) -> dict | None:
    _, bare = _split_qualified_tid(tid)
    return next((t for t in tasks if t["id"] == bare), None)


# --------------------------------------------------------------------------- progress engine (CELE-t161)
#
# Cards used to sit at 0% for their whole life because the working agent forgot to check off milestones
# and crest the bar. The fix is two-tier truth: the displayed bar = max(engine_floor, agent_set), capped
# 99 while doing. A DETERMINISTIC engine raises an honest floor from observable signals Celeborn already
# emits (commits carrying the `Celeborn-Task` trailer, touches, test runs, deploys); the working agent
# can always crest higher with semantic judgment (nudged via the UserPromptSubmit channel, never
# required). The engine is monotonic (only ever raises), idempotent, capped at 99 while doing, never
# overrides a higher manual value, and no-ops on todo/done. No LLM, no network — pure stdlib.
#
# State lives in `.context/progress.json` (local, gitignored, sibling of activity.md), one record per
# doing card. The bar ITSELF lives on the card (tasks.json `progress`), written through the normal
# task-save path so hosted autopush + the board just work. progress.json is engine bookkeeping only.
PROGRESS_NAME = "progress.json"
PROGRESS_SCHEMA = "celeborn-progress/1"
CLAIM_FLOOR = 5          # the instant a card goes doing
WORK_FLOOR = 10          # first observable work signal
SIGNAL_RAMP_STEP = 8     # no-milestone fallback: +8 per distinct hard signal …
SIGNAL_RAMP_CAP = 60     # … capped at 60 absolute (the band formula takes over once milestones exist)
NUDGE_T1, NUDGE_T2, NUDGE_T3 = 2, 4, 6   # turns-since-movement thresholds for the nudge ladder

# keyword↔signal map — data-driven so it's trivially extensible. A milestone's TEXT is matched by the
# same patterns against the evidence corpus: code ticks the machine-verifiable, the agent handles
# judgment ("reads cleanly", "UX feels right" — no pattern, left for the agent / Part C).
MILESTONE_SIGNALS = [
    (re.compile(r'\b(commit|committed)\b', re.I), 'commit'),
    (re.compile(r'\b(test|tests|suite|green|passing|tsc)\b', re.I), 'tests_green'),
    (re.compile(r'\b(deploy|deployed|ship|shipped|push|pushed|prod)\b', re.I), 'deploy'),
    (re.compile(r'\b(merge|merged|\bpr\b|pull request)\b', re.I), 'merge'),
]


def _progress_path(ctx: Path) -> Path:
    return ctx / PROGRESS_NAME


def _load_progress(ctx: Path) -> dict:
    p = _progress_path(ctx)
    if not p.is_file():
        return {"schema": PROGRESS_SCHEMA, "cards": {}}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {"schema": PROGRESS_SCHEMA, "cards": {}}
    data.setdefault("schema", PROGRESS_SCHEMA)
    data.setdefault("cards", {})
    return data


def _save_progress(ctx: Path, data: dict) -> None:
    """Atomic write (tmp + os.replace) so a crash mid-write never corrupts the registry."""
    import os
    data["schema"] = PROGRESS_SCHEMA
    p = _progress_path(ctx)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, p)


def _progress_rec(data: dict, tid: str) -> dict:
    rec = (data.get("cards") or {}).get(tid)
    if rec is None:
        rec = {"engine_floor": 0, "last_progress": 0, "last_change_ts": "", "claimed_at": "",
               "work_started": False, "auto_ticked_idx": [], "auto_ticked": [],
               "nudge_level": 0, "turns_since_change": 0}
    return rec


def _commits_for_task(ctx: Path, tid: str, since_iso: str | None = None, limit: int = 200) -> list[dict]:
    """Commits whose message carries the `Celeborn-Task: tN` trailer for this card. Returns
    [{hash, ts(epoch int), subject, body}], newest first. Empty if not a git repo / no history."""
    import subprocess
    _, bare = _split_qualified_tid(tid)
    try:
        out = subprocess.run(
            ["git", "-C", str(ctx.parent), "log", f"-n{int(limit)}",
             "--format=%H%x1f%ct%x1f%s%x1f%b%x1e"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    since_epoch = None
    if since_iso:
        try:
            since_epoch = _dt.datetime.fromisoformat(since_iso).timestamp()
        except (ValueError, TypeError):
            since_epoch = None
    trailer = re.compile(rf"Celeborn-Task:\s*{re.escape(bare)}\b", re.I)
    rows = []
    for rec in out.stdout.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        parts = rec.split("\x1f")
        if len(parts) < 4:
            continue
        h, ts, subject, body = parts[0], parts[1], parts[2], parts[3]
        if not trailer.search(subject + "\n" + body):
            continue
        try:
            ets = int(ts)
        except ValueError:
            ets = 0
        if since_epoch is not None and ets and ets < since_epoch - 1:
            continue
        rows.append({"hash": h, "ts": ets, "subject": subject, "body": body})
    return rows


def _activity_signal_corpus(ctx: Path) -> str:
    """The mechanical 'Recent commands' / 'Recent commits' sections of activity.md — evidence of work
    actually run this session. Deliberately excludes 'Last prompt' (which would create false signals)."""
    p = ctx / "activity.md"
    if not p.is_file():
        return ""
    try:
        text = p.read_text()
    except OSError:
        return ""
    out, keep = [], False
    for line in text.splitlines():
        if line.startswith("## Recent commands") or line.startswith("## Recent commits"):
            keep = True
            continue
        if line.startswith("## "):
            keep = False
            continue
        if keep:
            out.append(line)
    return "\n".join(out)


def _task_has_touch(ctx: Path, tid: str) -> bool:
    _, bare = _split_qualified_tid(tid)
    for meta in (_load_touches(ctx).get("files") or {}).values():
        t = (meta or {}).get("task") or ""
        if t and (_split_qualified_tid(t)[1] == bare):
            return True
    return False


def _progress_signals(ctx: Path, card: dict, rec: dict) -> dict:
    """Gather observable evidence for the card since it was claimed. Returns
    {present: set(tokens), commits: int, touched: bool, corpus: str}."""
    since = rec.get("claimed_at") or ""
    commits = _commits_for_task(ctx, card["id"], since_iso=since or None)
    corpus = _activity_signal_corpus(ctx) + "\n" + "\n".join(
        c["subject"] + "\n" + c["body"] for c in commits)
    present = set()
    if commits:
        present.add("commit")
    for rx, token in MILESTONE_SIGNALS:
        if rx.search(corpus):
            present.add(token)
    return {"present": present, "commits": len(commits), "touched": _task_has_touch(ctx, card["id"]),
            "corpus": corpus}


def _auto_tick_milestones(card: dict, signals: dict, rec: dict) -> list[int]:
    """Tick each unchecked milestone whose text matches a present signal — once per milestone (a human
    uncheck is respected). Judgment milestones (no matching pattern) are left for the agent."""
    present = signals.get("present") or set()
    already = set(rec.get("auto_ticked_idx") or [])
    credited = list(rec.get("auto_ticked") or [])
    newly = []
    for i, s in enumerate(card.get("subtasks") or []):
        if s.get("done") or i in already:
            continue
        for rx, token in MILESTONE_SIGNALS:
            if token in present and rx.search(s.get("text", "")):
                s["done"] = True
                newly.append(i)
                already.add(i)
                if token not in credited:
                    credited.append(token)
                break
    rec["auto_ticked_idx"] = sorted(already)
    rec["auto_ticked"] = credited
    return newly


def _engine_floor(card: dict, signals: dict, rec: dict) -> int:
    """The monotonic floor formula. Lifecycle (5 on claim, 10 on first work) is baked into
    rec['engine_floor'] by the stamps; here we add the milestone band (or a signal ramp when a card has
    no milestones) and take the high-water mark. Capped 99 while doing — ship is the only path to 100."""
    base = rec.get("engine_floor", 0) or 0
    # Before the first work signal a doing card rests at the claim floor (5) — the 10→99 band and the
    # signal ramp only kick in once work has started (so claim=5, first work=10, then climb).
    if not rec.get("work_started"):
        return min(99, base)
    subs = card.get("subtasks") or []
    if subs:
        total = sum(max(1, int(s.get("weight", 1))) for s in subs)
        done = sum(max(1, int(s.get("weight", 1))) for s in subs if s.get("done"))
        ratio = (done / total) if total else 0
        derived = WORK_FLOOR + round(ratio * 89)            # 10 → 99 span
    else:
        n_sig = len(signals.get("present") or set())
        derived = min(SIGNAL_RAMP_CAP, WORK_FLOOR + n_sig * SIGNAL_RAMP_STEP)
    return min(99, max(base, derived))


def _progress_engine_tick(ctx: Path, card: dict, *, count_turn: bool = False) -> dict:
    """Orchestrate one engine pass for a DOING card: detect work, auto-tick signal-backed milestones,
    raise the monotonic floor, and set card['progress'] = max(floor, current) capped 99. Idempotent;
    no-op on todo/done. Persists progress.json. Does NOT save the card (caller saves via _save_tasks).
    Returns {moved, newly, floor, signals, rec}."""
    if card.get("state") != "doing":
        return {"moved": False, "newly": [], "floor": 0, "signals": {"present": set()}, "rec": {}}
    data = _load_progress(ctx)
    rec = _progress_rec(data, card["id"])
    if not rec.get("claimed_at"):
        rec["claimed_at"] = now_iso()
    # high-water across both stores; doing ⇒ at least the claim floor.
    rec["engine_floor"] = max(int(rec.get("engine_floor", 0) or 0),
                              int(card.get("engine_floor", 0) or 0), CLAIM_FLOOR)
    signals = _progress_signals(ctx, card, rec)
    # First observable work signal → raise the floor to WORK_FLOOR (sticky). A checked milestone counts:
    # ticking a box is itself unambiguous evidence work has started.
    any_done = any(s.get("done") for s in (card.get("subtasks") or []))
    if not rec.get("work_started") and (signals["commits"] or signals["touched"]
                                        or signals["present"] or any_done):
        rec["work_started"] = True
        rec["engine_floor"] = max(int(rec.get("engine_floor", 0) or 0), WORK_FLOOR)
    newly = _auto_tick_milestones(card, signals, rec)
    floor = _engine_floor(card, signals, rec)
    rec["engine_floor"] = max(int(rec.get("engine_floor", 0) or 0), floor)   # monotonic high-water
    card["engine_floor"] = rec["engine_floor"]                               # persist so reload-recompute respects it
    new_prog = max(rec["engine_floor"], int(card.get("progress", 0) or 0))   # never override higher manual
    card["progress"] = new_prog
    _normalize_progress(card)                                                # caps 99 unless done
    moved = card["progress"] != rec.get("last_progress")
    if moved or newly:
        rec["last_progress"] = card["progress"]
        rec["last_change_ts"] = now_iso()
        rec["nudge_level"] = 0
        rec["turns_since_change"] = 0
    elif count_turn:
        rec["turns_since_change"] = int(rec.get("turns_since_change", 0) or 0) + 1
    data.setdefault("cards", {})[card["id"]] = rec
    _save_progress(ctx, data)
    return {"moved": moved, "newly": newly, "floor": rec["engine_floor"], "signals": signals, "rec": rec}


def _progress_stamp_claim(ctx: Path, card: dict) -> None:
    """The instant a card goes doing (in the claim path): floor 5 + a claimed_at anchor. Mutates
    card['progress'] so the claim's own _save_tasks(autopush) carries the 5% to the hosted board."""
    if card.get("state") != "doing":
        return
    data = _load_progress(ctx)
    rec = _progress_rec(data, card["id"])
    rec.setdefault("claimed_at", now_iso())
    rec["engine_floor"] = max(int(rec.get("engine_floor", 0) or 0), CLAIM_FLOOR)
    data.setdefault("cards", {})[card["id"]] = rec
    _save_progress(ctx, data)
    card["engine_floor"] = max(int(card.get("engine_floor", 0) or 0), CLAIM_FLOOR)  # marks the card engine-tracked
    card["progress"] = max(int(card.get("progress", 0) or 0), CLAIM_FLOOR)
    _normalize_progress(card)


def _session_task_id(ctx: Path, session: str | None) -> str:
    """The doing card this session is signed in to (via agent_sessions), or ''."""
    sid = (session or "").strip()
    if not sid:
        return ""
    tid = ((_load_metrics(ctx).get("agent_sessions") or {}).get(sid) or {}).get("task") or ""
    tid = tid.strip()
    if not tid:
        return ""
    card = next((t for t in _load_tasks(ctx) if t["id"] == _split_qualified_tid(tid)[1]), None)
    return tid if (card is not None and card.get("state") == "doing") else ""


def _progress_nudge_line(ctx: Path, card: dict, res: dict) -> str:
    """Compute the escalation level from turns-since-movement + unaccounted signals, craft a
    copy-pasteable line (real id + the obvious milestone number), and persist the new level. Empty when
    there is nothing to say. Tagged for surfacing by _compose_user_prompt_envelope."""
    rec = res.get("rec") or {}
    if card.get("state") != "doing":
        return ""
    disp = _display_tid(ctx, card["id"])
    pct = int(card.get("progress", 0) or 0)
    turns = int(rec.get("turns_since_change", 0) or 0)
    signals = res.get("signals") or {}
    n_commits = int(signals.get("commits", 0) or 0)
    subs = card.get("subtasks") or []
    # The obvious move: first unchecked milestone whose text matches a present signal, else first unchecked.
    present = signals.get("present") or set()
    target = None
    for i, s in enumerate(subs):
        if s.get("done"):
            continue
        if any((token in present and rx.search(s.get("text", ""))) for rx, token in MILESTONE_SIGNALS):
            target = i + 1
            break
    if target is None:
        target = next((i + 1 for i, s in enumerate(subs) if not s.get("done")), None)
    check_cmd = f"celeborn tasks check {card['id']} {target}" if target else f"celeborn tasks edit {card['id']} --progress {min(99, pct + 10)}"

    level = 0
    if turns >= NUDGE_T3:
        level = 3
    elif turns >= NUDGE_T2 or (n_commits and target is not None):
        level = 2
    elif turns >= NUDGE_T1:
        level = 1
    rec["nudge_level"] = level
    data = _load_progress(ctx)
    data.setdefault("cards", {})[card["id"]] = rec
    _save_progress(ctx, data)
    if level == 0:
        return ""
    if level == 1:
        return f"🏹 Celeborn —> {disp} at {pct}% — tick any finished milestones: {check_cmd}"
    if level == 2:
        ev = f"{n_commits} commit{'s' if n_commits != 1 else ''} landed" if n_commits else "work is moving"
        return (f"🏹 Celeborn —> {disp}: {ev}, bar hasn't moved. Check off completed milestones now → "
                f"{check_cmd}")
    return (f"🏹 Celeborn —> {disp}: auto-advanced to {pct}% from commit/test signals. Crest higher if "
            f"more is done: celeborn tasks edit {card['id']} --progress {min(99, pct + 10)}")


def _progress_hook(ctx: Path, session: str | None) -> str:
    """UserPromptSubmit entry point: tick the session's doing card and return the nudge line (or '')."""
    tid = _session_task_id(ctx, session)
    if not tid:
        return ""
    tasks = _load_tasks(ctx)
    card = _find_task(tasks, tid)
    if card is None or card.get("state") != "doing":
        return ""
    res = _progress_engine_tick(ctx, card, count_turn=True)
    if res["moved"] or res["newly"]:
        card["updated"] = now_iso()
        _save_tasks(ctx, tasks, autopush_ids=[card["id"]])
    return _progress_nudge_line(ctx, card, res)


def cmd_progress(args):
    """`celeborn progress [<id>] [--explain]` — run the deterministic progress engine once for a card
    (or every doing card) and show the signals → floor derivation. Debug/inspection; also moves the bar."""
    ctx = require_context(args)
    tasks = _load_tasks(ctx)
    tid = getattr(args, "id", None)
    if tid:
        resolved = _split_qualified_tid(_resolve_task_arg(ctx, tid))[1]
        targets = [t for t in tasks if t["id"] == resolved]
        if not targets:
            die(f"no task with id {tid!r}")
    else:
        targets = [t for t in tasks if t.get("state") == "doing"]
    if not targets:
        info("no doing cards to evaluate.")
        return
    explain = getattr(args, "explain", False)
    for card in targets:
        disp = _display_tid(ctx, card["id"])
        if card.get("state") != "doing":
            info(f"[{disp}] is {card.get('state')} — the engine only runs on doing cards.")
            continue
        before = int(card.get("progress", 0) or 0)
        res = _progress_engine_tick(ctx, card)
        if res["moved"] or res["newly"]:
            card["updated"] = now_iso()
            _save_tasks(ctx, tasks, autopush_ids=[card["id"]])
        print(f"[{disp}] {before}% → {card['progress']}%  (engine floor {res['floor']})")
        if explain:
            sig = res["signals"]
            present = ", ".join(sorted(sig.get("present") or [])) or "none"
            subs = card.get("subtasks") or []
            done = sum(1 for s in subs if s.get("done"))
            rec = res["rec"]
            print(f"    signals present  : {present}")
            print(f"    commits (trailer): {sig.get('commits', 0)}    touched: {sig.get('touched', False)}")
            print(f"    milestones       : {done}/{len(subs)} checked"
                  + (f"  (auto-ticked this run: {res['newly']})" if res["newly"] else ""))
            print(f"    work_started     : {rec.get('work_started')}    nudge_level: "
                  f"{rec.get('nudge_level')}    turns_since_change: {rec.get('turns_since_change')}")


def cmd_alert(args):
    """`celeborn alert <id> [--message …] [--kind permission|idle|stopped] [--session …]` — the
    reusable "coding progress is blocked, the user's input is needed" service (CELE-t169). Raises a
    live alert on a DOING card so it surfaces on the board (locally + celeborn.thot.ai). The
    Notification/Stop hooks are its first callers; any external system can call it the same way.
      celeborn alert <id> --message "…"      raise/refresh an alert
      celeborn alert <id> --clear            drop it (also happens automatically when the user replies)
      celeborn alert --list                  show the live alerts on this board
    No focus-stealing OS dialog — the alert rides the card (dialogs rejected t47/t50/t62)."""
    ctx = require_context(args)
    if getattr(args, "list", False) or not getattr(args, "id", None):
        alerts = _load_alerts(ctx).get("alerts") or {}
        if not alerts:
            info("no live alerts.")
            return
        for tid, rec in sorted(alerts.items()):
            disp = _display_tid(ctx, tid)
            print(f"🔔 [{disp}] {rec.get('kind', 'idle')} — {rec.get('message') or '(no message)'}"
                  f"  ({rec.get('at', '')})")
        return
    resolved = _split_qualified_tid(_resolve_task_arg(ctx, args.id))[1]
    card = next((t for t in _load_tasks(ctx) if t["id"] == resolved), None)
    if card is None:
        die(f"no task with id {args.id!r}")
    disp = _display_tid(ctx, resolved)
    if getattr(args, "clear", False):
        cleared = _clear_alert(ctx, resolved)
        _refresh_alerted_card(ctx, resolved)
        info(f"cleared alert on [{disp}]" if cleared else f"[{disp}] had no alert")
        return
    if card.get("state") != "doing":
        die(f"[{disp}] is {card.get('state')} — only a doing card can be blocked.")
    kind = getattr(args, "kind", None) or "idle"
    if kind not in ALERT_KINDS:
        die(f"--kind must be one of {', '.join(ALERT_KINDS)}")
    rec = _set_alert(ctx, resolved, kind, getattr(args, "message", "") or "", getattr(args, "session", "") or "")
    _refresh_alerted_card(ctx, resolved)
    # Push the alert to the hosted board now (not on the throttled heartbeat) so a remote watcher
    # sees the block promptly. Best-effort; a no-op when hosted sync isn't configured.
    try:
        __import__("celeborn_sync").schedule_agents_push(ctx, min_interval_s=0)
    except Exception:  # noqa: BLE001
        pass
    ok(f"🔔 alerted [{disp}] — {rec['kind']}" + (f": {rec['message']}" if rec.get("message") else ""))


def _reorder_task(tasks: list[dict], tid: str, direction: str) -> list[dict]:
    """Reprioritize a task within its own column (state group). Display order within a column is
    list order, so we permute only the same-state siblings among the slots they already occupy —
    tasks in other states keep their absolute positions. `direction`: up | down | top | bottom."""
    target = _find_task(tasks, tid)
    if not target:
        return tasks
    slots = [i for i, t in enumerate(tasks) if t["state"] == target["state"]]
    sibs = [tasks[i] for i in slots]
    pos = next(i for i, t in enumerate(sibs) if t["id"] == tid)
    if direction == "up":
        new = max(0, pos - 1)
    elif direction == "down":
        new = min(len(sibs) - 1, pos + 1)
    elif direction == "top":
        new = 0
    elif direction == "bottom":
        new = len(sibs) - 1
    else:
        return tasks
    sibs.insert(new, sibs.pop(pos))
    out = list(tasks)
    for slot, sib in zip(slots, sibs):
        out[slot] = sib
    return out


def _bring_to_state_front(tasks: list[dict], tid: str) -> list[dict]:
    """Move task `tid` to the front of its own state group in the flat list, so it renders at the
    *top* of that column. Used when a task is completed: the newest-done card arrives on top and
    pushes older done cards down (design: card-assignment.md / done-column ordering)."""
    t = _find_task(tasks, tid)
    if not t:
        return tasks
    rest = [x for x in tasks if x["id"] != tid]
    idx = next((i for i, x in enumerate(rest) if x["state"] == t["state"]), len(rest))
    rest.insert(idx, t)
    return rest


def _done_tasks_ordered(tasks: list[dict]) -> list[dict]:
    """Done cards in board column order (top/newest first, bottom/oldest last)."""
    return [t for t in tasks if t["state"] == "done"]


def _append_done_archive(ctx: Path, cards: list[dict], cfg: dict) -> None:
    """Append overflow done cards to done-archive.md; drop oldest entries past the FIFO cap."""
    if not cards:
        return
    path = ctx / DONE_ARCHIVE_FILE
    existing = _parse_tasks(path.read_text()) if path.is_file() else []
    combined = existing + cards
    cap = int(cfg.get("done_archive_keep_cards", DEFAULTS["done_archive_keep_cards"]))
    if len(combined) > cap:
        combined = combined[len(combined) - cap:]
    path.write_text(_render_tasks(combined, header=DONE_ARCHIVE_HEADER))


def _archive_overflow_done(ctx: Path, tasks: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Move done cards past `done_keep_cards` off the board into done-archive.md."""
    keep = int(cfg.get("done_keep_cards", DEFAULTS["done_keep_cards"]))
    done_ordered = _done_tasks_ordered(tasks)
    if len(done_ordered) <= keep:
        return tasks, []
    overflow = done_ordered[keep:]
    overflow_ids = {t["id"] for t in overflow}
    remaining = [t for t in tasks if t["id"] not in overflow_ids]
    _append_done_archive(ctx, overflow, cfg)
    return remaining, overflow


def cmd_tasks(args):
    ctx = require_context(args)
    tasks = _load_tasks(ctx)
    action = getattr(args, "task_cmd", None) or "list"
    # Accept project-qualified ids (SLUG-tN, slug/tN) anywhere a single card id is taken.
    if getattr(args, "id", None):
        args.id = _resolve_task_arg(ctx, args.id)

    if action == "add":
        tid = _next_task_id(tasks)
        stamp = now_iso()
        t = {
            "id": tid,
            "title": args.title.strip(),
            "state": args.state,
            "owner": (args.owner or "").strip(),
            "tags": _csv(args.tags),
            "blocked_by": _csv(args.blocked_by),
            "phase": (args.phase or "").strip(),
            # Stop condition (CELE-t81): use the supplied --stop, else auto-fill the generic default so
            # no card is ever stop-less. The agent protocol nudges the owner to replace the default.
            "stop": (getattr(args, "stop", "") or "").strip() or DEFAULT_STOP,
            "progress": _clamp_pct(getattr(args, "progress", 0) or 0),
            "jira": "",
            "created": stamp,
            "updated": stamp,
            "subtasks": [],
            "notes": (args.note or "").strip(),
        }
        tasks.append(t)
        tasks = _bring_to_state_front(tasks, tid)  # newest card lands on top of its column
        _save_tasks(ctx, tasks, autopush_ids=[tid])
        print(f"Added [{_display_tid(ctx, tid)}] {t['title']}  ({t['state']})")
        if getattr(args, "claim", False):
            by = _claim_identity(args)
            _claim_preflight(ctx, tasks, by, [tid], force=getattr(args, "force", False))
            t["owner"] = by
            if t["state"] == "todo":
                t["state"] = "doing"
            t["updated"] = now_iso()
            tasks = _bring_to_state_front(tasks, tid)
            _save_tasks(ctx, tasks, autopush_ids=[tid])
            # Write the session→card link too (CELE-t194) so an add-and-claim from a Bash call gets a
            # context-token chip like a pasted claim — the fix isn't complete if only `claim` links.
            _record_agent_session(ctx, _resolve_session(args), by, [tid])
            print(f"Claimed [{_display_tid(ctx, tid)}] {t['title']} → {by or 'unassigned'}")
        return

    if action == "move":
        t = _find_task(tasks, args.id)
        if not t:
            die(f"no task with id {args.id!r}")
        prev = t["state"]
        if args.state == "done" and prev == "doing":
            _require_crest_for_done(ctx, t)   # CELE-t176: must be crested to 99 before leaving DOING
        t["state"] = args.state
        t["updated"] = now_iso()
        if args.state == "done" and prev != "done":
            tasks = _bring_to_state_front(tasks, t["id"])   # newest-done lands on top of the column
        _save_tasks(ctx, tasks, autopush_ids=[t["id"]])
        print(f"[{_display_tid(ctx, t['id'])}] {t['title']} → {args.state}")
        return

    if action == "reorder":
        t = _find_task(tasks, args.id)
        if not t:
            die(f"no task with id {args.id!r}")
        tasks = _reorder_task(tasks, args.id, args.dir)
        _save_tasks(ctx, tasks)
        print(f"[{_display_tid(ctx, t['id'])}] {t['title']} → {args.dir} (within {t['state']})")
        return

    if action == "edit":
        t = _find_task(tasks, args.id)
        if not t:
            die(f"no task with id {args.id!r}")
        prev_state = t["state"]
        if args.title is not None:
            t["title"] = args.title.strip()
        if args.state is not None:
            t["state"] = args.state
        if args.owner is not None:
            t["owner"] = args.owner.strip()
        if args.tags is not None:
            t["tags"] = _csv(args.tags)
        if args.blocked_by is not None:
            t["blocked_by"] = _csv(args.blocked_by)
        if args.phase is not None:
            t["phase"] = args.phase.strip()
        if getattr(args, "stop", None) is not None:
            t["stop"] = args.stop.strip()
        if getattr(args, "progress", None) is not None:
            t["progress"] = _clamp_pct(args.progress)
        if args.note is not None:
            t["notes"] = args.note.strip()
        # CELE-t176: a DOING card edited to `done` must be crested. Honor any --progress set in the
        # SAME call (applied just above) before gating — so `edit --progress 99 --state done` works.
        if t["state"] == "done" and prev_state == "doing":
            _require_crest_for_done(ctx, t)
        t["updated"] = now_iso()
        if t["state"] == "done" and prev_state != "done":
            tasks = _bring_to_state_front(tasks, t["id"])   # newest-done lands on top of the column
        _save_tasks(ctx, tasks, autopush_ids=[t["id"]])
        print(f"Updated [{_display_tid(ctx, t['id'])}] {t['title']}")
        return

    if action in ("subtasks", "check", "uncheck"):
        t = _find_task(tasks, args.id)
        if not t:
            die(f"no task with id {args.id!r}")
        t.setdefault("subtasks", [])
        disp = _display_tid(ctx, t["id"])

        if action in ("check", "uncheck"):
            n = getattr(args, "n", 0)
            if not t["subtasks"]:
                die(f"[{disp}] has no subtasks — add some with `celeborn tasks subtasks {t['id']} add \"...\"`")
            if n < 1 or n > len(t["subtasks"]):
                die(f"subtask {n} out of range (1..{len(t['subtasks'])})")
            t["subtasks"][n - 1]["done"] = (action == "check")
        else:  # subtasks add | set | rm | list
            sub_cmd = getattr(args, "subtask_cmd", None) or "list"
            if sub_cmd == "add":
                spec = " ".join(args.text)
                item = _parse_subtask_spec(spec)
                if getattr(args, "weight", None):
                    item["weight"] = max(1, int(args.weight))
                t["subtasks"].append(item)
            elif sub_cmd == "set":
                t["subtasks"] = [_parse_subtask_spec(s) for s in args.items if s.strip()]
            elif sub_cmd == "rm":
                n = args.n
                if n < 1 or n > len(t["subtasks"]):
                    die(f"subtask {n} out of range (1..{len(t['subtasks'])})")
                t["subtasks"].pop(n - 1)
            # list falls through to the print below

        _recompute_progress(t)
        # CELE-t161: for an ENGINE-TRACKED doing card the engine owns the bar — its floor encodes the
        # milestone band and is monotonic, so re-assert it here (it never lowers, and preserves a higher
        # manual value). Gated to cards already tracked (claimed through the engine) so a plain
        # `add --state doing` card keeps the pure-ratio CELE-t106 behavior, uncheck included.
        if t.get("state") == "doing" and t["id"] in (_load_progress(ctx).get("cards") or {}):
            try:
                _progress_engine_tick(ctx, t)
            except Exception:  # noqa: BLE001 — progress is best-effort; never break a check
                pass
        if action != "subtasks" or getattr(args, "subtask_cmd", None):
            t["updated"] = now_iso()
            _save_tasks(ctx, tasks, autopush_ids=[t["id"]])
        # render the checklist
        subs = t["subtasks"]
        if not subs:
            print(f"[{disp}] no subtasks yet. Add: `celeborn tasks subtasks {t['id']} add \"<text>\" [--weight N]`")
            return
        done_w = sum(max(1, int(s.get("weight", 1))) for s in subs if s.get("done"))
        tot_w = sum(max(1, int(s.get("weight", 1))) for s in subs)
        print(f"[{disp}] {t['title']}  —  {t['progress']}%  ({done_w}/{tot_w} weighted)")
        for i, s in enumerate(subs, 1):
            box = "✓" if s.get("done") else "○"
            w = f"  ×{int(s.get('weight', 1))}" if int(s.get("weight", 1)) != 1 else ""
            print(f"  {i:>2}. {box} {s.get('text', '')}{w}")
        return

    if action == "rm":
        t = _find_task(tasks, args.id)
        if not t:
            die(f"no task with id {args.id!r}")
        removed = t["id"]
        tasks = [x for x in tasks if x["id"] != removed]
        # autopush the removed id: the live push drops the now-gone card from the hosted board too.
        _save_tasks(ctx, tasks, autopush_ids=[removed])
        print(f"Removed [{_display_tid(ctx, removed)}] {t['title']}")
        return

    if action == "archive":
        cfg = load_config(ctx)
        keep = args.keep if args.keep is not None else cfg["done_keep_cards"]
        done_n = len(_done_tasks_ordered(tasks))
        tasks, archived = _archive_overflow_done(ctx, tasks, {**cfg, "done_keep_cards": keep})
        if not archived:
            print(f"Done column has {done_n} card(s) (keep {keep}); nothing to archive.")
            return
        _tasks_path(ctx).write_text(_render_tasks(tasks))
        _write_tasks_json(ctx, tasks)
        print(f"Archived {len(archived)} done card(s) → {DONE_ARCHIVE_FILE}; kept {keep} on board.")
        print("Re-run `celeborn index` to refresh search.")
        return

    if action == "show":
        t = _find_task(tasks, args.id)
        if not t:
            die(f"no task with id {args.id!r}")
        print(f"[{_display_tid(ctx, t['id'])}] {t['title']}")
        print(f"  state:      {t['state']}")
        print(f"  owner:      {t['owner'] or '—'}")
        print(f"  tags:       {', '.join(t['tags']) or '—'}")
        print(f"  blocked-by: {', '.join(t['blocked_by']) or '—'}")
        print(f"  phase:      {t['phase'] or '—'}")
        print(f"  stop:       {t.get('stop') or '—'}")
        print(f"  jira:       {t.get('jira') or '—'}")
        print(f"  created:    {t['created'] or '—'}")
        print(f"  updated:    {t['updated'] or '—'}")
        if t["notes"]:
            print("\n" + t["notes"])
        print("\n" + _agent_card_protocol(t["id"]))
        return

    if action == "json":
        _write_tasks_json(ctx, tasks)
        if getattr(args, "out", None):
            Path(args.out).write_text(json.dumps(_tasks_doc(ctx, tasks), indent=2) + "\n")
            print(f"Wrote {_tasks_json_path(ctx)} and {args.out}")
        else:
            print(json.dumps(_tasks_doc(ctx, tasks), indent=2))
        return

    # default: list (the text board). Always refresh the derived JSON so the viewer is current.
    _write_tasks_json(ctx, tasks)
    if getattr(args, "json", False):
        print(json.dumps(_tasks_doc(ctx, tasks), indent=2))
        return
    if not tasks:
        print('No tasks yet. Add one:  celeborn tasks add "your first task"')
        return
    cfg = load_config(ctx)
    for s in TASK_STATES:
        col = [t for t in tasks if t["state"] == s]
        print(f"\n{TASK_STATE_LABELS[s]} ({len(col)})")
        for t in col:
            owner = f"  @{t['owner']}" if t["owner"] else ""
            blocked = f"  ⛔ {', '.join(t['blocked_by'])}" if t["blocked_by"] else ""
            print(f"  [{_display_tid(ctx, t['id'], cfg=cfg)}] {t['title']}{owner}{blocked}")


def _ambient_session_id() -> str:
    """The current Claude Code session id, as the harness injects it into EVERY tool subprocess
    (`CLAUDE_CODE_SESSION_ID`). This is the linchpin of CELE-t194: it lets an agent-initiated
    `celeborn claim` / `tasks add --claim` run from a Bash tool call be session-owned WITHOUT the
    agent remembering to pass `--session` — the hook's stdin session id never reaches a Bash
    subprocess, but this env var does. It's per-window (each session's shell inherits its own id),
    so it's multi-agent-safe where a repo-wide cursor (`last_session_id`) would misattribute.
    Empty outside a Claude Code window (a plain terminal) — there a manual `--by` still attributes."""
    import os
    return (os.environ.get("CLAUDE_CODE_SESSION_ID") or "").strip()


def _resolve_session(args) -> str:
    """The session id owning this command: explicit `--session` (the hook passes it) → the ambient
    `CLAUDE_CODE_SESSION_ID` the harness sets in every Bash tool call. Empty only outside a Claude
    window. Used for BOTH owner attribution (`_claim_identity`) and the session→card link
    (`_record_agent_session`) so an agent-typed claim tracks context exactly like a pasted one."""
    return (getattr(args, "session", None) or "").strip() or _ambient_session_id()


def _claim_identity(args) -> str:
    """Who owns the card. The SESSION is authoritative: whenever a real session id is resolvable —
    the `--session` the hook passes, OR the `CLAUDE_CODE_SESSION_ID` the harness injects into every
    Bash tool call — the card is owned by that session's short-id (6-char head), and the agent
    CANNOT rename it. The session IS the agent's name (CELE-t131); the code grabs it, not the model
    (CELE-t194 — this is what kills the recurring `@claude` / `@unknown` whimsy at the source). `--by`
    / `$CELEBORN_AGENT` attribute ONLY a session-less manual CLI run (a human at a plain terminal).

    Short head, not the full UUID, so it reads as a clean handle and the board never treats it as a
    raw session id. Mirrors _outbox_identity. Guard (CELE-t172): a model-shaped handle is never an
    owner — declare your model with `celeborn identify --model "…"`, not by stuffing it into --by."""
    explicit = (getattr(args, "by", None) or "").strip()
    sess = _resolve_session(args)
    import os
    env = (os.environ.get("CELEBORN_AGENT") or "").strip()
    if sess:
        short = sess[:6]
        # A live session owns the card, full stop. Surface (but ignore) any superseded --by so the
        # agent learns the code names the card, not it — with an extra nudge when the --by was a model
        # name (record that with `celeborn identify`, don't stuff it into the owner).
        if explicit and explicit != short:
            tail = (f" Record your model with `celeborn identify --model \"{explicit}\"`."
                    if _looks_like_model_handle(explicit) or explicit.lower() in _GENERIC_MODEL_FAMILIES
                    else "")
            warn(f"--by {explicit!r} is ignored — this card is owned by its session ({short}), not a "
                 f"name the agent chooses (CELE-t194).{tail}")
        return short
    handle = explicit or env
    if handle and _looks_like_model_handle(handle):
        warn(f"'{handle}' looks like a model, not an identity — a card should be owned by its "
             f"session or a handle, not a model. Record your model with `celeborn identify`.")
    return handle


def cmd_identify(args):
    """`celeborn identify --family <F> --model <M>` — declare who you are ONCE per session so every
    later touch/claim/ship shows your family + specific model (no per-command flags needed). Stored
    in the local `.context/.agents.json` registry, keyed by your handle (--as / $CELEBORN_AGENT).
    `--show` prints the known agents and exits."""
    ctx = require_context(args)
    if getattr(args, "show", False):
        agents = (_load_agents(ctx).get("agents") or {})
        if getattr(args, "json", False):
            print(json.dumps({"agents": agents}, indent=2))
            return
        if not agents:
            print("(no agents identified yet — run `celeborn identify --family … --model …`)")
            return
        for handle, e in sorted(agents.items()):
            label = _agent_label(e.get("family", ""), e.get("model", "")) or "(unknown model)"
            print(f"@{handle} — {label}")
        return
    handle = (getattr(args, "as_", None) or _claim_identity(args) or "").strip()
    if not handle:
        die("who are you? pass --as <handle> or set $CELEBORN_AGENT first.")
    family = (getattr(args, "family", None) or "").strip()
    model = (getattr(args, "model", None) or "").strip()
    if not family and not model:
        die("nothing to record — pass --family <Claude|Grok|GPT…> and/or --model \"<e.g. Opus 4.8>\".")
    entry = _register_agent(ctx, handle, family, model)
    label = _agent_label(entry.get("family", ""), entry.get("model", "")) or "(unknown model)"
    ok(f"identified @{handle} as {label}")


def cmd_claim(args):
    """`celeborn claim t13 [t14 …] [--by <agent>]` — claim-on-receipt. The act of a model receiving a
    card (its marker pasted into the chat, parsed by the UserPromptSubmit hook) assigns it: owner ←
    claimer, and a TODO card advances to DOING. Last claim wins — if a different owner held it, we
    reassign and say so (the board reflects the new owner; contention is surfaced, not silently lost)."""
    ctx = require_context(args)
    tasks = _load_tasks(ctx)
    by = _agent_identity(args, ctx)["handle"]  # resolves handle + records family/model for the board
    if getattr(args, "ids", None):  # accept project-qualified ids (SLUG-tN, slug/tN)
        args.ids = [_resolve_task_arg(ctx, x) for x in args.ids]
    claim_ids = list(getattr(args, "ids", None) or [])
    _claim_preflight(ctx, tasks, by, claim_ids, force=getattr(args, "force", False))
    results = []
    for tid in (getattr(args, "ids", None) or []):
        t = _find_task(tasks, tid)
        if not t:
            continue
        prev = (t.get("owner") or "").strip()
        t["owner"] = by
        if t["state"] == "todo":
            t["state"] = "doing"
        t["updated"] = now_iso()
        if t["state"] == "doing":
            _progress_stamp_claim(ctx, t)  # CELE-t161: engine floor 5 the instant a card goes doing
        tasks = _bring_to_state_front(tasks, tid)  # claimed card surfaces at top of its column
        who = by or "unassigned"
        disp = _display_tid(ctx, tid)
        if prev and prev != by:
            results.append(f"Reassigned [{disp}] {t['title']}: {prev} → {who} (last claim wins)")
        elif prev != by:
            results.append(f"Claimed [{disp}] {t['title']} → {who}")
    if results:
        claimed = [tid for tid in (getattr(args, "ids", None) or []) if _find_task(tasks, tid)]
        _save_tasks(ctx, tasks, autopush_ids=claimed)
    # Active-agents bridge (CELE-t131/t194): remember which session owns which card so `celeborn
    # agents` (and the fleet's context-token chip) can attribute that session's live context window to
    # this DOING card. The session is the RESOLVED one — the `--session` the hook passes OR the
    # ambient `CLAUDE_CODE_SESSION_ID` every Bash tool call inherits — so an agent-typed `celeborn
    # claim` tracks context identically to a pasted marker (the fix for cards with no context chip).
    # Runs even on a re-claim of a card you already own (no owner change → no `results` line, but the
    # session is still linked).
    owned_now = [tid for tid in claim_ids if (_find_task(tasks, tid) or {}).get("owner") == by]
    _record_agent_session(ctx, _resolve_session(args), by, owned_now)
    print("\n".join(results))


def cmd_ship(args):
    """`celeborn ship t42` — P0 close-out: release all touches tagged with the task, move it to Done.
    Prevents stale DOING cards after an agent releases files but forgets the kanban move."""
    ctx = require_context(args)
    tid = (getattr(args, "id", None) or "").strip()
    if not tid:
        die("usage: celeborn ship <task-id> [--note <ship note>]")
    tid = _resolve_task_arg(ctx, tid)  # accept project-qualified ids (SLUG-tN, slug/tN)
    tasks = _load_tasks(ctx)
    t = _find_task(tasks, tid)
    if not t:
        die(f"no task with id {tid!r}")
    # CELE-t176: a card leaving DOING for Done must be crested to 99 first. Gate BEFORE any side
    # effect (touch release / note append) so a refused ship leaves the card exactly as it was.
    # Only DOING is gated — a todo/blocked card shipped as triage isn't "in-flight work vanishing".
    if t["state"] == "doing":
        _require_crest_for_done(ctx, t)
    who = _agent_identity(args, ctx)["handle"]  # resolves handle + records family/model for the board
    _clear_alert(ctx, tid)   # CELE-t195: a shipped card awaits nothing — drop any stale blocked-alert
    released = _release_touches_for_task(ctx, tid)
    note = (getattr(args, "note", None) or "").strip()
    if note:
        t["notes"] = f"{t['notes']}\n\n{note}".strip() if t.get("notes") else note
    if who and not (t.get("owner") or "").strip():
        t["owner"] = who
    prev = t["state"]
    t["state"] = "done"
    t["updated"] = now_iso()
    if prev != "done":
        tasks = _bring_to_state_front(tasks, tid)
    _save_tasks(ctx, tasks, autopush_ids=[tid])
    # Verify the write stuck — agents must not assume ship succeeded from exit code alone.
    saved = _find_task(_load_tasks(ctx), tid)
    if not saved or saved["state"] != "done":
        die(f"ship [{_display_tid(ctx, tid)}] failed — board still shows {saved['state'] if saved else 'missing'}; re-run or check for a parallel session overwrite")
    ok(f"Shipped [{_display_tid(ctx, tid)}] {t['title']} → done")
    if released:
        print(f"  released {len(released)} touch(es): {', '.join(released)}")
    elif prev == "doing":
        print("  (no active touches for this card)")


# --------------------------------------------------------------------------- outbox (Phase 12)
#
# The prompt hand-off queue. The board's "Handoff" button (or `celeborn outbox push`) appends a
# prompt here; the UserPromptSubmit hook `drain`s pending entries each turn and injects them as the
# model's next instruction — the bridge from "I prioritized this card" to "the agent is now working
# on it". Local-only, gitignored, disposable: once drained, an entry moves to `outbox/sent.md` for
# provenance and the pending queue is emptied. No network — same machine, same .context/.
#
# Multi-agent routing (v0, design: references/card-assignment.md): the outbox is ONE FILE PER AGENT
# (`outbox/<agent>.md`), so several agents on one project can drain concurrently without clobbering
# each other (one writer, one reader per file). A card's `owner` is its assignee; pushing addresses
# the hand-off to that owner; an agent drains only its own file, its identity from $CELEBORN_AGENT.
# Unaddressed prompts land in `outbox/_unassigned.md` (today's single-queue behavior, claimable).

OUTBOX_DIR = "outbox"
OUTBOX_SENT_FILE = "sent.md"        # archive, lives inside OUTBOX_DIR
OUTBOX_UNASSIGNED = "_unassigned"   # agent slug for unaddressed prompts (any agent may claim them)


def _outbox_header(agent: str) -> str:
    who = agent or OUTBOX_UNASSIGNED
    return (
        f"# Prompt outbox · {who}\n\n"
        "<!-- Celeborn prompt hand-off queue (Phase 12). ONE FILE PER AGENT — the board's Handoff\n"
        "     button / `celeborn outbox push [--for <agent>]` appends here; that agent's UserPromptSubmit\n"
        "     hook drains it (`outbox drain`, identity from $CELEBORN_AGENT). One writer, one reader →\n"
        "     concurrency-safe across agents. Local-only, gitignored, disposable — drained entries move\n"
        "     to sent.md. One prompt per `##` block. -->\n"
    )


def _agent_slug(name: str) -> str:
    """A filesystem-safe agent id. Empty/blank → the shared unassigned queue."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", (name or "").strip()).strip("-").lower()
    return s or OUTBOX_UNASSIGNED


def _outbox_dir(ctx: Path) -> Path:
    return ctx / OUTBOX_DIR


def _outbox_file(ctx: Path, agent: str) -> Path:
    return _outbox_dir(ctx) / f"{_agent_slug(agent)}.md"


def _outbox_identity(args) -> str:
    """Who am I, for draining: explicit --for wins, else $CELEBORN_AGENT, else unassigned. This is
    how each concurrent agent pulls only the cards addressed to it (design: card-assignment.md)."""
    explicit = getattr(args, "for_", None)
    if explicit:
        return explicit.strip()
    import os
    return (os.environ.get("CELEBORN_AGENT") or "").strip()


CARD_REF_RE = re.compile(r"celeborn:\s*(?:(?P<slug>[\w.-]+)\s*/\s*)?(?P<tid>t\d+)", re.I)


def _card_marker(tid: str, slug: str) -> str:
    """Project-qualified card stamp: ⟨celeborn:slug/tN⟩. Prevents claiming tN in the wrong repo when
    a marker is pasted across projects. Parser still accepts legacy ⟨celeborn:tN⟩ in the same repo."""
    return f"⟨celeborn:{slug}/{tid}⟩"


def _find_card_refs(text: str, *, expected_slug: str | None = None) -> tuple[list[str], list[str]]:
    """Card ids to claim (first-seen order) and rejection lines for cross-project markers.
    Tolerant of stripped brackets; qualified markers must match expected_slug when it is set."""
    seen, out, rejects = set(), [], []
    for m in CARD_REF_RE.finditer(text or ""):
        slug, tid = (m.group("slug") or "").strip(), m.group("tid")
        if slug and expected_slug and slug.lower() != expected_slug.lower():
            rejects.append(
                f"  [{tid}] — marker project {slug!r} ≠ this repo {expected_slug!r} (not claimed)")
            continue
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out, rejects


# A project-qualified card id written in PROSE (the displayed `SLUG-tN` form, e.g. "continue with
# CELE-t131") — distinct from the pasted `celeborn:tN` marker that CARD_REF_RE catches. The literal
# `t` before the digits keeps it from matching Jira-style ids (SCRUM-115). Bare `tN` is deliberately
# NOT matched: a slug-less number in prose is too ambiguous to auto-claim.
PROSE_CARD_REF_RE = re.compile(r"\b(?P<slug>[A-Za-z][\w.]*)-(?P<tid>t\d+)\b", re.I)


def _find_prose_card_refs(text: str, *, expected_slug: str | None, claimable_ids: set[str]) -> list[str]:
    """Card ids (first-seen order) named in PROSE as `SLUG-tN` that belong to THIS board and are still
    open — the free-text counterpart to `_find_card_refs`'s pasted markers (CELE-t131). A human who
    types "continue with CELE-t131" into an unowned session is steering it onto that card; this lets
    the active-agents chip show the right owner+card without a manual `claim`. Mentions of another
    project's card, or of a shipped/unknown card, are silently skipped (no reject noise — unlike a
    pasted marker, a prose mention isn't an explicit claim attempt)."""
    seen, out = set(), []
    for m in PROSE_CARD_REF_RE.finditer(text or ""):
        slug, tid = m.group("slug"), m.group("tid").lower()
        if expected_slug and not _slug_matches(slug, expected_slug):
            continue                              # different project (or not a slug) — not ours
        if tid in seen or tid not in claimable_ids:
            continue                              # already taken this scan, or shipped/unknown card
        seen.add(tid)
        out.append(tid)
    return out


def _session_has_task(ctx: Path, session: str | None) -> bool:
    """Whether this session is already signed in to a still-live card. Gate for the prose sign-in: it
    only fills a VACUUM (a session not yet on a card), so a later casual mention of another card can't
    thrash the board. A shipped/abandoned card no longer counts, so the session can sign in to its next."""
    sid = (session or "").strip()
    if not sid:
        return False
    tid = ((_load_metrics(ctx).get("agent_sessions") or {}).get(sid) or {}).get("task")
    tid = (tid or "").strip()
    if not tid:
        return False
    card = next((t for t in _load_tasks(ctx) if t["id"] == tid), None)
    return card is not None and card.get("state") in ("doing",)


def _task_prompt(t: dict, ctx: Path) -> str:
    """Render a task into the prompt text that gets handed off / copied. Title is the instruction;
    notes ride along as detail. Kept deliberately plain so it reads as a natural user request — then
    the agent protocol block and a trailing card marker so the receiving session can claim it
    (design: card-assignment.md)."""
    body = t["title"].strip()
    if t.get("notes"):
        body += "\n\n" + t["notes"].strip()
    proto = t.get("agent_protocol") or _agent_card_protocol(t["id"])
    slug = project_slug(ctx)
    return body + "\n\n" + proto + "\n\n" + _card_marker(t["id"], slug)


def _outbox_blocks(text: str) -> list[str]:
    """Split outbox markdown into entry blocks (everything under each `## ` heading, heading included)."""
    return [("## " + b).rstrip() for b in re.split(r"(?m)^##[ \t]+", text)[1:]]


def _outbox_body(block: str) -> str:
    """The prompt text of one outbox block — the lines after the `## …` heading."""
    return "\n".join(block.splitlines()[1:]).strip()


def cmd_outbox(args):
    ctx = require_context(args)
    action = getattr(args, "outbox_cmd", None) or "list"
    d = _outbox_dir(ctx)

    if action == "push":
        if getattr(args, "task", None):
            t = _find_task(_load_tasks(ctx), args.task)
            if not t:
                die(f"no task with id {args.task!r}")
            prompt, tag = _task_prompt(t, ctx), f" [{_display_tid(ctx, t['id'])}]"
            # Addressing: explicit --for wins, else the card's owner (assigning the card addresses it).
            addressee = (getattr(args, "for_", None) or t.get("owner") or "").strip()
        elif getattr(args, "text", None):
            prompt, tag = args.text.strip(), ""
            addressee = (getattr(args, "for_", None) or "").strip()
        else:
            die("nothing to push — pass --task <id> or --text <prompt>")
        slug = _agent_slug(addressee)
        f = _outbox_file(ctx, addressee)
        d.mkdir(parents=True, exist_ok=True)
        existing = f.read_text() if f.is_file() else _outbox_header(addressee)
        forhint = f" for={slug}" if slug != OUTBOX_UNASSIGNED else ""
        entry = f"\n## queued {now_iso()}{tag}{forhint}\n{prompt}\n"
        f.write_text(existing.rstrip("\n") + "\n" + entry)
        print(f"Queued prompt to outbox{tag} → {slug if slug != OUTBOX_UNASSIGNED else 'unassigned'}")
        return

    if action == "drain":
        f = _outbox_file(ctx, _outbox_identity(args))
        if not f.is_file():
            return
        blocks = _outbox_blocks(f.read_text())
        if not blocks:
            return
        # Archive raw blocks for provenance, then empty this agent's pending queue.
        sent = d / OUTBOX_SENT_FILE
        prior = sent.read_text() if sent.is_file() else "# Prompt outbox — sent\n"
        sent.write_text(prior.rstrip("\n") + "\n\n" + "\n\n".join(blocks) + "\n")
        f.write_text(_outbox_header(_outbox_identity(args)))
        prompts = [_outbox_body(b) for b in blocks if _outbox_body(b)]
        print("\n\n---\n\n".join(prompts))
        return

    if action == "clear":
        target = getattr(args, "for_", None)
        if target:
            _outbox_file(ctx, target).write_text(_outbox_header(target))
            print(f"Cleared outbox for {_agent_slug(target)}")
            return
        if d.is_dir():
            for f in d.glob("*.md"):
                if f.name != OUTBOX_SENT_FILE:
                    f.unlink()
        print("Cleared the prompt outbox (all agents)")
        return

    # default: list pending entries, grouped by agent
    files = [f for f in sorted(d.glob("*.md")) if f.name != OUTBOX_SENT_FILE] if d.is_dir() else []
    groups, total = [], 0
    for f in files:
        blocks = _outbox_blocks(f.read_text())
        if not blocks:
            continue
        total += len(blocks)
        lines = [f"{f.stem} ({len(blocks)}):"]
        for b in blocks:
            head = b.splitlines()[0][3:].strip()
            first = next((ln for ln in b.splitlines()[1:] if ln.strip()), "")
            lines.append(f"  · {head} — {first[:60]}")
        groups.append("\n".join(lines))
    if total == 0:
        print("Outbox empty — nothing queued.")
        return
    print(f"{total} queued prompt(s):")
    for g in groups:
        print(g)


# --------------------------------------------------------------------------- argparse

# --------------------------------------------------------------------------- architecture (CELE-t187)

INFRA_LOCAL_NAME = "infra-local.json"
INFRA_SCHEMA = "celeborn-architecture/1"

# Node id → default fields for a vendor detected from repo signals / env-var names. Detection is
# best-effort and non-authoritative: it seeds a starter map the human/agent then edits. We read only
# file existence and env-var NAMES — never any secret value.
_INFRA_ENV_VENDORS = [
    ("ANTHROPIC", {"id": "anthropic", "name": "Anthropic API", "kind": "vendor", "vendor": "Anthropic",
                   "control_surface": "https://console.anthropic.com"}),
    ("OPENAI", {"id": "openai", "name": "OpenAI API", "kind": "vendor", "vendor": "OpenAI",
                "control_surface": "https://platform.openai.com"}),
    ("STRIPE", {"id": "stripe", "name": "Stripe", "kind": "vendor", "vendor": "Stripe",
                "control_surface": "https://dashboard.stripe.com"}),
    ("SUPABASE", {"id": "db", "name": "Database", "kind": "database", "vendor": "Supabase",
                  "control_surface": "https://supabase.com/dashboard"}),
    ("VERCEL", {"id": "web", "name": "Hosted App", "kind": "app", "vendor": "Vercel",
                "control_surface": "https://vercel.com/dashboard"}),
    ("CHATWOOT", {"id": "chatwoot", "name": "Chatwoot", "kind": "vendor", "vendor": "Chatwoot",
                  "control_surface": ""}),
    ("JIRA", {"id": "jira", "name": "Jira", "kind": "vendor", "vendor": "Atlassian",
              "control_surface": "https://admin.atlassian.com"}),
]

_INFRA_NODE_FIELDS = ("id", "name", "kind", "vendor", "role", "endpoint", "ip", "control_surface", "notes")

# Dependency-name → node template (CELE-t201). A NEW dependency in a manifest is the clearest "a piece
# entered the stack" signal, so the auto-trace (and init) map distinctive package tokens to vendor nodes.
# We read dependency NAMES only (never lockfile hashes or any secret). Tokens are chosen to be distinctive
# enough that a substring match over the manifest text won't false-positive on unrelated packages.
_INFRA_DEP_VENDORS = [
    (("@anthropic-ai", "anthropic"), {"id": "anthropic", "name": "Anthropic API", "kind": "vendor",
        "vendor": "Anthropic", "control_surface": "https://console.anthropic.com", "notes": "detected: dependency"}),
    (("openai",), {"id": "openai", "name": "OpenAI API", "kind": "vendor", "vendor": "OpenAI",
        "control_surface": "https://platform.openai.com", "notes": "detected: dependency"}),
    (("openrouter",), {"id": "openrouter", "name": "OpenRouter", "kind": "vendor", "vendor": "OpenRouter",
        "control_surface": "https://openrouter.ai", "notes": "detected: dependency"}),
    (("stripe",), {"id": "stripe", "name": "Stripe", "kind": "vendor", "vendor": "Stripe",
        "control_surface": "https://dashboard.stripe.com", "notes": "detected: dependency"}),
    (("@supabase", "supabase"), {"id": "db", "name": "Database", "kind": "database", "vendor": "Supabase",
        "control_surface": "https://supabase.com/dashboard", "notes": "detected: dependency"}),
    (("@vercel/",), {"id": "web", "name": "Hosted App", "kind": "app", "vendor": "Vercel",
        "control_surface": "https://vercel.com/dashboard", "notes": "detected: dependency"}),
    (("@aws-sdk", "boto3", "aws-sdk"), {"id": "aws", "name": "AWS", "kind": "vendor", "vendor": "AWS",
        "control_surface": "https://console.aws.amazon.com", "notes": "detected: dependency"}),
    (("mongoose", "mongodb"), {"id": "mongo", "name": "MongoDB", "kind": "database", "vendor": "MongoDB",
        "control_surface": "https://cloud.mongodb.com", "notes": "detected: dependency"}),
    (("ioredis", "@upstash/redis"), {"id": "redis", "name": "Redis", "kind": "database", "vendor": "Redis",
        "control_surface": "", "notes": "detected: dependency"}),
    (("twilio",), {"id": "twilio", "name": "Twilio", "kind": "vendor", "vendor": "Twilio",
        "control_surface": "https://console.twilio.com", "notes": "detected: dependency"}),
    (("@sendgrid", "sendgrid"), {"id": "sendgrid", "name": "SendGrid", "kind": "vendor", "vendor": "SendGrid",
        "control_surface": "https://app.sendgrid.com", "notes": "detected: dependency"}),
    (("resend",), {"id": "resend", "name": "Resend", "kind": "vendor", "vendor": "Resend",
        "control_surface": "https://resend.com", "notes": "detected: dependency"}),
]

# Manifest basenames (lowercased) whose EDIT means a dependency may have been added → trace immediately.
# The same set is scanned (root + one dir level) for the dependency tokens above.
_INFRA_MANIFESTS = ("package.json", "requirements.txt", "pyproject.toml", "go.mod", "gemfile",
                    "cargo.toml", "composer.json", "pipfile")


def _infra_path(ctx: Path) -> Path:
    return ctx / INFRA_LOCAL_NAME


def load_infra(ctx: Path) -> dict:
    """Read .context/infra-local.json (gitignored, per-machine). {} when absent/invalid."""
    p = _infra_path(ctx)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        warn(f"{INFRA_LOCAL_NAME} is not valid JSON; treating as empty.")
        return {}


def _full_node(partial: dict) -> dict:
    """Fill a partial node dict out to the full field set (empty strings for missing fields)."""
    return {f: partial.get(f, "") for f in _INFRA_NODE_FIELDS}


def _detect_infra_nodes(root: Path) -> list[dict]:
    """Best-effort starter nodes from repo signals + env-var NAMES (never values). Deduped by id."""
    nodes: dict[str, dict] = {}
    # The local CLI is always a node (the client edge of every flow).
    nodes["cli"] = {"id": "cli", "name": "Celeborn CLI", "kind": "client", "vendor": "local",
                    "role": "developer machine", "endpoint": "localhost", "notes": ""}
    # File signals.
    if any((root / f).exists() for f in ("vercel.json", "vercel.ts")) or \
       any((root / d / "next.config.js").is_file() for d in (".", "web")):
        nodes.setdefault("web", {"id": "web", "name": "Hosted App", "kind": "app", "vendor": "Vercel",
                                 "control_surface": "https://vercel.com/dashboard", "notes": "detected: Vercel/Next"})
    if (root / "supabase").is_dir():
        nodes.setdefault("db", {"id": "db", "name": "Database", "kind": "database", "vendor": "Supabase",
                                "role": "postgres", "control_surface": "https://supabase.com/dashboard",
                                "notes": "detected: supabase/"})
    # Env-var NAME signals (from any .env* at the root — names only, never values).
    env_names: set[str] = set()
    try:
        for envf in root.glob(".env*"):
            if not envf.is_file():
                continue
            for line in envf.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                env_names.add(line.split("=", 1)[0].strip().upper())
    except OSError:
        pass
    for prefix, tmpl in _INFRA_ENV_VENDORS:
        if any(name.startswith(prefix) for name in env_names):
            nodes.setdefault(tmpl["id"], dict(tmpl))
    # Dependency-manifest NAME signals (CELE-t201): scan known manifests at the root and one dir level
    # for distinctive package tokens → vendor nodes. Names only; lockfiles and values are never read.
    manifest_text = ""
    seen_manifest: set[Path] = set()
    for name in _INFRA_MANIFESTS:
        for cand in (root / name, *root.glob(f"*/{name}")):
            if cand in seen_manifest or not cand.is_file():
                continue
            seen_manifest.add(cand)
            try:
                manifest_text += "\n" + cand.read_text(errors="ignore")[:200_000].lower()
            except OSError:
                pass
    if manifest_text:
        for tokens, tmpl in _INFRA_DEP_VENDORS:
            if any(tok in manifest_text for tok in tokens):
                nodes.setdefault(tmpl["id"], dict(tmpl))
    return [_full_node(n) for n in nodes.values()]


def _architecture_init(ctx: Path, force: bool = False) -> None:
    p = _infra_path(ctx)
    if p.is_file() and not force:
        die(f"{INFRA_LOCAL_NAME} already exists (use --force to overwrite).")
    nodes = _detect_infra_nodes(ctx.parent)
    # Seed a naive flow from the CLI to the first detected server-side node so `sync` renders something.
    flows: list[dict] = []
    server_ids = [n["id"] for n in nodes if n["id"] != "cli"]
    if server_ids:
        flows.append({"from": "cli", "to": server_ids[0], "label": "sync push", "protocol": "https"})
    doc = {
        "schema": INFRA_SCHEMA,
        "updated": now_iso(),
        "_readme": ("Per-project architecture diagram (CELE-t187). Non-secret topology only. "
                    "`celeborn architecture sync` pushes nodes+flows to your hosted board with the "
                    "`credentials` block STRIPPED. Never put keys/tokens/passwords here — env NAMES only."),
        "nodes": nodes,
        "flows": flows,
        "credentials": {"_note": "NEVER synced — store env-var NAMES only, never values."},
    }
    p.write_text(json.dumps(doc, indent=2) + "\n")
    ok(f"wrote {INFRA_LOCAL_NAME} with {len(nodes)} detected node(s).")
    print("  Edit it to add IPs, endpoints, control-surface URLs, and information flows, then")
    print("  `celeborn architecture sync` to push it to your hosted board (Pro). It's gitignored.")


def _architecture_show(ctx: Path) -> None:
    doc = load_infra(ctx)
    nodes = doc.get("nodes") or []
    flows = doc.get("flows") or []
    if not nodes:
        print(f"No {INFRA_LOCAL_NAME} yet. Run `celeborn architecture init`.")
        return
    print(f"Architecture — {len(nodes)} node(s), {len(flows)} flow(s):")
    for n in nodes:
        head = f"  [{n.get('kind', '?')}] {n.get('name') or n.get('id')}"
        if n.get("vendor"):
            head += f" · {n['vendor']}"
        print(head)
        for label, key in (("endpoint", "endpoint"), ("ip", "ip"), ("control", "control_surface")):
            if n.get(key):
                print(f"      {label}: {n[key]}")
    if flows:
        print("  flows:")
        for f in flows:
            arrow = f"    {f.get('from')} → {f.get('to')}"
            if f.get("label"):
                arrow += f"  ({f['label']})"
            print(arrow)


# --------------------------------------------------------------------------- auto-architecture-trace (CELE-t201)
#
# The Stack is captured once (`architecture init`) then kept current AUTOMATICALLY: a lightweight "trace"
# re-detects the topology and ADDITIVELY merges any newly-discovered pieces into infra-local.json, then
# remaps the hosted Stack. It runs on a cadence (once every N turns — not every turn; topology changes
# rarely) and immediately when a dependency manifest is edited (the "a piece entered the stack" event).
# Two hard rules keep it safe: (1) it is a NO-OP unless the project already opted in (infra-local.json
# exists) — it never auto-creates a diagram in a random repo; (2) the merge is purely additive — it only
# APPENDS newly-detected nodes, never overwriting or removing anything a human authored.
ARCH_TRACE_STATE_NAME = ".arch-trace.json"     # gitignored per-project trace bookkeeping (turn counter + pending)
ARCH_TRACE_EVERY_TURNS = 3                       # cadence: run a trace once every N user turns


def _arch_trace_state_path(ctx: Path) -> Path:
    return ctx / ARCH_TRACE_STATE_NAME


def _load_arch_trace_state(ctx: Path) -> dict:
    try:
        d = json.loads(_arch_trace_state_path(ctx).read_text())
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_arch_trace_state(ctx: Path, state: dict) -> None:
    try:
        _arch_trace_state_path(ctx).write_text(json.dumps(state) + "\n")
    except OSError:
        pass


def _merge_infra_nodes(doc: dict, detected: list[dict]) -> tuple[dict, list[str]]:
    """Additively merge detected nodes into the doc's node list. A detected node is ADDED only when
    neither its id NOR its (vendor, kind) pair already exists — so hand-authored nodes are never
    duplicated or overwritten. Returns (doc, added_display_names). Pure."""
    existing = list(doc.get("nodes") or [])
    ids = {str(n.get("id")) for n in existing}
    vendor_kinds = {(str(n.get("vendor")).lower(), str(n.get("kind"))) for n in existing if n.get("vendor")}
    added: list[str] = []
    for d in detected:
        nid = str(d.get("id"))
        vk = (str(d.get("vendor")).lower(), str(d.get("kind")))
        if nid in ids or (d.get("vendor") and vk in vendor_kinds):
            continue
        existing.append(_full_node(d))
        ids.add(nid)
        if d.get("vendor"):
            vendor_kinds.add(vk)
        added.append(d.get("name") or nid)
    doc["nodes"] = existing
    return doc, added


def _architecture_trace(ctx: Path, *, reason: str, allow_push: bool = True) -> list[str]:
    """Re-detect the topology and additively merge new pieces into infra-local.json; on a change, remap
    the hosted Stack (detached best-effort push). NO-OP unless infra-local.json already exists (opt-in).
    Returns the display names of any nodes added this trace ([] when nothing changed). Never raises."""
    try:
        if not _infra_path(ctx).is_file():
            return []                                    # not opted in — the trace stays silent
        doc = load_infra(ctx)
        if not doc:
            return []
        detected = _detect_infra_nodes(ctx.parent)
        doc, added = _merge_infra_nodes(doc, detected)
        if not added:
            return []
        doc["updated"] = now_iso()
        _infra_path(ctx).write_text(json.dumps(doc, indent=2) + "\n")
        if allow_push:
            try:
                __import__("celeborn_sync").schedule_architecture_push(ctx)
            except Exception:
                pass                                     # remap is best-effort; local capture already landed
        return added
    except Exception:
        return []


def _arch_trace_note(added: list[str], reason: str) -> str:
    """The SURFACE-THIS line for a trace that added pieces. Empty when nothing changed."""
    if not added:
        return ""
    what = ", ".join(added[:6]) + ("…" if len(added) > 6 else "")
    return (f"🏹 Celeborn —> architecture trace ({reason}): added {len(added)} new node(s) to the stack "
            f"— {what}. Remapped your hosted Stack.")


def _maybe_arch_trace_on_edit(ctx: Path, rel_path: str) -> str:
    """PostToolUse hook (CELE-t201): if the edited file is a dependency manifest, trace NOW (bypassing
    the cadence throttle) and reset the turn counter. Stashes the surface note so the next
    user-prompt-submit relays it reliably (PostToolUse output is model-only). Returns a note or ""."""
    if not _infra_path(ctx).is_file():
        return ""                                        # opt-in only — no footprint until `architecture init`
    base = Path(rel_path).name.lower()
    if base not in _INFRA_MANIFESTS:
        return ""
    added = _architecture_trace(ctx, reason=f"{base} edited")
    state = _load_arch_trace_state(ctx)
    state["turns_since_trace"] = 0                        # the manifest trace resets the cadence clock
    note = _arch_trace_note(added, f"{base} edited")
    if note:
        pending = state.get("pending") or []
        pending.append(note)
        state["pending"] = pending
    _save_arch_trace_state(ctx, state)
    return note


def _maybe_arch_trace_on_turn(ctx: Path) -> str:
    """UserPromptSubmit hook (CELE-t201): tick the turn counter; every ARCH_TRACE_EVERY_TURNS run a
    cadence trace. Also drain any pending note a manifest-edit trace stashed. Returns a SURFACE-THIS
    block (possibly several lines) or ""."""
    if not _infra_path(ctx).is_file():
        return ""                                        # opt-in only — no footprint until `architecture init`
    state = _load_arch_trace_state(ctx)
    notes: list[str] = list(state.get("pending") or [])
    state["pending"] = []
    n = int(state.get("turns_since_trace") or 0) + 1
    if n >= ARCH_TRACE_EVERY_TURNS:
        state["turns_since_trace"] = 0
        added = _architecture_trace(ctx, reason="cadence")
        note = _arch_trace_note(added, "every 3 turns")
        if note:
            notes.append(note)
    else:
        state["turns_since_trace"] = n
    _save_arch_trace_state(ctx, state)
    return "\n".join(notes)


def cmd_architecture(args):
    """`celeborn architecture [init|show|trace]` — capture non-secret infrastructure topology locally.
    `sync` is handled in celeborn_sync (needs the network); init/show/trace are pure-local and stay here."""
    ctx = require_context(args)
    sub = getattr(args, "arch_cmd", None)
    if sub == "init":
        _architecture_init(ctx, force=getattr(args, "force", False))
    elif sub == "trace":
        if not _infra_path(ctx).is_file():
            die(f"No {INFRA_LOCAL_NAME} yet. Run `celeborn architecture init` first.")
        added = _architecture_trace(ctx, reason="manual")
        if added:
            ok(f"trace added {len(added)} node(s): {', '.join(added)} — remapping hosted Stack.")
        else:
            print("trace: no new pieces detected — the stack is up to date.")
    else:
        _architecture_show(ctx)


# --------------------------------------------------------------------------- product federation (CELE-t190)
#
# Layer A of CELE-t188 (plan/cele-t188-multi-repo-oss-stewardship.md). A product spans several repo-facets
# with different roles + publish policies (client:public → PyPI; server:private → never; oss:* → fork+PR).
# The registry mirrors Celeborn's own authored-vs-machine split EXACTLY:
#   • product.md            — authored, COMMITTED. Product FACTS only: facet keys, roles, publish policy,
#                             canonical repo URLs, OSS provenance (Layer C). Portable across every clone.
#   • product-local.json    — gitignored, PER-MACHINE. Binds each facet key → this machine's checkout path.
#                             A facet with no binding here degrades gracefully to "not present on this machine".
# Layers B (git/PR ops) and C (OSS provenance + guard) read this registry; Layer D (README) is gated on C.
PRODUCT_MD_NAME = "product.md"
PRODUCT_LOCAL_NAME = "product-local.json"
PRODUCT_LOCAL_SCHEMA = "celeborn-product-local/1"
# The role vocabulary from the t188 plan (§3). Determines publish policy + guard posture (guards land on B/C).
PRODUCT_ROLES = ("client:public", "server:private", "oss:upstream", "oss:dependency", "oss:fork")


def _product_md_path(ctx: Path) -> Path:
    return ctx / PRODUCT_MD_NAME


def _product_local_path(ctx: Path) -> Path:
    return ctx / PRODUCT_LOCAL_NAME


def load_product_local(ctx: Path) -> dict:
    """Read .context/product-local.json (gitignored, per-machine). {} when absent/invalid."""
    p = _product_local_path(ctx)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        warn(f"{PRODUCT_LOCAL_NAME} is not valid JSON; treating as empty.")
        return {}


def parse_product(text: str) -> dict:
    """Parse product.md → {'name': str, 'facets': [{key, role, publish, repo, upstream, ...}],
    'provenance': [raw '- …' lines]}. Section-aware: facet lines live under a line beginning 'Facets',
    provenance (Layer C) under a line beginning 'Provenance'. HTML comments are stripped first so the
    managed header never parses as data. Never raises on malformed input — it returns what it can."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    name, facets, provenance, mode = "", [], [], None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            h = s.lstrip("#").strip()
            if h.lower().startswith("product"):
                for sep in ("—", ":", " - ", "-"):
                    if sep in h:
                        name = h.split(sep, 1)[1].strip()
                        break
            mode = None
            continue
        low = s.lower()
        if low.startswith("facets"):
            mode = "facets"
            continue
        if low.startswith("provenance"):
            mode = "provenance"
            continue
        if not s.startswith("-"):
            continue
        body = s[1:].strip()
        if not body or body.lower().startswith("(none"):
            continue
        if mode == "facets":
            toks = body.split()
            facet = {"key": toks[0]}
            for t in toks[1:]:
                if "=" in t:
                    k, v = t.split("=", 1)
                    facet[k] = v
            facets.append(facet)
        elif mode == "provenance":
            provenance.append("- " + body)
    return {"name": name, "facets": facets, "provenance": provenance}


def load_product(ctx: Path) -> dict:
    """Parsed product.md plus an `exists` flag. Empty/absent → {'exists': False}."""
    p = _product_md_path(ctx)
    if not p.is_file():
        return {"name": "", "facets": [], "provenance": [], "exists": False}
    d = parse_product(p.read_text())
    d["exists"] = True
    return d


def _render_product(name: str, facets: list, provenance: list) -> str:
    """Serialize product.md canonically (Celeborn-maintained file). Provenance lines round-trip verbatim
    so a Layer-C write is preserved when Layer A rewrites the facet block."""
    lines = [f"# Product — {name}", ""]
    lines += [
        "<!-- Celeborn product registry (CELE-t190, Layer A of CELE-t188). Authored + COMMITTED — product",
        "     FACTS only: facet keys, roles, publish policy, canonical repo URLs, and OSS provenance",
        "     (Layer C). No local paths here — this machine's checkout paths live in product-local.json",
        "     (gitignored). Roles: client:public · server:private · oss:upstream · oss:dependency ·",
        "     oss:fork. Edit via `celeborn product add|bind`; the orient banner reads this file. -->",
        "",
        "Facets (key · role · publish · repo):",
    ]
    if facets:
        for f in facets:
            parts = [f"- {f['key']}", f"role={f.get('role', '')}"]
            if f.get("publish"):
                parts.append(f"publish={f['publish']}")
            if f.get("repo"):
                parts.append(f"repo={f['repo']}")
            if f.get("upstream"):
                parts.append(f"upstream={f['upstream']}")
            lines.append("   ".join(parts))
    else:
        lines.append("- (none yet — add with `celeborn product add <key> --role <role>`)")
    lines += ["", "Provenance (portions of the tree that are OSS — Layer C, CELE-t192):"]
    lines += provenance if provenance else ["- (none yet)"]
    lines.append("")
    return "\n".join(lines)


def _product_init(ctx: Path, name: str | None = None, force: bool = False) -> None:
    p = _product_md_path(ctx)
    if p.is_file() and not force:
        die(f"{PRODUCT_MD_NAME} already exists (use --force to overwrite).")
    nm = (name or _project_name(ctx) or ctx.parent.name).strip()
    p.write_text(_render_product(nm, [], []))
    ok(f"wrote {PRODUCT_MD_NAME} for product '{nm}'.")
    print("  Add a facet:  celeborn product add client --role client:public --repo github.com/you/app")
    print("  Bind it here: celeborn product bind client /path/to/checkout   (gitignored, per-machine)")


def _product_add(ctx: Path, key: str, role: str, publish: str | None,
                 repo: str | None, upstream: str | None) -> None:
    """Upsert a facet into product.md (add-or-edit by key). Scaffolds product.md if absent."""
    p = _product_md_path(ctx)
    if p.is_file():
        cur = parse_product(p.read_text())
        name, facets, provenance = cur["name"], cur["facets"], cur["provenance"]
    else:
        name, facets, provenance = (_project_name(ctx) or ctx.parent.name), [], []
    facet = {"key": key, "role": role}
    if publish:
        facet["publish"] = publish
    if repo:
        facet["repo"] = repo
    if upstream:
        facet["upstream"] = upstream
    existing = next((f for f in facets if f.get("key") == key), None)
    verb = "updated" if existing else "added"
    if existing:
        facets = [facet if f is existing else f for f in facets]
    else:
        facets.append(facet)
    p.write_text(_render_product(name, facets, provenance))
    ok(f"{verb} facet '{key}' (role={role}) in {PRODUCT_MD_NAME}.")


def _product_bind(ctx: Path, key: str, checkout: str) -> None:
    """Bind a facet key → this machine's checkout path in product-local.json (gitignored)."""
    prod = load_product(ctx)
    if prod["exists"] and not any(f.get("key") == key for f in prod["facets"]):
        warn(f"'{key}' is not a facet in {PRODUCT_MD_NAME} yet — binding it anyway "
             f"(add it with `celeborn product add {key} --role <role>`).")
    abspath = str(Path(checkout).expanduser().resolve())
    if not Path(abspath).is_dir():
        warn(f"{abspath} is not a directory on this machine — binding recorded, but the facet "
             f"shows as unbound (—) until the path exists.")
    local = load_product_local(ctx)
    if local.get("schema") != PRODUCT_LOCAL_SCHEMA:
        local["schema"] = PRODUCT_LOCAL_SCHEMA
    bindings = local.setdefault("bindings", {})
    bindings[key] = abspath
    _product_local_path(ctx).write_text(json.dumps(local, indent=2) + "\n")
    ok(f"bound '{key}' → {abspath} (product-local.json, gitignored).")


def _product_list(ctx: Path) -> None:
    prod = load_product(ctx)
    if not prod["exists"]:
        print(f"No {PRODUCT_MD_NAME} yet. Run `celeborn product init` to create the registry.")
        return
    facets = prod["facets"]
    bindings = (load_product_local(ctx).get("bindings") or {})
    name = prod["name"] or "product"
    print(f"Product — {name} · {len(facets)} facet(s)")
    if not facets:
        print("  (no facets — add with `celeborn product add <key> --role <role>`)")
    for f in facets:
        key, role = f.get("key", "?"), f.get("role", "?")
        path = bindings.get(key)
        bound = path and Path(path).is_dir()
        marker = "✓" if bound else "—"
        line = f"  [{role} {marker}] {key}"
        if f.get("repo"):
            line += f"   repo={f['repo']}"
        print(line)
        if path:
            print(f"        → {path}" + ("" if bound else "   (path missing — unbound)"))
        else:
            print("        (unbound on this machine — `celeborn product bind %s <path>`)" % key)
    if prod["provenance"]:
        print(f"  provenance (OSS — Layer C): {len(prod['provenance'])} entr(y/ies)")


def _product_banner(ctx: Path) -> str:
    """One-line orient banner: product name + facets with ✓ (bound + present here) / — (unbound) markers.
    '' when no product.md (silent for single-repo projects). Best-effort — never raises."""
    try:
        p = _product_md_path(ctx)
        if not p.is_file():
            return ""
        data = parse_product(p.read_text())
        facets = data.get("facets") or []
        if not facets:
            return ""
        bindings = load_product_local(ctx).get("bindings") or {}
        parts = []
        for f in facets:
            key, role = f.get("key", "?"), f.get("role", "?")
            path = bindings.get(key)
            marker = "✓" if (path and Path(path).is_dir()) else "—"
            parts.append(f"{key} ({role} {marker})")
        name = data.get("name") or "product"
        head = f"🏹 Celeborn product —> {name} · {len(facets)} facet{'s' if len(facets) != 1 else ''}: "
        shown, budget = [], 220
        for i, part in enumerate(parts):
            if shown and len(head) + len(" · ".join(shown + [part])) > budget:
                shown.append(f"+{len(parts) - i} more")
                break
            shown.append(part)
        return head + " · ".join(shown)
    except Exception:
        return ""


def cmd_product(args):
    """`celeborn product [list|init|add|bind]` — the product federation registry (Layer A of CELE-t188).
    Pure-local markdown/JSON maintenance; no network."""
    ctx = require_context(args)
    sub = getattr(args, "product_cmd", None)
    if sub == "init":
        _product_init(ctx, name=getattr(args, "name", None), force=getattr(args, "force", False))
    elif sub == "add":
        _product_add(ctx, args.key, args.role, getattr(args, "publish", None),
                     getattr(args, "repo", None), getattr(args, "upstream", None))
    elif sub == "bind":
        _product_bind(ctx, args.key, args.checkout)
    else:
        _product_list(ctx)


# --------------------------------------------------------------------------- Multi-repo git/PR ops (t191)
#
# Layer B of CELE-t188. `celeborn commit/push/pr --facet <key>` routes git (and a drafted `gh pr create`)
# to the facet's bound checkout, so a single board coordinates work across every repo of the product.
# Each op is attributed automatically — commits carry Celeborn-Task/-Agent/-Model trailers and register a
# cross-repo touch, exactly the multi-agent protocol the single-repo flow already uses. The publish guard
# (above) is the role enforcement; commit/push/pr are the routing. Reads the Layer A registry (t190).


def _facet_role_for_path(ctx: Path, path) -> tuple:
    """(key, role) of the bound facet whose checkout is `path` or an ancestor of it — longest match wins,
    so a nested facet resolves to the closest one. (None, None) when no product.md or no enclosing facet."""
    prod = load_product(ctx)
    if not prod.get("exists"):
        return (None, None)
    roles = {f.get("key"): f.get("role") for f in prod["facets"] if f.get("key")}
    bindings = load_product_local(ctx).get("bindings") or {}
    try:
        target = Path(path).expanduser().resolve()
    except Exception:
        return (None, None)
    best_key, best_role, best_len = None, None, -1
    for key, co in bindings.items():
        try:
            cop = Path(co).expanduser().resolve()
        except Exception:
            continue
        if target == cop or cop in target.parents:
            if len(str(cop)) > best_len:
                best_key, best_role, best_len = key, roles.get(key), len(str(cop))
    return (best_key, best_role)


def _publish_guard_targets(ctx: Path, cmd: str, project_dir: str) -> list:
    """The (key, role) of every forbidden-to-publish facet a publish command targets: a path token in the
    command that resolves into a bound checkout, else — when the command names no such path — the facet the
    command's own project resolves into. Resolving the command's OWN tokens (not string-matching the stored
    binding) makes detection symlink-robust (macOS /var → /private/var). Only server:private/oss:* facets
    are returned (client:public publishes are allowed), so an empty list means 'let it through'."""
    if not load_product(ctx).get("exists"):
        return []
    hits, seen = [], set()
    for tok in re.split(r"[\s'\"=]+", cmd):
        if "/" not in tok:
            continue                                   # only path-shaped tokens can name a checkout
        cand = tok.split("*", 1)[0].rstrip("/")        # drop a glob tail (dist/* → dist)
        if not cand:
            continue
        key, role = _facet_role_for_path(ctx, cand)
        if key and key not in seen and _role_forbids_publish(role):
            seen.add(key)
            hits.append((key, role))
    if hits:
        return hits
    key, role = _facet_role_for_path(ctx, project_dir)
    if key and _role_forbids_publish(role):
        return [(key, role)]
    return []


def _facet_resolve(ctx: Path, key: str) -> dict:
    """The facet dict (key/role/repo/upstream/publish) plus its resolved `checkout` Path on this machine.
    die()s with a corrective message when the facet is undeclared, unbound here, or the bound path is
    missing / not a git repo — the same graceful-degradation contract as the Layer A banner, but a hard
    stop because a git op has nowhere to run without a real checkout."""
    prod = load_product(ctx)
    if not prod.get("exists"):
        die("no product registry — run `celeborn product init` and declare facets first (CELE-t190).")
    facet = next((f for f in prod["facets"] if f.get("key") == key), None)
    if facet is None:
        declared = ", ".join(f.get("key", "?") for f in prod["facets"]) or "(none yet)"
        die(f"'{key}' is not a facet in product.md (declared: {declared}). "
            f"Add it: celeborn product add {key} --role <role>.")
    co = (load_product_local(ctx).get("bindings") or {}).get(key)
    if not co:
        die(f"facet '{key}' is not bound on this machine. Bind it: celeborn product bind {key} <checkout>.")
    cop = Path(co).expanduser()
    if not cop.is_dir():
        die(f"facet '{key}' is bound to {cop}, which is not a directory here — re-bind it "
            f"(`celeborn product bind {key} <checkout>`).")
    if not (cop / ".git").exists():
        die(f"facet '{key}' checkout {cop} is not a git repository.")
    return {**facet, "checkout": cop}


def _run_git(checkout: Path, git_args: list, timeout: int = 30):
    """Run `git -C <checkout> <args>` and return the CompletedProcess. die()s only if git can't be spawned
    at all; a non-zero git exit is left to the caller to report with context."""
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(checkout), *git_args],
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        die(f"could not run git in {checkout}: {e}")


def _celeborn_trailers(ident: dict, task: str) -> list:
    """The commit trailers that attribute a facet-routed commit — bare tN per the machine-parsed
    convention (CLAUDE.md), agent handle, and model label. Omits any part that isn't known."""
    trailers = []
    bare = _split_qualified_tid(task)[1] if task else ""
    if bare:
        trailers.append(f"Celeborn-Task: {bare}")
    handle = (ident.get("handle") or "").strip()
    if handle and handle != "unknown":
        trailers.append(f"Celeborn-Agent: {handle}")
    label = _agent_label(ident.get("family", ""), ident.get("model", ""))
    if label:
        trailers.append(f"Celeborn-Model: {label}")
    return trailers


def _facet_touch(ctx: Path, key: str, filepath: str, ident: dict, task: str, why: str) -> None:
    """Register a cross-repo touch (path namespaced `<key>:<file>`) so agents sharing this .context/ see
    the facet activity on orient. The registry is this project's — a facet-routed op is coordinated on the
    one board, even though the file lives in another repo."""
    data = _load_touches(ctx)
    files = data.setdefault("files", {})
    files[f"{key}:{filepath}"] = {
        "by": ident.get("handle") or "unknown",
        "family": ident.get("family", ""),
        "model": ident.get("model", ""),
        "at": now_iso(),
        "task": _split_qualified_tid(task)[1] if task else "",
        "why": why,
    }
    _save_touches(ctx, data)


def cmd_commit(args):
    """`celeborn commit --facet KEY -m MSG [files…]` — route a git commit into a bound facet checkout,
    appending Celeborn-Task/-Agent/-Model trailers automatically and registering a cross-repo touch. Files
    are staged by name (never `git add -A`); omit them to commit what's already staged. Layer B of CELE-t188."""
    ctx = require_context(args)
    facet = _facet_resolve(ctx, args.facet)
    co = facet["checkout"]
    ident = _agent_identity(args, ctx)
    task = (getattr(args, "task", None) or "").strip() or _session_task_id(ctx, _resolve_session(args))
    files = list(getattr(args, "files", None) or [])
    trailers = _celeborn_trailers(ident, task)
    full = args.message.rstrip() + ("\n\n" + "\n".join(trailers) if trailers else "")
    if files:
        r = _run_git(co, ["add", "--", *files])
        if r.returncode != 0:
            die(f"git add failed in facet '{args.facet}' ({co}):\n{(r.stderr or r.stdout).strip()}")
    commit_args = ["commit", "-m", full] + (["--", *files] if files else [])
    r = _run_git(co, commit_args)
    if r.returncode != 0:
        die(f"git commit failed in facet '{args.facet}' ({co}):\n{(r.stderr or r.stdout).strip()}")
    for f in (files or ["(staged)"]):
        _facet_touch(ctx, args.facet, f, ident, task, f"committed to {args.facet}")
    ok(f"committed to facet '{args.facet}' ({co})" + (f" · {_split_qualified_tid(task)[1]}" if task else ""))
    print("  trailers: " + (", ".join(trailers) if trailers
                            else "(none — run `celeborn identify` so commits show who you are)"))
    head = _run_git(co, ["log", "-1", "--oneline"])
    if head.returncode == 0 and head.stdout.strip():
        print("  " + head.stdout.strip())


def cmd_push(args):
    """`celeborn push --facet KEY [remote] [branch]` — route `git push` to a bound facet checkout. A branch
    push (even into a private repo's own remote) is fine; a RELEASE push (`--tags`/`--follow-tags`) into a
    server:private/oss:* facet is refused under the same publish policy the PreToolUse guard enforces —
    caught here too because that guard can't see the git that runs inside celeborn. Layer B of CELE-t188."""
    ctx = require_context(args)
    facet = _facet_resolve(ctx, args.facet)
    co = facet["checkout"]
    tags = bool(getattr(args, "tags", False)) or bool(getattr(args, "follow_tags", False))
    if tags and _role_forbids_publish(facet.get("role", "")):
        die(_publish_policy_reason(args.facet, facet.get("role", ""), "a tag/release push"))
    git_args = ["push"]
    if getattr(args, "set_upstream", False):
        git_args.append("--set-upstream")
    if getattr(args, "follow_tags", False):
        git_args.append("--follow-tags")
    if getattr(args, "tags", False):
        git_args.append("--tags")
    if getattr(args, "remote", None):
        git_args.append(args.remote)
    if getattr(args, "branch", None):
        git_args.append(args.branch)
    r = _run_git(co, git_args, timeout=120)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        die(f"git push failed in facet '{args.facet}' ({co}):\n{out}")
    ok(f"pushed facet '{args.facet}' ({co})")
    if out:
        print("  " + out.replace("\n", "\n  "))


def cmd_pr(args):
    """`celeborn pr --facet KEY [--base B] [--title T] [--body B]` — DRAFT a pull request for a bound facet
    checkout: compute branch/base/commits, compose title+body with provenance, and print a ready-to-run
    `gh pr create` command. Celeborn NEVER auto-opens a PR — for oss:* facets it also prints the fork→PR
    steps. The human/agent reviews and sends it. Layer B of CELE-t188 (draft-don't-send)."""
    import shlex
    ctx = require_context(args)
    facet = _facet_resolve(ctx, args.facet)
    co, role = facet["checkout"], facet.get("role", "")
    repo, upstream = facet.get("repo", ""), facet.get("upstream", "")
    base = (getattr(args, "base", None) or "main").strip()
    br = _run_git(co, ["rev-parse", "--abbrev-ref", "HEAD"])
    branch = br.stdout.strip() if br.returncode == 0 else ""
    if not branch or branch == "HEAD":
        die(f"facet '{args.facet}' has no current branch (detached HEAD?) — check out a branch first.")
    log = _run_git(co, ["log", f"{base}..{branch}", "--oneline"])
    commits = [l for l in (log.stdout or "").splitlines() if l.strip()] if log.returncode == 0 else []
    task = (getattr(args, "task", None) or "").strip() or _session_task_id(ctx, _resolve_session(args))
    ident = _agent_identity(args, ctx)
    title = (getattr(args, "title", None) or "").strip()
    if not title:
        title = (commits[0].split(" ", 1)[1] if commits and " " in commits[0] else f"{branch} → {base}")
    body = (getattr(args, "body", None) or "").strip()
    if not body:
        lines = ["## Changes", ""] + (
            [f"- {c.split(' ', 1)[1] if ' ' in c else c}" for c in commits] or ["- (no commits ahead of base)"])
        body = "\n".join(lines)
    foot = []
    bare = _split_qualified_tid(task)[1] if task else ""
    if bare:
        foot.append(f"Celeborn-Task: {bare}")
    handle = (ident.get("handle") or "").strip()
    label = _agent_label(ident.get("family", ""), ident.get("model", ""))
    if handle and handle != "unknown":
        foot.append(f"Drafted-by: @{handle}" + (f" ({label})" if label else ""))
    if foot:
        body = body + "\n\n" + "\n".join(foot)

    print(f"🏹 Celeborn PR draft — facet '{args.facet}' ({role})")
    print(f"  repo:        {repo or '(no repo url in product.md — add `--repo` via product add)'}")
    print(f"  base ← head: {base} ← {branch}")
    print(f"  commits:     {len(commits)} ahead of {base}")
    print()
    print(f"  title: {title}")
    print("  body:")
    for line in body.splitlines():
        print("    " + line)
    print()
    if role.startswith("oss:"):
        print("  ⓘ Stewarded OSS — contribute via a fork, never publish/push as ours:")
        if upstream:
            print(f"      upstream: {upstream}")
        print(f"      1) gh repo fork {repo or upstream or '<upstream>'} --clone=false")
        print("      2) push this branch to YOUR fork, then open the PR against upstream.")
    ghr = f" -R {repo}" if repo else ""
    print("  Ready to send — Celeborn drafts, it never auto-opens a PR. Review the diff, then run:")
    print(f"      gh pr create{ghr} --base {base} --head {branch} \\")
    print(f"        --title {shlex.quote(title)} \\")
    print(f"        --body {shlex.quote(body)}")
    warn("drafted, not sent (CELE-t191) — send it yourself with the gh command above.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="celeborn", description="Celeborn Code — a long-term context "
        "substrate for coding agents (memory for Claude Code / Codex / Grok). Not Apache Celeborn "
        "(Spark shuffle) or the frkngksl/Celeborn Windows tool. `celeborn about` for identity + links.")
    p.add_argument("--path", default=".", help="project dir to operate in (default: cwd)")
    sub = p.add_subparsers(dest="command", required=True)

    ip = sub.add_parser("init", help="scaffold .context/")
    ip.add_argument("--private", action="store_true",
                    help="gitignore .context/ — keep working memory local and sync it across devices "
                         "instead of committing. Auto-enabled when a public repo is detected.")
    ip.add_argument("--public", action="store_true",
                    help="commit .context/ to git (default for private repos; .context travels via git).")
    ip.add_argument("--no-claude-md", dest="claude_md", action="store_false",
                    help="don't annotate CLAUDE.md (by default init adds a managed block so Claude Code, "
                         "which auto-loads CLAUDE.md, knows Celeborn maintains context in .context/).")
    ip.add_argument("--no-agents-md", dest="agents_md", action="store_false",
                    help="don't annotate AGENTS.md (by default init adds the same managed block for "
                         "Codex/Grok-style hosts that auto-load AGENTS.md).")
    ip.add_argument("--no-scan", dest="scan", action="store_false",
                    help="don't read the repo (README, build manifest, git log) to pre-seed the Hot "
                         "tier; leave the empty template for you to fill in by hand.")
    ip.add_argument("--no-cmm", dest="no_cmm", action="store_true",
                    help="don't auto-engage Codebase Memory (CMM) for this project. By default init "
                         "pre-clears CMM's read-only tools (fewer 'Allow' prompts) and indexes the repo "
                         "if the CMM binary is installed; reverse anytime with `celeborn cmm off`. "
                         "($CELEBORN_NO_CMM=1 opts out globally.)")
    ip.add_argument("--name", dest="name", default=None,
                    help="name this project for the kanban board (skips the interactive prompt). "
                         "Persisted as project_name in .celebornrc; defaults to the repo folder name.")
    ip.add_argument("--no-open", dest="open_board", action="store_false",
                    help="don't launch or open the kanban board after init (by default init seeds an "
                         "empty board and starts the localhost viewer — Celeborn's UI).")
    ip.add_argument("--no-browser", dest="open_browser", action="store_false",
                    help="start the kanban viewer but don't pop a browser tab (the board stays "
                         "reachable on localhost). Implied when init isn't run from a terminal.")
    ip.set_defaults(func=cmd_init, claude_md=True, agents_md=True, scan=True, no_cmm=False,
                    open_board=True, open_browser=True)
    sp = sub.add_parser("status", help="print the Hot tier (Orient load)")
    sp.add_argument("--full", action="store_true",
                    help="print the Hot tier unclipped (bypass the Orient-load size budgets)")
    sp.set_defaults(func=cmd_status)

    cp = sub.add_parser("checkpoint",
                        help="safely update session.json (focus/next/branch/status) — writes valid JSON, "
                             "clips over-long fields, repairs a corrupt file; replaces hand-editing")
    cp.add_argument("--focus", help="current focus (one line; long-form belongs in state.md/notes.md)")
    cp.add_argument("--next", help="next action")
    cp.add_argument("--branch", help="working branch")
    cp.add_argument("--status", help="session status (e.g. in-progress, blocked, green)")
    cp.add_argument("--stop-allowed", dest="stop_allowed", action="store_true",
                    help="mark the session safe to stop/clear")
    cp.add_argument("--no-stop-allowed", dest="no_stop_allowed", action="store_true",
                    help="mark the session NOT safe to stop (work in flight)")
    cp.set_defaults(func=cmd_checkpoint)

    sub.add_parser("index", help="(re)build the SQLite FTS index").set_defaults(func=cmd_index)

    sp = sub.add_parser("search", help="full-text recall")
    sp.add_argument("query")
    sp.add_argument("-n", "--limit", type=int, default=None, help="max results")
    sp.set_defaults(func=cmd_search)

    ap = sub.add_parser("archive", help="archive old journal entries")
    ap.add_argument("--keep", type=int, default=None, help="entries to keep in journal.md")
    ap.set_defaults(func=cmd_archive)

    pp = sub.add_parser("promote", help="distill a note to a higher tier")
    pp.add_argument("--to", choices=["learnings", "durable"], required=True)
    pp.add_argument("--title", required=True)
    pp.add_argument("--note", default="", help="body text")
    pp.add_argument("--doc", default=None, help="durable doc name (default: gotchas)")
    pp.set_defaults(func=cmd_promote)

    sub.add_parser("handoff", help="regenerate handoff.md").set_defaults(func=cmd_handoff)
    sub.add_parser("doctor", help="health check + memory-drift + secret scan").set_defaults(func=cmd_doctor)
    pgp = sub.add_parser("progress", help="run the deterministic progress engine for a card (debug + explain)")
    pgp.add_argument("id", nargs="?", help="card id (default: every doing card)")
    pgp.add_argument("--explain", action="store_true", help="show the signals → floor derivation")
    pgp.set_defaults(func=cmd_progress)

    al = sub.add_parser("alert", help="raise/clear a 'coding blocked — needs the user' alert on a card (surfaces on the board)")
    al.add_argument("id", nargs="?", help="card id to alert on (omit with --list)")
    al.add_argument("--message", "-m", default="", help="what is blocking (e.g. the permission request text)")
    al.add_argument("--kind", choices=list(ALERT_KINDS), default=None,
                    help="permission (needs approval) · idle (stalled) · stopped (turn ended, awaiting you)")
    al.add_argument("--session", default="", help="the blocked session's id (attribution; hooks pass this)")
    al.add_argument("--clear", action="store_true", help="clear the alert (also happens when the user replies)")
    al.add_argument("--list", action="store_true", help="list the live alerts on this board")
    al.set_defaults(func=cmd_alert)

    tp = sub.add_parser("tasks", help="lightweight task/kanban board (tasks.md truth + derived tasks.json)")
    tp.set_defaults(func=cmd_tasks, task_cmd=None, json=False)
    tsub = tp.add_subparsers(dest="task_cmd")

    ta = tsub.add_parser("add", help="add a task")
    ta.add_argument("title")
    ta.add_argument("--state", default="todo", choices=TASK_STATES, help="initial state (default: todo)")
    ta.add_argument("--owner", default="", help="who owns it (e.g. an agent or person)")
    ta.add_argument("--tags", default="", help="comma/space-separated tags")
    ta.add_argument("--blocked-by", dest="blocked_by", default="", help="task id(s) blocking this one")
    ta.add_argument("--phase", default="", help="plan phase id this task belongs to (e.g. p11)")
    ta.add_argument("--stop", default="",
                    help="logical Stop condition — a clean `/clear` point for this card "
                         "(auto-filled with a generic default if omitted)")
    ta.add_argument("--progress", type=int, default=None,
                    help="percent complete 0-100 (drives the In-Progress card's sand-fill bar)")
    ta.add_argument("--note", default="", help="freeform notes / body")
    ta.add_argument("--claim", action="store_true",
                    help="claim the new card immediately (avoids guessing the new id in a second command)")
    ta.add_argument("--by", default=None, help="claimer for --claim (default: $CELEBORN_AGENT)")
    ta.add_argument("--force", action="store_true", help="with --claim: claim even if you have other DOING cards")
    ta.set_defaults(func=cmd_tasks, task_cmd="add")

    tm = tsub.add_parser("move", help="move a task to a new state")
    tm.add_argument("id")
    tm.add_argument("state", choices=TASK_STATES)
    tm.set_defaults(func=cmd_tasks, task_cmd="move")

    tre = tsub.add_parser("reorder", help="reprioritize a task within its column (up | down | top | bottom)")
    tre.add_argument("id")
    tre.add_argument("dir", choices=["up", "down", "top", "bottom"])
    tre.set_defaults(func=cmd_tasks, task_cmd="reorder")

    te = tsub.add_parser("edit", help="edit task fields (only the flags you pass change)")
    te.add_argument("id")
    te.add_argument("--title", default=None)
    te.add_argument("--state", default=None, choices=TASK_STATES)
    te.add_argument("--owner", default=None)
    te.add_argument("--tags", default=None)
    te.add_argument("--blocked-by", dest="blocked_by", default=None)
    te.add_argument("--phase", default=None)
    te.add_argument("--stop", default=None, help="set the logical Stop condition (clean `/clear` point)")
    te.add_argument("--progress", type=int, default=None,
                    help="percent complete 0-100 (drives the In-Progress card's sand-fill bar)")
    te.add_argument("--note", default=None)
    te.set_defaults(func=cmd_tasks, task_cmd="edit")

    # Subtask checklist (CELE-t106): map a card's steps; checking them auto-derives the progress bar.
    tst = tsub.add_parser("subtasks", help="manage a card's subtask checklist (auto-derives the progress percent)")
    tst.add_argument("id")
    tst.set_defaults(func=cmd_tasks, task_cmd="subtasks", subtask_cmd=None)
    tstsub = tst.add_subparsers(dest="subtask_cmd")
    _sa = tstsub.add_parser("add", help="append a subtask (weight via --weight or a trailing '*N')")
    _sa.add_argument("text", nargs="+")
    _sa.add_argument("--weight", type=int, default=None, help="effort weight (default 1)")
    _ss = tstsub.add_parser("set", help="define the whole checklist at once; each item may end with '*N' for weight")
    _ss.add_argument("items", nargs="+")
    _sr = tstsub.add_parser("rm", help="remove subtask N (1-based)")
    _sr.add_argument("n", type=int)
    tstsub.add_parser("list", help="show the checklist (also the default)")

    tck = tsub.add_parser("check", help="mark subtask N done → pours the progress bar to the new level")
    tck.add_argument("id")
    tck.add_argument("n", type=int)
    tck.set_defaults(func=cmd_tasks, task_cmd="check")
    tuck = tsub.add_parser("uncheck", help="mark subtask N not-done → recomputes progress")
    tuck.add_argument("id")
    tuck.add_argument("n", type=int)
    tuck.set_defaults(func=cmd_tasks, task_cmd="uncheck")

    tr = tsub.add_parser("rm", help="remove a task")
    tr.add_argument("id")
    tr.set_defaults(func=cmd_tasks, task_cmd="rm")

    tarch = tsub.add_parser("archive", help="archive done cards past the column cap to done-archive.md")
    tarch.add_argument("--keep", type=int, default=None, help="done cards to keep on the board (default: done_keep_cards)")
    tarch.set_defaults(func=cmd_tasks, task_cmd="archive")

    tshow = tsub.add_parser("show", help="show one task in full")
    tshow.add_argument("id")
    tshow.set_defaults(func=cmd_tasks, task_cmd="show")

    tj = tsub.add_parser("json", help="(re)write .context/tasks.json (the board's data) and print it")
    tj.add_argument("--out", default=None, help="also write the JSON to this path")
    tj.set_defaults(func=cmd_tasks, task_cmd="json")

    tl = tsub.add_parser("list", help="show the text board (this is also the default)")
    tl.add_argument("--json", action="store_true", help="print tasks.json to stdout instead of the board")
    tl.set_defaults(func=cmd_tasks, task_cmd="list")

    cl = sub.add_parser("claim", help="claim a card (owner ← you, TODO → DOING) — what receiving a pasted card does")
    cl.add_argument("ids", nargs="+", help="task id(s) to claim, e.g. t13")
    cl.add_argument("--by", default=None, help="claimer identity (default: $CELEBORN_AGENT)")
    cl.add_argument("--family", default=None, help="record your agent family (else `celeborn identify` / $CELEBORN_AGENT_FAMILY)")
    cl.add_argument("--model", default=None, help="record your specific model (else `celeborn identify` / $CELEBORN_AGENT_MODEL)")
    cl.add_argument("--force", action="store_true",
                    help="claim even if you already have other DOING cards (not recommended)")
    cl.add_argument("--session", default=None, help=argparse.SUPPRESS)  # active-agents bridge (t131)
    cl.set_defaults(func=cmd_claim)

    sh = sub.add_parser("ship", help="close out a card: release its touches + move to done")
    sh.add_argument("id", help="task id to ship, e.g. t42")
    sh.add_argument("--note", default=None, help="append a ship note to the card")
    sh.add_argument("--by", default=None, help="agent shipping (default: $CELEBORN_AGENT)")
    sh.add_argument("--family", default=None, help="record your agent family (else `celeborn identify` / $CELEBORN_AGENT_FAMILY)")
    sh.add_argument("--model", default=None, help="record your specific model (else `celeborn identify` / $CELEBORN_AGENT_MODEL)")
    sh.set_defaults(func=cmd_ship)

    idp = sub.add_parser("identify", help="declare your agent family + specific model once per session (multi-agent attribution)")
    idp.add_argument("--family", default=None, help="agent family, e.g. Claude / Grok / GPT / Gemini")
    idp.add_argument("--model", default=None, help='specific model, e.g. "Opus 4.8"')
    idp.add_argument("--as", dest="as_", default=None, help="handle to record under (default: $CELEBORN_AGENT / --by)")
    idp.add_argument("--by", default=None, help=argparse.SUPPRESS)
    idp.add_argument("--session", default=None, help=argparse.SUPPRESS)
    idp.add_argument("--show", action="store_true", help="list the agents already identified and exit")
    idp.add_argument("--json", action="store_true", help="with --show: JSON output")
    idp.set_defaults(func=cmd_identify)

    ob = sub.add_parser("outbox", help="prompt hand-off queue — drained into the live session each turn")
    ob.set_defaults(func=cmd_outbox, outbox_cmd=None)
    obsub = ob.add_subparsers(dest="outbox_cmd")
    obp = obsub.add_parser("push", help="queue a prompt (from a task card or raw text)")
    obp.add_argument("--task", default=None, help="render this task id into the queued prompt")
    obp.add_argument("--text", default=None, help="queue this literal prompt text")
    obp.add_argument("--for", dest="for_", default=None,
                     help="address the hand-off to this agent (default: the card's owner)")
    obp.set_defaults(func=cmd_outbox, outbox_cmd="push")
    obd = obsub.add_parser("drain", help="print + clear pending prompts (used by the UserPromptSubmit hook)")
    obd.add_argument("--for", dest="for_", default=None,
                     help="drain this agent's queue (default: $CELEBORN_AGENT, else unassigned)")
    obd.set_defaults(func=cmd_outbox, outbox_cmd="drain")
    obsub.add_parser("list", help="show pending prompts (all agents)").set_defaults(func=cmd_outbox, outbox_cmd="list")
    obc = obsub.add_parser("clear", help="discard pending prompts (all agents, or one with --for)")
    obc.add_argument("--for", dest="for_", default=None, help="clear only this agent's queue")
    obc.set_defaults(func=cmd_outbox, outbox_cmd="clear")

    vp = sub.add_parser("version", help="print version; --check looks back at GitHub for updates")
    vp.add_argument("--check", action="store_true",
                    help="check GitHub (cloud-dancer-labs/celeborn) for a newer Celeborn; offline-safe")
    vp.set_defaults(func=cmd_version)

    abp = sub.add_parser("about", help="identify Celeborn Code + canonical links (disambiguates the same-named projects)")
    abp.set_defaults(func=cmd_about)

    ip = sub.add_parser("integrity", help="verify the install matches the published release (detects in-place edits)")
    ip.add_argument("--write", action="store_true",
                    help="(re)generate the per-version checksum manifest — a release/build step, not for end users")
    ip.set_defaults(func=cmd_integrity)

    adv = sub.add_parser("advise", help="print the throughput/quality recommendations that apply right now")
    adv.add_argument("--harness", default=None, help="render for a specific harness adapter (default: autodetect)")
    adv.add_argument("--json", action="store_true", help="emit the recommendations as JSON")
    adv.add_argument("--dismiss", metavar="ID", default=None,
                     help="permanently silence one recommendation by id (e.g. reduce-permission-friction)")
    adv.add_argument("--restore", metavar="ID", default=None, help="un-silence a previously dismissed recommendation")
    adv.add_argument("--throughput", action="store_true",
                     help="also list on-demand throughput recommendations (spawn_task, /loop, /elves)")
    adv.set_defaults(func=cmd_advise)

    hsp = sub.add_parser("harness", help="show or pin the active harness adapter in .celebornrc (claude|grok|codex|neutral)")
    hsp.add_argument("name", nargs="?", default=None, metavar="claude|grok|codex|neutral",
                     help="harness to pin; omit to print the currently resolved adapter")
    hsp.set_defaults(func=cmd_harness)

    pmp = sub.add_parser("permissions",
                         help="generalize repeated permission approvals into reusable wildcard rules (Claude)")
    pmp.add_argument("--suggest", action="store_true", help="preview the proposed allow-list (default; read-only)")
    pmp.add_argument("--apply", action="store_true", help="write the generalized allow-list to the target file")
    pmp.add_argument("--shared", action="store_true",
                     help="target the committed settings.json (shared) instead of personal settings.local.json")
    pmp.add_argument("--yes", action="store_true", help="skip the confirm prompt on --apply / required to ARM the Danger Zone")
    pmp.add_argument("--harness", default=None, help="use a specific harness adapter (default: autodetect)")
    pmp.add_argument("--json", action="store_true",
                     help="emit the current permission state (baseline active-flags + Danger spectrum + resolved allow-list) as JSON")
    pmp.add_argument("--baseline", action="store_true",
                     help="apply the SAFE t100 auto-allow baseline (default target: global ~/.claude); pair with --remove to strip it")
    pmp.add_argument("--remove", action="store_true", help="with --baseline: remove the baseline rules instead of adding")
    pmp.add_argument("--danger-zone", dest="danger_zone", action="store_true",
                     help="arm (default, needs --yes) or --disarm the FULL UNSAFE auto-allow spectrum + bypassPermissions")
    pmp.add_argument("--disarm", action="store_true", help="with --danger-zone: remove the unsafe spectrum and restore safe defaults")
    pmp.add_argument("--global", dest="global_", action="store_true",
                     help="target the global ~/.claude/settings.json instead of the project file")
    pmp.set_defaults(func=cmd_permissions)

    skp = sub.add_parser("skills",
                         help="list Celeborn / recommended / Matt-Pocock skills; install the Matt Pocock suite")
    skp.add_argument("skills_cmd", nargs="?", choices=["list", "install-mattpocock", "update"], default="list",
                     help="list (default), install-mattpocock, or update (re-pull the Matt Pocock suite @latest)")
    skp.add_argument("--json", action="store_true", help="emit JSON (consumed by the board Settings page)")
    skp.add_argument("--global", dest="global_", action="store_true", help="install into the global ~/.claude scope")
    skp.set_defaults(func=cmd_skills)

    rp = sub.add_parser("record", help="record a memory event for the economy estimate")
    rp.add_argument("event", choices=["orient", "compaction", "handoff", "turn", "clear"])
    rp.add_argument("--session", default=None, help="session id (dedupes repeat orients)")
    rp.add_argument("--tokens", type=int, default=None, help="for `turn`: tokens to add to the rolling context estimate")
    rp.set_defaults(func=cmd_record)

    mp = sub.add_parser("metrics", help="show the tokens-saved / restarts-avoided estimate")
    mp.add_argument("--json", action="store_true", help="emit raw metrics JSON")
    mp.set_defaults(func=cmd_metrics)

    agp = sub.add_parser("agents", help="live per-session context windows (who's working + how full) — the board's /clear-nudge chips")
    agp.add_argument("--json", action="store_true", help="emit the active-agents snapshot as JSON (for the board /api/agents route)")
    agp.add_argument("--window-min", dest="window_min", type=float, default=None,
                     help=f"a transcript touched within this many minutes counts as live (default {int(AGENT_ACTIVE_WINDOW_MIN)})")
    agp.add_argument("--all", action="store_true", help="include idle sessions (ignore the window)")
    agp.add_argument("action", nargs="?", choices=["forget"],
                     help="`agents forget <session>`: wipe a ghost chip — tombstone a session so it leaves the board")
    agp.add_argument("session", nargs="?", help="session id (full or 8-char) to forget")
    agp.set_defaults(func=cmd_agents)

    for _kind, _help, _dd in (("standup", "what happened recently (done cards + commits + journal)", 1),
                              ("changelog", "a wider-window changelog of recent progress", 7)):
        _sp = sub.add_parser(_kind, help=_help)
        _sp.add_argument("--days", type=int, default=None, help=f"window in days (default {_dd})")
        _sp.add_argument("--tweet", action="store_true", help="emit a build-in-public X post (≤280 chars) instead")
        _sp.add_argument("--json", action="store_true", help="emit the raw aggregated activity as JSON")
        _sp.set_defaults(func=cmd_standup, kind=_kind)

    bdp = sub.add_parser("board", help="show this project's kanban URL + de-collided per-project port (and whether it's live)")
    bdp.add_argument("--json", action="store_true", help="emit {port,url,live} as JSON")
    bdp.add_argument("--port", dest="port_only", action="store_true", help="print just the resolved port")
    bdp.add_argument("--url", dest="url_only", action="store_true", help="print just the URL")
    bdp.add_argument("--start", action="store_true", help="ensure-on-orient: launch the viewer (detached) if its port is down")
    # Hidden: the detached restart-loop entrypoint `_spawn_board` re-invokes (keeps `next dev` alive).
    bdp.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    bdp.add_argument("--supervise-port", type=int, help=argparse.SUPPRESS)
    bdp.add_argument("--supervise-tasks", help=argparse.SUPPRESS)
    bdp.set_defaults(func=cmd_board)

    flp = sub.add_parser("fleet", help="live multi-project agent dashboard (register projects, then watch who's working/stuck)")
    flp.add_argument("fleet_action", nargs="?", default="", metavar="register|unregister|repair",
                     help="register this repo (or --path) / unregister <dir> / repair (re-dedup all slugs)")
    flp.add_argument("fleet_target", nargs="?", default=None, metavar="project-dir",
                     help="project directory to unregister (or register when no --path)")
    flp.add_argument("--path", dest="fleet_path", default=None,
                     help="project directory for register (default: the orienting repo)")
    flp.add_argument("--json", action="store_true", help="emit the fleet snapshot as JSON (for the board viewer)")
    flp.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="for `repair`: preview the slug changes without writing the registry or .celebornrc files")
    flp.set_defaults(func=cmd_fleet)

    rp = sub.add_parser("run", help="real-time tracker for ONE multi-agent swarm (the Elves): per-worker heartbeat, progress, and a shared learning blackboard")
    rsub = rp.add_subparsers(dest="run_cmd")
    r_start = rsub.add_parser("start", help="begin a run (clears prior workers + blackboard)")
    r_start.add_argument("run_id", nargs="?", default=None, help="stable run id (default: run-<ts>)")
    r_start.add_argument("--goal", default=None, help="one-line goal of the run")
    r_start.add_argument("--shards", type=int, default=0, help="number of shards/workers")
    r_start.add_argument("--units", type=int, default=0, help="total units of work (e.g. records)")
    r_start.add_argument("--keep", action="store_true", help="do NOT clear prior workers/blackboard")
    r_start.set_defaults(func=cmd_run, run_cmd="start")
    r_beat = rsub.add_parser("beat", help="heartbeat + progress upsert for one worker (call often)")
    r_beat.add_argument("--worker", required=True, help="worker id (e.g. ik_07)")
    r_beat.add_argument("--shard", default=None, help="shard label this worker owns")
    r_beat.add_argument("--phase", default=None, help="phase label (e.g. Crosswalk)")
    r_beat.add_argument("--item", default=None, help="current item being worked")
    r_beat.add_argument("--done", type=int, default=None, help="units completed so far")
    r_beat.add_argument("--total", type=int, default=None, help="units in this worker's shard")
    r_beat.add_argument("--found", type=int, default=None, help="units resolved (a hit)")
    r_beat.add_argument("--missed", type=int, default=None, help="units that resolved to nothing")
    r_beat.add_argument("--source-ok", dest="source_ok", default=None, help="increment ok for a source (e.g. wikidata)")
    r_beat.add_argument("--source-fail", dest="source_fail", default=None, help="increment fail for a source")
    r_beat.add_argument("--source-rl", dest="source_rl", default=None, help="increment rate-limited for a source")
    r_beat.add_argument("--quiet", action="store_true", help="suppress the per-beat echo")
    r_beat.set_defaults(func=cmd_run, run_cmd="beat")
    for _act, _help in (("done", "mark a worker finished"), ("fail", "mark a worker failed")):
        _rp = rsub.add_parser(_act, help=_help)
        _rp.add_argument("--worker", required=True, help="worker id")
        _rp.add_argument("--found", type=int, default=None)
        _rp.add_argument("--missed", type=int, default=None)
        _rp.add_argument("--done", type=int, default=None)
        _rp.add_argument("--total", type=int, default=None)
        if _act == "fail":
            _rp.add_argument("--error", default=None, help="failure reason")
        _rp.set_defaults(func=cmd_run, run_cmd=_act)
    r_learn = rsub.add_parser("learn", help="append a deduped lesson to the shared blackboard")
    r_learn.add_argument("lesson", help="the lesson (short, reusable)")
    r_learn.add_argument("--worker", default=None, help="who learned it")
    r_learn.add_argument("--quiet", action="store_true")
    r_learn.set_defaults(func=cmd_run, run_cmd="learn")
    r_lrn = rsub.add_parser("learnings", help="print recent blackboard lessons (elves read this at shard-start)")
    r_lrn.add_argument("-n", "--limit", type=int, default=30)
    r_lrn.add_argument("--json", action="store_true")
    r_lrn.set_defaults(func=cmd_run, run_cmd="learnings")
    r_st = rsub.add_parser("status", help="print the live run snapshot (also the default)")
    r_st.add_argument("--json", action="store_true", help="emit the snapshot as JSON (for the board)")
    r_st.set_defaults(func=cmd_run, run_cmd="status")
    r_w = rsub.add_parser("watch", help="live-refreshing terminal dashboard until all workers finish")
    r_w.add_argument("--interval", type=float, default=2.0)
    r_w.set_defaults(func=cmd_run, run_cmd="watch")
    rp.set_defaults(func=cmd_run, run_cmd=None)

    fp = sub.add_parser("flex", help="the shareable 🏹💪 '$ Wrapped' brag card (tokens→$ saved + restarts avoided)")
    fp.add_argument("--tweet", action="store_true", help="emit a ≤280-char build-in-public X post instead of the card")
    fp.add_argument("--json", action="store_true", help="emit the raw figures as JSON")
    fp.set_defaults(func=cmd_flex)

    sv = sub.add_parser("savings", help="running savings totals (this project + whole fleet) — the board's economy bar (t68)")
    sv.add_argument("--json", action="store_true", help="emit {generated_at, project, fleet} as JSON")
    sv.set_defaults(func=cmd_savings)

    bp = sub.add_parser("blame", help='git blame for the "why" — commits on a file + linked Celeborn memory')
    bp.add_argument("path_arg", metavar="file", help="repo-relative path (or absolute path inside the project)")
    bp.add_argument("-n", "--limit", type=int, default=8, help="max git commits to show (default 8)")
    bp.add_argument("--memory", type=int, default=5, help="max memory sections to show (default 5)")
    bp.add_argument("--json", action="store_true", help="emit {file, commits, memory} as JSON")
    bp.set_defaults(func=cmd_blame)

    wp = sub.add_parser("why", help='decision archaeology — why "<topic>"? (decision + date + rationale)')
    wp.add_argument("query", metavar="topic", help="topic / keyword to recall the reasoning for")
    wp.add_argument("-n", "--limit", type=int, default=None, help="max results (default 5)")
    wp.add_argument("--json", action="store_true", help="emit {query, hits} as JSON")
    wp.set_defaults(func=cmd_why)

    tch = sub.add_parser("touch", help="register who is editing which file (multi-agent; see references/multi-agent-editing.md)")
    tch.add_argument("words", nargs="*", metavar="file|command",
                     help="<file> to register; or: list | clear | release <file>")
    tch.add_argument("--by", default=None, help="agent id (default: $CELEBORN_AGENT)")
    tch.add_argument("--task", default=None, help="kanban task id (e.g. t28)")
    tch.add_argument("--why", default=None, help="short reason you're editing this file (shown on orient)")
    tch.add_argument("--family", default=None, help="agent family override (else `celeborn identify` / $CELEBORN_AGENT_FAMILY)")
    tch.add_argument("--model", default=None, help="specific model override (else `celeborn identify` / $CELEBORN_AGENT_MODEL)")
    tch.add_argument("--json", action="store_true", help="JSON output (list)")
    tch.add_argument("--force", action="store_true", help="release even if another agent owns the touch")
    tch.set_defaults(func=cmd_touch)

    rmp = sub.add_parser("remind", help="reassuring checkpoint-and-renew reminder (host supplies --tokens)")
    rmp.add_argument("--tokens", type=int, default=None, help="current context size in tokens (the host supplies this)")
    rmp.add_argument("--every", type=int, default=100_000, help="reminder increment in tokens (default 100k)")
    rmp.add_argument("--last", type=int, default=None, help="token count at last reminder; stay silent unless a new increment is crossed")
    rmp.add_argument("--auto", action="store_true", help="use Celeborn's own rolling context estimate (metrics.context_estimate) instead of --tokens; tracks its own last-reminded mark")
    rmp.add_argument("--transcript", default=None, help="path to a Claude Code transcript (JSONL); read the live context size from its latest usage record. Overrides --tokens/--auto and persists the reading to metrics")
    rmp.add_argument("--soft-limit", type=int, default=None, help="token ceiling; at/above it the reminder becomes an urgent plain-language warning (e.g. 150000)")
    rmp.add_argument("--clear-cmd", default=None, help="host-specific clear instruction to display")
    rmp.add_argument("--force", action="store_true", help="print even if no new increment was crossed")
    rmp.set_defaults(func=cmd_remind)

    psp = sub.add_parser("panic-save", help="snapshot the authored tiers to a restore point + print a visible '🏹 Celeborn saved your session' (runs automatically pre-compaction)")
    psp.add_argument("--reason", default="manual", help="why the save fired (compaction / alarm / manual); recorded in the snapshot meta.json")
    psp.add_argument("--session", default=None, help="session id to record in the snapshot meta")
    psp.add_argument("--keep", type=int, default=PANIC_KEEP, help=f"FIFO retention: keep the most recent N snapshots (default {PANIC_KEEP})")
    psp.add_argument("--quiet", action="store_true", help="save but print nothing")
    psp.add_argument("--json", action="store_true", help="print the snapshot record as JSON")
    psp.set_defaults(func=cmd_panic_save)

    rsp = sub.add_parser("restore", help="bring back a pre-compaction panic-save (most recent by default); current state is backed up first")
    rsp.add_argument("--from", dest="from_", default=None, help="restore a specific snapshot by stamp (see --list); default is the most recent")
    rsp.add_argument("--list", action="store_true", help="list available panic-saves (newest first) instead of restoring")
    rsp.add_argument("--keep", type=int, default=PANIC_KEEP, help=f"FIFO retention for the pre-restore backup (default {PANIC_KEEP})")
    rsp.add_argument("--json", action="store_true", help="print the restore result as JSON")
    rsp.set_defaults(func=cmd_restore)

    cap = sub.add_parser("capture", help="mechanically ingest a Claude Code transcript into the local Automatic Context Record (no model)")
    cap.add_argument("--transcript", required=True, help="path to the Claude Code transcript JSONL")
    cap.add_argument("--session", default=None, help="session id (from the Stop-hook stdin); cursor resets when it changes")
    cap.add_argument("--quiet", action="store_true", help="suppress the summary line (for hooks)")
    cap.add_argument("--global", dest="global_", action="store_true",
                     help="force the global ~/.context sink even inside a repo (the hybrid fallback "
                          "used for sessions run outside any .context/ repo)")
    cap.add_argument("--note", action="store_true",
                     help="also print a per-turn `{\"systemMessage\": ...}` heartbeat (for the Stop "
                          "hook): kept unique each turn (growing session total, or 'idle ×K') so "
                          "Claude Code can't suppress it as a duplicate. Terminal-only — see `heartbeat`.")
    cap.set_defaults(func=cmd_capture)

    hbp = sub.add_parser("heartbeat", help="print the per-turn capture heartbeat to plain stdout "
                                           "(for the UserPromptSubmit hook; visible in the Claude app)")
    hbp.add_argument("--session", default=None, help="session id (from the hook stdin); reads that "
                                                     "session's cursor instead of the most-recent one")
    hbp.set_defaults(func=cmd_heartbeat)

    slp = sub.add_parser("statusline", help="render Celeborn's Claude Code statusLine (persistent, "
                                            "can't be suppressed like a hook systemMessage)")
    slp.add_argument("--transcript", default=None, help="transcript JSONL; adds the live context size")
    slp.add_argument("--session", default=None, help="session id (from the hook stdin); reads that "
                                                     "session's cursor instead of the most-recent one")
    slp.set_defaults(func=cmd_statusline)

    wp = sub.add_parser("wire", help="merge Celeborn's `celeborn hook <event>` hooks + statusLine into a "
                                     "Claude Code settings.json (idempotent; migrates a legacy bash install)")
    wp.add_argument("--global", dest="global_", action="store_true",
                    help="write ~/.claude/settings.json (every session) instead of the project's .claude/settings.json")
    wp.add_argument("--force", action="store_true", help="replace an existing non-Celeborn statusLine")
    wp.add_argument("--no-permission-baseline", dest="no_permission_baseline", action="store_true",
                    help="with --global: do NOT merge the safe read-only permission baseline (the "
                         "'big three') into ~/.claude/settings.json")
    wp.add_argument("--no-skills", dest="no_skills", action="store_true",
                    help="with --global: do NOT install the Matt Pocock skill suite (on by default)")
    wp.add_argument("--grok", action="store_true",
                    help="also wire Grok Build hooks + .grok/rules/celeborn.md for this project")
    wp.set_defaults(func=cmd_wire)

    # t120 — Modal-clean first run: one guided command over wire + init + login.
    stp = sub.add_parser("setup", help="one-command first run (Modal-style): wire Claude Code + scaffold "
                                       "this project + sign in (browser). Idempotent — re-run to resume.")
    stp.add_argument("--project", action="store_true",
                     help="wire the project's .claude/settings.json instead of ~/.claude (the default is a "
                          "global wire, so every session is covered)")
    stp.add_argument("--force", action="store_true", help="replace an existing non-Celeborn statusLine when wiring")
    stp.add_argument("--no-permission-baseline", dest="no_permission_baseline", action="store_true",
                     help="don't merge the safe read-only permission baseline (passed through to `wire`)")
    stp.add_argument("--no-skills", dest="no_skills", action="store_true",
                     help="don't install the Matt Pocock skill suite (passed through to `wire`)")
    stp.add_argument("--no-init", dest="no_init", action="store_true",
                     help="skip the per-project scaffold step (wire/sign-in only — e.g. a machine with no project yet)")
    stp.add_argument("--no-cmm", dest="no_cmm", action="store_true",
                     help="don't auto-engage Codebase Memory for this project (passed through to `init`)")
    stp.add_argument("--name", dest="name", default=None,
                     help="name this project for the board, skipping the prompt (passed through to `init`)")
    stp.add_argument("--no-open", dest="no_open", action="store_true",
                     help="don't launch/open the kanban board after scaffolding (passed through to `init`)")
    stp.add_argument("--no-browser", dest="no_browser", action="store_true",
                     help="start the board but don't pop a browser tab (passed through to `init`)")
    stp.add_argument("--no-login", dest="no_login", action="store_true",
                     help="skip the sign-in step. Login is required by default (Modal parity); this is the "
                          "documented opt-out for a purely local-first install.")
    stp.add_argument("--email", help="sign in with email + password instead of the GitHub browser flow")
    stp.set_defaults(func=cmd_setup)

    wq = sub.add_parser("wire-quality", help="opt-in deterministic quality gates: auto-test-on-edit + "
                                             "board `tsc --noEmit` (PostToolUse + Stop hooks; AGENTS.md fallback)")
    wq.add_argument("--local", action="store_true",
                    help="write the personal settings.local.json instead of the shared settings.json")
    wq.set_defaults(func=cmd_wire_quality)

    gkp = sub.add_parser("grok", help="Grok Build integration — wire hooks + per-project orient rules")
    gkp.add_argument("grok_action", nargs="?", default="wire", metavar="wire|sync-rules",
                     help="wire = install hooks + bootstrap; sync-rules = refresh .grok/rules/celeborn.md")
    gkp.set_defaults(func=cmd_grok)

    hp = sub.add_parser("hook", help="in-process Claude Code hook entry point (reads the event JSON on "
                                     "stdin); what `wire` points every hook at")
    hp.add_argument("event", choices=list(HOOK_EVENTS),
                    help="which hook event to run: " + ", ".join(HOOK_EVENTS))
    hp.set_defaults(func=cmd_hook)

    cp = sub.add_parser("consent", help="review the click-reducing automations (all opt-out) + record "
                                        "your agreement to the Celeborn User Agreement")
    cp.add_argument("--name", help="your full name — records agreement non-interactively")
    cp.add_argument("--opt-out", dest="opt_out",
                    help="comma-separated item numbers or keys to DISABLE (e.g. '5' or 'cd-redirect-autoallow')")
    cp.add_argument("--yes", action="store_true",
                    help="skip the interactive opt-out prompt (keep all enabled unless --opt-out given)")
    cp.add_argument("--show", action="store_true", help="print the recorded consent and exit")
    cp.set_defaults(func=cmd_consent)

    # Account + premium Supabase-backed sync (Phase 8b). Lazily imported so the core stays
    # network-free. Identity is Supabase Auth (email+password, TOTP MFA, GitHub OAuth); the free
    # account is OPTIONAL — the local core never needs it.
    rgp = sub.add_parser("register", help="create a free Celeborn account (email + password + username)")
    rgp.add_argument("--email", help="account email (prompted if omitted)")
    rgp.add_argument("--username", help="display username (prompted if omitted)")
    rgp.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_register(a))

    lgp = sub.add_parser("login", help="sign in (email+password +TOTP, or --github) to enable hosted sync")
    lgp.add_argument("--github", action="store_true", help="sign in with GitHub (browser PKCE) instead of a password")
    lgp.add_argument("--email", help="account email (prompted if omitted)")
    lgp.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_login(a))

    sub.add_parser("logout", help="revoke the session and delete local credentials").set_defaults(
        func=lambda a: __import__("celeborn_sync").cmd_logout(a))
    sub.add_parser("whoami", help="show the signed-in account (email, username, MFA, tier)").set_defaults(
        func=lambda a: __import__("celeborn_sync").cmd_whoami(a))
    acp = sub.add_parser("account", help="account: show identity (default), or `migrate` to heal a CLI/GitHub split")
    acp.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_whoami(a))
    acsub = acp.add_subparsers(dest="account_cmd")
    amp = acsub.add_parser("migrate",
                           help="move hosted projects from an old account into the one you're signed in as (CELE-t107)")
    amp.add_argument("--email", help="source (old) account email; prompted if omitted")
    amp.add_argument("--yes", action="store_true", help="skip the 'not signed in with GitHub' confirmation")
    amp.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_account_migrate(a))

    upg = sub.add_parser("upgrade", help="start a Celeborn Pro/Team subscription (opens Stripe Checkout)")
    upg.add_argument("--tier", choices=["pro", "team"], default="pro", help="plan to subscribe to (default: pro)")
    upg.add_argument("--annual", action="store_true", help="bill annually (≈2 months free) instead of monthly")
    upg.add_argument("--seats", type=int, default=1, help="number of seats to purchase (default: 1)")
    upg.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_upgrade(a))

    sub.add_parser("billing", help="manage your subscription (opens the Stripe billing portal)").set_defaults(
        func=lambda a: __import__("celeborn_sync").cmd_billing(a))

    mfp = sub.add_parser("mfa", help="manage TOTP MFA (Google Authenticator)")
    mfp.add_argument("action", nargs="?", default="status", choices=["enroll", "status", "disable"],
                     help="enroll a new TOTP factor, show status, or disable (default: status)")
    mfp.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_mfa(a))

    syp = sub.add_parser("sync", help="push/pull .context/ to the hosted (Supabase) backend")
    syp.add_argument("--watch", action="store_true", help="keep syncing on an interval instead of once")
    syp.add_argument("--interval", type=int, default=5, help="seconds between syncs in --watch mode (default 5)")
    syp.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_sync(a))

    # Architecture (CELE-t187): capture NON-SECRET infrastructure topology (vendor names, IPs,
    # control-surface URLs, DB endpoints) into .context/infra-local.json (gitignored). init/show are
    # local; sync pushes the topology (credentials stripped) to the hosted architecture diagram (Pro).
    arp = sub.add_parser("architecture",
        help="capture infrastructure topology for the hosted architecture diagram")
    arsub = arp.add_subparsers(dest="arch_cmd")
    ari = arsub.add_parser("init", help="scaffold .context/infra-local.json (auto-detects vendors)")
    ari.add_argument("--force", action="store_true", help="overwrite an existing infra-local.json")
    ari.set_defaults(func=cmd_architecture)
    arsub.add_parser("show", help="print the captured topology + control-surface links").set_defaults(
        func=cmd_architecture)
    arsub.add_parser("sync", help="push the topology (credentials stripped) to the hosted board").set_defaults(
        func=lambda a: __import__("celeborn_sync").cmd_architecture_sync(a))
    arsub.add_parser("trace", help="re-detect the stack + additively merge new pieces, then remap the hosted "
                                   "Stack (runs automatically every 3 turns + on a dependency-manifest edit)").set_defaults(
        func=cmd_architecture)
    arp.set_defaults(func=cmd_architecture, arch_cmd=None)  # bare `architecture` → show

    # Product federation registry (CELE-t190, Layer A of CELE-t188): name the repo-facets of one product,
    # their roles + publish policy (committed product.md) and this machine's checkout paths (gitignored
    # product-local.json). The orient banner reads it; Layers B/C/D build on it.
    pdp = sub.add_parser("product",
        help="product federation registry — repo-facets, roles, publish policy + orient banner (CELE-t190)")
    pdsub = pdp.add_subparsers(dest="product_cmd")
    pdsub.add_parser("list", help="print the facet table (roles · publish · bound/unbound here)").set_defaults(
        func=cmd_product, product_cmd="list")
    pdi = pdsub.add_parser("init", help="scaffold .context/product.md (the committed registry)")
    pdi.add_argument("--name", default=None, help="product name (default: project name / folder)")
    pdi.add_argument("--force", action="store_true", help="overwrite an existing product.md")
    pdi.set_defaults(func=cmd_product, product_cmd="init")
    pda = pdsub.add_parser("add", help="add or update a facet in product.md (product FACTS only, no paths)")
    pda.add_argument("key", help="facet key (e.g. client, server)")
    pda.add_argument("--role", required=True, choices=list(PRODUCT_ROLES), help="facet role")
    pda.add_argument("--publish", default=None, help="publish policy (e.g. never, pypi, fork+PR)")
    pda.add_argument("--repo", default=None, help="canonical remote URL (portable — no local checkout path)")
    pda.add_argument("--upstream", default=None, help="upstream remote (for oss:* facets)")
    pda.set_defaults(func=cmd_product, product_cmd="add")
    pdb = pdsub.add_parser("bind", help="bind a facet → this machine's checkout path (gitignored, per-machine)")
    pdb.add_argument("key", help="facet key to bind")
    pdb.add_argument("checkout", help="absolute path to the local checkout on this machine")
    pdb.set_defaults(func=cmd_product, product_cmd="bind")
    pdp.set_defaults(func=cmd_product, product_cmd=None)  # bare `product` → list

    # Multi-repo git/PR ops (CELE-t191, Layer B of CELE-t188): route git + a drafted `gh pr create` to a
    # bound facet checkout, auto-attributing each op (touch + Celeborn-Task/-Agent/-Model trailers). The
    # publish guard (PreToolUse) enforces role policy; these route. All read the Layer A registry (t190).
    cmp_ = sub.add_parser("commit",
        help="facet-routed git commit into a bound checkout, with auto touch + trailers (CELE-t191)")
    cmp_.add_argument("--facet", required=True, help="facet key to route the commit to (must be bound here)")
    cmp_.add_argument("-m", "--message", required=True,
                      help="commit message (Celeborn-Task/-Agent/-Model trailers appended automatically)")
    cmp_.add_argument("--task", default=None,
                      help="task id for the Celeborn-Task trailer (default: this session's doing card)")
    cmp_.add_argument("--by", default=None, help="agent handle (default: this session)")
    cmp_.add_argument("--family", default=None, help="agent family for attribution (e.g. Claude)")
    cmp_.add_argument("--model", default=None, help="agent model for attribution (e.g. Opus 4.8)")
    cmp_.add_argument("--session", default=None, help=argparse.SUPPRESS)
    cmp_.add_argument("files", nargs="*",
                      help="files to stage+commit (paths relative to the facet checkout); omit to commit staged")
    cmp_.set_defaults(func=cmd_commit)

    psp_ = sub.add_parser("push",
        help="facet-routed git push to a bound checkout (release/tag push guarded by role) (CELE-t191)")
    psp_.add_argument("--facet", required=True, help="facet key to route the push to (must be bound here)")
    psp_.add_argument("remote", nargs="?", default=None, help="git remote (e.g. origin); default: git's default")
    psp_.add_argument("branch", nargs="?", default=None, help="branch/refspec to push; default: git's default")
    psp_.add_argument("-u", "--set-upstream", dest="set_upstream", action="store_true",
                      help="pass --set-upstream to git push")
    psp_.add_argument("--tags", action="store_true",
                      help="push tags too (a RELEASE push — refused on server:private/oss:* facets)")
    psp_.add_argument("--follow-tags", dest="follow_tags", action="store_true",
                      help="pass --follow-tags (a RELEASE push — refused on server:private/oss:* facets)")
    psp_.set_defaults(func=cmd_push)

    prd_ = sub.add_parser("pr",
        help="DRAFT a facet-routed pull request (prints a gh command; never auto-sends) (CELE-t191)")
    prd_.add_argument("--facet", required=True, help="facet key to route the PR to (must be bound here)")
    prd_.add_argument("--base", default=None, help="base branch for the PR (default: main)")
    prd_.add_argument("--title", default=None, help="PR title (default: the top commit subject)")
    prd_.add_argument("--body", default=None, help="PR body (default: a bullet list of the commits)")
    prd_.add_argument("--task", default=None, help="task id for the PR provenance (default: session's doing card)")
    prd_.add_argument("--by", default=None, help="agent handle for the drafted-by line (default: session)")
    prd_.add_argument("--family", default=None, help=argparse.SUPPRESS)
    prd_.add_argument("--model", default=None, help=argparse.SUPPRESS)
    prd_.add_argument("--session", default=None, help=argparse.SUPPRESS)
    prd_.set_defaults(func=cmd_pr)

    # Manage hosted projects on celeborn.thot.ai (t97): list them, or remove one (incl. an orphan whose
    # repo was deleted — removal is hosted-only, no local .context/ needed). RM cascades server-side.
    prp = sub.add_parser("project", help="manage hosted projects (list / remove) on celeborn.thot.ai")
    prsub = prp.add_subparsers(dest="project_cmd")
    prsub.add_parser("list", help="list your hosted projects (name · id)").set_defaults(
        func=lambda a: __import__("celeborn_sync").cmd_project(a))
    prr = prsub.add_parser("rm", help="remove a hosted project by name or id (cascades; PERMANENT)")
    prr.add_argument("name", metavar="name|id", help="project name (exact) or its uuid")
    prr.add_argument("--yes", action="store_true", help="skip the type-the-name confirmation")
    prr.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_project(a))
    prp.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_project(a))  # bare `project` → list

    # Hidden: best-effort live push of changed cards to the hosted board, spawned detached by a local
    # task mutation so celeborn.thot.ai updates in ~realtime. Not for direct use.
    hpp = sub.add_parser("hosted-push", help=argparse.SUPPRESS)
    hpp.add_argument("--ids", default="", help="comma-separated task ids to push")
    hpp.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_hosted_push(a))

    # Hidden: best-effort live push of the active-agents windows to the hosted board, spawned detached
    # by the per-turn capture so hosted token chips track the live local windows (CELE-t131). Not for
    # direct use.
    hpa = sub.add_parser("hosted-push-agents", help=argparse.SUPPRESS)
    hpa.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_hosted_push_agents(a))

    # GitHub App: bind a repo so the App ingests its PR/issue threads (capture free; pull = Pro sync).
    ghp = sub.add_parser("github", help="GitHub App integration (link a repo to ingest PR/issue threads)")
    ghsub = ghp.add_subparsers(dest="github_cmd", required=True)
    ghl = ghsub.add_parser("link", help="link <owner/repo> to this project for GitHub App ingest")
    ghl.add_argument("repo", metavar="owner/repo", help="GitHub repository as <owner>/<repo>")
    ghl.add_argument("--installation", help="App installation id (shown on the App's post-install page)")
    ghl.set_defaults(func=lambda a: __import__("celeborn_sync").cmd_github_link(a))

    # Bidirectional Jira Cloud integration (Phase 10/11). Lazily imported; transport is the Jira REST
    # API + an API token (works headless, unlike the OAuth MCP server). See celeborn_jira.py.
    jp = sub.add_parser("jira", help="bidirectional Jira Cloud sync (issues ↔ tasks/phases)")
    jsub = jp.add_subparsers(dest="jira_cmd", required=True)
    jc = jsub.add_parser("connect", help="connect a Jira Cloud site (hidden API-token prompt)")
    jc.add_argument("--site", help="https://yourname.atlassian.net (prompted if omitted)")
    jc.add_argument("--email", help="Atlassian account email (prompted if omitted)")
    jc.add_argument("--project", help="project key to sync, e.g. CEL (prompted if omitted)")
    jc.add_argument("--token", help="API token (prefer CELEBORN_JIRA_TOKEN env — never commit tokens)")
    jc.add_argument("--json", action="store_true", help="print connection JSON after success")
    jc.set_defaults(func=lambda a: __import__("celeborn_jira").cmd_jira(a))
    js = jsub.add_parser("status", help="verify the stored Jira connection")
    js.add_argument("--json", action="store_true", help="print JSON (for the board API)")
    js.set_defaults(func=lambda a: __import__("celeborn_jira").cmd_jira(a))
    jrec = jsub.add_parser("reconcile", help="audit Jira vs Celeborn (Celeborn wins); --apply pushes outward")
    jrec.add_argument("--apply", action="store_true", help="push all Celeborn cards → Jira (no orphan import)")
    jrec.add_argument("--json", action="store_true", help="print JSON report")
    jrec.set_defaults(func=lambda a: __import__("celeborn_jira").cmd_jira(a))
    jpull = jsub.add_parser("pull", help="pull Jira issues → tasks (idempotent; links via the issue key)")
    jpull.add_argument("--dry-run", dest="dry_run", action="store_true", help="preview without writing tasks.md")
    jpull.set_defaults(func=lambda a: __import__("celeborn_jira").cmd_jira(a))
    jpush = jsub.add_parser("push", help="push tasks → Jira (PREVIEW by default; --apply to write)")
    jpush.add_argument("ids", nargs="*", help="task ids to push (default: all cards)")
    jpush.add_argument("--apply", action="store_true", help="actually write to Jira (default is a safe preview)")
    jpush.add_argument("--type", help="issue type for NEW issues (default: Task)")
    jpush.add_argument("--sprint", help="where new issues land: active | backlog | <sprint-id> (default: active)")
    jpush.set_defaults(func=lambda a: __import__("celeborn_jira").cmd_jira(a))
    jflush = jsub.add_parser("flush", help="drain the auto-push queue now (also runs after capture)")
    jflush.add_argument("--force", action="store_true", help="ignore per-task debounce")
    jflush.set_defaults(func=lambda a: __import__("celeborn_jira").cmd_jira(a))

    # codebase-memory-mcp (CMM) integration (Sprint 1 "Zero Prompts"). Lazily imported; all glue
    # lives in celeborn_cmm.py (interface-level, depends on CMM's public surface, never internals).
    cmp_ = sub.add_parser("cmm", help="codebase-memory-mcp: pre-clear permissions + engage structural memory")
    cmsub = cmp_.add_subparsers(dest="cmm_cmd", required=True)
    cme = cmsub.add_parser("engage", help="pre-clear CMM's read-only tools, register MCP, index, install the flow-first North Star")
    cme.add_argument("--global", dest="global_", action="store_true",
                     help="write the allow-list to ~/.claude/settings.json (default: project .claude/settings.json)")
    cme.add_argument("--force", action="store_true", help="re-engage even if this project is opted out")
    cme.add_argument("--no-provision", dest="no_provision", action="store_true",
                     help="skip auto-provisioning the pinned CMM binary (S2)")
    cme.set_defaults(func=lambda a: __import__("celeborn_cmm").cmd_cmm(a))
    cmo = cmsub.add_parser("off", help="disengage CMM for this project (revert added entries; sticky opt-out)")
    cmo.add_argument("--global", dest="global_", action="store_true", help="revert the global allow-list")
    cmo.set_defaults(func=lambda a: __import__("celeborn_cmm").cmd_cmm(a))
    cms = cmsub.add_parser("status", help="report engaged/indexed/version/allow-list state")
    cms.add_argument("--global", dest="global_", action="store_true", help="inspect the global allow-list")
    cms.add_argument("--json", action="store_true", help="print JSON")
    cms.set_defaults(func=lambda a: __import__("celeborn_cmm").cmd_cmm(a))
    # S2 "Zero Touch": provisioning + upstream tracking. Glue lives in celeborn_cmm_provision.py.
    cmpv = cmsub.add_parser("provision", help="fetch + checksum-verify + cache the pinned CMM binary (S2)")
    cmpv.add_argument("--force", action="store_true", help="re-download even if a valid cached copy exists")
    cmpv.set_defaults(func=lambda a: __import__("celeborn_cmm_provision").cmd_provision(a))
    cmct = cmsub.add_parser("contract", help="run the CMM interface contract test (14 tools + ids); exits non-zero on drift")
    cmct.add_argument("--json", action="store_true", help="print JSON")
    cmct.set_defaults(func=lambda a: __import__("celeborn_cmm_provision").cmd_contract(a))
    cmsc = cmsub.add_parser("sync-check", help="watch upstream for a newer pinned release; gate it behind the contract test, plan a PR (S2)")
    cmsc.add_argument("--apply", action="store_true", help="execute a green plan as a branch + gh PR (default: dry-run plan)")
    cmsc.set_defaults(func=lambda a: __import__("celeborn_cmm_provision").cmd_sync_check(a))
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
