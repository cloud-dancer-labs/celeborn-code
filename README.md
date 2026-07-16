# Celeborn Code

> Celeborn Code — persistent context and a multi-agent kanban for AI coding agents ([celeborncode.ai](https://celeborncode.ai)).
> Source-available under the [Business Source License 1.1](https://github.com/cloud-dancer-labs/celeborn-code/blob/main/LICENSE) (© Cloud Dancer; distributed by Thot Technologies LLC). Saves time, money, and tokens — and mitigates session restarts.

**A long-term context substrate for coding agents.** Celeborn Code gives any repository a small,
disciplined memory on disk so AI coding sessions stay light, survive compaction, and keep their
place across days or weeks — instead of degrading as the context window fills with chatter.

> ### Is this the Celeborn you're looking for?
> **Yes — if you want memory for an AI coding agent.** *Celeborn Code* is this project: a context
> substrate for AI coding agents (Claude Code, Codex, Grok). Install it with **`uv tool install celeborn-code`**
> or **`pip install celeborn-code`**; home is **[celeborncode.ai](https://celeborncode.ai)** and
> **[github.com/cloud-dancer-labs/celeborn-code](https://github.com/cloud-dancer-labs/celeborn-code)**.
>
> It is **not** these same-named projects:
> - **[Apache Celeborn](https://celeborn.apache.org)** — a remote shuffle service for Spark/Flink/MapReduce (big-data infra). Different product, different company.
> - **[`frkngksl/Celeborn`](https://github.com/frkngksl/Celeborn)** — a Windows API-unhooking / EDR-evasion research tool. Unrelated.
>
> If an agent told you to "install Celeborn" for coding-agent memory, **this** is the one — `uv tool install celeborn-code`.
> (Releases up to 0.2.0 were published on PyPI under the name `celeborn`; that project is deprecated — `celeborn-code` is the package.)

---

## Install

**Two commands.** Install the CLI, then run it from inside your project folder:

```bash
uv tool install celeborn-code   # installs the `celeborn` command — macOS, Windows, and Linux
celeborn init                   # wires the hooks, scaffolds the project, and opens your board
```

No [uv](https://docs.astral.sh/uv/)? `pip install celeborn-code` works too. Then open your AI coding tool
— **Claude Code, Codex, or Grok** — in that project and just say hi: Celeborn orients itself, asks
what you want to build, and remembers everything from there.

<sub>Prefer Homebrew or winget? See the **Install — manual &amp; advanced** section below — every
path ends at the same `celeborn init`.</sub>

---

*Everything below is for builders who want the details — how it works, manual setup, and the full
architecture. If you just ran the command above, you're already done.*

## The idea

LLM sessions degrade as they grow: the window fills with old tool output and stale state, then
compaction erases working memory and the agent drifts. Celeborn fixes this not with a bigger window
but with a **layered memory on disk**:

- only a tiny **Hot tier** loads by default — its size is bounded *no matter how much total memory
  the project has accumulated*;
- everything deeper is reached **on demand** by targeted search, returning snippets, not whole files;
- memory lives in files, so it **survives compaction and full thread restarts**;
- resuming gets **cheaper over time**, not more expensive.

**The tradeoff, stated honestly:** every unit of work pays a small write-tax — update one live brief,
append one journal line, occasionally refresh an index. In exchange you get a bounded context
footprint and continuity that spans days. *Slightly slower per step; dramatically longer-lived per
session.* It is an opt-in bet, and Celeborn makes the tax as small and automatable as possible.

## How it works

Memory is a per-repo `.context/` directory, organized in tiers:

| Tier | File(s) | Loaded at session start? |
|---|---|---|
| **Hot** | `state.md` (headline), `session.json`, `durable/manifest.md` | **Always** |
| **Warm** | `notes.md` (unbounded working detail), `journal.md` (tail) | On demand |
| **Cold** | `journal-archive/` | Only via search |
| **Distilled** | `learnings.md`, `decisions.md` | On demand |
| **Durable** | `durable/*.md` | Manifest always; bodies on demand |
| **Automatic Context Record** | `activity.md`, `auto/*.md` (mechanical capture) | `activity.md` always; turns via search |
| **Index** | `index.db` (SQLite FTS, gitignored) | Queried, never read into context |

The **Automatic Context Record** is what makes "stores context automatically" literally true. A `Stop` hook runs
`celeborn capture` every turn — it reads the Claude Code transcript and records a *faithful, near-complete*
account of the turn (your prompt, the assistant's text, every tool call with its full input, and tool
outputs) deterministically, **with no model in the loop** and **with secrets redacted on the way in**.
The full record lives in the cold `auto/*` tier (reached by search + sync); a small `activity.md` digest
is derived from it and is the only part that loads on Orient, so the Hot tier stays bounded no matter how
much gets recorded. It never touches the judgment-authored tiers; it's an always-current safety net so that
even if `state.md` drifts, `activity.md` + search recover what actually happened. **Capture fires for every
session** — inside a repo it writes that repo's `.context/`; run from anywhere else it falls back to a
global `~/.context/` so nothing goes unrecorded. (Sync the global record with `celeborn --path "$HOME" sync`.)

Because the Automatic Context Record holds your **verbatim prompts, file paths, and commands**, Celeborn treats it
specially. The authored tiers are curated summaries safe to share; raw prompts are not — they can
contain half-formed ideas, internal names, customer details, or anything you happened to type. If that
rode along in git, then **anyone with access to the repo would see it** — and in a *public* or shared
repo that means publishing your raw working transcript to the world. So Celeborn **keeps the Automatic Context Record
out of git entirely** (it's gitignored, never committed, never in the shared history).

But you still want that history on **your own other machines**. That's the gap git can't fill once a
file is gitignored — and it's a core reason Celeborn offers **hosted sync**: `celeborn sync` carries the
Automatic Context Record (and the rest of `.context/`) **privately across your devices** — your data only, behind
row-level security, with secrets redacted before upload — *without* ever exposing verbatim prompts in a
repo others can read. Git is the **shared/public** channel (curated, safe to publish); `celeborn sync`
is the **private, cross-device** channel (everything, for your eyes only).

**Where the work happens — context management is client-side.** Celeborn's core loop intercepts the
*agent's* loop, which runs on your machine; the cloud is an optional backbone, never the brain:

```text
   CLIENT (your machine) — THE PRODUCT          CLOUD (optional)
   ┌─────────────────────────────────┐          ┌──────────────────────┐
   │ Claude Code / Codex / Grok       │          │ Supabase (sync)      │
   │   ▲ agent loop                   │   sync   │  ▲ ingested entries  │
   │ celeborn CLI + hooks  ───────────┼─────────►│  │                   │
   │   • capture / orient / search    │          │ GitHub App webhooks  │
   │   • the .context/ engine ◄───────┼──────────┼──┘ Ingest (sink)     │
   └─────────────────────────────────┘          │   Drift  (reporter)  │
        ↑ context management lives HERE          └──────────────────────┘
```

The CLI + hooks are the product — they're what manages context, because only code running inside the
session can see the live agent loop. The cloud side is optional: `celeborn sync` moves your managed
context between devices, and the **GitHub App** *(planned)* hangs off the edge as a data source — it
**ingests** PR/issue threads into your store and **reports** context drift as a PR check, but it never
manages context and never publishes your memory.

Markdown is the source of truth (git-diffable, human- and model-readable). The SQLite index is
**derived** and disposable — if it goes stale or missing, regenerate it from the markdown. That one
design choice is what makes a Celeborn project *portable*: the truth is plain text that travels in
git, and the fast path rebuilds itself anywhere in milliseconds.

The agent works five verbs, taught by the bundled skill:
**Orient** (cheap rehydration) · **Checkpoint** (write current state) · **Forget** (archive/prune to
keep Hot small) · **Promote** (distill `journal → learnings → durable`) · **Handoff** (a tiny resume
prompt so a fresh thread can pick up where the last died).

There is one ground rule: **`.context/` is the single source of truth.** No parallel `NOTES.md` or
`PROGRESS.md` to drift out of sync — everything durable goes into a tier, and every session hydrates
from Celeborn.

## Install — manual & advanced

Celeborn is a **CLI plus editor hooks** — the hooks invoke a `celeborn` command every turn. It runs on
**macOS, Windows, and Linux**, and works with **Claude Code, Codex, and Grok** (Claude Code gets native
hooks; Codex and Grok read the `AGENTS.md` + `.grok/rules` that `init` writes). **Copying the files into
your project is not enough:** that gets you the scaffold but no installed command for the hooks to run,
so nothing fires.

### Two steps

```bash
# 1 — install the command (pick your platform)
brew install cloud-dancer-labs/celeborn/celeborn                        # macOS
winget install ThotTechnologies.Celeborn                         # Windows (or: scoop — see below)
uv tool install celeborn-code                                    # any OS (or: pip install celeborn-code)

# 2 — one guided command: wires your coding agent, scaffolds this project, signs you in, opens your board
cd your-project
celeborn init
```

That's the whole first run. **`celeborn init` is the one command** — it does the hook wiring
(`wire --global`), the per-project scaffold (which also writes the `AGENTS.md` + Grok rules Codex/Grok
auto-load), a sign-in (`login` — email + password; GitHub linkable afterwards), and then opens your kanban board — in one pass. It's
**idempotent and resumable** — re-run it any time and it skips the steps already done. Useful flags:
`--no-login` (purely local-first, skip the account), `--project` (wire just this repo, not `~/.claude`),
`--name "<name>"` / `--no-open` (board), `--no-skills` / `--no-permission-baseline` (wiring). **Grok
users:** for Grok's full hooks (not just project rules) also run `bash grok/scripts/install.sh` (see
[Grok Build](#grok-build)). Run `celeborn init --help` for the full list. *(Only need the per-project
files, agent already wired? `celeborn scaffold` does just the scaffold step.)*

Everything below is the **detailed / manual breakdown** of what `init` runs — read it when you want to
run the pieces by hand, wire CI, or tune the defaults.

#### Manual / advanced — the individual steps

**1 — Install the `celeborn` / `cel` command** (pick one):

```bash
# macOS (Homebrew) — no Python toolchain needed:
brew install cloud-dancer-labs/celeborn/celeborn

# Windows (winget or Scoop) — no Python toolchain needed:
winget install ThotTechnologies.Celeborn
#   …or:  scoop bucket add celeborn https://github.com/cloud-dancer-labs/scoop-celeborn && scoop install celeborn

# Any OS with a Python toolchain — uv (recommended) or pip:
uv tool install celeborn-code        # or:  pip install celeborn-code
```

Every path delivers the same **compiled Celeborn binary**. Homebrew/winget/Scoop fetch it straight
from the release; the PyPI package is a **thin installer** — the one small module in this repository —
that on first run downloads the version-pinned tarball from
[`celeborn-releases`](https://github.com/cloud-dancer-labs/celeborn-releases/releases), verifies its
sha256 against the checksum baked in at release time, and places the binary under `~/.celeborn/bin/`
(a mismatched download is never placed). Verify it's on your PATH: `celeborn version` — and
`celeborn --installer-info` shows what the installer resolved without touching the network.

**2 — Wire the hooks into Claude Code** (once). The easy way — run it from your clone:

```bash
celeborn wire --global      # merges statusLine + the 6 hook groups into ~/.claude/settings.json
                            # (omit --global to wire just the current project's .claude/settings.json)
```

Each hook is a single in-process `celeborn hook <event>` command — no bash wrapper, no inline
`python3`, no `$CELEBORN_HOME` (so the `celeborn` you installed in step 1 must be on PATH). `wire` is
idempotent (re-running adds nothing), preserves anything already in the file, won't replace a
non-Celeborn `statusLine` without `--force`, and **migrates an older bash-based install** in place.

`wire --global` also merges a small, **safe permission baseline** into `~/.claude/settings.json` so
you stop re-approving the same read-only commands: the read-only built-ins (`Read`/`Glob`/`Grep`),
prefix-wildcards for read-only / trivially-reversible shell commands (`grep`, `ls`, `git log/diff/
show/status`, `gh … view/list`, `curl -sS http://localhost`, …), and `defaultMode: acceptEdits` (file
edits auto-approve; Bash and anything outward-facing still prompts). It is **ask-wins** — it never
replaces an entry, never touches a command you've put in `permissions.deny`, and never changes a
`defaultMode` you've already set — and **idempotent**. It complements `celeborn permissions --suggest`
(which learns wildcards from your *own* approval history). Opt out with
`celeborn wire --global --no-permission-baseline`; revert later via `/permissions` or by editing the
file. Changes apply to **new** sessions (`Shift+Tab` toggles acceptEdits in the current one).

`wire --global` also installs **[Matt Pocock's skill suite](https://github.com/mattpocock/skills)**
default-on (via `npx --yes skills@latest add mattpocock/skills` → `.claude/skills/`, auto-discovered by
Claude Code) and **keeps it current**: the SessionStart hook fires a detached, throttled (~weekly)
background refresh so the skills stay up to date without blocking orient. Opt out with
`celeborn wire --global --no-skills` (or `autoupdate:false` in `~/.config/celeborn/skills.json`, or the
`CELEBORN_NO_SKILLS` env); refresh on demand with `celeborn skills update`. `celeborn skills` lists
what's available — Celeborn's own verbs, the advisor's recommended skills, and the Matt Pocock suite.
**Scope note:** the recommended slash-command skills and the Matt Pocock suite are **Claude-only**
(they live under `.claude/skills/`); on Grok/Codex the advisor surfaces the *same* recommendations as
prose, not installable skills. The board's **Settings** page (the ⚙️ on the kanban board) shows the
skills (with an Update button + last-refresh), which auto-allows are active, and a red **Danger Zone**
to toggle the full (unsafe) auto-allow spectrum.

Prefer it over hand-editing — but if you'd rather own `~/.claude/settings.json` yourself, run
`celeborn wire` once and take over the managed block it writes:

```jsonc
{ "statusLine": { "type": "command", "command": "celeborn hook statusline" },
  "hooks": { /* SessionStart · UserPromptSubmit · PreCompact · SessionEnd · Stop · Notification — copy from the snippet */ } }
```

This is what makes capture, the per-turn note, the statusLine, and context reminders fire automatically. The hooks
no-op in repos without a `.context/` (so they're safe to enable everywhere), except capture, which
falls back to a global `~/.context/`. See [Auto-load on Claude Code](#auto-load-on-claude-code) for the
per-hook detail. On agents without hooks (Codex, Claude.ai), skip this — the bundled skill drives the
same verbs by hand.

**3 — Scaffold each project:**

```bash
cd your-project
celeborn scaffold    # creates .context/ (always private) + annotates CLAUDE.md
celeborn status      # what an agent loads on Orient
```

`celeborn init` already runs this scaffold step; `celeborn scaffold` is the same step on its own, for
when your agent is already wired and you just want a new project's files. When you run it at a terminal
it **asks for a project name** (defaulting to the folder name) and then **opens the project's localhost
kanban board** — that board is Celeborn's UI, where you see the control surfaces (tasks, run/fleet,
settings), and the SessionStart hook keeps it live from then on. Pass `--name "<name>"` to skip the
prompt, `--no-browser` to start the board without popping a tab, or `--no-open` to skip the board
entirely. Non-interactive installs (CI, scripts, headless agents) skip the prompt and the browser
automatically — the board still comes up on the next session's Orient.

Your `.context/` is **always private** — gitignored, never committed (see the note below). From here
Celeborn records every turn and rehydrates on each new session. Scaffolding also **engages Codebase
Memory (CMM)** for the project if the CMM binary is installed (step 4) — pre-clearing its read-only
tools so structural questions never stop to ask permission. Opt out per project with
`celeborn scaffold --no-cmm` (or globally with `CELEBORN_NO_CMM=1`); reverse anytime with `celeborn cmm off`.

**4 — Strongly recommended: install Codebase Memory (CMM).** This is the single biggest accelerator
Celeborn can give a vibe coder, and it's worth understanding why:

> **The human is the bottleneck.** Every time the agent reaches for `Bash`, `grep`, `rg`, `find`, or
> `cat` to understand your code, Claude Code stops and waits for you to click **"Allow."** Step away
> from the laptop and the *entire* project blocks on that one click. **[CMM](https://github.com/DeusData/codebase-memory-mcp)**
> ([DeusData/codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp)) indexes your
> codebase into a queryable graph and answers structural questions — call chains, impact, routes, "where
> is X defined/used" — through pre-cleared, read-only tools. The agent stops shelling out to search, so
> the prompts stop firing, so **you stop being the thing it waits on.** Removing that human-in-the-loop
> stall is the key to uninterrupted flow; the token savings and richer code intelligence are a bonus.

CMM is **its own binary**, installed once in your dev environment (not vendored into Celeborn). Then
Celeborn engages it per project automatically:

```bash
# Install the CMM binary (one time, per machine) — pick one:
curl -fsSL https://raw.githubusercontent.com/DeusData/codebase-memory-mcp/main/install.sh | bash
#   …or from source:  git clone https://github.com/DeusData/codebase-memory-mcp && cd codebase-memory-mcp && scripts/build.sh
#   …then put the binary on PATH, or point Celeborn at it:  export CELEBORN_CMM_BIN=/path/to/codebase-memory-mcp

# Now, in any Celeborn project:
celeborn cmm engage    # (also runs automatically on `celeborn init`) — pre-clears CMM's read-only
                       # tools + Grep/Glob, registers the MCP server, indexes the repo, and installs
                       # the "flow is primary" note into the project's agent instructions
celeborn cmm status    # engaged? indexed? 14 tools cleared?
celeborn cmm off       # disengage + revert everything CMM added (sticky per-project opt-out)
```

The 11 read-only CMM tools (plus native `Grep`/`Glob`) are auto-allowed; the 3 mutating tools
(`delete_project`, `manage_adr`, `ingest_traces`) always prompt by design. If the CMM binary isn't
installed, Celeborn still pre-clears what it can and degrades gracefully — it never blocks a session —
and reminds you to install CMM for the structural half.

> **Keep the CLI current.** A stale install silently shadows newer hooks and skips features. If you
> installed via uv/pip, run `uv tool upgrade celeborn-code` (or `pip install -U celeborn-code`) after
> Celeborn updates; Homebrew/winget/Scoop upgrade the usual way. Check anytime with `celeborn version --check`.

## Quickstart

> **Your memory is always private — never in git.** `.context/` holds your prompts, notes, and working
> memory, so Celeborn **always gitignores it** — there is no option to commit it. (Committing it to a repo
> that is, or ever becomes, public would leak all of that permanently into git history.) It lives on your
> machine and travels **between your devices via your account** — `celeborn init` (or
> `celeborn login` later), then `celeborn sync` — **not** git. See the [FAQ](#faq) for the why.

```bash
# from your project root
celeborn scaffold        # scaffold .context/ (always gitignored/private)
celeborn status          # what an agent loads on Orient
celeborn index           # build the search index
celeborn search "that decision about auth"
celeborn doctor          # health check + secret scan
celeborn integrity       # verify the install matches the published release
celeborn metrics         # estimated tokens saved + restarts avoided
celeborn version --check # is a newer Celeborn available?
```

`version --check` looks back at the published releases and compares your installed version to the
latest. It's offline-safe (skips quietly if GitHub is unreachable), and the `CLAUDE.md` block reminds
the agent to run it now and then.

`integrity` verifies that the install matches its published release — it **detects** an install that
was tampered with in place (it doesn't prevent edits), and `doctor` surfaces the same status. The
line it protects: the file formats, the documented CLI verbs, and the hook protocol are the **stable
public contract**; module internals change freely between versions. A modified install reports
`modified` and points you back to a clean reinstall — and per the LICENSE Supplemental Terms,
warranty and support cover only checksum-valid installs.

(Prefer the short `celeborn` / `cel` command? That's the one-time [Install](#install) above.)

### Product federation (multi-repo) — optional

A product often spans several repos — a public client, a private server, vendored or forked
dependencies — each with its own rules. `celeborn product` names those **facets** in a small registry
so every agent knows, on orient, what facets exist and which are present on this machine:

```bash
celeborn product init                                             # scaffold .context/product.md
celeborn product add client --role client:public --repo github.com/you/app
celeborn product add server --role server:private --repo github.com/you/app-cloud
celeborn product bind  client /path/to/checkout                   # per-machine, gitignored
celeborn product                                                  # the facet table
```

Product **facts** (facet keys, roles, publish policy, repo URLs) live in the committed `product.md`;
this machine's **checkout paths** live in a gitignored `product-local.json`, so a facet you haven't
checked out here simply shows as *not present* rather than following a dead path. When a `product.md`
exists, orient leads with a one-line banner of the facets and their bound/unbound state.

Once facets are bound, git/PR ops route to the right repo and attribute themselves automatically:

```bash
celeborn commit --facet client -m "fix parser" parser.py   # commit in the client checkout + trailers
celeborn push   --facet client                             # git push, routed to that checkout
celeborn pr     --facet client --base main                 # DRAFT a PR (prints a gh command; never sends)
```

`commit` appends `Celeborn-Task`/`-Agent`/`-Model` trailers and registers a cross-repo touch, so one
board coordinates work across every repo. A **publish guard** hard-DENYs a release (`twine`/`npm publish`,
`gh release create`, a tag push, …) targeting a `server:private` or `oss:*` facet — the former never
publishes, the latter is contributed back via fork → PR.

### Sync across devices (optional)

Because `.context/` is always private (gitignored), git never carries it between machines — so Celeborn can.
**Celeborn is a paid subscription with a 7-day free trial** — every account starts as a full Pro
trial (card upfront, $0 today, cancel anytime in one click), because that's what lets us keep
building and supporting it:

- **Pro — $8/seat/mo (starts with a 7-day free trial):** the full product. Everything local —
  capture, the tiered store, search, the board, git-daemon sync, bring-your-own-Supabase — plus
  hosted sync: `celeborn login` → `celeborn sync`, cross-device, real-time, zero-setup, unlimited
  projects. Secrets are redacted out before upload; your local SQLite index is never synced. Pro
  also includes the [encrypted secrets manager](#encrypted-secrets-manager-pro) (`celeborn
  secrets`). Annual ($80/yr) is available from the billing portal after your trial converts.
- **Team — $12/seat/mo:** everything in Pro, plus shared projects, org admin, shared context, and shared
  agent telepathy (the multi-agent bus). A straight upgrade from inside Pro — no separate trial.
  **Enterprise:** SSO + custom terms — [get in touch](#support).

Whatever you decide on day 8, your `.context/` memory stays on your disk as plain Markdown,
readable by anything — your memory is yours, the machinery is ours.

### Encrypted secrets manager (Pro)

New to secrets? The rule is: **API keys never belong in your repo, your `.env`, or a chat prompt.**
Celeborn Pro gives them one safe home — an encrypted vault ([Infisical](https://infisical.com), your
own free account) — and reads them back *locally, at run time* when a command needs them:

```bash
celeborn secrets setup                      # once: browser login + Celeborn creates your vault project
celeborn secrets set ANTHROPIC_API_KEY      # hidden prompt — the value never touches your repo's disk
celeborn secrets run -- vercel deploy       # runs with vault secrets injected as env vars
```

`setup` is fully hands-off: it provisions a pinned, checksum-verified `infisical` CLI, opens the
browser login (signup happens right there), creates the vault project over Infisical's API with your
own session, and writes the committable `.infisical.json` (project id only — no sensitive data). You
never see a dashboard. `secrets list` shows names, never values; `secrets status` shows the wiring;
self-hosters can point `--host` at their own Infisical.

Celeborn also **enforces the discipline**: `celeborn doctor` (and the advisor nudges every agent
harness gets) flag any live-looking secret *value* sitting in a repo `.env*` file and walk you
through moving it into the vault. Non-secret config stays in `.celebornrc`; the login token lives in
your OS keyring, managed by the Infisical CLI itself — Celeborn never stores a vault secret anywhere.

### The economy estimate

Celeborn keeps an honest running estimate in `.context/metrics.json`:

- **Tokens saved** — `tokens(all of .context/) − tokens(Hot tier)`, summed over load events. The
  counterfactual is "if you'd naively loaded all project memory every session."
- **Restarts avoided** — fresh sessions that resumed onto existing memory + compactions the memory
  bridged (the `PreCompact` hook fires there).

`status` and `metrics` are read-only — inspecting never inflates the numbers; only the hooks (or a
manual `celeborn record`) do.

With **hosted sync**, each `celeborn sync` also writes that project's current totals to the backend, and
the server keeps a **per-user running total across all your projects** (the `user_savings` view, sums
the per-project values so there's no double-counting). `sync` prints it back — *"Σ N tokens saved across
M project(s) — your running total on Celeborn"* — so you see your whole footprint, not just this repo's.

### Auto-load on Claude Code

`celeborn init` annotates your project's **`CLAUDE.md`** with a small managed block (between
`<!-- BEGIN CELEBORN -->` markers) telling the agent that context lives in `.context/` and how to
orient. Since Claude Code auto-loads `CLAUDE.md`, this is the zero-config baseline — Celeborn announces
itself even before any hooks or the skill are active. It's idempotent (re-running `init` refreshes the
block, never duplicates it) and preserves the rest of the file; opt out with `init --no-claude-md`.

For the live, per-turn behaviour, run `celeborn wire`. Each
hook is a single in-process `celeborn hook <event>` command — no bash wrapper, no inline `python3`, no
`$CELEBORN_HOME`; just have `celeborn` on PATH. The six hooks: `SessionStart` pre-hydrates the Hot
tier, `Stop` auto-captures each turn, `UserPromptSubmit` surfaces the context reminder and the capture
heartbeat, `PreCompact` forces a checkpoint, `SessionEnd` records the close, and `Notification` raises
a blocked-progress alert on the DOING card when a session needs the user (a permission prompt or an
idle stall — see `celeborn alert`). The authored-tier hooks **no-op in repos without a `.context/`**, so they're safe to enable globally; the
`Stop`/capture hook instead falls back to the global `~/.context/` (above), so every session is
recorded. On tools without hooks (Codex, Claude.ai), the skill still works — the agent runs the Orient
read manually.

**The Orient load is size-bounded.** `SessionStart` injects `celeborn status` into the session as
context, and hosts cap how much hook output they'll inline — past that they spill it to a file and hand
the model only a short preview, which would quietly defeat the auto-rehydration. Two things keep the
load small so you never police it by hand:

- **A headline / detail split.** `state.md` is the small Hot **headline** (focus, next action, where you
  are, pointers) — it's what loads every Orient. Its unbounded companion **`notes.md`** holds the
  working detail: open threads, constraints, extended context. `notes.md` is *not* auto-loaded (so it has
  no size limit — write freely); the agent reads it on demand, and `celeborn search` indexes it. If a
  thought doesn't fit the headline, it goes in `notes.md` — you don't trim `state.md`.
- **A budget backstop.** Should the headline (or the `activity.md` digest, or a `session.json` focus
  string) still overflow, `status` clips it to a character budget (`hot_state_max_chars`,
  `hot_activity_max_chars`, `hot_focus_max_chars`, tunable in `.celebornrc`), leaving a `… [Hot tier
  clipped — N more chars in <file>]` pointer. `celeborn status --full` prints everything unclipped for
  humans, and `celeborn doctor` warns when a Hot file is over budget.

> **Where the per-turn heartbeat shows — it depends on your surface.** Hook output reaches the *model's*
> context everywhere, but whether it's *painted on your screen* is host-dependent, so Celeborn surfaces
> the heartbeat through every channel it can:
> - **`statusLine`** (`🏹 Celeborn —> M tok banked · ctx ~N`) — painted persistently in the UI chrome and
>   **can't be suppressed**. The deterministic, cross-surface channel; the recommended one. Configure it
>   from the snippet's `statusLine` block.
> - **`Stop` `systemMessage`** (`🏹 Celeborn —> +N tok this turn · M this session`) — shown inline in a
>   **terminal**, but the **Claude desktop/web app does not paint it** (hidden transcript attachment; cc
>   issue [#50542](https://github.com/anthropics/claude-code/issues/50542)). Kept unique each turn so
>   Claude Code can't drop it as a duplicate of the previous one.
> - **`UserPromptSubmit` line** (`🏹 Celeborn —> M tok banked this session · +N last turn`) — reliably
>   visible in a terminal; reaches the model but the app may not paint it either.
> - **CLAUDE.md echo** — the managed `init` block asks the agent to reprint the heartbeat at the top of
>   its reply, so it shows even on a surface that paints nothing else (the agent's reply is always
>   rendered). Belt-and-suspenders for the Claude app.
>
> (No need to isolate the `Stop`/`celeborn hook stop` capture in its own `Stop` group — hooks across scopes are additive.)

### Context reminders

As a session's window fills, Celeborn surfaces a gentle nudge to `/clear` — calm, because nothing is
lost: the memory lives on disk and the bounded Hot tier **reloads automatically** next session (kept
small enough to inline — see [*Auto-load on Claude Code*](#auto-load-on-claude-code)). On Claude Code
the `celeborn hook user-prompt-submit` hook (`UserPromptSubmit`) reads the live size from the
transcript each turn and speaks at most once per band; on other agents the skill invites the same
`celeborn remind` call by hand.

Three knobs, resolved wherever `remind` is invoked:

- **`--soft-limit <n>` / `--hard-limit <n>`** — the context-pressure thresholds. Newly crossing one
  turns the nudge into an explicit warning: ⚠ at soft (*wrap the current step, checkpoint, then
  clear*), ⛔ at hard (*stop and checkpoint NOW*). Defaults come from `.celebornrc` —
  `context_soft_tokens` (100k) and `context_hard_tokens` (125k), the same lines the board's
  "clear now"/"clear urgent" bands draw — so configure them once per project; the flags override
  per call, and ≤ 0 disables a threshold. The DOING cards on the board carry matching ⚠/⛔ chips.
- **`--every <n>`** — the band (default 100k; the hook uses 50k). The nudge fires once per band
  crossed, then stays silent until the next — so it never nags.

The `UserPromptSubmit` hook reads the thresholds from `.celebornrc` automatically — on Claude Code
from the live transcript, on OpenCode from the window the session reported via `celeborn record
tokens` (the warning rides the per-turn envelope into the TUI).

**Seamless clear-and-continue (OpenCode, opt-in).** On OpenCode, Celeborn can take the last step for
you: when a session crosses the hard threshold it clears *itself* and resumes the same card, no human
in the loop. Turn it on with `"opencode_autoclear": true` in `.celebornrc`. Then, at the next turn
boundary (never mid-turn), the plugin runs `celeborn autoclear`, which verify-gates that a clear
would lose nothing (the same freshness check as `celeborn checkpoint --for-clear`) and — if the Hot
tier is fresh — regenerates the handoff, takes a restorable snapshot, and queues a resume brief to the
session's outbox. The plugin then compacts the session (losslessly: the compaction summary *is* your
Celeborn memory, verbatim) and re-prompts it to continue, draining that brief on the first turn back.
If the Hot tier is stale, `autoclear` instead hands the coder the exact fix-list so it freshens and
retries — so a clear never strands you in a stub. `autoclear_cooldown_minutes` (default 10) keeps it
from thrashing. Claude Code keeps the calm human-run `/clear` nudge above.

### Troubleshooting

**"GitHub CLI authentication expired. Run gh auth login to refresh pull request status."**
This banner is **Claude Code's**, not Celeborn's. Claude Code's PR-status panel periodically shells out
to the `gh` CLI; if `gh` is installed but you've never logged in, it shows that message. Celeborn
doesn't call `gh` and can't trigger or suppress it. The fix is to authenticate once — `gh auth login` —
or simply ignore it: your `git` push/pull keep working via your system keychain, which is independent of
`gh`. (`celeborn doctor` prints the same heads-up when it detects this state.)

## Elves, included

Celeborn integrates **[Elves](https://github.com/aigorahub/elves)** (by John Ennis, MIT) — the
autonomous multi-batch "night shift" development skill whose context-economy technique Celeborn grew
from. The `/elves` skill ships with the product. The division of labour:

- **Elves drives the work** — the loop, batch planning, multi-agent execution, testing, PR review.
- **Celeborn holds the memory** — the tiered `.context/` store, bounded Hot tier, recall, rehydration.

In this edition, Elves' working surfaces map straight onto Celeborn's tiers (survival guide →
`state.md`, execution log → `journal.md`, learnings → `learnings.md`, `.ai-docs/*` → `durable/*`), and
its **constant git pushes for state are replaced by cheap local `celeborn` checkpoints** — git/PRs are
kept for code review only.

With deep gratitude to the Father of Elves — go star
[the original](https://github.com/aigorahub/elves).

## Grok Build

Celeborn also runs under **[Grok Build](https://x.ai)**. Grok has its own hook format and doesn't feed
`SessionStart` output to the model, so a thin adapter under [`grok/`](grok/) bridges it to the stock
`celeborn` CLI — it converts Grok transcripts for `celeborn capture`, reads token usage for the
reminders, and writes the Orient load to `.context/.grok-orient-pending.md` (read once per session, then
deleted). **Celeborn core is untouched** — `grok/` is a host overlay, the same way Elves is a
workflow overlay. One command:

```bash
bash grok/scripts/install.sh --project /path/to/your-project
```

Global hooks then load on every new Grok session — no manual reload (mid-session installs just need
`/clear` once). See [`grok/SKILL.md`](grok/SKILL.md).

## Standing on the shoulders of giants

Celeborn is built almost entirely out of other people's generosity, and it would be wrong to close
this README any other way than by saying thank you.

- **To the open-source ecosystem as a whole.** The interpreters, the compilers, the databases, the
  ten thousand libraries quietly maintained by people who will never meet the strangers they help.
  SQLite — the engine at Celeborn's core — sits in the public domain and runs in more places than
  almost any software ever written, asking nothing in return. Python's standard library gave us
  everything else; there are *zero* third-party dependencies here because so much was already given.
  That spirit of freely-given work is the water all of us swim in.

- **To [elves](https://github.com/aigorahub/elves)** — the project that lit the spark. Celeborn
  generalizes the context-economy technique elves pioneered for long unattended runs: chats are for
  execution, handoff docs are for memory, archives are for history, fresh threads are for speed.
  Without elves, there is no Celeborn. Go star it.

- **To Marc Andreessen**, whose one-word reply — *"Interesting…"* — about the elves project was the
  small, generous nudge that told a builder the idea was worth chasing. Encouragement costs the giver
  almost nothing and means almost everything to the receiver. Thank you for spending it.

- **To every engineer shipping open source** for the rest of us to build on — credited in a changelog
  or thanked by no one at all. This project is *for* you and *because of* you. Celeborn itself is a
  commercial product rather than open source, so it isn't given back in kind — but the debt is real,
  and we try to honor it the ways that are ours to give: crediting your work plainly, building on it
  with care, and never pretending the foundation was our own.

> **Celeborn** — a Sindarin elf, Lord of Lothlórien (*celeb* "silver" + *orn* "tree"). A tended
> tree of memory that stays rooted across sessions and keeps growing. The natural successor to the
> [elves](https://github.com/aigorahub/elves) skill, whose context-economy technique this
> generalizes.

---

## FAQ

**What's the one command to get started?**
`celeborn init`, run in your project. It wires your coding agent, scaffolds this project, signs you in
(optional), and opens your kanban board — all in one pass, and it's safe to re-run. That's it. (If your
agent is already wired and you only want a new project's files, `celeborn scaffold` does just that step.)

**Is my code or memory ever uploaded or made public?**
No. Everything runs locally by default. Your `.context/` (prompts, notes, working memory) is **always
gitignored — never committed, and there is no option to commit it.** Cross-device sync is opt-in, goes
only to *your* private account, and redacts secrets before upload.

**Why can't I commit `.context/` to git, even if I want to?**
Because it's the single easiest way to leak your working memory forever. `.context/` contains your
prompts, notes, and decisions. If it's committed to a repo that is public — or private today but made
public later, or forked, or its history published — all of that becomes permanently world-visible in git
history, and history is very hard to scrub. Rather than leave that footgun armed, Celeborn keeps
`.context/` private by design and moves it between your machines a safer way (below). There is no
`--public` flag.

**Then how does my memory follow me to another machine?**
Through your account, not git. Run `celeborn init` (or `celeborn login` later — email + password),
then `celeborn sync` — hosted, cross-device, real-time sync is part of your Pro plan. See
[Sync across devices](#sync-across-devices-optional).

**Where do I actually *use* Celeborn once it's installed?**
Your **kanban board** (`celeborn board` opens it) is Celeborn's UI — tasks, run/fleet, and settings live
there, and your coding agent orients from `.context/` automatically at the start of every session. You
mostly just talk to your agent; Celeborn remembers.

---

## Support

The Celeborn client ships as a **compiled binary** (all rights reserved — the license travels inside
each release tarball). This repository — the thin installer that fetches and sha256-verifies that
binary — is **source-available under the [Business Source License 1.1](https://github.com/cloud-dancer-labs/celeborn-code/blob/main/LICENSE)**
(© Cloud Dancer; distributed by Thot Technologies LLC), converting to **Apache-2.0** on its Change
Date (four years after each version's release). Releases **up to 0.2.1** shipped the full client as
BUSL source; they remain so forever — on PyPI and in this repo's git history — and convert to
Apache-2.0 on their own Change Dates. Production use of the unmodified, checksum-valid client is
granted at no charge; the **only** use restriction is that you may not operate a competing hosted
service built on it. See the [LICENSE](https://github.com/cloud-dancer-labs/celeborn-code/blob/main/LICENSE) for the full terms.

It earns its keep in tokens and time. If you want to know how much, run `celeborn metrics`.

**Plans**

- **Pro — $8/seat/mo** — the full product: the local engine, hosted sync (cross-device, real-time,
  zero-setup, unlimited projects), and the hosted board. **Every new account starts with a 7-day
  free trial** — card upfront, $0 today, cancel anytime in one click; the day-8 charge has a
  published no-questions 7-day refund window. Annual ($80/yr) is available from the billing portal
  after the trial converts.
- **Team — $12/seat/mo** — Pro plus shared projects, org admin, shared context, and shared agent
  telepathy (the multi-agent bus). A straight upgrade from inside Pro — no separate trial.
- **Enterprise** — SSO/SAML, custom terms, volume pricing. **Ask in the support chat at [celeborncode.ai/faq](https://celeborncode.ai/faq).**

There is no free tier — we charge because we intend to still exist next year. Whatever you decide,
your `.context/` memory stays yours: plain files on your disk, readable by anything. Start the
trial with `celeborn register` (email + password + MFA — you can link GitHub afterwards if you
like). ⭐ Starring the repo and telling another developer is free, and it genuinely helps.

## Status

Early and moving fast. The installer in this repository is source-available under the
[Business Source License 1.1](https://github.com/cloud-dancer-labs/celeborn-code/blob/main/LICENSE) —
© Cloud Dancer; distributed by Thot Technologies LLC (converts to Apache-2.0 four years after release).
The client binary it installs is © Cloud Dancer, all rights reserved; distributed by Thot Technologies LLC.
