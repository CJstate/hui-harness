# HUI harness

**A small, dependency-free, replayable coding agent.** HUI runs a tool loop against any
OpenAI-compatible or Anthropic endpoint, writes an append-only session log, and can
re-execute that log later to prove the run is reproducible.

```console
$ hui demo                       # no API key needed
$ hui run "fix the failing test" --provider deepseek
$ hui replay .sessions/session-20261006-114248.jsonl
```

[![CI](https://github.com/CJstate/hui-harness/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CJstate/hui-harness/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![Runtime dependencies](https://img.shields.io/badge/runtime%20dependencies-0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

- **~3,000 lines of stdlib Python** across 10 modules; `pip install` pulls in nothing.
- **138 tests**, ruff-clean, and a CI matrix over Linux / Windows / macOS × Python 3.10–3.13.
- **CI proves the replay claim** on Linux and Windows instead of asking you to trust it.

---

## Why another coding agent?

Most agent harnesses are excellent but large. HUI exists to make five invariants small enough
to read in one sitting:

| Invariant | What it means in practice |
| --- | --- |
| **Zero runtime dependencies** | Pure standard library. No `requests`, no `httpx`, no `pydantic`. `urllib`, `json`, `subprocess`, `difflib`. Auditing the whole thing is a single afternoon. |
| **Replayable sessions** | Every run appends a hash-chained JSONL event log. `hui replay` re-executes the recorded tool calls and compares the **sha256 of every tool result** against the recording. A non-deterministic run fails loudly. |
| **Encoding and newline safety** | Files are read as UTF-8, the platform encoding (GBK/CP936 on a Chinese Windows box), then Latin-1 — so a GBK source file or a UTF-8 BOM never crashes a run. Edits keep the file's original CRLF/LF style. |
| **Atomic context compaction** | When the token budget is exceeded, older steps are dropped in whole groups: an assistant `tool_call` is **never** separated from its tool result. Sending half a pair corrupts a conversation; HUI cannot. |
| **Policy before actuators** | `read-only`, `workspace-write`, `danger-full-access`. Writes are confined to the workspace, and ~20 destructive command patterns need an explicit approval hook. Commands never run through `shell=True`. |

## Quickstart

```console
# 1. See it work offline (fixed script, real tool execution)
$ python -m hui demo --dir ./demo
workspace: /home/you/demo
  [ok] todo_write (0 ms)
  [ok] write_file (5 ms)
  [ok] run_command (399 ms)
      exit_code=0
      --- stdout ---
      hello from HUI
  [ok] todo_write (0 ms)

Demo done. Those tool outputs are real: HUI wrote and ran hello_hui.py here.
session: .sessions/session-20261006-114248.jsonl

# 2. Replay that session and verify every hash
$ python -m hui replay .sessions/session-20261006-114248.jsonl -C ./demo
session sess_a95027c539eb83e5: 5 recorded assistant turn(s), 4 tool call(s)
ok     0 todo_write   recorded=5c56e32cc806 replayed=5c56e32cc806
ok     1 write_file   recorded=06e47d7cd0de replayed=06e47d7cd0de
ok     2 run_command  recorded=5d82fb714802 replayed=5d82fb714802
ok     3 todo_write   recorded=78bbc7c4924c replayed=78bbc7c4924c
REPLAY OK — every tool result reproduced its recorded hash (stop=end_turn)

# 3. Talk to a real model
$ export DEEPSEEK_API_KEY=sk-...
$ hui run "add a regression test for the GBK decode path" --provider deepseek
$ hui            # or start the REPL: /help /tools /mode /cost /compact /exit
```

## Architecture

| Module | Responsibility |
| --- | --- |
| `hui/common.py` | Messages, tool calls, stream events (`TextDelta`, `ToolCallReady`, `UsageEvent`, `StopEvent`), token estimation (CJK-aware), hashing. |
| `hui/providers.py` | `OpenAICompatProvider`, `AnthropicProvider`, `ScriptedProvider`. SSE parsing, retry with backoff on 429/5xx, tool-delta accumulation. |
| `hui/tools.py` | `ToolRegistry` + 8 built-ins: `read_file`, `write_file`, `edit_file`, `glob_files`, `grep_files`, `list_dir`, `run_command`, `todo_write`. |
| `hui/policy.py` | Permission modes, workspace containment, the destructive-command gate, argv construction. |
| `hui/fileio.py` | Encoding-safe read/write, newline detection, numbered views, unified diffs. |
| `hui/session.py` | Append-only JSONL event log with a sha256 hash chain, resume, `verify()`, and the replay script. |
| `hui/replay.py` | `RecordedProvider`, hash-by-hash comparison, `REPLAY OK` / `REPLAY MISMATCH` reports. |
| `hui/loop.py` | The step loop: stream → execute tools → record → compact → repeat, bounded by `max_steps` and a token budget. |
| `hui/cli.py` | `run`, `demo`, `replay`, `sessions`, and the REPL. |

```text
  prompt ──▶ Agent.run ──▶ provider.stream ──▶ TextDelta / ToolCallReady / UsageEvent
                  ▲                                        │
                  │                             ┌──────────▼──────────┐
                  │                             │ ToolRegistry.execute │◀── Policy gate
                  │                             └──────────┬──────────┘
                  │                                        │
        compact whole groups only               Session.append(tool_result + sha256)
                  ▲                                        │
                  └────────────── transcript ◀─────────────┘
```

## The replay guarantee

A session log is JSONL, one event per line, each event carrying the hash of the previous one:

```json
{"seq":0,"ts":"2026-10-06T03:42:48Z","kind":"start","data":{"session_id":"sess_a950…","workspace":"/home/you/demo"},"prev_hash":"000…","hash":"3f1c…"}
{"seq":6,"ts":"2026-10-06T03:42:48Z","kind":"tool_result","data":{"id":"call_…","name":"run_command","arguments":{"command":"python hello_hui.py"},"content":"exit_code=0\n--- stdout ---\nhello from HUI","sha256":"5d82fb714802…","ok":true,"duration_ms":399},"prev_hash":"b17e…","hash":"c93d…"}
```

`hui replay` feeds the recorded assistant turns back through the real loop (so the real tools run
again) and compares each fresh `sha256` with the recorded one. Editing one byte of the log breaks
the chain (`Session.verify()` says exactly which event); editing the file a tool read shows up as a
`REPLAY MISMATCH`. This is what makes an agent run auditable after the fact.

## Providers

| Preset | Endpoint | Key |
| --- | --- | --- |
| `deepseek` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| `glm` | `https://open.bigmodel.cn/api/paas/v4` | `GLM_API_KEY` |
| `qwen` | DashScope compatible mode | `DASHSCOPE_API_KEY` |
| `ollama` | `http://localhost:11434/v1` | *(none — localhost skips the key check)* |

Anything else: `--base-url` + `--model` + `--api-key` (or `HUI_BASE_URL`, `HUI_MODEL`, `HUI_API_KEY`).

## Permissions

| Mode | Reads | Writes | Commands |
| --- | --- | --- | --- |
| `read-only` | anywhere | refused | refused |
| `workspace-write` *(default)* | anywhere | workspace + temp only | allowed, destructive patterns need approval |
| `danger-full-access` | anywhere | anywhere | allowed |

The gate covers `rm -rf /…`, `rd /s`, `del /f`, `mkfs`, `dd if=`, drive formats, `shutdown`,
`reg delete`, `sudo`/`runas`, `git push --force`, `git reset --hard`, `curl … | sh`,
`chmod -R 777`, package publishes, `gh repo delete`, and `drop table` — with a unit test naming
each pattern, plus tests that ordinary commands such as `rm -rf ./build` are *not* flagged.

## Windows, encodings and consoles

Cross-platform agents usually fail on exactly these five things, so each one has a test:

1. `read_text()` with no `encoding=` on a GBK machine → `UnicodeEncodeError`. `read_text_safe`
   tries UTF-8, then the locale encoding, then Latin-1 and reports which one it used.
2. A UTF-8 BOM breaking the first parse of a config file → handled by `utf-8-sig`, and preserved on
   write if it was there before.
3. An `edit_file` that rewrites every line ending in a CRLF repository → the file's dominant
   newline is detected and restored.
4. `subprocess` with `shell=True` and a command containing quotes → never used; commands go through
   an argv list (`powershell -NoProfile -NonInteractive -Command` on Windows, `/bin/sh -c` elsewhere).
5. A cp936 console that cannot print an em dash or CJK output → `main()` reconfigures stdout/stderr
   with `errors="replace"`, so a result is never lost to a `UnicodeEncodeError`.

## Non-goals

Honest scope, because an agent that claims everything does nothing well:

- **No MCP client, subagents, web fetch, or IDE integration** (yet). The core is the loop, the
  policy, and the log.
- **No Windows/macOS sandbox integration** — the policy is enforced in-process, so treat
  `danger-full-access` as "no protection".
- **No provider-specific prompt tuning.** HUI sends the same system prompt to every model.
- **Interface stability:** `0.1.0` — the modules are small and may move before `1.0`.

## Development

```console
$ python -m pip install -e . pytest ruff
$ python -m pytest -q          # 138 tests, offline (ScriptedProvider, no network)
$ ruff check . && ruff format --check .
$ python -m build              # sdist + wheel
```

The test suite never touches the network: `ScriptedProvider` replays scripted stream events, which
is also what the demo, the CLI tests, and the replay tests run against.

## License

MIT © 2026 CJstate. Written from scratch; no code copied from any other agent project.
