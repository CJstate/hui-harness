"""Command line interface: ``hui run``, ``hui demo``, ``hui replay``, and a REPL."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hui import __version__
from hui.common import Message, ProviderError, TextDelta, ToolCall, Usage
from hui.loop import DEFAULT_MAX_STEPS, DEFAULT_MAX_TOKENS, Agent, AgentResult
from hui.policy import DANGER, MODES, WORKSPACE_WRITE, policy_from_env
from hui.providers import (
    PRESETS,
    ScriptedProvider,
    build_provider,
    config_from_env,
)
from hui.replay import list_script, replay_session
from hui.session import Session
from hui.tools import ToolContext, default_registry

DEFAULT_SESSION_DIR = Path(os.environ.get("HUI_SESSION_DIR") or (Path.home() / ".hui" / "sessions"))

BANNER = f"""\
HUI harness {__version__} — a small, dependency-free coding agent.
/tools  list tools      /mode   show or set the permission mode
/cost   token usage     /session  where this conversation is recorded
/compact  drop old steps from the working transcript
/exit   leave (Ctrl-D also works)
"""


def session_path(directory: Path, name: str | None) -> Path:
    if name:
        return Path(name)
    import time

    return Path(directory) / f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-C", "--workspace", default=None, help="workspace root (default: current directory)"
    )
    common.add_argument(
        "--session-dir", default=str(DEFAULT_SESSION_DIR), help="where session logs are written"
    )
    common.add_argument("--quiet", action="store_true", help="do not stream assistant text")
    with_mode = argparse.ArgumentParser(add_help=False, parents=[common])
    with_mode.add_argument(
        "--mode", choices=MODES, default=None, help="permission mode (default: workspace-write)"
    )

    parser = argparse.ArgumentParser(
        prog="hui",
        description="HUI harness — a replayable coding agent.",
        parents=[with_mode],
    )
    parser.add_argument("--version", action="version", version=f"hui-harness {__version__}")

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run one prompt", parents=[with_mode])
    run.add_argument("prompt", nargs="?", help="the task to perform")
    run.add_argument("-p", "--print", dest="prompt_opt", help="the task to perform")
    run.add_argument("--provider", default=None, help=f"one of {', '.join(sorted(PRESETS))}")
    run.add_argument("--model", default=None)
    run.add_argument("--base-url", default=None)
    run.add_argument("--api-key", default=None)
    run.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--json", action="store_true", help="print a machine-readable summary")

    demo = sub.add_parser("demo", help="offline demo (no API key needed)", parents=[common])
    demo.add_argument("--dir", default=None, help="demo workspace (default: ./.hui-demo)")

    replay = sub.add_parser(
        "replay", help="re-execute a recorded session and verify it", parents=[common]
    )
    replay.add_argument("session", help="path to a .jsonl session file")
    replay.add_argument(
        "--list", action="store_true", help="list the recorded turns without executing"
    )
    replay.add_argument("--json", action="store_true")
    replay.add_argument(
        "--mode", choices=MODES, default=None, help="mode used while replaying (default: danger)"
    )

    sessions = sub.add_parser("sessions", help="list recorded sessions", parents=[common])
    sessions.add_argument("-n", type=int, default=20, help="how many to show")

    return parser


def _make_agent(args: argparse.Namespace, *, session: Session | None, echo: bool) -> Agent:
    workspace = Path(args.workspace or Path.cwd()).resolve()
    mode = args.mode or None
    policy = policy_from_env(workspace, mode)
    provider = build_provider(config_from_env(_provider_overrides(args)))
    on_event = None
    if echo:

        def on_event(event: Any) -> None:  # noqa: ANN401 - stream events are a closed union
            if isinstance(event, TextDelta):
                sys.stdout.write(event.text)
                sys.stdout.flush()

    return Agent(
        provider,
        registry=default_registry(),
        context=ToolContext(workspace=workspace, policy=policy),
        session=session,
        max_steps=getattr(args, "max_steps", DEFAULT_MAX_STEPS),
        max_tokens=getattr(args, "max_tokens", DEFAULT_MAX_TOKENS),
        on_event=on_event,
    )


def _provider_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in ("provider", "model", "base_url", "api_key"):
        value = getattr(args, key, None)
        if value:
            overrides[key] = value
    return overrides


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    prompt = args.prompt_opt or args.prompt
    if not prompt:
        print("hui run: a prompt is required (positional or -p)", file=sys.stderr)
        return 2
    path = session_path(Path(args.session_dir), None)
    session = Session.create(path, {"workspace": str(Path(args.workspace or Path.cwd()).resolve())})
    agent = _make_agent(args, session=session, echo=not args.quiet)
    try:
        result = agent.run(prompt)
    except ProviderError as exc:
        print(f"\nprovider error: {exc}", file=sys.stderr)
        return 3
    if not args.quiet:
        sys.stdout.write("\n" if result.final_text() else "")
    if args.json:
        import json

        print(json.dumps(result.summary(), indent=2, sort_keys=True))
    else:
        print(f"\n[{_summary_line(result)}]\nsession: {path}")
    return 0 if result.stop_reason != "error" else 3


def cmd_demo(args: argparse.Namespace) -> int:
    workspace = Path(args.dir or Path.cwd() / ".hui-demo").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    python = sys.executable or "python"
    shell_python = f'& "{python}" hello_hui.py' if os.name == "nt" else f'"{python}" hello_hui.py'
    script = [
        [
            ready(
                "todo_write",
                {"todos": [{"content": "write hello_hui.py", "status": "in_progress"}]},
            )
        ],
        [ready("write_file", {"path": "hello_hui.py", "content": 'print("hello from HUI")\n'})],
        [ready("run_command", {"command": shell_python})],
        [
            ready(
                "todo_write", {"todos": [{"content": "write hello_hui.py", "status": "completed"}]}
            )
        ],
        [TextDelta("Demo done. Those tool outputs are real: HUI wrote and ran hello_hui.py here.")],
    ]
    provider = ScriptedProvider(script)
    session = Session.create(
        session_path(Path(args.session_dir), None), {"workspace": str(workspace)}
    )
    agent = Agent(
        provider,
        registry=default_registry(),
        context=ToolContext(
            workspace=workspace, policy=policy_from_env(workspace, WORKSPACE_WRITE)
        ),
        session=session,
        on_event=None,
    )
    result = agent.run("Run the offline HUI demo.")
    print(f"workspace: {workspace}")
    for step in result.steps:
        for outcome in step.outcomes:
            status = "ok" if outcome.ok else "FAIL"
            print(f"  [{status}] {outcome.call.name} ({outcome.duration_ms} ms)")
            if outcome.call.name == "run_command":
                for line in outcome.content.splitlines():
                    print(f"      {line}")
    print(f"\n{result.final_text()}\nsession: {session.path}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    if args.list:
        print(list_script(args.session))
        return 0
    report = replay_session(args.session, workspace=args.workspace, mode=args.mode or DANGER)
    if args.json:
        import json

        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.render())
    return 0 if report.ok else 1


def cmd_sessions(args: argparse.Namespace) -> int:
    directory = Path(args.session_dir)
    if not directory.exists():
        print(f"no sessions yet in {directory}")
        return 0
    files = sorted(directory.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in files[: args.n]:
        try:
            summary = Session.load(path).summary()
        except Exception as exc:  # a truncated log should not break the listing
            print(f"{path.name}  (unreadable: {exc})")
            continue
        print(
            f"{summary['started']}  {path.name}  msgs={summary['messages']} "
            f"tools={summary['tool_calls']} stop={summary['stop_reason']}"
        )
    return 0


# --------------------------------------------------------------------------- #
# REPL
# --------------------------------------------------------------------------- #
def repl(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace or Path.cwd()).resolve()
    path = session_path(Path(args.session_dir), None)
    session = Session.create(path, {"workspace": str(workspace)})
    agent = _make_agent(args, session=session, echo=not args.quiet)
    history: list[Message] = []
    total = Usage()
    print(BANNER)
    print(f"workspace: {workspace}\nmode: {agent.context.policy.mode}\nsession: {path}\n")
    while True:
        try:
            line = input("hui> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            command, _, rest = line.partition(" ")
            if command in {"/exit", "/quit", "/q"}:
                break
            if command == "/help":
                print(BANNER)
                continue
            if command == "/tools":
                for tool in agent.registry:
                    print(f"  {tool.name:<12} {tool.description}")
                continue
            if command == "/mode":
                if rest.strip():
                    mode = rest.strip()
                    try:
                        agent.context.policy = policy_from_env(workspace, mode)
                    except Exception as exc:
                        print(f"  {exc}")
                        continue
                print(f"  mode: {agent.context.policy.mode}")
                continue
            if command == "/cost":
                billed = f"prompt={total.prompt_tokens} completion={total.completion_tokens}"
                print(f"  tokens: {billed} total={total.total_tokens}")
                continue
            if command == "/session":
                print(f"  {path}")
                continue
            if command == "/compact":
                history = history[-2:]
                print(f"  transcript compacted to {len(history)} message(s)")
                continue
            print(f"  unknown command {command}; /help lists them")
            continue
        try:
            result = agent.run(line, messages=history)
        except ProviderError as exc:
            print(f"provider error: {exc}")
            continue
        except KeyboardInterrupt:
            print("\ninterrupted")
            continue
        history = result.messages
        total = total + result.usage
        if not args.quiet and result.final_text():
            sys.stdout.write("\n")
        print(f"[{_summary_line(result)}]")
    print(f"session: {path}")
    return 0


def _summary_line(result: AgentResult) -> str:
    summary = result.summary()
    return (
        f"steps={summary['steps']} tools={summary['tool_calls']} "
        f"failures={summary['tool_failures']} stop={summary['stop_reason']} "
        f"tokens={summary['total_tokens']}"
    )


def ready(name: str, arguments: dict[str, Any]):
    """Build a scripted ``ToolCallReady`` turn for the offline demo."""
    from hui.common import ToolCallReady

    return ToolCallReady(ToolCall.from_arguments(name, arguments))


def harden_console() -> None:
    """Never crash while printing a result on a cp936/GBK Windows console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError):  # redirected or detached
            reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    harden_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "demo":
            return cmd_demo(args)
        if args.command == "replay":
            return cmd_replay(args)
        if args.command == "sessions":
            return cmd_sessions(args)
        return repl(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
