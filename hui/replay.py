"""Deterministic, offline replay of a recorded session.

A session log records every assistant message and the SHA-256 of every tool
result. :func:`replay_session` feeds the recorded assistant turns back through
the loop — no network, no model — re-runs the tools, and checks that each result
hashes to what was recorded.

That is the honest guarantee: *the tool half of a past session can be re-executed
and verified byte-for-byte from the log alone.* If a file changed, a command
started printing different output, or the log was edited, replay says so.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hui.common import (
    Message,
    ReasoningDelta,
    StopEvent,
    TextDelta,
    ToolCallReady,
    dumps,
)
from hui.loop import Agent, ToolOutcome
from hui.policy import DANGER
from hui.session import Session
from hui.tools import ToolRegistry, default_registry


@dataclass(slots=True)
class ReplayStep:
    index: int
    name: str
    expected: str | None
    actual: str | None
    ok: bool
    duration_ms: int = 0

    def render(self) -> str:
        mark = "ok  " if self.ok else "DIFF"
        expected = (self.expected or "?")[:12]
        actual = (self.actual or "-")[:12]
        return f"{mark} {self.index:>3} {self.name:<12} recorded={expected} replayed={actual}"


@dataclass(slots=True)
class ReplayReport:
    session_id: str
    turns: int
    steps: list[ReplayStep] = field(default_factory=list)
    stop_reason: str = ""
    final_text: str = ""
    executed: bool = True

    @property
    def mismatches(self) -> list[ReplayStep]:
        return [step for step in self.steps if not step.ok]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def render(self) -> str:
        lines = [
            f"session {self.session_id}: {self.turns} recorded assistant turn(s), "
            f"{len(self.steps)} tool call(s)"
        ]
        lines.extend(step.render() for step in self.steps)
        if not self.executed:
            lines.append("(listing only: tools were not re-executed)")
            return "\n".join(lines)
        if self.ok:
            lines.append(
                f"REPLAY OK — every tool result reproduced its recorded hash "
                f"(stop={self.stop_reason or 'unknown'})"
            )
        else:
            verdict = f"REPLAY MISMATCH — {len(self.mismatches)} of {len(self.steps)}"
            lines.append(f"{verdict} tool result(s) differ")
        if self.final_text:
            lines.append("--- final message ---")
            lines.append(self.final_text)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": self.turns,
            "executed": self.executed,
            "ok": self.ok,
            "stop_reason": self.stop_reason,
            "steps": [
                {
                    "index": step.index,
                    "name": step.name,
                    "recorded": step.expected,
                    "replayed": step.actual,
                    "ok": step.ok,
                }
                for step in self.steps
            ],
        }


class RecordedProvider:
    """Serves the assistant turns read back from a session file."""

    tool_spec_style = "openai"

    def __init__(self, turns: Sequence[dict[str, Any]]):
        self.turns = [dict(turn) for turn in turns]
        self.requests: list[list[Message]] = []

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        system: str | None = None,
    ) -> Iterator[Any]:
        self.requests.append(list(messages))
        if not self.turns:
            yield TextDelta("[replay: recording exhausted]")
            yield StopEvent("end_turn")
            return
        turn = self.turns.pop(0)
        message = Message.from_dict(turn["assistant"])
        if message.reasoning:
            yield ReasoningDelta(message.reasoning)
        if message.content:
            yield TextDelta(message.content)
        for call in message.tool_calls:
            yield ToolCallReady(call)
        yield StopEvent("tool_use" if message.tool_calls else "end_turn")


def load_session(source: str | Path | Session) -> Session:
    return source if isinstance(source, Session) else Session.load(source)


def list_script(source: str | Path | Session) -> str:
    """Describe a session's assistant turns without executing anything."""
    session = load_session(source)
    lines = [f"session {session.id} · {session.path}", dumps(session.summary())]
    for index, turn in enumerate(session.script()):
        message = Message.from_dict(turn["assistant"])
        preview = (message.content or "").strip().replace("\n", " ")[:100]
        lines.append(f"turn {index}: {preview or '(no text)'}")
        for call in turn["calls"]:
            expected = (call.get("expected") or "unrecorded")[:12]
            name = call["call"]["name"]
            args = dumps(call["call"]["arguments"])
            lines.append(f"  -> {name}({args}) recorded={expected}")
    return "\n".join(lines)


def replay_session(
    source: str | Path | Session,
    *,
    workspace: str | Path | None = None,
    registry: ToolRegistry | None = None,
    mode: str = DANGER,
    execute: bool = True,
) -> ReplayReport:
    """Re-run a recorded session's tools and verify the recorded hashes."""
    session = load_session(source)
    turns = session.script()
    expected: list[dict[str, Any]] = [call for turn in turns for call in turn["calls"]]

    report = ReplayReport(
        session_id=session.id,
        turns=len(turns),
        stop_reason=session.stop_reason() or "",
        executed=execute,
    )
    if not execute:
        report.steps = [
            ReplayStep(
                index=index,
                name=str(item["call"]["name"]),
                expected=item.get("expected"),
                actual=None,
                ok=True,
            )
            for index, item in enumerate(expected)
        ]
        report.final_text = _last_text(session)
        return report

    provider = RecordedProvider(turns)
    cursor = 0
    mismatches: list[str] = []

    def compare(outcome: ToolOutcome) -> None:
        nonlocal cursor
        item = expected[cursor] if cursor < len(expected) else None
        wanted = item.get("expected") if item else None
        matched = bool(wanted) and wanted == outcome.sha256
        report.steps.append(
            ReplayStep(
                index=cursor,
                name=outcome.call.name,
                expected=wanted,
                actual=outcome.sha256,
                ok=matched,
                duration_ms=outcome.duration_ms,
            )
        )
        if not matched:
            mismatches.append(outcome.call.name)
        cursor += 1

    agent = Agent(
        provider,
        registry=registry or default_registry(),
        workspace=workspace or _workspace_hint(session),
        mode=mode,
        max_steps=max(len(turns), 1),
        on_tool_result=compare,
    )
    seed = _first_user(session)
    result = agent.run(messages=[seed] if seed else None)
    report.stop_reason = result.stop_reason
    report.final_text = result.final_text()
    # A turn whose tools never ran still needs a visible, failing entry.
    for index in range(cursor, len(expected)):
        report.steps.append(
            ReplayStep(
                index=index,
                name=str(expected[index]["call"]["name"]),
                expected=expected[index].get("expected"),
                actual=None,
                ok=False,
            )
        )
    return report


def _first_user(session: Session) -> Message | None:
    for message in session.messages():
        if message.role == "user":
            return message
    return None


def _last_text(session: Session) -> str:
    for message in reversed(session.messages()):
        if message.role == "assistant" and message.content:
            return message.content
    return ""


def _workspace_hint(session: Session) -> Path | None:
    value = session.meta.get("workspace")
    return Path(str(value)) if value else None
