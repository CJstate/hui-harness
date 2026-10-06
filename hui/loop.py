"""The agent loop.

One step is: stream a model reply, run every tool it asked for, append the
results, repeat. Two invariants hold at all times:

1. **A tool call is never separated from its result.** When the history has to
   be compacted to fit the token budget, whole ``assistant(tool_calls) + tool
   result(s)`` groups are dropped together — never half of a pair, which is what
   makes an API reject the transcript.
2. **Every step is recorded before it is used.** The session log is written
   before the loop moves on, so a crash leaves a replayable prefix.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hui.common import (
    Message,
    PolicyError,
    ReasoningDelta,
    StopEvent,
    TextDelta,
    ToolCall,
    ToolCallReady,
    ToolError,
    Usage,
    UsageEvent,
    estimate_messages_tokens,
    sha256_text,
    truncate,
)
from hui.policy import policy_from_env
from hui.session import Session
from hui.tools import ToolContext, ToolRegistry, default_registry

DEFAULT_MAX_STEPS = 24
DEFAULT_MAX_TOKENS = 120_000
DEFAULT_MAX_OUTPUT = 20_000
DEFAULT_SYSTEM = """\
You are HUI, a coding agent working inside {workspace}.

How you work:
- Look before you touch: read a file, or search with grep_files / glob_files, before editing it.
- Prefer a small exact `edit_file` over rewriting a file with `write_file`.
- Verify with the project's own tests or commands instead of assuming.
- For multi-step work keep the plan in `todo_write` and update it as you go.
- Report what you observed. If a command failed, read the error and adapt; never invent output.
"""


def _groups(messages: Sequence[Message]) -> list[list[Message]]:
    """Split a transcript into atomic groups (an assistant call + its results)."""
    groups: list[list[Message]] = []
    current: list[Message] = []
    for message in messages:
        if (
            message.role == "tool"
            and current
            and current[0].role == "assistant"
            and current[0].tool_calls
        ):
            current.append(message)
            continue
        if current:
            groups.append(current)
        current = [message]
    if current:
        groups.append(current)
    return groups


@dataclass(slots=True)
class ToolOutcome:
    call: ToolCall
    content: str
    ok: bool
    duration_ms: int

    @property
    def sha256(self) -> str:
        return sha256_text(self.content)


@dataclass(slots=True)
class Step:
    index: int
    assistant: Message
    outcomes: list[ToolOutcome] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return self.assistant.content or ""

    @property
    def failed(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.ok)


@dataclass(slots=True)
class AgentResult:
    messages: list[Message]
    steps: list[Step]
    usage: Usage
    stop_reason: str
    session: Session | None = None

    @property
    def tool_calls(self) -> list[ToolOutcome]:
        return [outcome for step in self.steps for outcome in step.outcomes]

    def final_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "assistant" and message.content:
                return message.content
        return ""

    def summary(self) -> dict[str, Any]:
        outcomes = self.tool_calls
        return {
            "steps": len(self.steps),
            "stop_reason": self.stop_reason,
            "tool_calls": len(outcomes),
            "tool_failures": sum(1 for outcome in outcomes if not outcome.ok),
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "session": str(self.session.path) if self.session else None,
        }

    def render(self) -> str:
        summary = self.summary()
        header = (
            f"steps={summary['steps']} stop={summary['stop_reason']} "
            f"tools={summary['tool_calls']} failures={summary['tool_failures']} "
            f"tokens={summary['total_tokens']}"
        )
        return f"{header}\n{self.final_text()}"


class Agent:
    """Runs the streaming tool loop against one provider."""

    def __init__(
        self,
        provider: Any,
        *,
        registry: ToolRegistry | None = None,
        context: ToolContext | None = None,
        workspace: str | Path | None = None,
        mode: str | None = None,
        system: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        keep_recent: int = 6,
        max_output: int = DEFAULT_MAX_OUTPUT,
        session: Session | None = None,
        on_event: Callable[[Any], None] | None = None,
        on_tool_result: Callable[[ToolOutcome], None] | None = None,
    ):
        root = Path(workspace or (context.workspace if context else Path.cwd()))
        policy = context.policy if context else policy_from_env(root, mode)
        self.provider = provider
        self.registry = registry or default_registry()
        self.context = context or ToolContext(workspace=root, policy=policy)
        self.system = system or DEFAULT_SYSTEM.format(workspace=self.context.workspace)
        self.max_steps = max(1, int(max_steps))
        self.max_tokens = int(max_tokens)
        self.keep_recent = max(2, int(keep_recent))
        self.max_output = int(max_output)
        self.session = session
        self.on_event = on_event
        self.on_tool_result = on_tool_result

    # -- public API --------------------------------------------------------
    def run(
        self, prompt: str | None = None, *, messages: Sequence[Message] | None = None
    ) -> AgentResult:
        history: list[Message] = list(messages or [])
        if prompt is not None:
            user = Message.user(prompt)
            history.append(user)
            self._record(user)

        steps: list[Step] = []
        total = Usage()
        stop_reason = "max_steps"
        try:
            for index in range(self.max_steps):
                history = self._enforce_budget(history)
                assistant, calls, usage, stop = self._stream_step(history)
                history.append(assistant)
                self._record(assistant)
                total = total + usage
                if not calls:
                    steps.append(Step(index=index, assistant=assistant, usage=usage))
                    stop_reason = stop or "end_turn"
                    break
                outcomes: list[ToolOutcome] = []
                for call in calls:
                    outcome = self._execute(call)
                    outcomes.append(outcome)
                    history.append(Message.tool_result(call.id, outcome.content, call.name))
                    if self.session:
                        self.session.record_tool_result(
                            call, outcome.content, ok=outcome.ok, duration_ms=outcome.duration_ms
                        )
                    if self.on_tool_result:
                        self.on_tool_result(outcome)
                steps.append(Step(index=index, assistant=assistant, outcomes=outcomes, usage=usage))
        except BaseException as exc:
            if self.session:
                self.session.note(f"error: {type(exc).__name__}: {exc}")
                self.session.stop("error")
            raise

        if self.session:
            self.session.record_usage(total)
            self.session.stop(stop_reason)
        return AgentResult(
            messages=history,
            steps=steps,
            usage=total,
            stop_reason=stop_reason,
            session=self.session,
        )

    # -- internals ---------------------------------------------------------
    def _record(self, message: Message) -> None:
        if self.session:
            self.session.record(message)

    def _tool_specs(self) -> list[dict[str, Any]]:
        style = getattr(self.provider, "tool_spec_style", None)
        if style is None:
            style = "anthropic" if "anthropic" in type(self.provider).__name__.lower() else "openai"
        return self.registry.specs(style)

    def _stream_step(
        self, history: Sequence[Message]
    ) -> tuple[Message, list[ToolCall], Usage, str]:
        text: list[str] = []
        reasoning: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage()
        stop = "end_turn"
        for event in self.provider.stream(list(history), self._tool_specs(), self.system):
            if self.on_event:
                self.on_event(event)
            if isinstance(event, TextDelta):
                text.append(event.text)
            elif isinstance(event, ReasoningDelta):
                reasoning.append(event.text)
            elif isinstance(event, ToolCallReady):
                calls.append(event.call)
            elif isinstance(event, UsageEvent):
                usage = usage + event.usage
            elif isinstance(event, StopEvent):
                stop = event.reason or "end_turn"
        assistant = Message.assistant(
            content="".join(text) or None,
            tool_calls=calls,
            reasoning="".join(reasoning) or None,
        )
        return assistant, calls, usage, stop

    def _execute(self, call: ToolCall) -> ToolOutcome:
        started = time.perf_counter()
        try:
            content = self.registry.execute(call, self.context)
            ok = True
        except ToolError as exc:
            content, ok = f"tool error: {exc}", False
        except PolicyError as exc:
            content, ok = f"denied by policy: {exc}", False
        elapsed = int((time.perf_counter() - started) * 1000)
        return ToolOutcome(
            call=call, content=truncate(content, self.max_output), ok=ok, duration_ms=elapsed
        )

    def _enforce_budget(self, history: list[Message]) -> list[Message]:
        if self.max_tokens <= 0 or estimate_messages_tokens(history) <= self.max_tokens:
            return history
        groups = _groups(history)
        if len(groups) <= 2:
            return history
        keep_from = max(1, len(groups) - self.keep_recent)
        dropped = groups[1:keep_from]
        if not dropped:
            return history
        note = Message.system(
            f"[hui] compacted {len(dropped)} earlier step(s) to stay inside the token budget."
        )
        self._record(note)
        kept = [message for group in groups[keep_from:] for message in group]
        return [*groups[0], note, *kept]


def run_agent(provider: Any, prompt: str, **kwargs: Any) -> AgentResult:
    """One-shot convenience wrapper around :class:`Agent`."""
    return Agent(provider, **kwargs).run(prompt)
