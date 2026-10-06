"""Shared primitives: messages, tool calls, provider stream events, utilities.

Everything here is standard library only. The types are deliberately plain
dataclasses so that a whole session can be serialised to JSON and replayed
byte-for-byte (see :mod:`hui.session` and :mod:`hui.providers`).
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


class HuiError(Exception):
    """Base class for every error HUI raises on purpose."""


class ProviderError(HuiError):
    """The model provider answered with an error or something unparsable."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


class ToolError(HuiError):
    """A tool refused or failed. Surfaced to the model as ``is_error`` output."""


class PolicyError(HuiError):
    """The sandbox / permission policy denied an action."""


# --------------------------------------------------------------------------
# messages
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCall:
    """A tool invocation requested by the model.

    ``raw_arguments`` keeps the exact JSON text the model produced so that a
    replay can prove it re-fed the same bytes to the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    @classmethod
    def from_raw(cls, name: str, raw: str, call_id: str | None = None) -> ToolCall:
        call_id = call_id or new_id("call")
        text = (raw or "").strip()
        if not text:
            return cls(id=call_id, name=name, arguments={}, raw_arguments=raw or "")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolError(f"arguments for {name} are not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ToolError(
                f"arguments for {name} must be a JSON object, got {type(parsed).__name__}"
            )
        return cls(id=call_id, name=name, arguments=parsed, raw_arguments=text)

    @classmethod
    def from_arguments(
        cls, name: str, arguments: dict[str, Any] | str, call_id: str | None = None
    ) -> ToolCall:
        if isinstance(arguments, str):
            return cls.from_raw(name, arguments, call_id)
        return cls(
            id=call_id or new_id("call"),
            name=name,
            arguments=dict(arguments),
            raw_arguments=json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "raw_arguments": self.raw_arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=dict(data.get("arguments") or {}),
            raw_arguments=data.get("raw_arguments", ""),
        )


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None

    # -- constructors ------------------------------------------------------
    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: Sequence[ToolCall] | None = None,
        reasoning: str | None = None,
    ) -> Message:
        return cls(
            role="assistant",
            content=content,
            tool_calls=list(tool_calls or []),
            reasoning=reasoning,
        )

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str, name: str | None = None) -> Message:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)

    # -- helpers -----------------------------------------------------------
    def is_tool_use(self) -> bool:
        return self.role == "assistant" and bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.reasoning:
            data["reasoning"] = self.reasoning
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=[ToolCall.from_dict(item) for item in data.get("tool_calls") or []],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            reasoning=data.get("reasoning"),
        )


# --------------------------------------------------------------------------
# provider stream events
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    index: int
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    index: int
    partial: str


@dataclass(frozen=True, slots=True)
class ToolCallReady:
    call: ToolCall


@dataclass(frozen=True, slots=True)
class UsageEvent:
    usage: Usage


@dataclass(frozen=True, slots=True)
class StopEvent:
    reason: str = "end_turn"


StreamEvent = (
    TextDelta
    | ReasoningDelta
    | ToolCallStart
    | ToolCallDelta
    | ToolCallReady
    | UsageEvent
    | StopEvent
)


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Usage:
        data = data or {}
        return cls(int(data.get("prompt_tokens", 0)), int(data.get("completion_tokens", 0)))


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3000 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF
        or 0x20000 <= code <= 0x3FFFF
    )


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate.

    Latin text averages ~4 characters per token; CJK is closer to 1.5. This is
    only used for budget warnings and compaction triggers, never billed.
    """
    if not text:
        return 0
    cjk = sum(1 for char in text if _is_cjk(char))
    latin = len(text) - cjk
    return int(latin / 4 + cjk / 1.5) + 1


def estimate_messages_tokens(messages: Iterable[Message]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(message.content) + 4
        for call in message.tool_calls:
            total += estimate_tokens(call.name) + estimate_tokens(call.raw_arguments) + 8
    return total


def truncate(text: str, limit: int, *, marker: str = "\n… [truncated {dropped} chars]") -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    dropped = len(text) - limit
    return text[:limit] + marker.format(dropped=dropped)


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "HuiError",
    "Message",
    "PolicyError",
    "ProviderError",
    "ReasoningDelta",
    "Role",
    "StopEvent",
    "StreamEvent",
    "TextDelta",
    "ToolCall",
    "ToolCallDelta",
    "ToolCallReady",
    "ToolCallStart",
    "ToolError",
    "Usage",
    "UsageEvent",
    "dumps",
    "estimate_messages_tokens",
    "estimate_tokens",
    "new_id",
    "now_iso",
    "sha256_text",
    "truncate",
]
