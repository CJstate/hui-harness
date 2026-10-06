"""Append-only session log — the substrate for ``hui replay``.

Every session is a JSONL file. Each line is one event, and each event carries
the hash of the previous line, so the log is tamper-evident: rewriting history
in the middle of a file is detectable with :meth:`Session.verify`.

The log is deliberately boring. It stores the messages the model produced, the
raw arguments it sent, and the *hash* of every tool result — enough to re-run
the deterministic half of the session offline (see :mod:`hui.replay`) without
storing anything the agent is not allowed to keep.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hui.common import (
    Message,
    ToolCall,
    Usage,
    dumps,
    new_id,
    now_iso,
    sha256_text,
)

GENESIS = "0" * 64
KINDS = ("start", "message", "tool_result", "usage", "note", "stop")

__all__ = ["GENESIS", "KINDS", "Session", "SessionEvent"]


@dataclass(frozen=True, slots=True)
class SessionEvent:
    seq: int
    ts: str
    kind: str
    data: dict[str, Any]
    prev_hash: str
    hash: str

    def payload(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "kind": self.kind,
            "data": self.data,
            "prev_hash": self.prev_hash,
        }

    def to_line(self) -> str:
        return dumps({**self.payload(), "hash": self.hash})

    @classmethod
    def from_line(cls, line: str) -> SessionEvent:
        raw = json.loads(line)
        return cls(
            seq=int(raw["seq"]),
            ts=str(raw["ts"]),
            kind=str(raw["kind"]),
            data=dict(raw.get("data") or {}),
            prev_hash=str(raw.get("prev_hash") or ""),
            hash=str(raw.get("hash") or ""),
        )

    @classmethod
    def chain(
        cls, seq: int, kind: str, data: dict[str, Any], prev_hash: str, *, ts: str | None = None
    ) -> SessionEvent:
        stamp = ts or now_iso()
        body = dumps({"seq": seq, "ts": stamp, "kind": kind, "data": data, "prev_hash": prev_hash})
        return cls(
            seq=seq,
            ts=stamp,
            kind=kind,
            data=data,
            prev_hash=prev_hash,
            hash=sha256_text(f"{prev_hash}\n{body}"),
        )


class Session:
    """A durable, append-only record of one agent run."""

    def __init__(
        self,
        path: str | Path,
        events: list[SessionEvent] | None = None,
        meta: dict[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.events: list[SessionEvent] = list(events or [])
        self.meta: dict[str, Any] = dict(meta or {})

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def create(cls, path: str | Path, meta: dict[str, Any] | None = None) -> Session:
        session = cls(path, meta=meta)
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
        session.append("start", {"session_id": new_id("sess"), **(meta or {})})
        return session

    @classmethod
    def load(cls, path: str | Path) -> Session:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"session not found: {target}")
        events = [
            SessionEvent.from_line(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        meta = dict(events[0].data) if events else {}
        return cls(target, events=events, meta=meta)

    @property
    def id(self) -> str:
        return str(self.meta.get("session_id") or self.path.stem)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[SessionEvent]:
        return iter(self.events)

    # -- writing -----------------------------------------------------------
    def append(self, kind: str, data: dict[str, Any]) -> SessionEvent:
        if kind not in KINDS:
            raise ValueError(
                f"unknown session event kind {kind!r}; expected one of {', '.join(KINDS)}"
            )
        prev = self.events[-1].hash if self.events else GENESIS
        event = SessionEvent.chain(len(self.events), kind, data, prev)
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(event.to_line() + "\n")
        self.events.append(event)
        return event

    def record(self, message: Message) -> SessionEvent:
        return self.append("message", message.to_dict())

    def record_usage(self, usage: Usage) -> SessionEvent:
        return self.append("usage", usage.to_dict())

    def record_tool_result(
        self,
        call: ToolCall,
        content: str,
        *,
        ok: bool,
        duration_ms: int,
    ) -> SessionEvent:
        return self.append(
            "tool_result",
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "content": content,
                "sha256": sha256_text(content),
                "ok": ok,
                "duration_ms": duration_ms,
            },
        )

    def note(self, text: str) -> SessionEvent:
        return self.append("note", {"text": text})

    def stop(self, reason: str) -> SessionEvent:
        return self.append("stop", {"reason": reason})

    # -- reading -----------------------------------------------------------
    def messages(self) -> list[Message]:
        """The full conversation, with tool results rebuilt from their receipts.

        Tool results are stored as their own event so they carry a content hash
        for replay; rebuilding them here means a resumed session sees exactly the
        transcript the model saw, without storing the content twice.
        """
        restored: list[Message] = []
        for event in self.events:
            if event.kind == "message":
                restored.append(Message.from_dict(event.data))
            elif event.kind == "tool_result":
                restored.append(
                    Message.tool_result(
                        str(event.data["id"]),
                        str(event.data.get("content", "")),
                        event.data.get("name"),
                    )
                )
        return restored

    def tool_results(self) -> list[dict[str, Any]]:
        return [event.data for event in self.events if event.kind == "tool_result"]

    def usage(self) -> Usage:
        total = Usage()
        for event in self.events:
            if event.kind == "usage":
                total = total + Usage.from_dict(event.data)
        return total

    def stop_reason(self) -> str | None:
        for event in reversed(self.events):
            if event.kind == "stop":
                return str(event.data.get("reason"))
        return None

    def script(self) -> list[dict[str, Any]]:
        """Assistant turns paired with the tool results they produced.

        This is the exact input a replay needs: the model's own output plus the
        fingerprints of what the tools returned.
        """
        turns: list[dict[str, Any]] = []
        pending: dict[str, dict[str, Any]] = {str(item["id"]): item for item in self.tool_results()}
        for message in self.messages():
            if message.role != "assistant":
                continue
            calls = []
            for call in message.tool_calls:
                recorded = pending.get(call.id)
                calls.append(
                    {
                        "call": call.to_dict(),
                        "expected": recorded["sha256"] if recorded else None,
                        "recorded_content": recorded["content"] if recorded else None,
                        "recorded_ok": recorded["ok"] if recorded else None,
                    }
                )
            turns.append({"assistant": message.to_dict(), "calls": calls})
        return turns

    def verify(self) -> tuple[bool, str | None]:
        """Recompute the hash chain; return ``(ok, problem)``."""
        prev = GENESIS
        for index, event in enumerate(self.events):
            if event.seq != index:
                return False, f"event {index} has seq={event.seq}"
            if event.prev_hash != prev:
                return False, f"event {index} does not chain to the previous hash"
            rebuilt = SessionEvent.chain(event.seq, event.kind, event.data, prev, ts=event.ts)
            if rebuilt.hash != event.hash:
                return False, f"event {index} content does not match its hash"
            prev = event.hash
        return True, None

    def summary(self) -> dict[str, Any]:
        usage = self.usage()
        results = self.tool_results()
        return {
            "session_id": self.id,
            "path": str(self.path),
            "events": len(self.events),
            "messages": len(self.messages()),
            "tool_calls": len(results),
            "tool_failures": sum(1 for item in results if not item.get("ok", True)),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "stop_reason": self.stop_reason(),
            "started": self.events[0].ts if self.events else None,
        }
