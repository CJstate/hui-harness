from __future__ import annotations

import pytest

from hui.common import (
    Message,
    ToolCall,
    ToolError,
    Usage,
    dumps,
    estimate_messages_tokens,
    estimate_tokens,
    sha256_text,
    truncate,
)


def test_tool_call_from_raw_parses_json_object() -> None:
    call = ToolCall.from_raw("read_file", '{"path": "a.txt", "limit": 3}', "call_1")
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.txt", "limit": 3}
    assert call.raw_arguments == '{"path": "a.txt", "limit": 3}'


def test_tool_call_from_raw_rejects_invalid_json() -> None:
    with pytest.raises(ToolError, match="not valid JSON"):
        ToolCall.from_raw("read_file", "{nope")


def test_tool_call_from_raw_rejects_non_object() -> None:
    with pytest.raises(ToolError, match="must be a JSON object"):
        ToolCall.from_raw("read_file", "[1, 2]")


def test_tool_call_from_raw_accepts_empty_arguments() -> None:
    call = ToolCall.from_raw("todo_write", "")
    assert call.arguments == {}
    assert call.id.startswith("call_")


def test_tool_call_round_trip() -> None:
    call = ToolCall.from_arguments("grep", {"pattern": "x"}, "call_9")
    assert ToolCall.from_dict(call.to_dict()) == call


def test_message_round_trip_keeps_tool_calls_and_reasoning() -> None:
    message = Message.assistant(
        "thinking out loud", [ToolCall.from_arguments("ls", {}, "c1")], reasoning="why"
    )
    restored = Message.from_dict(message.to_dict())
    assert restored == message
    assert restored.is_tool_use()


def test_message_constructors() -> None:
    assert Message.system("s").role == "system"
    assert Message.user("u").role == "user"
    result = Message.tool_result("c1", "ok", "read_file")
    assert (result.role, result.tool_call_id, result.name) == ("tool", "c1", "read_file")
    assert not Message.assistant("hi").is_tool_use()


def test_estimate_tokens_counts_cjk_denser_than_latin() -> None:
    assert estimate_tokens("") == 0
    latin = estimate_tokens("a" * 400)
    cjk = estimate_tokens("汉" * 400)
    assert cjk > latin * 2


def test_estimate_messages_tokens_includes_tool_arguments() -> None:
    with_tool = Message.assistant("", [ToolCall.from_arguments("edit_file", {"old": "x" * 200})])
    assert estimate_messages_tokens([with_tool]) > estimate_messages_tokens([Message.assistant("")])


def test_truncate_keeps_limit_and_reports_dropped() -> None:
    text = "x" * 100
    cut = truncate(text, 10)
    assert cut.startswith("x" * 10)
    assert "truncated 90 chars" in cut
    assert truncate("short", 100) == "short"
    assert truncate("x" * 100, -1) == "x" * 100


def test_usage_addition_and_serialisation() -> None:
    total = Usage(10, 5) + Usage(1, 2)
    assert total.total_tokens == 18
    assert Usage.from_dict(total.to_dict()) == total
    assert Usage.from_dict(None) == Usage(0, 0)


def test_dumps_is_stable_and_utf8_first() -> None:
    assert dumps({"b": 1, "a": "汉"}) == '{"a":"汉","b":1}'
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")
