from __future__ import annotations

import json

import pytest

from hui import providers as mod
from hui.common import (
    Message,
    ProviderError,
    ReasoningDelta,
    StopEvent,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallReady,
    ToolCallStart,
    UsageEvent,
)
from hui.providers import (
    AnthropicProvider,
    OpenAICompatProvider,
    ProviderConfig,
    ScriptedProvider,
    anthropic_messages,
    config_from_env,
    openai_messages,
)


class FakeResponse:
    def __init__(self, text: str):
        self._lines = [(line + "\n").encode("utf-8") for line in text.split("\n")]
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self) -> None:
        self.closed = True


def openai_sse(chunks: list[dict]) -> FakeResponse:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return FakeResponse(body + "data: [DONE]\n\n")


def anthropic_sse(events: list[tuple[str, dict]]) -> FakeResponse:
    body = "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events)
    return FakeResponse(body)


def test_iter_sse_skips_comments_and_joins_multi_line_data() -> None:
    response = FakeResponse(": keepalive\nevent: ping\ndata: one\ndata: two\n\ndata: [DONE]\n\n")
    events = list(mod._iter_sse(response))
    assert events == [("ping", "one\ntwo"), ("", "[DONE]")]


def test_openai_messages_converts_tool_exchange() -> None:
    call = ToolCall.from_arguments("read_file", {"path": "a.txt"}, "call_1")
    messages = [
        Message.system("sys"),
        Message.user("hi"),
        Message.assistant("reading", [call]),
        Message.tool_result("call_1", "contents", "read_file"),
    ]
    payload = openai_messages(messages, system="extra")
    assert payload[0] == {"role": "system", "content": "extra"}
    assert payload[1] == {"role": "system", "content": "sys"}
    assert payload[3]["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": '{"path": "a.txt"}',
    }
    assert payload[4] == {"role": "tool", "tool_call_id": "call_1", "content": "contents"}


def test_anthropic_messages_merges_tool_results_into_one_user_turn() -> None:
    call = ToolCall.from_arguments("ls", {}, "toolu_1")
    messages = [
        Message.system("rules"),
        Message.user("go"),
        Message.assistant("", [call]),
        Message.tool_result("toolu_1", "a\nb", "ls"),
        Message.tool_result("toolu_2", "c", "ls"),
    ]
    system, converted = anthropic_messages(messages)
    assert system == "rules"
    assert converted[0] == {"role": "user", "content": [{"type": "text", "text": "go"}]}
    assert converted[1]["content"][0]["type"] == "tool_use"
    assert converted[1]["content"][0]["input"] == {}
    assert len(converted) == 3
    assert [block["type"] for block in converted[2]["content"]] == ["tool_result", "tool_result"]


def test_openai_provider_streams_text_tools_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_7",
                                "function": {"name": "read_file", "arguments": '{"pa'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.txt"}'}}]
                    }
                }
            ]
        },
        {"choices": [{"finish_reason": "tool_calls", "delta": {}}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 6}},
    ]
    captured: dict = {}

    def fake_open(url, payload, headers, timeout, retries):
        captured.update({"url": url, "payload": payload})
        return openai_sse(chunks)

    monkeypatch.setattr(mod, "_open_stream", fake_open)
    provider = OpenAICompatProvider(
        ProviderConfig(
            name="deepseek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="k",
        )
    )
    events = list(
        provider.stream(
            [Message.user("hi")], tools=[{"type": "function", "function": {"name": "read_file"}}]
        )
    )

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["Hel", "lo"]
    assert [event.text for event in events if isinstance(event, ReasoningDelta)] == ["hmm"]
    assert isinstance(events[0], TextDelta)
    call = next(event.call for event in events if isinstance(event, ToolCallReady))
    assert call.id == "call_7"
    assert call.arguments == {"path": "a.txt"}
    usage = next(event.usage for event in events if isinstance(event, UsageEvent))
    assert usage.prompt_tokens == 12 and usage.completion_tokens == 6
    assert isinstance(events[-1], StopEvent) and events[-1].reason == "tool_calls"
    assert provider.usage.total_tokens == 18
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["payload"]["tools"][0]["function"]["name"] == "read_file"


def test_openai_provider_raises_on_error_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod, "_open_stream", lambda *a, **k: openai_sse([{"error": {"message": "bad key"}}])
    )
    provider = OpenAICompatProvider(ProviderConfig(api_key="k"))
    with pytest.raises(ProviderError, match="bad key"):
        list(provider.stream([Message.user("hi")]))


def test_anthropic_provider_streams_tool_use_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 9}}}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "edit_file"},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": ' "a.py"}'},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 4},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    captured: dict = {}

    def fake_open(url, payload, headers, timeout, retries):
        captured.update({"url": url, "headers": headers, "payload": payload})
        return anthropic_sse(events)

    monkeypatch.setattr(mod, "_open_stream", fake_open)
    provider = AnthropicProvider(
        ProviderConfig(name="anthropic", model="claude-sonnet-4-5", api_key="sk-ant")
    )
    produced = list(
        provider.stream([Message.system("sys"), Message.user("hi")], tools=[{"name": "edit_file"}])
    )

    assert [event.text for event in produced if isinstance(event, TextDelta)] == ["ok"]
    starts = [event for event in produced if isinstance(event, ToolCallStart)]
    assert starts == [ToolCallStart(index=1, id="toolu_1", name="edit_file")]
    assert [event.partial for event in produced if isinstance(event, ToolCallDelta)] == [
        '{"path":',
        ' "a.py"}',
    ]
    call = next(event.call for event in produced if isinstance(event, ToolCallReady))
    assert call.arguments == {"path": "a.py"}
    usage = next(event.usage for event in produced if isinstance(event, UsageEvent))
    assert (usage.prompt_tokens, usage.completion_tokens) == (9, 4)
    assert captured["headers"]["x-api-key"] == "sk-ant"
    assert "Authorization" not in captured["headers"]
    assert captured["payload"]["system"] == "sys"
    assert captured["url"].endswith("/v1/messages")


def test_scripted_provider_walks_through_turns() -> None:
    provider = ScriptedProvider.tool_then_text("ls", {"path": "."}, "done")
    first = list(provider.stream([Message.user("go")]))
    second = list(provider.stream([Message.user("go")]))
    third = list(provider.stream([Message.user("go")]))
    assert isinstance(first[0], ToolCallReady)
    assert isinstance(second[0], TextDelta) and second[0].text == "done"
    assert "no more turns" in third[0].text
    assert len(provider.requests) == 3


def test_config_from_env_prefers_explicit_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUI_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.delenv("HUI_BASE_URL", raising=False)
    config = config_from_env({"model": "deepseek-reasoner"})
    assert config.name == "deepseek"
    assert config.model == "deepseek-reasoner"
    assert config.base_url == "https://api.deepseek.com/v1"
    assert config.api_key == "sk-x"


def test_config_from_env_preset_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUI_PROVIDER", "anthropic")
    monkeypatch.delenv("HUI_MODEL", raising=False)
    monkeypatch.delenv("HUI_API_KEY", raising=False)
    config = config_from_env()
    assert (config.name, config.base_url.endswith("/v1")) == ("anthropic", True)


def test_build_provider_requires_key_for_remote_endpoints() -> None:
    with pytest.raises(ProviderError, match="no API key"):
        mod.build_provider(
            ProviderConfig(name="openai", base_url="https://api.openai.com/v1", api_key="")
        )
    local = mod.build_provider(
        ProviderConfig(name="openai", base_url="http://127.0.0.1:11434/v1", api_key="")
    )
    assert isinstance(local, OpenAICompatProvider)
