"""Model providers.

Three concrete adapters, all implemented on top of the standard library:

* :class:`OpenAICompatProvider` — any ``/chat/completions`` endpoint
  (DeepSeek, OpenAI, GLM, Qwen, Kimi, vLLM, Ollama, …).
* :class:`AnthropicProvider` — the Messages API with its own streaming protocol.
* :class:`ScriptedProvider` — a deterministic, offline provider used by the test
  suite and by ``hui demo``. It is also the backbone of replay: a recorded run
  is just a scripted provider that reads events back from disk.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from hui.common import (
    Message,
    ProviderError,
    ReasoningDelta,
    StopEvent,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallReady,
    ToolCallStart,
    Usage,
    UsageEvent,
)

DEFAULT_MAX_TOKENS = 8192
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class ProviderConfig:
    """Everything a provider needs to talk to a model endpoint."""

    name: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float | None = None
    timeout: float = 120.0
    retries: int = 3
    extra_headers: dict[str, str] = field(default_factory=dict)

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers


# --------------------------------------------------------------------------
# HTTP plumbing (stdlib only)
# --------------------------------------------------------------------------


def _iter_sse(response: Any) -> Iterator[tuple[str, str]]:
    """Yield ``(event_name, data)`` pairs out of an SSE byte stream."""
    event_name = ""
    data_lines: list[str] = []
    for raw in response:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name, data_lines = "", []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield event_name, "\n".join(data_lines)


def _open_stream(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float, retries: int
) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urlrequest.Request(url, data=body, headers=headers, method="POST")
        try:
            return urlrequest.urlopen(request, timeout=timeout)  # noqa: S310 - explicit user-supplied endpoint
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            if exc.code in RETRY_STATUSES and attempt < retries:
                last_error = ProviderError(
                    f"HTTP {exc.code} from {url}", status=exc.code, body=detail
                )
                time.sleep(min(2**attempt * 0.5, 8.0))
                continue
            raise ProviderError(
                f"HTTP {exc.code} from {url}: {detail}", status=exc.code, body=detail
            ) from exc
        except urlerror.URLError as exc:
            if attempt < retries:
                last_error = exc
                time.sleep(min(2**attempt * 0.5, 8.0))
                continue
            raise ProviderError(f"cannot reach {url}: {exc}") from exc
    raise ProviderError(f"cannot reach {url}: {last_error}")


# --------------------------------------------------------------------------
# payload conversions
# --------------------------------------------------------------------------


def openai_messages(messages: Sequence[Message], system: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        if message.role == "system":
            out.append({"role": "system", "content": message.content})
        elif message.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id or "",
                    "content": message.content,
                }
            )
        elif message.role == "assistant":
            payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                    }
                    for call in message.tool_calls
                ]
            out.append(payload)
        else:
            out.append({"role": "user", "content": message.content})
    return out


def anthropic_messages(messages: Sequence[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush_results() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        elif message.role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.content or "",
                }
            )
        elif message.role == "assistant":
            flush_results()
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        else:
            flush_results()
            out.append({"role": "user", "content": [{"type": "text", "text": message.content}]})
    flush_results()
    return ("\n\n".join(part for part in system_parts if part) or None), out


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------


class OpenAICompatProvider:
    """Streaming client for any OpenAI-compatible ``/chat/completions``."""

    tool_spec_style = "openai"

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name
        self.model = config.model
        self.usage = Usage()

    def endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return f"{base}/chat/completions"

    def build_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        system: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages(messages, system),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        return payload

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        system: str | None = None,
    ) -> Iterator[StreamEvent]:
        payload = self.build_payload(messages, tools, system)
        response = _open_stream(
            self.endpoint(),
            payload,
            self.config.headers(),
            self.config.timeout,
            self.config.retries,
        )
        accumulator: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason = "end_turn"
        try:
            for _event, data in _iter_sse(response):
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("error"), dict):
                    message = chunk["error"].get("message", "unknown provider error")
                    raise ProviderError(message, body=data[:2000])
                raw_usage = chunk.get("usage")
                if isinstance(raw_usage, dict):
                    usage = Usage(
                        int(raw_usage.get("prompt_tokens") or 0),
                        int(raw_usage.get("completion_tokens") or 0),
                    )
                for choice in chunk.get("choices") or []:
                    finish = choice.get("finish_reason")
                    if finish:
                        stop_reason = str(finish)
                    delta = choice.get("delta") or choice.get("message") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        yield ReasoningDelta(str(reasoning))
                    content = delta.get("content")
                    if content:
                        yield TextDelta(str(content))
                    for call_delta in delta.get("tool_calls") or []:
                        index = int(call_delta.get("index") or 0)
                        slot = accumulator.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        if call_delta.get("id"):
                            slot["id"] = call_delta["id"]
                        function = call_delta.get("function") or {}
                        if function.get("name") and not slot["name"]:
                            slot["name"] = str(function["name"])
                            yield ToolCallStart(index=index, id=slot["id"], name=slot["name"])
                        if function.get("arguments"):
                            piece = str(function["arguments"])
                            slot["arguments"] += piece
                            yield ToolCallDelta(index=index, partial=piece)
        finally:
            response.close()

        for index in sorted(accumulator):
            slot = accumulator[index]
            if not slot["name"]:
                continue
            call = ToolCall.from_raw(slot["name"], slot["arguments"] or "{}", slot["id"] or None)
            yield ToolCallReady(call=call)
        self.usage = usage
        yield UsageEvent(usage=usage)
        yield StopEvent(reason=stop_reason)


class AnthropicProvider:
    """Streaming client for the Anthropic Messages API."""

    tool_spec_style = "anthropic"

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name or "anthropic"
        self.model = config.model
        self.usage = Usage()

    def endpoint(self) -> str:
        base = (self.config.base_url or "https://api.anthropic.com/v1").rstrip("/")
        return f"{base}/messages"

    def build_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        system: str | None = None,
    ) -> dict[str, Any]:
        merged_system, converted = anthropic_messages(messages)
        if system:
            merged_system = f"{system}\n\n{merged_system}" if merged_system else system
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if merged_system:
            payload["system"] = merged_system
        if tools:
            payload["tools"] = list(tools)
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        return payload

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        system: str | None = None,
    ) -> Iterator[StreamEvent]:
        headers = dict(self.config.headers())
        headers["x-api-key"] = self.config.api_key
        headers["anthropic-version"] = "2023-06-01"
        headers.pop("Authorization", None)
        payload = self.build_payload(messages, tools, system)
        response = _open_stream(
            self.endpoint(), payload, headers, self.config.timeout, self.config.retries
        )
        blocks: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason = "end_turn"
        try:
            for event_name, data in _iter_sse(response):
                if not data:
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                kind = chunk.get("type") or event_name
                if kind == "error":
                    error = chunk.get("error") or {}
                    raise ProviderError(
                        str(error.get("message", "unknown anthropic error")), body=data[:2000]
                    )
                if kind == "message_start":
                    raw = (chunk.get("message") or {}).get("usage") or {}
                    usage = Usage(int(raw.get("input_tokens") or 0), usage.completion_tokens)
                elif kind == "content_block_start":
                    index = int(chunk.get("index") or 0)
                    block = chunk.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        blocks[index] = {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "json": "",
                        }
                        yield ToolCallStart(
                            index=index, id=blocks[index]["id"], name=blocks[index]["name"]
                        )
                elif kind == "content_block_delta":
                    index = int(chunk.get("index") or 0)
                    delta = chunk.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield TextDelta(str(delta["text"]))
                    elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                        yield ReasoningDelta(str(delta["thinking"]))
                    elif delta.get("type") == "input_json_delta":
                        piece = str(delta.get("partial_json") or "")
                        if index in blocks:
                            blocks[index]["json"] += piece
                        yield ToolCallDelta(index=index, partial=piece)
                elif kind == "content_block_stop":
                    index = int(chunk.get("index") or 0)
                    block = blocks.pop(index, None)
                    if block and block["name"]:
                        yield ToolCallReady(
                            call=ToolCall.from_raw(
                                block["name"], block["json"] or "{}", block["id"] or None
                            )
                        )
                elif kind == "message_delta":
                    delta = chunk.get("delta") or {}
                    if delta.get("stop_reason"):
                        stop_reason = str(delta["stop_reason"])
                    raw = chunk.get("usage") or {}
                    usage = Usage(usage.prompt_tokens, int(raw.get("output_tokens") or 0))
        finally:
            response.close()
        self.usage = usage
        yield UsageEvent(usage=usage)
        yield StopEvent(reason=stop_reason)


class ScriptedProvider:
    """Deterministic provider for tests, demos and replay.

    ``script`` is either a list of "turns" — each turn being a list of
    :class:`StreamEvent` — or a callable receiving the message list and
    returning the next turn. Anything left over is answered with a plain
    assistant text so a test never hangs.
    """

    tool_spec_style = "openai"

    def __init__(
        self,
        turns: Sequence[Sequence[StreamEvent]] | None = None,
        *,
        responder: Callable[[Sequence[Message]], Sequence[StreamEvent]] | None = None,
        name: str = "scripted",
        model: str = "scripted-1",
    ):
        self.turns = [list(turn) for turn in (turns or [])]
        self.responder = responder
        self.name = name
        self.model = model
        self.requests: list[list[Message]] = []

    @classmethod
    def text(cls, content: str, **kwargs: Any) -> ScriptedProvider:
        return cls([[TextDelta(content), StopEvent("end_turn")]], **kwargs)

    @classmethod
    def tool_then_text(
        cls, name: str, arguments: dict[str, Any], content: str, **kwargs: Any
    ) -> ScriptedProvider:
        call = ToolCall.from_arguments(name, arguments)
        return cls(
            [
                [ToolCallReady(call=call), StopEvent("tool_use")],
                [TextDelta(content), StopEvent("end_turn")],
            ],
            **kwargs,
        )

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        system: str | None = None,
    ) -> Iterator[StreamEvent]:
        self.requests.append(list(messages))
        if self.responder is not None:
            yield from self.responder(messages)
            return
        if self.turns:
            yield from self.turns.pop(0)
            return
        yield TextDelta("[scripted provider: no more turns]")
        yield StopEvent("end_turn")


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------

PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-5",
        "env": "ANTHROPIC_API_KEY",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.5",
        "env": "GLM_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "env": "DASHSCOPE_API_KEY",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3:8b",
        "env": "OLLAMA_API_KEY",
    },
}


def config_from_env(overrides: dict[str, Any] | None = None) -> ProviderConfig:
    """Build a provider config from ``HUI_*`` environment variables.

    ``HUI_PROVIDER`` selects a preset (``deepseek``, ``openai``, ``anthropic``,
    ``glm``, ``qwen``, ``ollama``); ``HUI_BASE_URL`` / ``HUI_API_KEY`` /
    ``HUI_MODEL`` override any single field, which is how you point HUI at a
    self-hosted OpenAI-compatible server.
    """
    name = (os.environ.get("HUI_PROVIDER") or "deepseek").strip().lower()
    if overrides and overrides.get("provider"):
        name = str(overrides.pop("provider")).strip().lower()
    preset = PRESETS.get(name, PRESETS["deepseek"])
    base_url = os.environ.get("HUI_BASE_URL") or preset["base_url"]
    model = os.environ.get("HUI_MODEL") or preset["model"]
    api_key = os.environ.get("HUI_API_KEY") or os.environ.get(preset.get("env", ""), "")
    config = ProviderConfig(name=name, model=model, base_url=base_url, api_key=api_key)
    for key, value in (overrides or {}).items():
        if value is not None:
            setattr(config, key, value)
    return config


def build_provider(config: ProviderConfig) -> OpenAICompatProvider | AnthropicProvider:
    if config.name == "anthropic":
        return AnthropicProvider(config)
    if (
        not config.api_key
        and "127.0.0.1" not in config.base_url
        and "localhost" not in config.base_url
    ):
        env_name = PRESETS.get(config.name, {}).get("env", "HUI_API_KEY")
        raise ProviderError(f"no API key: set {env_name} or HUI_API_KEY (provider={config.name})")
    return OpenAICompatProvider(config)


__all__ = [
    "AnthropicProvider",
    "OpenAICompatProvider",
    "PRESETS",
    "ProviderConfig",
    "ScriptedProvider",
    "anthropic_messages",
    "build_provider",
    "config_from_env",
    "openai_messages",
]
