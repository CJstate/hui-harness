import json

import pytest

from hui.common import Message, ToolCall, Usage, sha256_text
from hui.session import GENESIS, Session, SessionEvent


def test_create_writes_a_chained_start_event(tmp_path):
    path = tmp_path / "s.jsonl"
    session = Session.create(path, {"workspace": "/w"})
    assert path.exists()
    assert len(session) == 1
    assert session.events[0].kind == "start"
    assert session.events[0].prev_hash == GENESIS
    assert session.meta["workspace"] == "/w"
    assert session.id
    assert session.verify() == (True, None)


def test_records_everything_and_reloads(tmp_path):
    path = tmp_path / "s.jsonl"
    session = Session.create(path)
    call = ToolCall.from_arguments("read_file", {"path": "a.txt"})
    session.record(Message.user("hello"))
    session.record(Message.assistant(content=None, tool_calls=[call]))
    session.record_tool_result(call, "file body", ok=True, duration_ms=3)
    session.record_usage(Usage(prompt_tokens=10, completion_tokens=2))
    session.note("a note")
    session.stop("end_turn")

    loaded = Session.load(path)
    assert [message.role for message in loaded.messages()] == ["user", "assistant", "tool"]
    assert loaded.tool_results()[0]["sha256"] == sha256_text("file body")
    assert loaded.tool_results()[0]["ok"] is True
    assert loaded.usage().total_tokens == 12
    assert loaded.stop_reason() == "end_turn"
    assert loaded.verify() == (True, None)
    summary = loaded.summary()
    assert summary["messages"] == 3  # the tool result is rebuilt from its receipt
    assert summary["tool_calls"] == 1
    assert summary["tool_failures"] == 0
    assert summary["stop_reason"] == "end_turn"


def test_unknown_kind_is_refused(tmp_path):
    session = Session.create(tmp_path / "s.jsonl")
    with pytest.raises(ValueError, match="unknown session event kind"):
        session.append("telepathy", {})


def test_tampering_with_a_line_breaks_the_chain(tmp_path):
    path = tmp_path / "s.jsonl"
    session = Session.create(path)
    session.record(Message.user("unique-marker-value"))
    session.stop("end_turn")

    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.replace("unique-marker-value", "tampered-marker-xx"), encoding="utf-8")

    ok, problem = Session.load(path).verify()
    assert ok is False
    assert "does not match its hash" in problem


def test_resume_appends_onto_the_existing_chain(tmp_path):
    path = tmp_path / "s.jsonl"
    Session.create(path).record(Message.user("first"))
    resumed = Session.load(path)
    assert len(resumed) == 2
    resumed.record(Message.user("second"))
    again = Session.load(path)
    assert len(again) == 3
    assert [message.content for message in again.messages()] == ["first", "second"]
    assert again.verify() == (True, None)


def test_script_pairs_assistant_turns_with_recorded_hashes(tmp_path):
    path = tmp_path / "s.jsonl"
    session = Session.create(path)
    call = ToolCall.from_arguments("write_file", {"path": "a.txt", "content": "x"})
    session.record(Message.assistant(content="writing", tool_calls=[call]))
    session.record_tool_result(call, "wrote 1 bytes", ok=True, duration_ms=1)
    session.record(Message.assistant(content="done"))

    turns = session.script()
    assert len(turns) == 2  # every assistant turn is listed, tool-using or not
    assert turns[0]["assistant"]["content"] == "writing"
    assert turns[0]["calls"][0]["expected"] == sha256_text("wrote 1 bytes")
    assert turns[0]["calls"][0]["recorded_content"] == "wrote 1 bytes"
    assert turns[1]["calls"] == []


def test_event_round_trip_is_lossless():
    event = SessionEvent.chain(
        0, "message", {"content": "héllo\n"}, GENESIS, ts="2026-01-01T00:00:00Z"
    )
    again = SessionEvent.from_line(event.to_line())
    assert again == event
    assert event.payload()["kind"] == "message"


def test_loading_a_missing_session_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Session.load(tmp_path / "nope.jsonl")


def test_session_file_is_jsonl_with_one_object_per_line(tmp_path):
    path = tmp_path / "s.jsonl"
    session = Session.create(path)
    session.record(Message.user("hi"))
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == len(session)
    assert all(json.loads(line)["kind"] in {"start", "message"} for line in lines)
