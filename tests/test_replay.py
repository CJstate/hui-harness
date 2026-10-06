import os

import pytest

from hui.common import Message, StopEvent, TextDelta, ToolCall, ToolCallReady
from hui.loop import Agent
from hui.policy import WORKSPACE_WRITE, Policy
from hui.providers import ScriptedProvider
from hui.replay import list_script, replay_session
from hui.session import Session
from hui.tools import ToolContext


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "data.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    return root


def record(tmp_path, workspace, calls, final_text="done"):
    """Record a real run whose tool results we can try to reproduce."""
    turns = [[ToolCallReady(call=call), StopEvent("tool_use")] for call in calls]
    turns.append([TextDelta(final_text), StopEvent("end_turn")])
    session = Session.create(tmp_path / "session.jsonl", {"workspace": str(workspace)})
    context = ToolContext(
        workspace=workspace, policy=Policy(workspace=workspace, mode=WORKSPACE_WRITE)
    )
    result = Agent(ScriptedProvider(turns), context=context, session=session).run("do the work")
    assert result.stop_reason == "end_turn"
    return session


def test_replay_reproduces_a_deterministic_read(tmp_path, workspace):
    call = ToolCall.from_arguments("read_file", {"path": "data.txt"})
    session = record(tmp_path, workspace, [call])

    report = replay_session(session, workspace=workspace)
    assert report.executed is True
    assert report.ok, report.render()
    assert report.steps[0].expected == report.steps[0].actual
    assert report.steps[0].name == "read_file"
    assert report.final_text == "done"
    assert "REPLAY OK" in report.render()
    assert report.to_dict()["steps"][0]["ok"] is True


def test_replay_detects_a_changed_tool_result(tmp_path, workspace):
    call = ToolCall.from_arguments("read_file", {"path": "data.txt"})
    session = record(tmp_path, workspace, [call])
    (workspace / "data.txt").write_text("alpha\ngamma\n", encoding="utf-8")

    report = replay_session(session, workspace=workspace)
    assert report.ok is False
    assert [step.name for step in report.mismatches] == ["read_file"]
    assert report.mismatches[0].expected != report.mismatches[0].actual
    rendered = report.render()
    assert "REPLAY MISMATCH" in rendered
    assert "1 of 1 tool result(s) differ" in rendered


def test_replay_re_executes_commands_deterministically(tmp_path, workspace):
    command = "Write-Output hui-replay" if os.name == "nt" else "echo hui-replay"
    call = ToolCall.from_arguments("run_command", {"command": command})
    session = record(tmp_path, workspace, [call], final_text="ran it")

    report = replay_session(session, workspace=workspace)
    assert report.ok, report.render()
    assert report.stop_reason == "end_turn"


def test_replay_of_several_tool_calls(tmp_path, workspace):
    calls = [
        ToolCall.from_arguments("read_file", {"path": "data.txt"}),
        ToolCall.from_arguments("list_dir", {}),
    ]
    session = record(tmp_path, workspace, calls)
    report = replay_session(session, workspace=workspace)
    assert report.ok, report.render()
    assert [step.name for step in report.steps] == ["read_file", "list_dir"]


def test_replay_without_executing_only_lists_the_plan(tmp_path, workspace):
    call = ToolCall.from_arguments("write_file", {"path": "new.txt", "content": "x\n"})
    session = record(tmp_path, workspace, [call])
    (workspace / "new.txt").unlink()

    report = replay_session(session, execute=False)
    assert report.executed is False
    assert all(step.ok for step in report.steps)
    assert not (workspace / "new.txt").exists()  # nothing was executed


def test_replay_marks_turns_that_never_ran(tmp_path, workspace):
    call = ToolCall.from_arguments("read_file", {"path": "data.txt"})
    session = Session.create(tmp_path / "s.jsonl", {"workspace": str(workspace)})
    # A text-only turn stops the replayed loop, leaving the later call unreached.
    session.record(Message.assistant(content="nothing to do"))
    session.record(Message.assistant(content=None, tool_calls=[call]))
    session.record_tool_result(call, "alpha\nbeta\n", ok=True, duration_ms=1)
    session.stop("end_turn")

    report = replay_session(session, workspace=workspace)
    assert report.ok is False
    assert len(report.steps) == 1
    assert report.steps[0].actual is None
    assert report.steps[0].expected  # the recording did have a hash for it


def test_list_script_never_runs_anything(tmp_path, workspace):
    call = ToolCall.from_arguments("write_file", {"path": "created.txt", "content": "x\n"})
    session = record(tmp_path, workspace, [call])
    (workspace / "created.txt").unlink()

    described = list_script(session)
    assert "turn 0" in described
    assert "recorded=" in described
    assert "write_file" in described
    assert not (workspace / "created.txt").exists()
