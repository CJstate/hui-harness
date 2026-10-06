import pytest

from hui.common import Message, StopEvent, TextDelta, ToolCall, ToolCallReady
from hui.loop import Agent, run_agent
from hui.policy import READ_ONLY, WORKSPACE_WRITE, Policy
from hui.providers import ScriptedProvider
from hui.session import Session
from hui.tools import ToolContext


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture
def context(workspace):
    return ToolContext(
        workspace=workspace, policy=Policy(workspace=workspace, mode=WORKSPACE_WRITE)
    )


def make_agent(context, provider, **kwargs):
    return Agent(provider, context=context, **kwargs)


def assert_calls_stay_paired(messages):
    """No assistant tool call may appear without its tool results right behind it."""
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.tool_calls:
            ids = [call.id for call in message.tool_calls]
            following = [item.tool_call_id for item in messages[index + 1 : index + 1 + len(ids)]]
            assert following == ids, f"tool call {ids} is not followed by its results: {following}"
            index += 1 + len(ids)
        else:
            index += 1


def test_text_only_run_finishes_with_end_turn(context):
    provider = ScriptedProvider.text("hello there")
    result = make_agent(context, provider).run("say hi")
    assert result.stop_reason == "end_turn"
    assert result.final_text() == "hello there"
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert len(result.steps) == 1
    assert result.steps[0].text == "hello there"
    assert provider.requests[0][0].role == "user"


def test_run_without_a_prompt_replays_history(context):
    seed = [Message.user("earlier"), Message.assistant(content="noted")]
    result = make_agent(context, ScriptedProvider.text("continued")).run(messages=seed)
    assert result.messages[0].content == "earlier"
    assert result.final_text() == "continued"


def test_a_tool_call_is_executed_and_fed_back(context, workspace):
    provider = ScriptedProvider.tool_then_text(
        "write_file", {"path": "out.txt", "content": "x\n"}, "wrote the file"
    )
    result = make_agent(context, provider).run("write a file")
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "x\n"
    assert result.stop_reason == "end_turn"
    outcome = result.tool_calls[0]
    assert outcome.ok is True
    assert outcome.content.startswith("wrote")
    assert outcome.duration_ms >= 0
    assert result.summary()["tool_calls"] == 1
    second_request = provider.requests[1]
    assert second_request[-1].role == "tool"
    assert second_request[-1].tool_call_id == outcome.call.id
    assert_calls_stay_paired(result.messages)


def test_tool_errors_are_reported_and_not_raised(context):
    provider = ScriptedProvider.tool_then_text(
        "read_file", {"path": "missing.txt"}, "could not read it"
    )
    result = make_agent(context, provider).run("read it")
    outcome = result.tool_calls[0]
    assert outcome.ok is False
    assert outcome.content.startswith("tool error:")
    assert result.summary()["tool_failures"] == 1
    assert result.steps[0].failed == 1


def test_policy_denials_are_reported_as_tool_failures(workspace):
    provider = ScriptedProvider.tool_then_text(
        "write_file", {"path": "out.txt", "content": "x"}, "done"
    )
    agent = Agent(provider, workspace=workspace, mode=READ_ONLY)
    result = agent.run("write a file")
    assert result.tool_calls[0].ok is False
    assert result.tool_calls[0].content.startswith("denied by policy:")
    assert not (workspace / "out.txt").exists()


def test_max_steps_is_enforced(context):
    def responder(messages):
        call = ToolCall.from_arguments("list_dir", {})
        return [ToolCallReady(call=call), StopEvent("tool_use")]

    agent = make_agent(context, ScriptedProvider(responder=responder), max_steps=3)
    result = agent.run("loop forever")
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 3
    assert len(result.tool_calls) == 3


def test_streaming_events_reach_the_callback(context):
    seen = []
    provider = ScriptedProvider([[TextDelta("a"), TextDelta("b"), StopEvent("end_turn")]])
    make_agent(context, provider, on_event=seen.append).run("hi")
    assert "".join(event.text for event in seen if isinstance(event, TextDelta)) == "ab"


def test_session_records_the_whole_run(context, tmp_path):
    path = tmp_path / "s.jsonl"
    session = Session.create(path, {"workspace": str(context.workspace)})
    provider = ScriptedProvider.tool_then_text(
        "write_file", {"path": "out.txt", "content": "x\n"}, "wrote the file"
    )
    result = make_agent(context, provider, session=session).run("write a file")
    assert result.session is session

    loaded = Session.load(path)
    assert loaded.verify() == (True, None)
    assert loaded.stop_reason() == "end_turn"
    assert loaded.summary()["tool_calls"] == 1
    assert loaded.summary()["messages"] == 4  # user, assistant(call), tool result, assistant
    assert [message.content for message in loaded.messages()][0] == "write a file"


def test_compaction_keeps_tool_calls_paired_with_their_results(context):
    turns = []
    for index in range(4):
        call = ToolCall.from_arguments("write_file", {"path": f"f{index}.txt", "content": "x"})
        turns.append([ToolCallReady(call=call), StopEvent("tool_use")])
    turns.append([TextDelta("all four written"), StopEvent("end_turn")])

    agent = make_agent(context, ScriptedProvider(turns), max_tokens=1, keep_recent=2, max_steps=10)
    result = agent.run("write four files")
    assert result.stop_reason == "end_turn"
    assert len(result.steps) == 5
    assert_calls_stay_paired(result.messages)
    notes = [message.content or "" for message in result.messages if message.role == "system"]
    assert any("compacted" in note for note in notes)
    assert result.final_text() == "all four written"


def test_run_agent_helper(context):
    result = run_agent(ScriptedProvider.text("ok"), "hi", context=context)
    assert result.final_text() == "ok"
