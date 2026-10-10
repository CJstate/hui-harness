import pytest

from hui import __version__, cli
from hui.common import StopEvent, TextDelta, ToolCall, ToolCallReady
from hui.providers import ScriptedProvider


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    for name in ("DEEPSEEK_API_KEY", "HUI_API_KEY", "HUI_PROVIDER", "HUI_BASE_URL", "HUI_MODEL"):
        monkeypatch.delenv(name, raising=False)


def record_session(tmp_path, workspace, command=None):
    from hui.loop import Agent
    from hui.policy import WORKSPACE_WRITE, Policy
    from hui.session import Session
    from hui.tools import ToolContext

    call = (
        ToolCall.from_arguments("run_command", {"command": command})
        if command
        else ToolCall.from_arguments("read_file", {"path": "data.txt"})
    )
    turns = [
        [ToolCallReady(call=call), StopEvent("tool_use")],
        [TextDelta("finished"), StopEvent("end_turn")],
    ]
    session = Session.create(tmp_path / "recorded.jsonl", {"workspace": str(workspace)})
    context = ToolContext(
        workspace=workspace, policy=Policy(workspace=workspace, mode=WORKSPACE_WRITE)
    )
    Agent(ScriptedProvider(turns), context=context, session=session).run("go")
    return session.path


def test_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_run_without_a_prompt_is_a_usage_error(tmp_path, capsys):
    assert cli.main(["run", "--session-dir", str(tmp_path)]) == 2
    assert "a prompt is required" in capsys.readouterr().err


def test_run_without_credentials_reports_a_provider_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HUI_PROVIDER", "deepseek")
    code = cli.main(["run", "hello", "-p", "hello", "--session-dir", str(tmp_path)])
    assert code == 3
    assert "provider error" in capsys.readouterr().err


def test_run_provider_flag_builds_the_named_preset(tmp_path, capsys, monkeypatch):
    # `--provider` names a preset; passing it through the override dict used to
    # crash with AttributeError because ProviderConfig has no `provider` field.
    captured = {}

    def fake_build(config):
        captured["name"] = config.name
        return ScriptedProvider.text("ok")

    monkeypatch.setattr(cli, "build_provider", fake_build)
    code = cli.main(
        ["run", "hi", "--provider", "glm", "--session-dir", str(tmp_path), "--quiet"]
    )
    assert code == 0
    assert captured["name"] == "glm"


def test_run_with_a_scripted_provider_prints_a_summary(tmp_path, capsys, monkeypatch):
    from hui.providers import ScriptedProvider as SP

    monkeypatch.setattr(cli, "build_provider", lambda config: SP.text("scripted answer"))
    code = cli.main(["run", "hi", "--session-dir", str(tmp_path), "--quiet"])
    out = capsys.readouterr().out
    assert code == 0
    assert "steps=1" in out
    assert "stop=end_turn" in out
    assert list(tmp_path.glob("session-*.jsonl"))


def test_run_json_output_is_machine_readable(tmp_path, capsys, monkeypatch):
    import json

    from hui.providers import ScriptedProvider as SP

    monkeypatch.setattr(cli, "build_provider", lambda config: SP.text("ok"))
    assert cli.main(["run", "hi", "--json", "--quiet", "--session-dir", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stop_reason"] == "end_turn"


def test_demo_runs_offline_and_writes_the_file(tmp_path, capsys):
    demo_dir = tmp_path / "demo"
    code = cli.main(["demo", "--dir", str(demo_dir), "--session-dir", str(tmp_path / "s")])
    out = capsys.readouterr().out
    assert code == 0
    assert (demo_dir / "hello_hui.py").exists()
    assert "hello from HUI" in (demo_dir / "hello_hui.py").read_text(encoding="utf-8")
    assert "Demo done" in out
    assert "[ok] run_command" in out
    assert "hello from HUI" in out  # the command really printed it


def test_replay_command_verifies_a_good_recording(tmp_path, capsys):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "data.txt").write_text("alpha\n", encoding="utf-8")
    path = record_session(tmp_path, workspace)

    code = cli.main(["replay", str(path), "--json", "-C", str(workspace)])
    out = capsys.readouterr().out
    assert code == 0
    assert '"ok": true' in out


def test_replay_command_exits_one_on_mismatch(tmp_path, capsys):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "data.txt").write_text("alpha\n", encoding="utf-8")
    path = record_session(tmp_path, workspace)
    (workspace / "data.txt").write_text("changed\n", encoding="utf-8")

    code = cli.main(["replay", str(path), "-C", str(workspace)])
    assert code == 1
    assert "REPLAY MISMATCH" in capsys.readouterr().out


def test_replay_list_shows_the_plan(tmp_path, capsys):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "data.txt").write_text("alpha\n", encoding="utf-8")
    path = record_session(tmp_path, workspace)

    assert cli.main(["replay", str(path), "--list"]) == 0
    out = capsys.readouterr().out
    assert "turn 0" in out
    assert "read_file" in out


def test_sessions_lists_recent_recordings(tmp_path, capsys):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "data.txt").write_text("alpha\n", encoding="utf-8")
    record_session(tmp_path, workspace)

    assert cli.main(["sessions", "--session-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "recorded.jsonl" in out
    assert "stop=end_turn" in out


def test_sessions_reports_an_empty_directory(tmp_path, capsys):
    assert cli.main(["sessions", "--session-dir", str(tmp_path / "nope")]) == 0
    assert "no sessions yet" in capsys.readouterr().out


def test_repl_exits_on_eof(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "build_provider", lambda config: ScriptedProvider.text("ok"))
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    assert cli.main(["--session-dir", str(tmp_path / "s"), "-C", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "HUI harness" in out
    assert "workspace:" in out


def test_repl_handles_slash_commands(tmp_path, capsys, monkeypatch):
    inputs = iter(["/help", "/tools", "/mode", "/cost", "/session", "/compact", "/nope", "/exit"])
    monkeypatch.setattr(cli, "build_provider", lambda config: ScriptedProvider.text("ok"))
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs, "/exit"))
    assert cli.main(["--session-dir", str(tmp_path / "s"), "-C", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "read_file" in out
    assert "mode: workspace-write" in out
    assert "unknown command /nope" in out


class _Bare:
    """A console stand-in with no ``reconfigure`` support at all."""


class _Detached:
    def reconfigure(self, **kwargs):
        raise ValueError("underlying buffer has been detached")


def test_console_hardening_tolerates_odd_consoles(monkeypatch):
    for stream in (_Bare(), _Detached()):
        monkeypatch.setattr("sys.stdout", stream)
        monkeypatch.setattr("sys.stderr", stream)
        cli.harden_console()  # must never raise
