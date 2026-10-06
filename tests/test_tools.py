import locale
import os
from dataclasses import replace

import pytest

from hui.common import PolicyError, ToolCall, ToolError
from hui.policy import DANGER, READ_ONLY, WORKSPACE_WRITE, Policy
from hui.tools import Tool, ToolContext, ToolRegistry, default_registry, default_tools


@pytest.fixture
def env(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    policy = Policy(workspace=workspace, mode=WORKSPACE_WRITE)
    return ToolRegistry(default_tools()), ToolContext(workspace=workspace, policy=policy), workspace


def run(registry, ctx, name, **arguments):
    return registry.execute(ToolCall.from_arguments(name, arguments), ctx)


def test_default_registry_exposes_the_expected_tool_set(env):
    registry, _, _ = env
    assert registry.names() == [
        "read_file",
        "write_file",
        "edit_file",
        "glob_files",
        "grep_files",
        "list_dir",
        "run_command",
        "todo_write",
    ]


def test_openai_specs_put_required_arguments_at_the_top_level(env):
    registry, _, _ = env
    spec = next(
        item for item in registry.specs("openai") if item["function"]["name"] == "read_file"
    )
    assert spec["type"] == "function"
    assert spec["function"]["parameters"]["required"] == ["path"]
    assert "required" not in spec["function"]["parameters"]["properties"]["path"]


def test_anthropic_specs_use_input_schema(env):
    registry, _, _ = env
    spec = next(item for item in registry.specs("anthropic") if item["name"] == "edit_file")
    assert set(spec) == {"name", "description", "input_schema"}
    assert spec["input_schema"]["required"] == ["path", "old_string", "new_string"]


def test_tool_lookup_is_tolerant_of_model_casing(env):
    registry, _, _ = env
    assert registry.get("Read").name == "read_file"
    assert registry.get("BASH").name == "run_command"
    with pytest.raises(ToolError, match="unknown tool"):
        registry.get("teleport")


def test_duplicate_registration_is_refused(env):
    registry, _, _ = env
    with pytest.raises(ValueError, match="already registered"):
        registry.register(default_tools()[0])
    registry.register(replace(default_tools()[0]), replace=True)  # explicit replace is allowed
    assert registry.get("read_file")


def _locale_sample(encoding: str) -> str:
    """Non-ASCII text this machine's code page can represent (cp1252 cannot do CJK)."""
    for sample in ("中文注释", "café · résumé"):
        try:
            sample.encode(encoding)
        except UnicodeEncodeError:
            continue
        return sample
    raise AssertionError(f"no sample text is encodable as {encoding}")


def test_read_file_reports_locale_encoded_text(env):
    registry, ctx, workspace = env
    preferred = locale.getpreferredencoding(False)
    sample = _locale_sample(preferred)
    (workspace / "notes.txt").write_bytes(f"{sample}\nsecond\n".encode(preferred))
    out = run(registry, ctx, "read_file", path="notes.txt")
    assert sample in out
    assert "encoding=" in out
    assert "2 lines" in out


def test_read_file_missing_and_directory_errors(env):
    registry, ctx, workspace = env
    with pytest.raises(ToolError, match="file not found"):
        run(registry, ctx, "read_file", path="nope.txt")
    (workspace / "dir").mkdir()
    with pytest.raises(ToolError, match="is a directory"):
        run(registry, ctx, "read_file", path="dir")


def test_read_file_requires_path(env):
    registry, ctx, _ = env
    with pytest.raises(ToolError, match="missing required argument 'path'"):
        run(registry, ctx, "read_file")


def test_write_then_edit_round_trip(env):
    registry, ctx, workspace = env
    out = run(registry, ctx, "write_file", path="pkg/mod.py", content="VALUE = 1\n")
    assert "wrote" in out
    assert (workspace / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    diff = run(
        registry,
        ctx,
        "edit_file",
        path="pkg/mod.py",
        old_string="VALUE = 1",
        new_string="VALUE = 2",
    )
    assert "-VALUE = 1" in diff and "+VALUE = 2" in diff
    assert (workspace / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_edit_file_preserves_crlf_endings(env):
    registry, ctx, workspace = env
    target = workspace / "win.txt"
    target.write_bytes(b"a\r\nb\r\n")
    run(registry, ctx, "edit_file", path="win.txt", old_string="b", new_string="c")
    assert target.read_bytes() == b"a\r\nc\r\n"


def test_edit_file_refuses_ambiguous_and_missing_matches(env):
    registry, ctx, workspace = env
    (workspace / "dup.txt").write_text("x\nx\n", encoding="utf-8")
    with pytest.raises(ToolError, match="appears 2 times"):
        run(registry, ctx, "edit_file", path="dup.txt", old_string="x", new_string="y")
    run(
        registry, ctx, "edit_file", path="dup.txt", old_string="x", new_string="y", replace_all=True
    )
    assert (workspace / "dup.txt").read_text(encoding="utf-8") == "y\ny\n"
    with pytest.raises(ToolError, match="old_string not found"):
        run(registry, ctx, "edit_file", path="dup.txt", old_string="zzz", new_string="q")


def test_edit_file_rejects_identical_strings(env):
    registry, ctx, workspace = env
    (workspace / "a.txt").write_text("same\n", encoding="utf-8")
    with pytest.raises(ToolError, match="identical"):
        run(registry, ctx, "edit_file", path="a.txt", old_string="same", new_string="same")


def test_writes_are_refused_in_read_only_mode(tmp_path):
    workspace = tmp_path / "ro"
    workspace.mkdir()
    ctx = ToolContext(workspace=workspace, policy=Policy(workspace=workspace, mode=READ_ONLY))
    registry = default_registry()
    with pytest.raises(PolicyError, match="read-only"):
        run(registry, ctx, "write_file", path="a.txt", content="x")
    with pytest.raises(PolicyError, match="read-only"):
        run(registry, ctx, "edit_file", path="a.txt", old_string="a", new_string="b")


def test_writes_outside_the_workspace_are_refused(env):
    registry, ctx, _ = env
    with pytest.raises(PolicyError, match="outside the workspace"):
        run(registry, ctx, "write_file", path="../escape.txt", content="nope")


def test_glob_files_is_recursive_and_skips_noise_dirs(env):
    registry, ctx, workspace = env
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "deep.py").write_text("print('hi')\n", encoding="utf-8")
    (workspace / "top.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "junk.py").write_text("junk\n", encoding="utf-8")
    out = run(registry, ctx, "glob_files", pattern="**/*.py")
    assert "pkg/deep.py" in out and "top.py" in out
    assert "junk.py" not in out


def test_glob_files_reports_no_matches(env):
    registry, ctx, _ = env
    assert "no files matched" in run(registry, ctx, "glob_files", pattern="*.rs")


def test_grep_files_reports_file_and_line(env):
    registry, ctx, workspace = env
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("import os\nNEEDLE = 1\n", encoding="utf-8")
    out = run(registry, ctx, "grep_files", pattern="needle", ignore_case=True, include="*.py")
    assert "src/app.py:2" in out
    assert "NEEDLE = 1" in out


def test_grep_files_rejects_a_bad_regex(env):
    registry, ctx, _ = env
    with pytest.raises(ToolError, match="invalid regular expression"):
        run(registry, ctx, "grep_files", pattern="([unclosed")


def test_list_dir_marks_directories_and_sizes(env):
    registry, ctx, workspace = env
    (workspace / "sub").mkdir()
    (workspace / "file.txt").write_text("12345", encoding="utf-8")
    out = run(registry, ctx, "list_dir")
    assert "d sub/" in out
    assert "f file.txt  5 B" in out


def test_run_command_captures_output_and_exit_code(env):
    registry, ctx, _ = env
    command = "Write-Output hui-ok" if os.name == "nt" else "echo hui-ok"
    out = run(registry, ctx, "run_command", command=command)
    assert "exit_code=0" in out
    assert "hui-ok" in out


def test_run_command_is_refused_in_read_only_mode(tmp_path):
    workspace = tmp_path / "ro2"
    workspace.mkdir()
    ctx = ToolContext(workspace=workspace, policy=Policy(workspace=workspace, mode=READ_ONLY))
    with pytest.raises(PolicyError, match="read-only"):
        run(default_registry(), ctx, "run_command", command="echo hi")


def test_run_command_needs_approval_for_destructive_input(env):
    registry, ctx, _ = env
    with pytest.raises(PolicyError, match="no approval hook"):
        run(registry, ctx, "run_command", command="rm -rf /")


def test_run_command_checks_cwd_bounds(env):
    registry, ctx, _ = env
    with pytest.raises(PolicyError, match="outside the workspace"):
        run(registry, ctx, "run_command", command="echo hi", cwd="../..")


def test_todo_write_stores_and_renders(env):
    registry, ctx, _ = env
    out = run(
        registry,
        ctx,
        "todo_write",
        todos=[
            {"content": "write tests", "status": "completed"},
            {"content": "ship it", "status": "in_progress"},
        ],
    )
    assert "1/2 complete" in out
    assert "[x] write tests" in out and "[~] ship it" in out
    assert ctx.todos[1] == {"content": "ship it", "status": "in_progress"}


def test_todo_write_validates_status_and_content(env):
    registry, ctx, _ = env
    with pytest.raises(ToolError, match="must be one of"):
        run(registry, ctx, "todo_write", todos=[{"content": "x", "status": "doing"}])
    with pytest.raises(ToolError, match="missing 'content'"):
        run(registry, ctx, "todo_write", todos=[{"status": "pending"}])
    with pytest.raises(ToolError, match="must be a JSON array"):
        run(registry, ctx, "todo_write", todos="not json")


def test_registry_wraps_unexpected_tool_bugs(env):
    registry, _, workspace = env

    def explode(args, ctx):
        raise ValueError("boom")

    registry.register(Tool(name="explode", description="", parameters={}, handler=explode))
    ctx = ToolContext(workspace=workspace, policy=Policy(workspace=workspace, mode=DANGER))
    with pytest.raises(ToolError, match="explode failed: ValueError: boom"):
        run(registry, ctx, "explode")


def test_write_tools_require_a_writable_policy(tmp_path):
    workspace = tmp_path / "ro3"
    workspace.mkdir()
    ctx = ToolContext(workspace=workspace, policy=Policy(workspace=workspace, mode=READ_ONLY))
    with pytest.raises(PolicyError, match="read-only"):
        run(default_registry(), ctx, "edit_file", path="x.txt", old_string="a", new_string="b")
