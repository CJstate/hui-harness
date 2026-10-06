"""Built-in tools and the registry that exposes them to a model.

Every tool is an ordinary function ``(args, ctx) -> str``. Nothing here imports
a third-party package, and every filesystem read goes through the encoding-safe
helpers in :mod:`hui.fileio`, so the same code runs on a UTF-8 Linux box and a
GBK Chinese Windows box without surprises.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hui import fileio
from hui.common import PolicyError, ToolCall, ToolError
from hui.policy import Policy, command_argv

MAX_OUTPUT = 20000
DEFAULT_TIMEOUT = 120
MAX_MATCHES = 200
DEFAULT_GLOB_RESULTS = 200
MAX_SCAN_BYTES = 2_000_000
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
)
TODO_STATUSES = ("pending", "in_progress", "completed")


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n… [truncated {len(text) - limit} chars]"


def _decode(data: bytes) -> str:
    for encoding in fileio.candidate_encodings():
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _req_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ToolError(f"missing required argument {key!r}")
    return str(value)


def _int_arg(args: dict[str, Any], key: str, default: int, *, minimum: int = 0) -> int:
    value = args.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"argument {key!r} must be an integer, got {value!r}") from None
    return max(parsed, minimum)


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


@dataclass(slots=True)
class ToolContext:
    """Everything a tool needs: where it runs and what it is allowed to do."""

    workspace: Path
    policy: Policy
    todos: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()

    def resolve(self, path: str | os.PathLike[str]) -> Path:
        """Resolve ``path`` against the workspace (absolute paths stay absolute)."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate

    def display(self, path: str | os.PathLike[str]) -> str:
        target = Path(path)
        try:
            return target.resolve().relative_to(self.workspace).as_posix()
        except ValueError:
            return str(target)


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], str]
    requires_write: bool = False
    aliases: tuple[str, ...] = ()

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": self.parameters, "additionalProperties": False}

    def required(self) -> list[str]:
        return [key for key, spec in self.parameters.items() if spec.get("required")]

    def spec_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        key: {k: v for k, v in spec.items() if k != "required"}
                        for key, spec in self.parameters.items()
                    },
                    "required": self.required(),
                },
            },
        }

    def spec_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    key: {k: v for k, v in spec.items() if k != "required"}
                    for key, spec in self.parameters.items()
                },
                "required": self.required(),
            },
        }


class ToolRegistry:
    """An ordered, tolerant lookup table of tools."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._aliases: dict[str, str] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        key = tool.name.lower()
        if key in self._tools and not replace:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[key] = tool
        for alias in tool.aliases:
            self._aliases[alias.lower()] = key
        return tool

    def get(self, name: str) -> Tool:
        key = str(name).strip().lower()
        if key in self._tools:
            return self._tools[key]
        alias = self._aliases.get(key)
        if alias:
            return self._tools[alias]
        raise ToolError(f"unknown tool {name!r}; available: {', '.join(self.names())}")

    def names(self) -> list[str]:
        return [tool.name for tool in self._tools.values()]

    def specs(self, style: str = "openai") -> list[dict[str, Any]]:
        if style == "openai":
            return [tool.spec_openai() for tool in self._tools.values()]
        if style == "anthropic":
            return [tool.spec_anthropic() for tool in self._tools.values()]
        raise ValueError(f"unknown tool spec style {style!r}")

    def execute(self, call: ToolCall, ctx: ToolContext) -> str:
        tool = self.get(call.name)
        if tool.requires_write:
            ctx.policy.check_write(ctx.workspace)
        try:
            return tool.handler(call.arguments, ctx)
        except (ToolError, PolicyError):
            raise  # both are meaningful to the caller: keep the original type
        except Exception as exc:  # surface *any* tool bug as a tool result
            raise ToolError(f"{tool.name} failed: {type(exc).__name__}: {exc}") from exc

    def __contains__(self, name: object) -> bool:
        try:
            self.get(str(name))
        except ToolError:
            return False
        return True

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())


# --------------------------------------------------------------------------- #
# filesystem tools
# --------------------------------------------------------------------------- #
def _read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = ctx.resolve(_req_str(args, "path"))
    ctx.policy.check_read(path)
    if not path.exists():
        raise ToolError(f"file not found: {ctx.display(path)}")
    if path.is_dir():
        raise ToolError(f"{ctx.display(path)} is a directory; use list_dir")
    text, encoding = fileio.read_text_safe(path)
    total = fileio.count_lines(text)
    limit = _int_arg(args, "limit", 400, minimum=1)
    offset = _int_arg(args, "offset", 1, minimum=1)
    body = fileio.line_numbered(text, offset=offset, limit=limit)
    header = f"{ctx.display(path)} · {total} lines · encoding={encoding}"
    return _truncate(f"{header}\n{body}")


def _write_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = ctx.resolve(_req_str(args, "path"))
    ctx.policy.check_write(path)
    content = args.get("content")
    if not isinstance(content, str):
        raise ToolError("argument 'content' must be a string")
    newline = None
    if path.exists():
        existing, _ = fileio.read_text_safe(path)
        newline = fileio.detect_newline(existing)
    written = fileio.write_text_safe(path, content, newline=newline)
    return f"wrote {written} bytes to {ctx.display(path)} ({fileio.count_lines(content)} lines)"


def _edit_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = ctx.resolve(_req_str(args, "path"))
    ctx.policy.check_write(path)
    if not path.exists():
        raise ToolError(f"file not found: {ctx.display(path)}")
    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        raise ToolError("arguments 'old_string' and 'new_string' must both be strings")
    if old == new:
        raise ToolError("old_string and new_string are identical")
    raw, _ = fileio.read_text_safe(path)
    bom = raw.startswith(fileio.BOM_UTF8)
    text = raw.replace(fileio.BOM_UTF8, "", 1) if bom else raw
    newline = fileio.detect_newline(text)
    flat = text.replace("\r\n", "\n")
    flat_old = old.replace("\r\n", "\n")
    flat_new = new.replace("\r\n", "\n")
    count = flat.count(flat_old)
    if count == 0:
        raise ToolError(f"old_string not found in {ctx.display(path)}")
    replace_all = bool(args.get("replace_all", False))
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_string appears {count} times in {ctx.display(path)}; "
            "add unique context or pass replace_all=true"
        )
    updated = flat.replace(flat_old, flat_new)
    fileio.write_text_safe(path, updated, newline=newline, bom=bom)
    diff = fileio.unified_diff(flat, updated, ctx.display(path))
    changed = count if replace_all else 1
    return _truncate(f"edited {ctx.display(path)} ({changed} replacement(s))\n{diff}")


def _iter_files(root: Path) -> list[Path]:
    results: list[Path] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            results.append(Path(current) / name)
    return results


def _glob_files(args: dict[str, Any], ctx: ToolContext) -> str:
    pattern = _req_str(args, "pattern")
    root = ctx.resolve(args.get("path") or ".")
    ctx.policy.check_read(root)
    if not root.exists():
        raise ToolError(f"path not found: {ctx.display(root)}")
    max_results = _int_arg(args, "max_results", DEFAULT_GLOB_RESULTS, minimum=1)
    if root.is_file():
        candidates = [root]
    elif any(char in pattern for char in "*?["):
        candidates = [path for path in _iter_files(root) if fnmatch_path(path, root, pattern)]
    else:
        candidates = [path for path in _iter_files(root) if path.name == pattern]
    relative = sorted({ctx.display(path) for path in candidates if path.is_file()})
    if not relative:
        return f"no files matched {pattern!r} under {ctx.display(root)}"
    shown = relative[:max_results]
    suffix = f"\n… {len(relative) - len(shown)} more matches" if len(relative) > len(shown) else ""
    return _truncate("\n".join([f"{len(relative)} match(es) for {pattern!r}", *shown]) + suffix)


def fnmatch_path(path: Path, root: Path, pattern: str) -> bool:
    import fnmatch

    rel = path.relative_to(root).as_posix()
    normalized = pattern.replace("\\", "/").lstrip("./")
    if fnmatch.fnmatch(rel, normalized) or fnmatch.fnmatch(path.name, normalized):
        return True
    # `**/*.py` and `*.py` should both match a nested file.
    return fnmatch.fnmatch(rel, normalized.replace("**/", ""))


def _grep_files(args: dict[str, Any], ctx: ToolContext) -> str:
    pattern = _req_str(args, "pattern")
    root = ctx.resolve(args.get("path") or ".")
    ctx.policy.check_read(root)
    if not root.exists():
        raise ToolError(f"path not found: {ctx.display(root)}")
    try:
        regex = re.compile(pattern, re.IGNORECASE if args.get("ignore_case") else 0)
    except re.error as exc:
        raise ToolError(f"invalid regular expression: {exc}") from None
    include = args.get("include") or "*"
    max_matches = _int_arg(args, "max_matches", MAX_MATCHES, minimum=1)
    files = [root] if root.is_file() else _iter_files(root)
    hits: list[str] = []
    scanned = 0
    for path in files:
        if not fnmatch_path(path, root if root.is_dir() else path.parent, include):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if len(data) > MAX_SCAN_BYTES or _is_binary(data):
            continue
        scanned += 1
        text, _ = fileio.read_text_safe(path)
        for number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{ctx.display(path)}:{number}: {line.strip()[:400]}")
                if len(hits) >= max_matches:
                    break
        if len(hits) >= max_matches:
            break
    if not hits:
        return f"no matches for {pattern!r} in {scanned} file(s) under {ctx.display(root)}"
    header = f"{len(hits)} match(es) for {pattern!r} across {scanned} file(s)"
    return _truncate("\n".join([header, *hits]))


def _list_dir(args: dict[str, Any], ctx: ToolContext) -> str:
    path = ctx.resolve(args.get("path") or ".")
    ctx.policy.check_read(path)
    if not path.exists():
        raise ToolError(f"path not found: {ctx.display(path)}")
    if not path.is_dir():
        raise ToolError(f"{ctx.display(path)} is not a directory")
    rows: list[str] = []
    for entry in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
        if entry.is_dir():
            rows.append(f"d {entry.name}/")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            rows.append(f"f {entry.name}  {size} B")
    if not rows:
        return f"{ctx.display(path)} is empty"
    return _truncate("\n".join([f"{ctx.display(path)} ({len(rows)} entries)", *rows]))


# --------------------------------------------------------------------------- #
# shell + todos
# --------------------------------------------------------------------------- #
def _run_command(args: dict[str, Any], ctx: ToolContext) -> str:
    command = _req_str(args, "command")
    ctx.policy.check_command(command)
    cwd = ctx.workspace
    if args.get("cwd"):
        candidate = ctx.resolve(str(args["cwd"]))
        ctx.policy.check_write(candidate)
        if not candidate.is_dir():
            raise ToolError(f"cwd not found: {ctx.display(candidate)}")
        cwd = candidate
    timeout = _int_arg(args, "timeout", DEFAULT_TIMEOUT, minimum=1)
    argv = command_argv(command)
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out after {timeout}s: {command!r}") from None
    except OSError as exc:
        raise ToolError(f"could not start command: {exc}") from None
    stdout = _decode(completed.stdout or b"")
    stderr = _decode(completed.stderr or b"")
    parts = [f"exit_code={completed.returncode}"]
    if stdout.strip():
        parts.append(f"--- stdout ---\n{stdout.rstrip()}")
    if stderr.strip():
        parts.append(f"--- stderr ---\n{stderr.rstrip()}")
    if len(parts) == 1:
        parts.append("(no output)")
    return _truncate("\n".join(parts))


def _todo_write(args: dict[str, Any], ctx: ToolContext) -> str:
    raw = args.get("todos")
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolError(f"todos must be a JSON array: {exc}") from None
    if not isinstance(raw, list):
        raise ToolError("argument 'todos' must be an array of {content, status} objects")
    todos: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ToolError(f"todos[{index}] must be an object")
        content = str(item.get("content") or "").strip()
        if not content:
            raise ToolError(f"todos[{index}] is missing 'content'")
        status = str(item.get("status") or "pending").strip().lower()
        if status not in TODO_STATUSES:
            raise ToolError(f"todos[{index}].status must be one of {', '.join(TODO_STATUSES)}")
        todos.append({"content": content, "status": status})
    ctx.todos[:] = todos
    marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    done = sum(1 for todo in todos if todo["status"] == "completed")
    body = "\n".join(f"{marks[todo['status']]} {todo['content']}" for todo in todos)
    return f"{done}/{len(todos)} complete\n{body}"


def _str_param(description: str, *, required: bool = True) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": "string", "description": description}
    if required:
        spec["required"] = True
    return spec


def _int_param(description: str, *, required: bool = False) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": "integer", "description": description}
    if required:
        spec["required"] = True
    return spec


def default_tools() -> list[Tool]:
    """The built-in tool set, ordered as a model usually needs them."""
    return [
        Tool(
            name="read_file",
            description="Read a UTF-8/GBK/latin-1 text file, returning numbered lines.",
            parameters={
                "path": _str_param("File path, relative to the workspace."),
                "offset": _int_param("1-based first line to read (default 1)."),
                "limit": _int_param("Maximum number of lines (default 400)."),
            },
            handler=_read_file,
            aliases=("Read", "read"),
        ),
        Tool(
            name="write_file",
            description="Create or replace a text file (UTF-8, parent dirs created).",
            parameters={
                "path": _str_param("File path, relative to the workspace."),
                "content": _str_param("Full file content."),
            },
            handler=_write_file,
            requires_write=True,
            aliases=("Write", "write"),
        ),
        Tool(
            name="edit_file",
            description="Replace an exact string in a file and return a unified diff.",
            parameters={
                "path": _str_param("File path, relative to the workspace."),
                "old_string": _str_param(
                    "Exact text to replace (must be unique unless replace_all)."
                ),
                "new_string": _str_param("Replacement text."),
                "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
            },
            handler=_edit_file,
            requires_write=True,
            aliases=("Edit", "edit"),
        ),
        Tool(
            name="glob_files",
            description="List files matching a glob pattern (e.g. '**/*.py').",
            parameters={
                "pattern": _str_param("Glob pattern, '/' separated."),
                "path": _str_param("Directory to search (default workspace root).", required=False),
                "max_results": _int_param("Maximum results (default 200)."),
            },
            handler=_glob_files,
            aliases=("Glob",),
        ),
        Tool(
            name="grep_files",
            description="Search file contents with a regular expression.",
            parameters={
                "pattern": _str_param("Python regular expression."),
                "path": _str_param("File or directory to search.", required=False),
                "include": _str_param("Glob filter for file names (default '*').", required=False),
                "ignore_case": {"type": "boolean", "description": "Case-insensitive search."},
                "max_matches": _int_param("Maximum matches (default 200)."),
            },
            handler=_grep_files,
            aliases=("Grep",),
        ),
        Tool(
            name="list_dir",
            description="List a directory, one entry per line.",
            parameters={
                "path": _str_param("Directory path (default workspace root).", required=False)
            },
            handler=_list_dir,
            aliases=("LS", "ls"),
        ),
        Tool(
            name="run_command",
            description="Run a shell command in the workspace and return its output.",
            parameters={
                "command": _str_param("Command line to execute."),
                "cwd": _str_param("Working directory (default workspace root).", required=False),
                "timeout": _int_param("Timeout in seconds (default 120)."),
            },
            handler=_run_command,
            aliases=("Bash", "bash", "shell"),
        ),
        Tool(
            name="todo_write",
            description="Record the current task list for this session.",
            parameters={
                "todos": {
                    "type": "array",
                    "description": "Full list of {content, status} objects.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": list(TODO_STATUSES)},
                        },
                        "required": ["content"],
                    },
                    "required": True,
                }
            },
            handler=_todo_write,
            aliases=("TodoWrite",),
        ),
    ]


def default_registry() -> ToolRegistry:
    return ToolRegistry(default_tools())
