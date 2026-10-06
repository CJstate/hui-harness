"""Permission policy and sandbox rules.

Three modes, matching what reviewers expect from a modern coding agent:

``read-only``
    Reads anywhere, but every write and every shell command is refused.
``workspace-write`` (default)
    Writes are confined to the workspace (plus explicitly allowed roots, e.g.
    the system temp directory). Shell commands run, but a small set of
    genuinely destructive patterns needs an explicit approval hook.
``danger-full-access``
    No restrictions — for throwaway containers.

The policy is a plain object so it can be unit-tested without touching a shell.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hui.common import PolicyError

READ_ONLY = "read-only"
WORKSPACE_WRITE = "workspace-write"
DANGER = "danger-full-access"
MODES: tuple[str, ...] = (READ_ONLY, WORKSPACE_WRITE, DANGER)

#: Patterns that are never run without an explicit approval, even when the user
#: chose ``workspace-write``. Each entry is ``(regex, human description)``.
DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"(?<![\w-])rm\s+(?:-[A-Za-z]+\s+)*(?:/|~|\$HOME|\*)",
        "recursive delete outside the workspace",
    ),
    (r"(?<![\w-])(?:rd|rmdir)\s+/s\b", "recursive directory delete"),
    (r"(?<![\w-])del\s+/[sfq]\b", "forced file delete"),
    (r"\b(?:mkfs|fdisk|diskpart)\b", "filesystem formatting"),
    (r"\bdd\s+if=", "raw disk write"),
    (r">\s*/dev/sd", "raw disk write"),
    (r"\b(?:shutdown|reboot|halt|poweroff)\b", "power state change"),
    (r"\bformat\s+[A-Za-z]:", "drive format"),
    (r"\breg\s+delete\b", "registry deletion"),
    (r"Remove-Item\s+.*-(?:Recurse|Force).*[A-Za-z]:\\", "recursive delete of a drive path"),
    (r"\b(?:sudo|runas)\b", "privilege escalation"),
    (r"\bgit\s+push\b.*--force", "force push"),
    (r"\bgit\s+reset\s+--hard\b", "hard reset"),
    (
        r"(?:curl|wget|iwr|Invoke-WebRequest)\b[^|;]*\|\s*(?:ba|z|fi)?sh\b",
        "pipe a download into a shell",
    ),
    (r"\bchmod\s+-R\s+777\b", "world-writable recursive chmod"),
    (r"\b(?:npm|pip|pnpm|yarn)\s+(?:publish|unpublish)\b", "package publish"),
    (r"\bgh\s+repo\s+delete\b", "repository deletion"),
    (r"\bdrop\s+(?:database|table)\b", "destructive SQL"),
)

_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), description) for pattern, description in DANGEROUS_PATTERNS
)


def _normal(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


@dataclass(slots=True)
class Policy:
    workspace: Path
    mode: str = WORKSPACE_WRITE
    extra_writable: tuple[Path, ...] = ()
    approver: Callable[[str, str], bool] | None = None
    approved: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        if self.mode not in MODES:
            raise PolicyError(f"unknown mode {self.mode!r}; expected one of {', '.join(MODES)}")
        self.extra_writable = tuple(Path(root) for root in self.extra_writable)

    # -- paths -------------------------------------------------------------
    def roots(self) -> tuple[Path, ...]:
        return (self.workspace, *self.extra_writable)

    def is_writable(self, path: str | os.PathLike[str]) -> bool:
        target = _normal(path)
        for root in self.roots():
            root_norm = _normal(root)
            if target == root_norm or target.startswith(root_norm + os.sep):
                return True
        return False

    def check_read(self, path: str | os.PathLike[str]) -> Path:
        return Path(path)

    def check_write(self, path: str | os.PathLike[str]) -> Path:
        if self.mode == READ_ONLY:
            raise PolicyError(f"mode read-only: refusing to write {path}")
        if self.mode == DANGER:
            return Path(path)
        if not self.is_writable(path):
            raise PolicyError(
                f"mode workspace-write: {path} is outside the workspace {self.workspace}"
            )
        return Path(path)

    # -- commands ----------------------------------------------------------
    @staticmethod
    def risk(command: str) -> str | None:
        for pattern, description in _COMPILED:
            if pattern.search(command):
                return description
        return None

    def check_command(self, command: str) -> str:
        text = (command or "").strip()
        if not text:
            raise PolicyError("empty command")
        if self.mode == READ_ONLY:
            raise PolicyError(f"mode read-only: refusing to run {text!r}")
        if self.mode == DANGER:
            return text
        risk = self.risk(text)
        if risk and text not in self.approved:
            if self.approver is None:
                raise PolicyError(
                    f"refusing a command that would {risk}: {text!r} (no approval hook)"
                )
            if not self.approver(text, risk):
                raise PolicyError(f"command denied by the operator ({risk}): {text!r}")
            self.approved.add(text)
        return text

    def describe(self) -> str:
        roots = ", ".join(str(root) for root in self.roots())
        return f"mode={self.mode} workspace={roots}"


def policy_from_env(workspace: str | os.PathLike[str], mode: str | None = None) -> Policy:
    chosen = (mode or os.environ.get("HUI_MODE") or WORKSPACE_WRITE).strip()
    extra: list[Path] = []
    for key in ("TEMP", "TMP"):
        value = os.environ.get(key)
        if value:
            extra.append(Path(value))
    return Policy(workspace=Path(workspace), mode=chosen, extra_writable=tuple(extra))


def command_argv(command: str, *, shell: Sequence[str] | None = None) -> list[str]:
    """Wrap ``command`` for the platform shell without ever using ``shell=True``."""
    if shell:
        return [*shell, command]
    if os.name == "nt":
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
    return ["/bin/sh", "-c", command]
