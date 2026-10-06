"""Encoding- and newline-safe text I/O.

This is the part of HUI that refuses to fall over on a real developer machine.
A file written by a Chinese Windows editor is GBK, a file exported from CI is
UTF-8 with a BOM, and a checkout on Windows has CRLF endings. A coding agent
that crashes on any of those is useless, so every read goes through
:func:`read_text_safe`, which reports the encoding it actually used.
"""

from __future__ import annotations

import difflib
import locale
import os
from pathlib import Path

BOM_UTF8 = "\ufeff"
DEFAULT_MAX_LINE = 2000


def candidate_encodings() -> tuple[str, ...]:
    """UTF-8 first, then the platform's own encoding, then a lossless fallback."""
    preferred = locale.getpreferredencoding(False) or "utf-8"
    encodings = ["utf-8-sig"]
    if preferred.lower().replace("-", "") not in {"utf8", "utf8sig"}:
        encodings.append(preferred)
    encodings.append("latin-1")  # never raises; keeps bytes recoverable
    return tuple(encodings)


def read_text_safe(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Read ``path`` as UTF-8, falling back to the locale encoding.

    Returns ``(text, encoding)``. The BOM is stripped when present; elsewhere
    the bytes are preserved as-is so that an edit cannot silently rewrite a
    whole file's encoding.
    """
    target = Path(path)
    data = target.read_bytes()
    for encoding in candidate_encodings():
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text, encoding
    return data.decode("latin-1"), "latin-1"


def detect_newline(text: str) -> str:
    """Return the dominant newline of ``text`` (defaults to ``\\n``)."""
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def to_newline(text: str, newline: str) -> str:
    """Normalise every line ending in ``text`` to ``newline``."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)


def write_text_safe(
    path: str | os.PathLike[str],
    text: str,
    *,
    newline: str | None = None,
    bom: bool = False,
) -> int:
    """Write ``text`` as UTF-8, creating parent directories as needed.

    ``newline`` forces a line-ending style; ``None`` writes the bytes exactly
    as given (``newline=""`` in :meth:`Path.write_text` semantics). Returns the
    number of bytes written.
    """
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    payload = to_newline(text, newline) if newline else text
    if bom:
        payload = BOM_UTF8 + payload
    with open(target, "w", encoding="utf-8", newline="") as handle:
        handle.write(payload)
    return len(payload.encode("utf-8"))


def line_numbered(
    text: str,
    *,
    offset: int = 1,
    limit: int | None = None,
    max_line: int = DEFAULT_MAX_LINE,
) -> str:
    """Render ``text`` as ``   12| code`` lines, 1-based and bounded."""
    lines = text.splitlines()
    start = max(offset, 1)
    end = len(lines) if limit is None else min(start - 1 + limit, len(lines))
    width = len(str(max(end, 1)))
    rendered: list[str] = []
    for number in range(start, end + 1):
        line = lines[number - 1]
        if len(line) > max_line:
            line = line[:max_line] + f" … [+{len(line) - max_line} chars on this line]"
        rendered.append(f"{number:>{width}}| {line}")
    if end < len(lines):
        rendered.append(f"   … {len(lines) - end} more lines (offset={end + 1})")
    return "\n".join(rendered)


def unified_diff(old: str, new: str, path: str = "file", context: int = 3) -> str:
    """A compact unified diff for edit previews and tool receipts."""
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
        n=context,
    )
    return "\n".join(diff)


def count_lines(text: str) -> int:
    return len(text.splitlines())
