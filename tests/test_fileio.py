import locale

import pytest

from hui import fileio


def test_read_text_safe_strips_utf8_bom(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfhello\n")
    text, encoding = fileio.read_text_safe(path)
    assert text == "hello\n"
    assert encoding == "utf-8-sig"


def _locale_sample(encoding: str) -> str:
    """Non-ASCII text that the machine's own code page can actually represent.

    A US Windows runner is cp1252, where CJK fixture text cannot be encoded at
    all: the test would fail while building the fixture, before the library was
    ever called.
    """
    for sample in ("中文批注 · 编码", "café · résumé"):
        try:
            sample.encode(encoding)
        except UnicodeEncodeError:
            continue
        return sample
    raise AssertionError(f"no sample text is encodable as {encoding}")


def test_read_text_safe_uses_locale_encoding_for_local_files(tmp_path):
    """A file saved by a local editor is in the machine's code page, not UTF-8."""
    path = tmp_path / "gbk.txt"
    preferred = locale.getpreferredencoding(False)
    sample = _locale_sample(preferred)
    path.write_bytes(sample.encode(preferred))
    text, encoding = fileio.read_text_safe(path)
    assert text == sample
    assert encoding != "latin-1"


def test_read_text_safe_falls_back_to_a_us_windows_code_page(tmp_path, monkeypatch):
    """Reproduces the cp1252 CI runner: not UTF-8, and the locale is not CJK."""
    monkeypatch.setattr(fileio.locale, "getpreferredencoding", lambda *a, **k: "cp1252")
    path = tmp_path / "latin1-notes.txt"
    path.write_bytes("café · résumé".encode("cp1252"))
    text, encoding = fileio.read_text_safe(path)
    assert text == "café · résumé"
    assert encoding.lower() == "cp1252"


def test_candidate_encodings_always_ends_with_latin1():
    encodings = fileio.candidate_encodings()
    assert encodings[0] == "utf-8-sig"
    assert encodings[-1] == "latin-1"


def test_write_text_safe_creates_parents_and_counts_bytes(tmp_path):
    path = tmp_path / "a" / "b" / "c.txt"
    written = fileio.write_text_safe(path, "héllo\n")
    assert path.read_bytes() == "héllo\n".encode()
    assert written == len("héllo\n".encode())


def test_write_text_safe_forced_newline(tmp_path):
    path = tmp_path / "crlf.txt"
    fileio.write_text_safe(path, "a\nb\n", newline="\r\n")
    assert path.read_bytes() == b"a\r\nb\r\n"


def test_write_text_safe_bom(tmp_path):
    path = tmp_path / "bom.txt"
    fileio.write_text_safe(path, "x", bom=True)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a\nb\n", "\n"),
        ("a\r\nb\r\n", "\r\n"),
        ("a\r\nb\n", "\n"),
        ("a\r\nb\r\nc\n", "\r\n"),
        ("", "\n"),
    ],
)
def test_detect_newline(text, expected):
    assert fileio.detect_newline(text) == expected


def test_to_newline_normalises_mixed_endings():
    assert fileio.to_newline("a\r\nb\rc\n", "\n") == "a\nb\nc\n"


def test_line_numbered_offsets_and_trailer():
    rendered = fileio.line_numbered("a\nb\nc\nd\n", offset=2, limit=2)
    assert rendered.splitlines() == ["2| b", "3| c", "   … 1 more lines (offset=4)"]


def test_line_numbered_clips_absurdly_long_lines():
    rendered = fileio.line_numbered("x" * 2500, max_line=100)
    assert "… [+2400 chars on this line]" in rendered
    assert len(rendered.splitlines()[0]) < 200


def test_unified_diff_marks_changed_lines():
    diff = fileio.unified_diff("a\nb\n", "a\nc\n", "demo.txt")
    assert "-b" in diff and "+c" in diff
    assert diff.startswith("--- a/demo.txt")


def test_count_lines_ignores_trailing_newline():
    assert fileio.count_lines("a\nb\n") == 2
