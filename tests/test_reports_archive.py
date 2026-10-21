"""
Reports — discovery, archive / restore / delete, and the delete code.
"""

import msvcrt
import os
import sys
import zipfile
from datetime import datetime

import pytest

from _reports_util import tree_hashes
from modules.reports.backend import archive as arch
from modules.reports.backend.scan import (list_archives, list_reports,
                                          owning_reports_folder,
                                          parse_archive_name)


def _report(folder, name, files=None):
    d = folder / name
    d.mkdir(parents=True)
    (d / "report.py").write_text("import streamlit as st\nst.write('hi')\n")
    for rel, data in (files or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return d


# ── discovery ────────────────────────────────────────────────────────────────

def test_discovery_rules(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _report(a, "alpha")
    _report(a, "_draft")
    _report(a, ".hidden")
    (a / "no_entry").mkdir()
    (a / "no_entry" / "main.py").write_text("")
    (a / "README.md").write_text("not a report")
    _report(a, "_archive")
    _report(b, "alpha")                         # same name in a second folder
    _report(b, "beta", {"data/x.bin": b"1"})
    refs = list_reports([a, b, tmp_path / "missing"])
    assert [(r.name, r.folder.name) for r in refs] == [
        ("alpha", "a"), ("alpha", "b"), ("beta", "b")]
    assert all(r.entry.name == "report.py" for r in refs)


def test_last_modified_is_newest_file(tmp_path):
    d = _report(tmp_path, "r", {"data/deep/new.txt": b"x"})
    old = datetime(2020, 1, 1).timestamp()
    os.utime(d / "report.py", (old, old))
    new = datetime(2030, 1, 1).timestamp()
    os.utime(d / "data" / "deep" / "new.txt", (new, new))
    (ref,) = list_reports([tmp_path])
    assert ref.modified == datetime.fromtimestamp(new)


def test_owning_folder(tmp_path):
    root = tmp_path / "reports"
    (root / "x").mkdir(parents=True)
    assert owning_reports_folder(root / "x", [root]) == root
    assert owning_reports_folder(root / "x" / ".." / "..", [root]) is None
    assert owning_reports_folder(tmp_path, [root]) is None


# ── archive ──────────────────────────────────────────────────────────────────

def test_archive_then_restore_is_byte_identical(tmp_path):
    files = {"data/a.parquet": os.urandom(5000), "notes.md": b"# hi\n",
             "sub/deeper/b.bin": b"\x00" * 100, ".report_state.json": b"{}"}
    d = _report(tmp_path, "rep", files)
    (d / "empty_dir").mkdir()
    before = tree_hashes(d)

    z = arch.archive_report(d, now=datetime(2026, 10, 21, 9, 30, 5))
    assert z == tmp_path / "_archive" / "rep__2026-10-21_093005.zip"
    assert not d.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]
    with zipfile.ZipFile(z) as zf:
        assert all(n.startswith("rep/") for n in zf.namelist())
        assert zf.testzip() is None
    (aref,) = list_archives([tmp_path])
    assert aref.name == "rep" and aref.archived_at == datetime(2026, 10, 21, 9, 30, 5)
    assert list_reports([tmp_path]) == []       # _archive is not a report

    restored = arch.restore_archive(z)
    assert restored == d
    assert tree_hashes(d) == before             # saved views travel too
    assert (d / "empty_dir").is_dir()
    assert not z.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_rearchive_same_second_does_not_collide(tmp_path):
    when = datetime(2026, 1, 1, 12, 0, 0)
    z1 = arch.archive_report(_report(tmp_path, "rep"), now=when)
    z2 = arch.archive_report(_report(tmp_path, "rep"), now=when)
    assert z1 != z2 and z1.exists() and z2.exists()
    assert parse_archive_name(z2.name) == ("rep", when)


@pytest.mark.skipif(sys.platform != "win32", reason="byte-range locks are Windows-only")
def test_locked_file_aborts_and_leaves_folder(tmp_path):
    d = _report(tmp_path, "rep", {"data/locked.bin": b"x" * 4096})
    before = tree_hashes(d)
    fh = open(d / "data" / "locked.bin", "r+b")
    try:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 4096)
        with pytest.raises(arch.ArchiveError, match="locked.bin"):
            arch.archive_report(d)
    finally:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 4096)
        fh.close()
    assert tree_hashes(d) == before
    assert list((tmp_path / "_archive").iterdir()) == []   # no zip, no temp file


def test_cancelled_archive_leaves_nothing(tmp_path):
    d = _report(tmp_path, "rep", {"a.bin": b"1" * 100, "b.bin": b"2" * 100})

    def stop(cur, total, msg):
        if msg.startswith("Archived a.bin"):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        arch.archive_report(d, on_progress=stop)
    assert d.exists() and list((tmp_path / "_archive").iterdir()) == []


# ── restore guards ───────────────────────────────────────────────────────────

def test_restore_refuses_existing_folder(tmp_path):
    z = arch.archive_report(_report(tmp_path, "rep", {"x.txt": b"old"}))
    _report(tmp_path, "rep", {"x.txt": b"new"})
    with pytest.raises(arch.ArchiveError, match="already exists"):
        arch.restore_archive(z)
    assert z.exists()
    assert (tmp_path / "rep" / "x.txt").read_bytes() == b"new"


@pytest.mark.parametrize("bad", ["rep/../evil.txt", "/abs/evil.txt",
                                 "C:/evil.txt", "rep\\..\\evil.txt"])
def test_restore_rejects_escaping_entries(tmp_path, bad):
    (tmp_path / "_archive").mkdir()
    z = tmp_path / "_archive" / "rep__2026-01-01_000000.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("rep/report.py", "x")
        zf.writestr(bad, "evil")
    with pytest.raises(arch.ArchiveError, match="Unsafe|exactly one"):
        arch.restore_archive(z)
    assert z.exists()
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "rep").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_restore_corrupt_zip_keeps_zip(tmp_path):
    z = arch.archive_report(_report(tmp_path, "rep", {"a.bin": os.urandom(4000)}))
    data = bytearray(z.read_bytes())
    i = data.find(b"a.bin") + 40                # inside the first file's data
    data[i:i + 8] = b"\xff" * 8
    z.write_bytes(bytes(data))
    with pytest.raises(Exception):
        arch.restore_archive(z)
    assert z.exists() and not (tmp_path / "rep").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


# ── delete ───────────────────────────────────────────────────────────────────

def test_delete_folder_and_zip(tmp_path):
    d = _report(tmp_path, "rep", {"ro.txt": b"x"})
    os.chmod(d / "ro.txt", 0o444)                # read-only files go too
    z = arch.archive_report(_report(tmp_path, "other"))
    arch.delete_path(d)
    arch.delete_path(z)
    assert not d.exists() and not z.exists()
    with pytest.raises(arch.ArchiveError):
        arch.delete_path(d)


def test_delete_code_alphabet_and_freshness():
    codes = {arch.generate_delete_code() for _ in range(200)}
    assert len(codes) > 190                      # fresh every time
    for c in codes:
        assert len(c) == 6
        assert not set(c) & set("O0I1L")
        assert set(c) <= set(arch.CODE_ALPHABET)
