"""
Archive / restore / delete for report folders — pure, Qt-free.

Safety orders (nothing is removed before its replacement is verified):

archive_report   zip to `_archive/.<final>.tmp` -> verify (zip integrity
                 test, same file count and total uncompressed size as the
                 folder) -> rename to `<report>__<timestamp>.zip` -> move the
                 report folder aside (fails fast if Windows still holds a file
                 open) -> delete it. Any failure before the final step removes
                 the partial zip and leaves the folder untouched. An
                 unreadable (e.g. locked) file aborts with a message naming it.
restore_archive  refuse when `<reports folder>/<report>` exists -> validate
                 every entry (no absolute paths, no `..`, one top folder) ->
                 extract into a hidden temporary folder in the same reports
                 folder (zipfile checks each member's CRC) -> verify count +
                 sizes -> rename into place -> only then delete the zip.
delete_path      permanent (no recycle bin) — the UI's typed code is the
                 safeguard.

Every long step reports through `on_progress(current, total, message)`; an
exception raised inside it (the UI's Cancel) unwinds through the same
cleanup as a failure.
"""

import os
import secrets
import shutil
import stat
import time
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from .scan import ARCHIVE_DIR, archive_name, is_hidden_name

# Delete-confirmation code: uppercase letters + digits minus look-alikes
# (O/0, I/1/L).
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 6


class ArchiveError(Exception):
    """A user-facing archive/restore/delete failure (message is shown as is)."""


def generate_delete_code(length: int = CODE_LENGTH) -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(length))


def _noop(*_a):
    pass


def _walk(folder: Path) -> tuple[list[Path], list[Path]]:
    """(dirs, files) under `folder`, sorted, relative paths."""
    dirs, files = [], []
    for root, dnames, fnames in os.walk(folder):
        dnames.sort()
        rel_root = Path(root).relative_to(folder)
        for d in dnames:
            dirs.append(rel_root / d)
        for f in sorted(fnames):
            files.append(rel_root / f)
    return dirs, files


def _force_remove(func, path, _exc):
    """rmtree onexc: clear the read-only bit and retry once."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, onexc=_force_remove)


# ── archive ──────────────────────────────────────────────────────────────────

def archive_report(report_dir, on_progress=None, now: datetime | None = None) -> Path:
    """Zip the report folder into its reports folder's `_archive/` and remove
    the folder. Returns the final zip path."""
    on_progress = on_progress or _noop
    report_dir = Path(report_dir)
    if not report_dir.is_dir():
        raise ArchiveError(f"Report folder not found: {report_dir}")
    name = report_dir.name
    arch = report_dir.parent / ARCHIVE_DIR
    arch.mkdir(exist_ok=True)

    when = now or datetime.now()
    final = arch / archive_name(name, when)
    n = 2
    while final.exists():
        final = arch / f"{name}__{when:%Y-%m-%d_%H%M%S}_{n}.zip"
        n += 1
    tmp = arch / f".{final.name}.tmp"

    dirs, files = _walk(report_dir)
    sizes = []
    for rel in files:
        try:
            sizes.append((report_dir / rel).stat().st_size)
        except OSError as e:
            raise ArchiveError(f"Cannot read {report_dir / rel}: {e}") from e
    total_bytes = sum(sizes)

    try:
        on_progress(0, max(total_bytes, 1), f"Archiving {name} ({len(files)} files)…")
        done = 0
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            zf.writestr(zipfile.ZipInfo(f"{name}/"), b"")
            for rel in dirs:
                zf.writestr(zipfile.ZipInfo(f"{name}/{rel.as_posix()}/"), b"")
            for rel, size in zip(files, sizes):
                src = report_dir / rel
                try:
                    zf.write(src, f"{name}/{rel.as_posix()}")
                except OSError as e:
                    raise ArchiveError(
                        f"Cannot read {src} — is it open in another program? ({e})"
                    ) from e
                done += size
                on_progress(done, max(total_bytes, 1), f"Archived {rel.as_posix()}")

        on_progress(total_bytes, max(total_bytes, 1), "Verifying the archive…")
        _verify_zip(tmp, len(files), total_bytes)

        os.rename(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    # the zip is verified and final — now the folder goes. Move it aside
    # first: on Windows a rename fails cleanly while a file inside is still
    # open, so a half-deleted folder can't happen at that point.
    aside = report_dir.parent / f".{name}.archiving_{secrets.token_hex(3)}"
    try:
        os.rename(report_dir, aside)
    except OSError as e:
        final.unlink(missing_ok=True)
        raise ArchiveError(
            f"Could not remove {report_dir} (is a file inside open in another "
            f"program?) — nothing was archived. ({e})") from e
    try:
        _rmtree(aside)
    except OSError as e:
        raise ArchiveError(
            f"Archived to {final}, but the leftover folder {aside} could not "
            f"be fully deleted: {e}") from e
    on_progress(1, 1, f"Archived to {final.name}")
    return final


def _verify_zip(path: Path, n_files: int, total_bytes: int) -> None:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ArchiveError(f"Archive verification failed: {bad} is corrupt")
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) != n_files:
            raise ArchiveError(f"Archive verification failed: {len(infos)} files "
                               f"in the zip, {n_files} in the folder")
        size = sum(i.file_size for i in infos)
        if size != total_bytes:
            raise ArchiveError(f"Archive verification failed: {size} bytes in the "
                               f"zip, {total_bytes} in the folder")


# ── restore ──────────────────────────────────────────────────────────────────

def zip_top_folder(infos) -> str:
    """The single top folder every entry lives under; raises ArchiveError on
    an unsafe entry (absolute path, drive letter, `..`) or a zip that does
    not hold exactly one top folder."""
    tops = set()
    for info in infos:
        raw = info.filename
        if "\\" in raw:
            raw = raw.replace("\\", "/")
        p = PurePosixPath(raw)
        parts = [x for x in p.parts if x not in ("", ".")]
        if raw.startswith("/") or (parts and ":" in parts[0]) or ".." in parts:
            raise ArchiveError(f"Unsafe entry in the archive: {info.filename!r}")
        if not parts:
            continue
        tops.add(parts[0])
    if len(tops) != 1:
        raise ArchiveError("The archive must contain exactly one report folder "
                           f"(found: {', '.join(sorted(tops)) or 'nothing'})")
    top = tops.pop()
    if is_hidden_name(top):
        raise ArchiveError(f"Unexpected folder name in the archive: {top!r}")
    return top


def restore_archive(zip_path, on_progress=None) -> Path:
    """Extract an archived report back into its reports folder (the parent
    of `_archive/`) and delete the zip. Returns the restored folder."""
    on_progress = on_progress or _noop
    zip_path = Path(zip_path)
    reports_folder = zip_path.parent.parent
    try:
        zf = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as e:
        raise ArchiveError(f"Cannot open {zip_path.name}: {e}") from e

    with zf:
        infos = zf.infolist()
        top = zip_top_folder(infos)
        target = reports_folder / top
        if target.exists():
            raise ArchiveError(
                f"A folder named '{top}' already exists in {reports_folder} — "
                "rename or remove it first (restore never overwrites or merges).")

        file_infos = [i for i in infos if not i.is_dir()]
        total = sum(i.file_size for i in file_infos)
        tmp = reports_folder / f".restore_{top}_{secrets.token_hex(3)}"
        tmp.mkdir()
        try:
            done = 0
            on_progress(0, max(total, 1), f"Restoring {top}…")
            tmp_resolved = tmp.resolve()
            for info in infos:
                rel = PurePosixPath(info.filename.replace("\\", "/"))
                dest = tmp.joinpath(*[x for x in rel.parts if x not in ("", ".")])
                if tmp_resolved not in dest.resolve().parents:
                    raise ArchiveError(f"Unsafe entry in the archive: {info.filename!r}")
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, 1024 * 1024)   # CRC checked at EOF
                stamp = time.mktime(info.date_time + (0, 0, -1))
                os.utime(dest, (stamp, stamp))
                done += info.file_size
                on_progress(done, max(total, 1), f"Extracted {rel.as_posix()}")

            # verify: same files, same sizes
            extracted = tmp / top
            _dirs, files = _walk(extracted)
            sizes = {f.as_posix(): (extracted / f).stat().st_size for f in files}
            expected = {}
            for info in file_infos:
                parts = [x for x in PurePosixPath(info.filename.replace("\\", "/")).parts
                         if x not in ("", ".")]
                expected["/".join(parts[1:])] = info.file_size
            if sizes != expected:
                raise ArchiveError("Restore verification failed: the extracted files "
                                   "do not match the archive")
            os.rename(extracted, target)
        except BaseException:
            _rmtree(tmp)
            raise
    _rmtree(tmp)
    zip_path.unlink()
    on_progress(1, 1, f"Restored {top}")
    return target


# ── delete ───────────────────────────────────────────────────────────────────

def delete_path(path) -> None:
    """Permanently delete a report folder or an archived zip."""
    path = Path(path)
    if path.is_dir():
        _rmtree(path)
    elif path.exists():
        try:
            path.unlink()
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()
    else:
        raise ArchiveError(f"Nothing to delete — {path} no longer exists")
