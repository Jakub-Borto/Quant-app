"""
Reports window (offscreen Qt): discovery listing, Archive/Delete disabled
while a report runs, the typed-code delete dialog, the Archive tab's
restore, and the Compress tab.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QPushButton  # noqa: E402

from modules.reports.backend.archive import archive_report  # noqa: E402
from modules.reports.backend.scan import list_archives  # noqa: E402


@pytest.fixture(autouse=True)
def _theme(qapp):
    from modules.common.ui.theme import apply_theme
    apply_theme(qapp)


@pytest.fixture()
def win(qtbot, tmp_path):
    from modules.common.backend.settings import Settings
    from modules.reports.window import ReportsWindow
    extra = tmp_path / "my_reports"
    for name in ("alpha", "beta"):
        (extra / name).mkdir(parents=True)
        (extra / name / "report.py").write_text("import streamlit\n")
    (extra / "_skip").mkdir()
    (extra / "_skip" / "report.py").write_text("")
    w = ReportsWindow(Settings({"reports": [str(extra)]}, ["data"]))
    qtbot.addWidget(w)
    w.extra = extra
    return w


def _card(win, name):
    lay = win._report_rows
    for i in range(lay.count()):
        w = lay.itemAt(i).widget()
        if w is not None and any(lbl.text() == name for lbl in w.findChildren(QLabel)):
            return w
    return None


def _button(widget, text) -> QPushButton:
    (btn,) = [b for b in widget.findChildren(QPushButton) if b.text() == text]
    return btn


def test_lists_reports_from_extra_folders(win):
    assert _card(win, "alpha") is not None and _card(win, "beta") is not None
    assert _card(win, "_skip") is None
    assert win.folders()[0].name == "reports"           # in-repo default first


class _FakeInstance:
    state = "running"
    port = 51234
    label = "alpha #1"
    url = "http://localhost:51234"

    def is_alive(self):
        return True


def test_archive_and_delete_disabled_while_running(win):
    win._instances[win.extra / "alpha"] = _FakeInstance()
    try:
        win.refresh(force=True)
        alpha, beta = _card(win, "alpha"), _card(win, "beta")
        for text in ("Archive", "Delete", "Run"):
            assert not _button(alpha, text).isEnabled()
            assert _button(beta, text).isEnabled()
        assert "Stop the report first" in _button(alpha, "Archive").toolTip()
    finally:
        win._instances.clear()


def test_delete_dialog_requires_exact_code(qtbot):
    from modules.reports.delete_dialog import DeleteConfirmDialog
    dlg = DeleteConfirmDialog("this report folder", "C:/x/rep")
    qtbot.addWidget(dlg)
    assert not dlg.delete_btn.isEnabled()
    dlg.field.setText(dlg.code.lower())                 # case-sensitive
    assert not dlg.delete_btn.isEnabled()
    dlg.field.setText(dlg.code[:-1])
    assert not dlg.delete_btn.isEnabled()
    dlg.field.setText(dlg.code)
    assert dlg.delete_btn.isEnabled()
    codes = set()
    for _ in range(20):                                 # a new code every opening
        d = DeleteConfirmDialog("x", "y")
        qtbot.addWidget(d)
        codes.add(d.code)
    assert len(codes) > 15


def test_archive_tab_restore(win, qtbot):
    z = archive_report(win.extra / "beta")
    win.tabs.setCurrentIndex(1)
    texts = [b.text() for b in win.findChildren(QPushButton) if b.isVisibleTo(win)]
    assert "Restore" in texts and "Open archive folder" in texts
    (ref,) = [r for r in list_archives([win.extra]) if r.path == z]
    win._restore(ref)
    qtbot.waitUntil(lambda: (win.extra / "beta").exists() and not z.exists(),
                    timeout=10000)


def test_compress_tab_constructs(win):
    win.tabs.setCurrentIndex(2)
    assert win._source.count() >= 1
    assert not win._compress_btn.isEnabled()             # no runs -> nothing to do
