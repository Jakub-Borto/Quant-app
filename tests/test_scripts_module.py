"""
Scripts module tests: marker detection + port allocation (pure), plus a
cheap real-process integration test for the plain-Python path (no streamlit
needed) and window-level scan/sort checks.
"""

import os
import socket
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.scripts.backend.ports import find_free_port
from modules.scripts.backend.scan import is_streamlit_script

REPO = Path(__file__).resolve().parents[1]


# ── marker detection ─────────────────────────────────────────────────────────

def _script(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_marker_comment(tmp_path):
    p = _script(tmp_path, "a.py", "# app: streamlit\nimport streamlit\n")
    assert is_streamlit_script(p)


def test_marker_constant(tmp_path):
    p = _script(tmp_path, "b.py", '"""doc"""\nSTREAMLIT = True\n')
    assert is_streamlit_script(p)


def test_marker_case_and_spacing(tmp_path):
    p = _script(tmp_path, "c.py", "   #App :  Streamlit  \n")
    assert is_streamlit_script(p)


def test_marker_past_scan_limit_ignored(tmp_path):
    p = _script(tmp_path, "d.py", "\n" * 35 + "# app: streamlit\n")
    assert not is_streamlit_script(p)


def test_no_marker(tmp_path):
    p = _script(tmp_path, "e.py", "print('hi')\n")
    assert not is_streamlit_script(p)


def test_filename_never_marks(tmp_path):
    p = _script(tmp_path, "streamlit_thing.py", "print('hi')\n")
    assert not is_streamlit_script(p)


def test_missing_file_is_plain(tmp_path):
    assert not is_streamlit_script(tmp_path / "gone.py")


# ── ports ────────────────────────────────────────────────────────────────────

def test_find_free_port_is_bindable():
    port = find_free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))     # must still be free


def test_find_free_port_respects_exclude():
    p1 = find_free_port()
    p2 = find_free_port(exclude={p1})
    assert p2 != p1


def test_list_folder_splits_and_excludes(tmp_path):
    from modules.scripts.backend.scan import folder_summary, list_folder
    (tmp_path / "sub").mkdir()
    (tmp_path / "_private").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "base.py").write_text("", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    folders, scripts = list_folder(tmp_path)
    assert [f.name for f in folders] == ["sub"]
    assert [s.name for s in scripts] == ["a.py"]
    assert folder_summary(tmp_path) == (1, 1)


def test_find_app_browser_existing_or_none():
    from modules.scripts.backend.browser import find_app_browser
    browser = find_app_browser()
    assert browser is None or (browser.exists()
                               and browser.suffix == ".exe")


def test_launch_args_use_dedicated_profile():
    from modules.scripts.backend.browser import launch_args, profile_dir
    args = launch_args("http://localhost:8501")
    assert args[0] == f"--user-data-dir={profile_dir()}"
    assert args[-1] == "http://localhost:8501"
    assert profile_dir().is_dir()      # created on demand


# ── Qt-side tests ────────────────────────────────────────────────────────────

pytest.importorskip("PySide6")


@pytest.fixture(autouse=True)
def _theme(qapp):
    from modules.common.ui.theme import apply_theme
    apply_theme(qapp)


def test_settings_scripts_category(tmp_path):
    from modules.common.backend.settings import load_settings
    s = load_settings(tmp_path / "settings.json")
    assert s.plugin_dirs("scripts")[0].name == "scripts"


def test_plain_script_runs_to_exit(qtbot, tmp_path):
    from modules.common.backend.plugins import PluginRef
    from modules.scripts.process_manager import ScriptInstance
    script = tmp_path / "hello.py"
    script.write_text("print('alpha')\nprint('beta')\n", encoding="utf-8")
    inst = ScriptInstance(PluginRef("hello", script, tmp_path, "hello"),
                          "python", 1)
    inst.start()
    qtbot.waitUntil(
        lambda: inst.state in (ScriptInstance.EXITED, ScriptInstance.CRASHED),
        timeout=30000)
    assert inst.state == ScriptInstance.EXITED, inst.log_text()
    assert "alpha" in inst.log_text() and "beta" in inst.log_text()


def test_restart_reruns_current_file(qtbot, tmp_path):
    from modules.common.backend.plugins import PluginRef
    from modules.scripts.process_manager import ScriptInstance
    script = tmp_path / "ping.py"
    script.write_text("print('ping-v1')\n", encoding="utf-8")
    inst = ScriptInstance(PluginRef("ping", script, tmp_path, "ping"),
                          "python", 1)
    inst.start()
    qtbot.waitUntil(lambda: inst.state == ScriptInstance.EXITED,
                    timeout=30000)
    script.write_text("print('ping-v2')\n", encoding="utf-8")   # edit + restart
    inst.restart()
    qtbot.waitUntil(lambda: inst.state == ScriptInstance.EXITED,
                    timeout=30000)
    text = inst.log_text()
    assert "ping-v1" in text and "ping-v2" in text
    assert "restarted" in text


def test_folder_navigation(qtbot, tmp_path):
    from modules.common.backend.settings import Settings
    from modules.scripts.window import ScriptsWindow
    root = tmp_path / "s"
    (root / "deep" / "deeper").mkdir(parents=True)
    (root / "top.py").write_text("print(1)\n", encoding="utf-8")
    (root / "deep" / "mid.py").write_text("print(1)\n", encoding="utf-8")
    (root / "deep" / "deeper" / "leaf.py").write_text(
        "print(1)\n", encoding="utf-8")
    win = ScriptsWindow(Settings({"scripts": [str(root)]}, ["data"]))
    qtbot.addWidget(win)
    # root view merges the in-repo default folder too — check membership only
    assert any(r.name == "top" for r in win._refs)
    assert any(f.name == "deep" for f in win._folders)

    win._go_to(root / "deep")
    assert [r.name for r in win._refs] == ["mid"]
    assert [f.name for f in win._folders] == ["deeper"]
    win._go_to(root / "deep" / "deeper")
    assert [r.name for r in win._refs] == ["leaf"]
    assert win._folders == []
    win._go_back()
    assert win._cwd == root / "deep"
    win._go_back()
    assert win._cwd is None       # parent of "deep" is a configured root


def test_kill_script_clears_all_instances(qtbot, tmp_path):
    from modules.common.backend.settings import Settings
    from modules.scripts.window import ScriptsWindow
    folder = tmp_path / "s"
    folder.mkdir()
    (folder / "sleeper.py").write_text(
        "import time\nprint('up')\ntime.sleep(60)\n", encoding="utf-8")
    win = ScriptsWindow(Settings({"scripts": [str(folder)]}, ["data"]))
    qtbot.addWidget(win)
    win.show()
    ref = next(r for r in win._refs if r.name == "sleeper")
    win._run(ref)
    win._run(ref)
    qtbot.waitUntil(lambda: all(i.is_alive() for i in win._instances),
                    timeout=30000)
    assert len(win._instances) == 2
    win._kill_script(ref)
    assert win._instances == []
    assert win._panel.current_instance() is None


def test_crashing_script_keeps_traceback(qtbot, tmp_path):
    from modules.common.backend.plugins import PluginRef
    from modules.scripts.process_manager import ScriptInstance
    script = tmp_path / "boom.py"
    script.write_text("raise ValueError('kaput')\n", encoding="utf-8")
    inst = ScriptInstance(PluginRef("boom", script, tmp_path, "boom"),
                          "python", 1)
    inst.start()
    qtbot.waitUntil(
        lambda: inst.state in (ScriptInstance.EXITED, ScriptInstance.CRASHED),
        timeout=30000)
    assert inst.state == ScriptInstance.CRASHED
    assert "kaput" in inst.log_text()


def test_streamlit_script_serves_http(qtbot, tmp_path):
    """Full launch of a real streamlit server (skipped if not installed)."""
    pytest.importorskip("streamlit")
    import urllib.request
    from modules.common.backend.plugins import PluginRef
    from modules.scripts.backend.ports import find_free_port
    from modules.scripts.process_manager import ScriptInstance
    script = tmp_path / "page.py"
    script.write_text("# app: streamlit\nimport streamlit as st\n"
                      "st.write('served')\n", encoding="utf-8")
    inst = ScriptInstance(PluginRef("page", script, tmp_path, "page"),
                          "streamlit", 1, port=find_free_port())
    inst.start()
    try:
        qtbot.waitUntil(lambda: inst.state != ScriptInstance.STARTING,
                        timeout=60000)
        assert inst.state == ScriptInstance.RUNNING, inst.log_text()
        with urllib.request.urlopen(inst.url, timeout=10) as resp:
            assert resp.status == 200
    finally:
        inst.stop()
        inst.wait_finished(5000)
    qtbot.waitUntil(lambda: inst.state == ScriptInstance.EXITED,
                    timeout=10000)


def test_window_scan_sort_and_exclusions(qtbot, tmp_path):
    from modules.common.backend.settings import Settings
    from modules.scripts.window import ScriptsWindow
    extra = tmp_path / "extra_scripts"
    extra.mkdir()
    old = extra / "a_old.py"                 # alphabetically first, oldest
    old.write_text("print('old')\n", encoding="utf-8")
    new = extra / "z_new.py"                 # alphabetically last, newest
    new.write_text("# app: streamlit\nimport streamlit as st\n",
                   encoding="utf-8")
    (extra / "base.py").write_text("print('excluded')\n", encoding="utf-8")
    os.utime(old, (1_000_000_000, 1_000_000_000))
    os.utime(new, (2_000_000_000, 2_000_000_000))

    win = ScriptsWindow(Settings({"scripts": [str(extra)]}, ["data"]))
    qtbot.addWidget(win)
    win.show()

    names = [r.name for r in win._refs]
    assert "base" not in names
    assert names.index("z_new") < names.index("a_old")   # mtime desc default
    assert not win._instances                            # nothing spawned

    win._sort.setCurrentIndex(1)                         # Name A–Z
    names = [r.name for r in win._refs if r.name in ("a_old", "z_new")]
    assert names == ["a_old", "z_new"]
