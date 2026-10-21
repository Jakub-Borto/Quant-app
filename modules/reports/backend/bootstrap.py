"""
Report-process entry point:  python -u bootstrap.py <report.py> [streamlit flags]

Runs `streamlit run <report.py> [flags]` in THIS process after two setup
steps that every report needs and none should have to write itself:

1. Imports. The repo root is APPENDED to sys.path, so `modules.reports.kit`
   and the app's pure backend packages import, while the standard library
   stays ahead of the repo root (its scratch `inspect.py` would otherwise
   shadow stdlib `inspect` — which is why PYTHONPATH, a prepend, is not
   used). This file's own folder (which Python put at sys.path[0]) is
   removed. Streamlit itself puts the report's folder at sys.path[0] while
   the script runs, so helper modules next to report.py import as usual.
2. Script-start hook (each full rerun, never a fragment rerun):
   - the default page layout is "wide" (a report's own st.set_page_config
     runs later and wins);
   - a corrupt `.report_state.json` (kit saved views) is set aside as
     `.report_state.json.bad` and ONE warning is shown at the top of the page
     for the rest of the browser session.

`--apptest <report.py>` instead runs the report once headless through
streamlit.testing (same sys.path setup) and exits 0 / 1 — the test suite
uses it to prove a launched report can import the kit and the real stdlib.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]          # modules/reports/backend -> repo root

STATE_WARNING_KEY = "_rk_state_file_warning"


def prepare_sys_path() -> None:
    if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == HERE:
        sys.path.pop(0)
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))


def _on_script_start_extras(ctx) -> None:
    if ctx.fragment_ids_this_run:            # fragment rerun: leave the page alone
        return
    import streamlit as st
    try:
        st.set_page_config(layout="wide")
    except Exception:  # noqa: BLE001 — a layout default must never break a report
        pass
    try:
        from modules.reports.kit.state import check_state_file
        report_dir = Path(ctx.main_script_path).resolve().parent
        warning = check_state_file(report_dir)
        if warning:
            st.session_state[STATE_WARNING_KEY] = warning
        if st.session_state.get(STATE_WARNING_KEY):
            st.warning(st.session_state[STATE_WARNING_KEY])
    except Exception:  # noqa: BLE001
        pass


def install_script_start_hook() -> None:
    from streamlit.runtime.scriptrunner_utils.script_run_context import (
        ScriptRunContext)
    if getattr(ScriptRunContext.on_script_start, "_rk_hooked", False):
        return
    original = ScriptRunContext.on_script_start

    def on_script_start(self):
        original(self)
        _on_script_start_extras(self)

    on_script_start._rk_hooked = True
    ScriptRunContext.on_script_start = on_script_start


def _apptest(report: str) -> int:
    from streamlit.testing.v1 import AppTest
    # what `streamlit run` does before serving (web/bootstrap._fix_sys_path);
    # AppTest alone does not
    sys.path.insert(0, str(Path(report).resolve().parent))
    at = AppTest.from_file(report, default_timeout=120).run()
    if at.exception:
        for e in at.exception:
            print("EXCEPTION:", e.message, "\n", "\n".join(e.stack_trace))
        return 1
    print("APPTEST_OK")
    return 0


def main(argv: list[str]) -> int:
    prepare_sys_path()
    install_script_start_hook()
    if len(argv) >= 2 and argv[0] == "--apptest":
        return _apptest(argv[1])
    if not argv:
        print("usage: bootstrap.py <report.py> [streamlit flags]")
        return 2
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", *argv]
    return cli.main()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
