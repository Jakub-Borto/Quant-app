"""
How a report process is started — pure, Qt-free (the window's ReportInstance
hands these arguments to QProcess).

    python -u modules/reports/backend/bootstrap.py <report>/report.py
           --server.port N --server.headless true --server.runOnSave true
           --browser.gatherUsageStats false --theme.* (the shared theme)

cwd = the report folder. The bootstrap (not PYTHONPATH, which PREPENDS and
would let the repo root's scratch `inspect.py` shadow the standard library)
APPENDS the repo root to sys.path, then runs Streamlit in-process. Theme
options are passed on the command line, which beats any `.streamlit/
config.toml` a report folder might contain.
"""

from pathlib import Path

from modules.common.backend import palette

BOOTSTRAP = Path(__file__).resolve().parent / "bootstrap.py"


def theme_options() -> dict[str, str]:
    """The ONE Streamlit theme every report gets: the app's dark palette and
    fonts (modules/common/backend/palette.py)."""
    font = palette.FONT_FAMILY.replace('"', "")
    return {
        "theme.base": "dark",
        "theme.primaryColor": palette.ACCENT_HOVER,
        "theme.backgroundColor": palette.BG,
        "theme.secondaryBackgroundColor": palette.SURFACE,
        "theme.textColor": palette.TEXT,
        "theme.borderColor": palette.BORDER,
        "theme.linkColor": palette.ACCENT_SOFT,
        "theme.font": font,
        "theme.headingFont": font,
        "theme.codeFont": palette.MONO_FAMILY,
        "theme.baseRadius": "medium",
    }


def streamlit_flags(port: int) -> list[str]:
    flags = ["--server.port", str(port),
             "--server.headless", "true",
             "--server.runOnSave", "true",       # saving a file reruns the report
             "--browser.gatherUsageStats", "false"]
    for key, value in theme_options().items():
        flags += [f"--{key}", value]
    return flags


def launch_args(report_py, port: int) -> list[str]:
    """Interpreter arguments for one report process (program = the app's
    own sys.executable; cwd = the report folder)."""
    return ["-u", str(BOOTSTRAP), str(report_py), *streamlit_flags(port)]
