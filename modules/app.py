"""
QApplication bootstrap: theme, settings, main menu, event loop.

Called from main.py (inside its __main__ guard — see the spawn-safety note
there).
"""

import sys

from PySide6.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Quant Research Platform")
    app.setOrganizationName("QuantApp")

    from modules.common.backend.settings import load_settings
    from modules.common.ui.theme import apply_theme
    from modules.main_menu.window import MainMenuWindow

    apply_theme(app)
    settings = load_settings()

    menu = MainMenuWindow(settings)
    menu.show()

    return app.exec()
