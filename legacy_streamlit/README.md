# legacy_streamlit — frozen reference copy of the old Streamlit frontend

This folder holds the pre-rebuild Streamlit app (`app.py`, `views/`, `.streamlit/`)
exactly as it was before the PySide6 desktop rebuild.

**It is intentionally NOT runnable.** Its imports (`optimization`, `views`,
`transforms`) point at folder locations that no longer exist — the backend moved
into `modules/` and the plugin folders were renamed. Do not fix the imports, do
not import from this folder, and do not add it to `sys.path`. It exists only as
a line-by-line reference while verifying that the PySide6 app reproduces the old
behavior, and will be deleted once the new app is verified.

The last fully working Streamlit version lives on the `main` branch (commit
468c843 and earlier) — check that out if you need to actually run it.
