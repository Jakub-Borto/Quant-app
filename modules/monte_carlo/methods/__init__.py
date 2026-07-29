"""Monte-Carlo method plugins.

This folder is a plugin drop-zone FIRST — the picker scans it with
plugins.list_plugins, which excludes `__init__` and `base` from discovery, so
this file adds no entry to the UI and drop-a-file still works unchanged.

It exists so that `base.py` can be imported the normal absolute way
(`from modules.monte_carlo.methods.base import ...`) by pure backend code.
The plugin idiom used inside this folder — sys.path.insert + `from base
import ...` — must never spread to modules/: several plugin folders own a
`base.py` and the first one loaded wins `sys.modules['base']`
(regime_detectors/base.py documents the collision).
"""
