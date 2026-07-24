"""
Optional pull-style helper for regime-detector scripts. EXCLUDED from plugin
discovery (like every base.py). A script that wants to do everything itself
simply doesn't import this.

IMPORT MECHANISM (the section-5/13 decision from the build plan): detector
scripts import the helper DIRECTLY from the app package —

    from modules.regime_detector.backend.runner import (RegimeContext,
                                                        SHARED_PARAMS)
    from modules.regime_detector.backend.schema import UNKNOWN_STATE

— which works from any configured detector folder because plugins load
in-process (the app puts the repo root on sys.path).

Do NOT use the `sys.path.insert(own folder); from base import ...` idiom
here even though other plugin folders use it: several plugin folders own a
base.py (this one, modules/monte_carlo/methods/base.py, ...), the first one
imported anywhere wins sys.modules['base'], and every later `from base
import` silently resolves to the WRONG file. This base.py stays only as the
documented home of that decision, a convenience re-export for interactive
use, and the discovery-excluded slot the plugin scanner expects.

Loader gotcha that WILL bite: never combine `from __future__ import
annotations` with a @dataclass in a detector script — under the
exec-without-registration loader on Python 3.13 it crashes at UI load time
(dataclasses._is_type can't resolve the module). Plain classes or plain
dataclasses without the future import are both fine.

STATE CONVENTION: declare the FULL state set in REGIME_STATES up front (it
keeps chart colours stable across loaded ranges). 'unknown' is implicit —
every regime column may emit it for warm-up/missing data (never a silently
wrong number, never a real regime); do not declare it.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modules.regime_detector.backend.runner import (  # noqa: E402,F401
    SHARED_PARAMS, RegimeContext)
from modules.regime_detector.backend.schema import (  # noqa: E402,F401
    UNKNOWN_STATE)
