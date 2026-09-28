"""
STRATEGY_GUIDE.md must be exactly what docs/strategy_guide/build_guide.py
renders from the template — so its example code is always the tested code in
strategies/ (tests/test_strategy_examples.py). If this fails, run:

    python docs/strategy_guide/build_guide.py
"""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _builder():
    path = REPO / "docs" / "strategy_guide" / "build_guide.py"
    spec = importlib.util.spec_from_file_location("build_guide", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_guide_is_up_to_date():
    rendered = _builder().render_markdown()
    committed = (REPO / "STRATEGY_GUIDE.md").read_text(encoding="utf-8")
    assert committed == rendered, ("STRATEGY_GUIDE.md is stale — run "
                                   "python docs/strategy_guide/build_guide.py")


def test_guide_contains_the_example_files_verbatim():
    guide = (REPO / "STRATEGY_GUIDE.md").read_text(encoding="utf-8")
    examples = [REPO / "strategies" / "example_first_hour_breakout.py",
                *sorted((REPO / "strategies" / "example_prev_day_levels").glob("*.py"))]
    for f in examples:
        assert f.read_text(encoding="utf-8").rstrip("\n") in guide, f.name
    assert "<<FILE:" not in guide and "<<ASSET_TABLE>>" not in guide


def test_pdf_exists_next_to_the_guide():
    pdf = REPO / "Strategy_Guide.pdf"
    assert pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-"
