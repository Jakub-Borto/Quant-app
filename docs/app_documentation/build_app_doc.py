"""
Build Quant_app_documentation.pdf (repo root) from APP_DOCUMENTATION.md.

    python docs/app_documentation/build_app_doc.py

<<ASSET_TABLE>> is replaced by a table generated from ASSET_INFO; the PDF is
rendered with the strategy guide's reportlab renderer
(docs/strategy_guide/build_guide.py — same Markdown subset). Edit the .md,
never the PDF.
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE / "APP_DOCUMENTATION.md"
OUT_PDF = REPO / "Quant_app_documentation.pdf"


def _guide_builder():
    path = REPO / "docs" / "strategy_guide" / "build_guide.py"
    spec = importlib.util.spec_from_file_location("build_guide", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_markdown() -> str:
    guide = _guide_builder()
    return SOURCE.read_text(encoding="utf-8").replace("<<ASSET_TABLE>>", guide.asset_table())


def main():
    guide = _guide_builder()
    md = render_markdown()
    guide.build_pdf(md, out_pdf=OUT_PDF, doc_title="Platform Documentation")
    print(f"wrote {OUT_PDF.name} from {SOURCE.name}")


if __name__ == "__main__":
    main()
