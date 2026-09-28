"""
Build the strategy-author guide from its template.

    python docs/strategy_guide/build_guide.py

1. STRATEGY_GUIDE.md (repo root) = GUIDE_TEMPLATE.md with
     <<FILE:relative/path.py>>  replaced by that file's exact content, and
     <<ASSET_TABLE>>            replaced by a table generated from ASSET_INFO,
   so the guide's example code is ALWAYS the tested code
   (tests/test_strategy_guide.py fails when the committed .md is stale).
2. Strategy_Guide.pdf (repo root) rendered from that Markdown with reportlab —
   a small renderer for exactly the Markdown subset the guide uses: headings
   (with internal-link anchors), paragraphs, bullet / numbered / checkbox
   lists, tables, fenced code blocks, block quotes, rules, inline code, bold,
   italics and [links](#anchor).

Edit GUIDE_TEMPLATE.md, never STRATEGY_GUIDE.md.
"""

import html
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("GUIDE_TEMPLATE.md")
OUT_MD = REPO / "STRATEGY_GUIDE.md"
OUT_PDF = REPO / "Strategy_Guide.pdf"


# ══ Markdown assembly ═════════════════════════════════════════════════════════
def _num(x) -> str:
    """Plain decimal, never scientific notation (0.0000005, 2000000)."""
    return f"{x:.10f}".rstrip("0").rstrip(".")


def asset_table() -> str:
    sys.path.insert(0, str(REPO))
    from modules.common.backend.asset_info import ASSET_INFO
    rows = ["| Ticker | tick_size | ticks_per_point | $ per tick | commission / side | parent |",
            "|---|---|---|---|---|---|"]
    for t, i in ASSET_INFO.items():
        comm = i.get("commissions_per_contract")
        rows.append(f"| {t} | {_num(i['tick_size'])} | {_num(i['ticks_per_point'])} | "
                    f"{_num(i['dollars_per_tick'])} | {'—' if comm is None else _num(comm)} | "
                    f"{i.get('parent', '')} |")
    return "\n".join(rows)


def render_markdown() -> str:
    text = TEMPLATE.read_text(encoding="utf-8")

    def _file(m):
        return (REPO / m.group(1)).read_text(encoding="utf-8").rstrip("\n")

    text = re.sub(r"<<FILE:([^>]+)>>", _file, text)
    return text.replace("<<ASSET_TABLE>>", asset_table())


# ══ PDF rendering ═════════════════════════════════════════════════════════════
def slug(title: str) -> str:
    """GitHub-style heading anchor."""
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _\-]", "", s)
    return s.replace(" ", "-")


_PDF_SUBST = {"⚠": "[!]", "✓": "OK", "↪": "->"}


_LINK = re.compile(r"\[((?:[^\[\]`]|`[^`]*`)+)\]\(([^)\s]+)\)")


def inline(text: str) -> str:
    """Markdown inline -> reportlab paragraph markup. Links are resolved first
    (their text may itself contain `code`), then code spans, bold, italics."""
    for k, v in _PDF_SUBST.items():
        text = text.replace(k, v)
    out, pos = [], 0
    for m in _LINK.finditer(text):
        out.append(_spans(text[pos:m.start()]))
        out.append(f'<a href="{html.escape(m.group(2))}" color="#1f4e9c">'
                   f'{_spans(m.group(1))}</a>')
        pos = m.end()
    out.append(_spans(text[pos:]))
    return "".join(out)


def _spans(text: str) -> str:
    """Code spans become placeholders first, so **bold** / *italic* may wrap
    around them (and a `*` inside code is never taken as emphasis)."""
    codes = []

    def _code(m):
        codes.append(f'<font face="Mono" color="#8a2b2b">{html.escape(m.group(1))}</font>')
        return f"\x00{len(codes) - 1}\x00"

    p = re.sub(r"`([^`]+)`", _code, text)
    p = html.escape(p, quote=False)
    p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
    p = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", p)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], p)


def build_pdf(md: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, Preformatted,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)
    from reportlab.platypus.flowables import HRFlowable

    fonts = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Body", str(fonts / "segoeui.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(fonts / "segoeuib.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", str(fonts / "segoeuii.ttf")))
    pdfmetrics.registerFont(TTFont("Body-BoldItalic", str(fonts / "segoeuiz.ttf"))
                            if (fonts / "segoeuiz.ttf").exists()
                            else TTFont("Body-BoldItalic", str(fonts / "segoeuib.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(fonts / "consola.ttf")))
    pdfmetrics.registerFont(TTFont("Mono-Bold", str(fonts / "consolab.ttf")))
    from reportlab.lib.fonts import addMapping
    addMapping("Body", 0, 0, "Body"); addMapping("Body", 1, 0, "Body-Bold")
    addMapping("Body", 0, 1, "Body-Italic"); addMapping("Body", 1, 1, "Body-BoldItalic")
    addMapping("Mono", 0, 0, "Mono"); addMapping("Mono", 1, 0, "Mono-Bold")
    addMapping("Mono", 0, 1, "Mono"); addMapping("Mono", 1, 1, "Mono-Bold")

    page_w, page_h = A4
    margin = 18 * mm
    frame_w = page_w - 2 * margin

    body = ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=13.4,
                          alignment=TA_LEFT, spaceAfter=5)
    small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=10.6, spaceAfter=0)
    small_b = ParagraphStyle("small_b", parent=small, fontName="Body-Bold")
    quote = ParagraphStyle("quote", parent=body, leftIndent=10, textColor=colors.HexColor("#555555"),
                           borderPadding=(4, 6, 4, 8), backColor=colors.HexColor("#fff7e0"))
    heads = {
        1: ParagraphStyle("h1", fontName="Body-Bold", fontSize=20, leading=25, spaceAfter=8,
                          textColor=colors.HexColor("#1a2b4c"), keepWithNext=1),
        2: ParagraphStyle("h2", fontName="Body-Bold", fontSize=15, leading=19, spaceBefore=8,
                          spaceAfter=6, textColor=colors.HexColor("#1f4e9c"), keepWithNext=1),
        3: ParagraphStyle("h3", fontName="Body-Bold", fontSize=11.8, leading=15, spaceBefore=8,
                          spaceAfter=4, textColor=colors.HexColor("#1a2b4c"), keepWithNext=1),
        4: ParagraphStyle("h4", fontName="Body-Bold", fontSize=10.4, leading=13.5, spaceBefore=6,
                          spaceAfter=3, textColor=colors.HexColor("#333333"), keepWithNext=1),
    }
    code_size = 7.4
    code_style = ParagraphStyle("code", fontName="Mono", fontSize=code_size, leading=9.2)
    mono_char_w = pdfmetrics.stringWidth("M", "Mono", code_size)
    code_cols = int((frame_w - 12) / mono_char_w)

    story = []
    lines = md.splitlines()
    i = 0

    def split_code(block: list[str]):
        """A code block as ONE table with a row per few lines, so it can break
        across pages anywhere between rows (one grey box per page part)."""
        wrapped = []
        for ln in block:
            ln = ln.replace("\t", "    ")
            while len(ln) > code_cols:
                cut = ln.rfind(" ", 0, code_cols)
                cut = cut if cut > code_cols * 0.5 else code_cols
                wrapped.append(ln[:cut])
                ln = "    " + ln[cut:].lstrip()
            wrapped.append(ln)
        # one row per line: a mono Paragraph with non-breaking spaces keeps
        # indentation AND blank lines (Preformatted strips blank edge lines),
        # and the table can break across pages between any two lines
        rows = [[Paragraph(html.escape(ln).replace(" ", "&nbsp;") or "&nbsp;", code_style)]
                for ln in wrapped] or [[Paragraph("&nbsp;", code_style)]]
        t = Table(rows, colWidths=[frame_w])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f5f7")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5d8de")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, 0), 4), ("BOTTOMPADDING", (0, -1), (-1, -1), 4),
        ]))
        return [t]

    def word_w(text: str, font: str) -> float:
        """Width of the widest unbreakable word of a cell (markup stripped)."""
        plain = re.sub(r"<[^>]+>", "", inline(text))
        words = plain.split() or [""]
        mono = "`" in text
        return max(pdfmetrics.stringWidth(html.unescape(w), "Mono" if mono else font, 8.2)
                   for w in words)

    def table(rows: list[str]):
        # split on UNESCAPED pipes; "\|" inside a cell is a literal pipe
        cells = [[c.strip().replace("\\|", "|")
                  for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))] for r in rows]
        header, data = cells[0], [r for r in cells[2:]]
        ncol = len(header)
        data = [(r + [""] * ncol)[:ncol] for r in data]
        # column widths: at least the widest word (no mid-word breaks, capped
        # at 38% of the page), the rest shared by content length
        lens = [max(len(re.sub(r"[`*]", "", row[c])) for row in [header] + data) for c in range(ncol)]
        mins = [min(max(word_w(row[c], "Body-Bold" if row is header else "Body")
                        for row in [header] + data) + 9, frame_w * 0.38) for c in range(ncol)]
        weights = [max(l, 4) ** 0.75 for l in lens]
        spare = frame_w - sum(mins)
        if spare > 0:
            tot = sum(weights)
            widths = [m + spare * w / tot for m, w in zip(mins, weights)]
        else:
            widths = [m * frame_w / sum(mins) for m in mins]
        tdata = [[Paragraph(inline(c), small_b) for c in header]]
        tdata += [[Paragraph(inline(c), small) for c in r] for r in data]
        t = Table(tdata, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7ecf5")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8ced8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        return t

    list_re = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            i += 1
            story.extend(split_code(block))
            story.append(Spacer(1, 5))
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            if level == 2 and title.startswith("1. "):
                story.append(PageBreak())       # the contents page stands alone
            anchor = slug(re.sub(r"[`*]", "", title))
            story.append(Paragraph(f'<a name="{anchor}"/>{inline(title)}', heads[level]))
            i += 1
            continue
        if line.strip() == "---":
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#c8ced8"),
                                    spaceBefore=4, spaceAfter=6))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            story.append(table(rows))
            story.append(Spacer(1, 6))
            continue
        if line.startswith(">"):
            parts = []
            while i < len(lines) and lines[i].startswith(">"):
                parts.append(lines[i].lstrip(">").strip()); i += 1
            story.append(Paragraph(inline(" ".join(parts)), quote))
            continue
        lm = list_re.match(line)
        if lm:
            while i < len(lines):
                lm = list_re.match(lines[i])
                if not lm:
                    break
                indent = len(lm.group(1))
                marker, content = lm.group(2), lm.group(3)
                i += 1
                # continuation lines: indented, non-empty, not a new list item
                while (i < len(lines) and lines[i].startswith(" ") and lines[i].strip()
                       and not list_re.match(lines[i])):
                    content += " " + lines[i].strip()
                    i += 1
                level = indent // 2
                if content.startswith("[ ] "):
                    bullet, content = "[  ]", content[4:]
                elif marker in "-*":
                    bullet = "•" if level == 0 else "–"
                else:
                    bullet = marker
                st = ParagraphStyle(f"li{level}", parent=body, leftIndent=16 + level * 16,
                                    bulletIndent=3 + level * 16, spaceAfter=2.5)
                story.append(Paragraph(inline(content), st, bulletText=bullet))
            story.append(Spacer(1, 3))
            continue
        if not line.strip():
            i += 1
            continue
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,4}\s|```|\||>|---$)", lines[i]) and not list_re.match(lines[i]):
            para.append(lines[i].strip()); i += 1
        story.append(Paragraph(inline(" ".join(para)), body))

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Body", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(margin, 10 * mm, "Quant Research Platform — Writing a Strategy from Scratch")
        canvas.drawRightString(page_w - margin, 10 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=margin, rightMargin=margin,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Writing a Strategy from Scratch",
                            author="Quant Research Platform")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main():
    md = render_markdown()
    OUT_MD.write_text(md, encoding="utf-8")
    build_pdf(md)
    print(f"wrote {OUT_MD.name} ({len(md.splitlines())} lines) and {OUT_PDF.name}")


if __name__ == "__main__":
    main()
