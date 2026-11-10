"""High-fidelity PDF generator for Indian legal documents and court applications.

Renders publication-ready A4 PDFs adhering strictly to Indian court and legal
standards: standard filing margins (left margin clearance for docket/thread
binding), standard court typography (Times Roman serif, 1.4-1.5 leading),
cause title centering, memo of parties formatting, hanging paragraph indents,
running headers, pagination ('Page X of Y'), and statutory advocate review notices.
"""
from __future__ import annotations

import os
import re
from typing import Any
import pymupdf

A4_WIDTH = 595.32
A4_HEIGHT = 841.92

# Standard court filing margins in points
MARGIN_LEFT = 80.0    # 1.11 inches: clearance for thread/tag binding
MARGIN_RIGHT = 50.0   # 0.69 inches
MARGIN_TOP = 60.0
MARGIN_BOTTOM = 60.0
CONTENT_WIDTH = A4_WIDTH - MARGIN_LEFT - MARGIN_RIGHT

FONT_SIZE_BODY = 10.5
LEADING_BODY = 15.0
FONT_SIZE_HEADING = 11.5
LEADING_HEADING = 16.0

INDIC_RANGES: dict[str, tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}

FONT_CANDIDATES: dict[str, list[str]] = {
    "devanagari": [
        "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
        "/System/Library/Fonts/MuktaMahee.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    ],
    "bengali": [
        "/System/Library/Fonts/KohinoorBangla.ttc",
        "/System/Library/Fonts/Supplemental/NotoSansBengali-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    ],
    "telugu": [
        "/System/Library/Fonts/Supplemental/Telugu Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/Telugu MN.ttc",
        "/System/Library/Fonts/KohinoorTelugu.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
    ],
    "tamil": [
        "/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/Tamil MN.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
    ],
    "kannada": [
        "/System/Library/Fonts/Supplemental/Kannada Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/Kannada MN.ttc",
        "/System/Library/Fonts/NotoSansKannada.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
    ],
    "malayalam": [
        "/System/Library/Fonts/Supplemental/Malayalam Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/Malayalam MN.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansMalayalam-Regular.ttf",
    ],
    "gujarati": [
        "/System/Library/Fonts/Supplemental/Gujarati Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/GujaratiMT.ttc",
        "/System/Library/Fonts/KohinoorGujarati.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",
    ],
    "gurmukhi": [
        "/System/Library/Fonts/Supplemental/Gurmukhi Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/Gurmukhi MN.ttc",
        "/System/Library/Fonts/Supplemental/Gurmukhi.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansGurmukhi-Regular.ttf",
    ],
}


def _detect_indic_font(text: str) -> tuple[str | None, str | None]:
    """Detect dominant Indic script in text and locate an available system font file."""
    counts = {script: 0 for script in INDIC_RANGES}
    for ch in text:
        cp = ord(ch)
        for script, (start, end) in INDIC_RANGES.items():
            if start <= cp <= end:
                counts[script] += 1
                break
    best_script = max(counts, key=counts.get)
    if counts[best_script] > 0:
        candidates = FONT_CANDIDATES.get(best_script, [])
        for p in candidates:
            if os.path.exists(p):
                return best_script, p
    return None, None

CENTERED_PATTERNS = [
    r"^IN THE (?:COURT|HIGH COURT|SUPREME COURT|TRIBUNAL|COMMISSION|APPELLATE TRIBUNAL)",
    r"^BEFORE THE (?:HON'BLE|COURT|LD\.|LEARNED|DISTRICT|CHIEF|SUB-DIVISIONAL|SUPERINTENDENT|PUBLIC|APPROPRIATE|NATIONAL|STATE|CONTROLLING)",
    r"^(?:CRIMINAL|CIVIL|ORIGINAL|WRIT|SPECIAL|MISC|APPELLATE|EXECUTION)\s+(?:MISC\.?\s+)?(?:APPLICATION|PETITION|SUIT|APPEAL|CASE)",
    r"^(?:APPLICATION|PETITION|MEMORANDUM|SUIT|PLAINT|COMPLAINT|NOTICE|WRITTEN STATEMENT|CLAIM PETITION)\s+(?:UNDER|FOR|ON)",
    r"^(?:SUIT|CASE|BAIL APPLICATION|PETITION|WRIT PETITION|O\.?A\.?|C\.?P\.?)\s+(?:NO\.?|NUMBER)",
    r"^IN THE MATTER OF",
    r"^IN RE:",
    r"^MEMORANDUM OF (?:PARTIES|APPEAL|GROUNDS)",
    r"^MOST RESPECTFULLY SHOWETH",
    r"^STATEMENT OF (?:FACTS|CLAIM)",
    r"^SYNOPSIS AND LIST OF DATES",
    r"^GROUNDS",
    r"^PRAYER",
    r"^RELIEF SOUGHT",
    r"^VERIFICATION",
    r"^AFFIDAVIT",
    r"^FORMAL LEGAL NOTICE",
    r"^LEGAL NOTICE",
    r"^FORM ['\"]?[A-Z]['\"]?",
    r"^AGREEMENT FOR",
    r"^GENERAL POWER OF ATTORNEY",
    r"^SPECIAL POWER OF ATTORNEY",
    r"^DEED OF",
]
_CENTERED_RE = re.compile("|".join(CENTERED_PATTERNS), re.IGNORECASE)


def normalize_legal_draft_text(content: str) -> str:
    """Normalize draft whitespace, markdown code blocks, and legal paragraph breaks.

    Preserves exact sentences and natural wording from the LLM or caller,
    only ensuring that markdown code fences are stripped and unsegmented
    headers, metadata, salutations, numbered clauses, and signature lines
    receive clean paragraph breaks.
    """
    if not content:
        return ""
    s = content.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # 1. Separate document title if glued to Date or To, at the very start
    s = re.sub(r'^(LEGAL NOTICE|FORMAL LEGAL NOTICE|DEMAND NOTICE|STATUTORY NOTICE)\s+(Date:|Dated:|To,)', r'\1\n\n\2', s, flags=re.IGNORECASE)
    s = re.sub(r'^(IN THE COURT OF[^\n]+?)\s+(IN THE MATTER OF|VERSUS|APPLICATION|PETITION|SUIT|CRIMINAL)', r'\1\n\n\2', s, flags=re.IGNORECASE)

    # 2. Separate Date if glued to To,
    s = re.sub(r'(Date\s*:\s*\[?[^\]\n]+?\]?)\s+(To\s*,)', r'\1\n\n\2', s, flags=re.IGNORECASE)

    # 3. Separate Addressee if glued to Subject
    s = re.sub(r'(\bTo\s*,.*?)\s+(Subject\s*:\s*)', r'\1\n\n\2', s, flags=re.IGNORECASE)

    # 4. Separate Subject if glued to Salutation
    s = re.sub(r'(Subject\s*:\s*[^\n]+?)\s+(Sir\s*/\s*Madam\s*,|Respected Sir\s*,|Dear Sir\s*,)', r'\1\n\n\2', s, flags=re.IGNORECASE)

    # 5. Separate Salutation if glued to opening sentence or Point 1
    s = re.sub(r'(Sir\s*/\s*Madam\s*,|Respected Sir\s*,|Dear Sir\s*,)\s*(\d{1,2}\.|\bUnder\b)', r'\1\n\n\2', s, flags=re.IGNORECASE)

    # 6. Separate unsegmented numbered points after sentence-ending punctuation only
    s = re.sub(r'([.:;?!]|\bnotice:|\bgrounds:|\bfacts:)\s+(\d{1,2}\.|\([0-9a-zA-Z]{1,2}\))\s+', r'\1\n\n\2 ', s, flags=re.IGNORECASE)

    # 7. Separate standalone formal headings if glued to prior text without newlines
    for h in [r'MOST RESPECTFULLY SHOWETH\s*:', r'PRAYER\s*:', r'VERIFICATION\s*:', r'AFFIDAVIT\s*:', r'VERSUS\b']:
        s = re.sub(rf'(?<!\n)\s+(\b{h})', r'\n\n\1', s, flags=re.IGNORECASE)

    # 8. Separate closing / signature block at tail if glued to last sentence
    s = re.sub(r'([.?!])\s+(Copy retained\.?)', r'\1\n\n\2', s, flags=re.IGNORECASE)
    s = re.sub(r'([.?!]|\bCopy retained\.?)\s+(Advocate for\b)', r'\1\n\n\2', s, flags=re.IGNORECASE)

    # Standardize multiple blank lines into clean double newlines
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def generate_legal_pdf(
    title: str,
    content: str,
    document_type: str = "general_legal_draft",
    warning: str = "",
) -> bytes:
    """Produce an A4 court-standard PDF binary from formatted legal text."""
    doc = pymupdf.open()

    # Normalize whitespace without mutating sentences
    content = normalize_legal_draft_text(content)

    script_name, indic_font_path = _detect_indic_font(f"{title} {content} {warning}")
    use_indic = bool(indic_font_path)
    indic_font_name = f"indic_{script_name}" if script_name else "indic_font"

    if use_indic:
        font_roman = pymupdf.Font(fontfile=indic_font_path)
        font_bold = font_roman
        font_italic = font_roman
    else:
        font_roman = pymupdf.Font("times-roman")
        font_bold = pymupdf.Font("times-bold")
        font_italic = pymupdf.Font("times-italic")

    lines_to_draw: list[dict[str, Any]] = []
    raw_lines = content.split("\n")

    for raw in raw_lines:
        line_str = raw.strip()
        if not line_str:
            lines_to_draw.append({"type": "empty", "height": LEADING_BODY * 0.5})
            continue

        upper = line_str.upper()

        # 1. Centered Headings & Section Dividers
        if (
            line_str.startswith("#")
            or (_CENTERED_RE.match(upper) and len(line_str) < 85)
            or (upper in {"VERSUS", "VS.", "VS", "PRAYER:", "PRAYER", "VERIFICATION:", "VERIFICATION", "AFFIDAVIT:", "AFFIDAVIT", "LEGAL NOTICE", "MOST RESPECTFULLY SHOWETH:", "SYNOPSIS", "GROUNDS:"})
            or (line_str.isupper() and len(line_str) < 70 and not line_str.startswith("1.") and not line_str.startswith("THAT") and not line_str.startswith("AND") and not line_str.startswith("FOR"))
        ):
            clean_text = line_str.lstrip("#").strip()
            words = clean_text.split()
            current: list[str] = []
            for w in words:
                test_str = " ".join(current + [w])
                if font_bold.text_length(test_str, fontsize=FONT_SIZE_HEADING) > CONTENT_WIDTH:
                    if current:
                        lines_to_draw.append({
                            "type": "heading", "text": " ".join(current),
                            "size": FONT_SIZE_HEADING, "bold": True,
                            "height": LEADING_HEADING, "centered": True,
                        })
                        current = [w]
                    else:
                        lines_to_draw.append({
                            "type": "heading", "text": w,
                            "size": FONT_SIZE_HEADING, "bold": True,
                            "height": LEADING_HEADING, "centered": True,
                        })
                        current = []
                else:
                    current.append(w)
            if current:
                lines_to_draw.append({
                    "type": "heading", "text": " ".join(current),
                    "size": FONT_SIZE_HEADING, "bold": True,
                    "height": LEADING_HEADING, "centered": True,
                })
            lines_to_draw.append({"type": "empty", "height": LEADING_HEADING * 0.25})

        # 2. Party Status (... APPLICANT, ... RESPONDENT) - strictly lines starting with ellipsis or dash
        elif line_str.startswith("...") or line_str.startswith("…") or line_str.startswith("—"):
            lines_to_draw.append({
                "type": "party_status", "text": line_str,
                "size": FONT_SIZE_BODY, "bold": True,
                "height": LEADING_BODY, "right": True,
            })

        # 3. Subject Line (bold, distinct)
        elif upper.startswith("SUBJECT:") or upper.startswith("SUB:") or upper.startswith("REF:"):
            words = line_str.split()
            current = []
            for w in words:
                test_str = " ".join(current + [w])
                if font_bold.text_length(test_str, fontsize=FONT_SIZE_BODY) > CONTENT_WIDTH:
                    lines_to_draw.append({
                        "type": "subject", "text": " ".join(current),
                        "size": FONT_SIZE_BODY, "bold": True,
                        "height": LEADING_BODY, "indent": 0.0,
                    })
                    current = [w]
                else:
                    current.append(w)
            if current:
                lines_to_draw.append({
                    "type": "subject", "text": " ".join(current),
                    "size": FONT_SIZE_BODY, "bold": True,
                    "height": LEADING_BODY, "indent": 0.0,
                })
            lines_to_draw.append({"type": "empty", "height": LEADING_BODY * 0.3})

        # 4. Date Line
        elif (upper.startswith("DATE:") or upper.startswith("DATED:")) and len(line_str) < 40:
            lines_to_draw.append({
                "type": "date", "text": line_str,
                "size": FONT_SIZE_BODY, "bold": False,
                "height": LEADING_BODY, "right": True,
            })
            lines_to_draw.append({"type": "empty", "height": LEADING_BODY * 0.2})

        # 5. Signature / Closing Block
        elif any(upper.startswith(sig) for sig in [
            "ADVOCATE FOR", "COUNSEL FOR", "THROUGH: ADVOCATE", "THROUGH ADVOCATE",
            "THROUGH COUNSEL", "YOURS FAITHFULLY", "YOURS SINCERELY", "DEPONENT"
        ]) and len(line_str) < 60:
            lines_to_draw.append({"type": "empty", "height": LEADING_BODY * 0.4})
            lines_to_draw.append({
                "type": "signature", "text": line_str,
                "size": FONT_SIZE_BODY, "bold": True,
                "height": LEADING_BODY, "right": True,
            })

        # 6. Standard Paragraphs and Numbered Clauses
        else:
            words = line_str.split()
            first_word = words[0] if words else ""
            is_numbered = bool(re.match(r"^(?:\d+[\.\)]|\([0-9a-zA-Z]+\)|[a-z]\))\s*", first_word))
            indent = 18.0 if is_numbered else 0.0
            avail_width = CONTENT_WIDTH - indent

            current = []
            for w in words:
                test_str = " ".join(current + [w])
                if font_roman.text_length(test_str, fontsize=FONT_SIZE_BODY) > avail_width:
                    if current:
                        lines_to_draw.append({
                            "type": "body", "text": " ".join(current),
                            "size": FONT_SIZE_BODY, "bold": False,
                            "height": LEADING_BODY, "indent": indent,
                        })
                        current = [w]
                    else:
                        lines_to_draw.append({
                            "type": "body", "text": w,
                            "size": FONT_SIZE_BODY, "bold": False,
                            "height": LEADING_BODY, "indent": indent,
                        })
                        current = []
                else:
                    current.append(w)
            if current:
                lines_to_draw.append({
                    "type": "body", "text": " ".join(current),
                    "size": FONT_SIZE_BODY, "bold": False,
                    "height": LEADING_BODY, "indent": indent,
                })
            if is_numbered:
                lines_to_draw.append({"type": "empty", "height": LEADING_BODY * 0.3})

    # Pagination calculation
    pages_lines: list[list[tuple[dict[str, Any], float]]] = []
    current_page: list[tuple[dict[str, Any], float]] = []
    current_y = MARGIN_TOP + 20.0
    max_y = A4_HEIGHT - MARGIN_BOTTOM - 20.0

    if warning:
        current_y += 36.0

    for item in lines_to_draw:
        h = item["height"]
        if current_y + h > max_y:
            pages_lines.append(current_page)
            current_page = []
            current_y = MARGIN_TOP + 20.0
        current_page.append((item, current_y))
        current_y += h

    if current_page:
        pages_lines.append(current_page)

    total_pages = max(1, len(pages_lines))

    for page_idx, p_lines in enumerate(pages_lines, start=1):
        page = doc.new_page(width=A4_WIDTH, height=A4_HEIGHT)
        if use_indic:
            page.insert_font(fontname=indic_font_name, fontfile=indic_font_path)

        # Running Header
        header_y = MARGIN_TOP - 12.0
        page.draw_line(
            pymupdf.Point(MARGIN_LEFT, header_y + 4),
            pymupdf.Point(A4_WIDTH - MARGIN_RIGHT, header_y + 4),
            color=(0.7, 0.7, 0.7), width=0.5,
        )
        doc_header = (title[:65] + "...") if len(title) > 65 else title
        h_fn = indic_font_name if use_indic else "times-bold"
        page.insert_text(
            pymupdf.Point(MARGIN_LEFT, header_y),
            doc_header if use_indic else doc_header.upper(),
            fontname=h_fn, fontsize=8.0, color=(0.4, 0.4, 0.4),
        )

        # Page 1 Warning Banner
        if page_idx == 1 and warning:
            box_rect = pymupdf.Rect(MARGIN_LEFT, MARGIN_TOP + 4, A4_WIDTH - MARGIN_RIGHT, MARGIN_TOP + 36)
            page.draw_rect(box_rect, color=(0.85, 0.85, 0.85), fill=(0.96, 0.96, 0.96), width=0.5)
            notice_text = f"STATUTORY NOTICE: {warning}"
            w_fn = indic_font_name if use_indic else "times-italic"
            page.insert_textbox(
                box_rect + (6, 5, -6, -5), notice_text,
                fontname=w_fn, fontsize=8.0, color=(0.25, 0.25, 0.25),
            )

        # Running Footer
        footer_y = A4_HEIGHT - MARGIN_BOTTOM + 15.0
        page.draw_line(
            pymupdf.Point(MARGIN_LEFT, footer_y - 8),
            pymupdf.Point(A4_WIDTH - MARGIN_RIGHT, footer_y - 8),
            color=(0.7, 0.7, 0.7), width=0.5,
        )
        page.insert_text(
            pymupdf.Point(MARGIN_LEFT, footer_y),
            "Nyaya Legal Information System · Formatted for Indian Legal Practice",
            fontname="times-roman", fontsize=7.5, color=(0.4, 0.4, 0.4),
        )
        page_num_str = f"Page {page_idx} of {total_pages}"
        p_len = font_bold.text_length(page_num_str, fontsize=8.0)
        page.insert_text(
            pymupdf.Point(A4_WIDTH - MARGIN_RIGHT - p_len, footer_y),
            page_num_str, fontname="times-bold", fontsize=8.0, color=(0.2, 0.2, 0.2),
        )

        # Draw Lines
        for item, y_pos in p_lines:
            if item["type"] == "empty":
                continue
            text = item.get("text", "")
            size = item.get("size", FONT_SIZE_BODY)
            fn = indic_font_name if use_indic else ("times-bold" if item.get("bold") else "times-roman")

            if item.get("centered"):
                t_len = (font_bold if item.get("bold") else font_roman).text_length(text, fontsize=size)
                x_pos = MARGIN_LEFT + max(0.0, (CONTENT_WIDTH - t_len) / 2.0)
            elif item.get("right"):
                t_len = font_bold.text_length(text, fontsize=size)
                x_pos = A4_WIDTH - MARGIN_RIGHT - t_len
            else:
                x_pos = MARGIN_LEFT + item.get("indent", 0.0)

            page.insert_text(
                pymupdf.Point(x_pos, y_pos),
                text, fontname=fn, fontsize=size, color=(0.0, 0.0, 0.0),
            )

    return doc.tobytes()
