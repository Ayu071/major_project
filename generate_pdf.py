"""
generate_pdf.py — Converts presentation_notes.md and walkthrough.md to PDF
Run: python generate_pdf.py
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm

def parse_md_to_pdf(md_path, pdf_path):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN
    )

    styles = getSampleStyleSheet()

    # Custom Styles
    h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=18, spaceAfter=10, textColor=colors.HexColor('#1a1a2e'))
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, spaceAfter=8, textColor=colors.HexColor('#16213e'))
    h3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=12, spaceAfter=6, textColor=colors.HexColor('#0f3460'))
    body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=4, leading=14)
    code = ParagraphStyle('Code', parent=styles['Code'], fontSize=8, backColor=colors.HexColor('#f4f4f4'),
                          leftIndent=10, spaceAfter=6, fontName='Courier')
    bullet = ParagraphStyle('Bullet', parent=styles['Normal'], fontSize=10, leftIndent=15, spaceAfter=3, leading=13)

    story = []
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def flush_table():
        nonlocal table_rows, in_table
        if not table_rows:
            return
        # Filter out separator rows (---|---|---)
        data = [r for r in table_rows if not all(c.strip().startswith('-') for c in r)]
        if not data:
            table_rows = []
            in_table = False
            return

        col_count = max(len(r) for r in data)
        # Pad rows
        data = [r + [''] * (col_count - len(r)) for r in data]

        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9f9f9')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))
        table_rows = []
        in_table = False

    def flush_code():
        nonlocal code_lines, in_code_block
        if code_lines:
            text = '\n'.join(code_lines)
            # Escape for reportlab
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            text = text.replace('\n', '<br/>')
            story.append(Paragraph(text, code))
            story.append(Spacer(1, 4))
        code_lines = []
        in_code_block = False

    for line in lines:
        raw = line.rstrip('\n')

        # Code block toggle
        if raw.strip().startswith('```'):
            if in_code_block:
                flush_code()
            else:
                if in_table:
                    flush_table()
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(raw)
            continue

        # Table row
        if raw.strip().startswith('|'):
            if in_table is False:
                in_table = True
            cells = [c.strip() for c in raw.strip().strip('|').split('|')]
            table_rows.append(cells)
            continue
        else:
            if in_table:
                flush_table()

        # Blank line
        if not raw.strip():
            story.append(Spacer(1, 6))
            continue

        # Escape HTML chars
        text = raw.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Bold **text**
        import re
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        # Inline code `text`
        text = re.sub(r'`(.+?)`', r'<font name="Courier" size="8">\1</font>', text)

        # Headings
        if text.startswith('### '):
            story.append(Paragraph(text[4:], h3))
        elif text.startswith('## '):
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
            story.append(Paragraph(text[3:], h2))
        elif text.startswith('# '):
            story.append(Paragraph(text[2:], h1))
            story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
            story.append(Spacer(1, 6))
        elif text.startswith('- ') or text.startswith('* '):
            story.append(Paragraph('• ' + text[2:], bullet))
        elif text.startswith('---'):
            story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
            story.append(Spacer(1, 4))
        else:
            story.append(Paragraph(text, body))

    if in_code_block:
        flush_code()
    if in_table:
        flush_table()

    doc.build(story)
    print(f"✓ Created: {pdf_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    parse_md_to_pdf(
        os.path.join(base, "presentation_notes.md"),
        os.path.join(base, "Presentation_Notes.pdf")
    )
    parse_md_to_pdf(
        os.path.join(base, "walkthrough.md"),
        os.path.join(base, "Walkthrough.pdf")
    )
    print("\nDone! Both PDFs are in your project folder.")
