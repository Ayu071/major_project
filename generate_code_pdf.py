"""
generate_code_pdf.py
Combines all pipeline scripts + app.py into a single readable PDF.
Run: python generate_code_pdf.py
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, PageBreak, Preformatted
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm

# ── Files to include (in order) ───────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

FILES = [
    ("Step 1 — Image Basics",          os.path.join(BASE, "scripts", "01_image_basics.py")),
    ("Step 2 — Image Processing",      os.path.join(BASE, "scripts", "02_image_processing.py")),
    ("Step 3 — Depth Estimation",      os.path.join(BASE, "scripts", "03_depth_estimation.py")),
    ("Step 4 — Visualize Point Cloud", os.path.join(BASE, "scripts", "04_visualize_point_cloud.py")),
    ("Step 5 — Mesh Generation",       os.path.join(BASE, "scripts", "05_mesh_generation.py")),
    ("Step 6 — Textured Mesh",         os.path.join(BASE, "scripts", "06_mesh_textured.py")),
    ("Step 7 — Multi-View Fusion",     os.path.join(BASE, "scripts", "07_multi_view_fusion.py")),
    ("Frontend — app.py",              os.path.join(BASE, "app.py")),
]

def build_pdf(pdf_path):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=22,
        textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#555555'),
        spaceAfter=20,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.white,
        backColor=colors.HexColor('#1a1a2e'),
        spaceAfter=10,
        spaceBefore=6,
        leftIndent=-10,
        rightIndent=-10,
        leading=22,
    )
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=7.5,
        fontName='Courier',
        leading=11,
        leftIndent=0,
        spaceAfter=0,
        backColor=colors.HexColor('#f8f8f8'),
        textColor=colors.HexColor('#1a1a2e'),
    )

    story = []

    # ── Cover Page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("2D to 3D Reconstruction Pipeline", title_style))
    story.append(Paragraph("Complete Source Code — All Steps", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
    story.append(Spacer(1, 0.5 * cm))

    toc_lines = [f"&nbsp;&nbsp;&nbsp;{label}" for label, _ in FILES]
    for line in toc_lines:
        story.append(Paragraph(line, ParagraphStyle(
            'TOC', parent=styles['Normal'], fontSize=10,
            textColor=colors.HexColor('#333333'), spaceAfter=4
        )))

    story.append(PageBreak())

    # ── Each Script ───────────────────────────────────────────────────────────
    for label, filepath in FILES:
        # Section header
        story.append(Paragraph(f"  {label}", section_style))
        story.append(Spacer(1, 0.2 * cm))

        if not os.path.exists(filepath):
            story.append(Paragraph(
                f"<i>File not found: {filepath}</i>",
                ParagraphStyle('Missing', parent=styles['Normal'],
                               textColor=colors.red, fontSize=9)
            ))
            story.append(PageBreak())
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        # Split into lines, escape HTML, wrap in Preformatted
        lines = source.split('\n')
        # Chunk into blocks of 60 lines to avoid single huge flowable
        chunk_size = 60
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i + chunk_size]
            text = '\n'.join(chunk)
            story.append(Preformatted(text, code_style))

        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor('#cccccc')))
        story.append(PageBreak())

    doc.build(story)
    print(f"✓ Created: {pdf_path}")


if __name__ == "__main__":
    out = os.path.join(BASE, "Pipeline_Source_Code.pdf")
    build_pdf(out)
    print("Done!")
