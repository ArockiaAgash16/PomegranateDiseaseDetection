from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import os

def generate_pdf_report(image_path, disease, severity, severity_pct, treatment, chat_history, output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=50,
        bottomMargin=50
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Justify", alignment=4))  # Fully justified

    story = []

    # 🔶 Title
    story.append(Paragraph("🍎 Pomegranate Disease Diagnosis Report", styles['Title']))
    story.append(Spacer(1, 20))

    # 🔶 Uploaded Image (if exists)
    if os.path.exists(image_path):
        img = Image(image_path, width=200, height=200)
        story.append(img)
        story.append(Spacer(1, 12))

    # 🔶 Disease + Severity Table
    table_data = [
        ["Disease", disease],
        ["Severity", f"{severity} ({severity_pct:.2f}%)"]
    ]
    table = Table(table_data, colWidths=[100, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # 🔶 Treatment Plan
    story.append(Paragraph("🩺 Treatment Plan", styles['Heading2']))
    treatment_paragraph = Paragraph(treatment, styles['Justify'])
    story.append(treatment_paragraph)
    story.append(Spacer(1, 20))

    # 🔶 Follow-up Chat History
    if chat_history:
        story.append(Paragraph("💬 Follow-up Questions", styles['Heading2']))
        for chat in chat_history:
            story.append(Paragraph(f"<b>User Query:</b> {chat['user']}", styles['BodyText']))
            story.append(Paragraph(f"<b>AI's Reply:</b> {chat['bot']}", styles['Justify']))
            story.append(Spacer(1, 10))

    # 🔶 Build PDF
    doc.build(story)
    print(f"✅ PDF report saved at: {output_path}")

