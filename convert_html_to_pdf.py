#!/usr/bin/env python3
"""
Convert HTML report to PDF using WeasyPrint
"""

from weasyprint import HTML, CSS
import os

def convert_html_to_pdf(html_file, pdf_file):
    """Convert HTML file to PDF with proper styling"""

    # Check if HTML file exists
    if not os.path.exists(html_file):
        print(f"Error: {html_file} not found")
        return False

    try:
        # Convert HTML to PDF
        HTML(html_file).write_pdf(pdf_file)
        print(f"Successfully converted {html_file} to {pdf_file}")
        return True
    except Exception as e:
        print(f"Error converting HTML to PDF: {e}")
        return False

if __name__ == "__main__":
    html_file = "FINAL_REPORT.html"
    pdf_file = "FINAL_REPORT.pdf"

    success = convert_html_to_pdf(html_file, pdf_file)
    if success:
        print(f"PDF generated: {pdf_file}")
    else:
        print("Failed to generate PDF")
