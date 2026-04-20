"""
Authentify - PDF Report Generator
Generates a professional forensic verification report in PDF format.
"""

import os
import io
import base64
from datetime import datetime
from fpdf import FPDF
from typing import Dict, Any

class AuthentifyReport(FPDF):
    def header(self):
        # Draw accent bar at top
        self.set_fill_color(99, 102, 241)  # var(--accent-primary)
        self.rect(0, 0, 210, 15, 'F')
        
        # Title
        self.set_font('helvetica', 'B', 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 5, 'AUTHENTIFY', align='L')
        self.set_font('helvetica', '', 8)
        self.cell(0, 5, 'AI-POWERED FORENSIC VERIFICATION', align='R')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()} | Authentify Security System | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C')

def sanitize_text(text: Any) -> str:
    """Helper to convert any text to latin-1 compatible string for FPDF."""
    if text is None:
        return ""
    s = str(text)
    # Replace common non-latin-1 characters
    s = s.replace('•', '-').replace('—', '-').replace('–', '-').replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    # Remove any emojis or icons that might have slipped through
    return s.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_report(data: Dict[str, Any]) -> bytes:
    """
    Generates a PDF bytes buffer from the analysis results.
    """
    pdf = AuthentifyReport()
    pdf.add_page()
    
    # --- Summary Section ---
    pdf.set_font('helvetica', 'B', 16)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 10, 'Verification Report Summary', ln=True)
    pdf.ln(2)
    
    # Analysis Details Grid
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_fill_color(243, 244, 246)
    
    col_width = 95
    pdf.cell(col_width, 8, ' Analysis ID:', border=1, fill=True)
    pdf.set_font('helvetica', '', 10)
    pdf.cell(col_width, 8, f' {sanitize_text(data.get("analysis_id", "N/A"))}', border=1, ln=True)
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(col_width, 8, ' Filename:', border=1, fill=True)
    pdf.set_font('helvetica', '', 10)
    pdf.cell(col_width, 8, f' {sanitize_text(data.get("filename", "N/A"))}', border=1, ln=True)
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(col_width, 8, ' Processing Time:', border=1, fill=True)
    pdf.set_font('helvetica', '', 10)
    pdf.cell(col_width, 8, f' {sanitize_text(data.get("processing_time_seconds", 0))}s', border=1, ln=True)
    
    pdf.ln(10)
    
    # --- Authenticity Score & Verdict ---
    auth = data.get("authenticity", {})
    score = auth.get("score", 0)
    verdict = auth.get("verdict", "UNKNOWN")
    
    # Set color based on score
    if score >= 0.8:
        color = (16, 185, 129) # green
    elif score >= 0.4:
        color = (245, 158, 11) # orange
    else:
        color = (239, 68, 68)  # red
        
    pdf.set_draw_color(*color)
    pdf.set_line_width(1)
    pdf.rect(10, pdf.get_y(), 190, 30)
    
    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(50, 10, 'AUTHENTICITY SCORE:')
    
    pdf.set_font('helvetica', 'B', 24)
    pdf.set_text_color(*color)
    pdf.cell(40, 10, f'{int(score * 100)}%')
    
    pdf.set_xy(105, pdf.get_y() - 2)
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, f'VERDICT: {sanitize_text(verdict)}', ln=True)
    
    pdf.set_xy(105, pdf.get_y() - 2)
    pdf.set_font('helvetica', '', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, sanitize_text(auth.get("verdict_description", "")))
    
    pdf.ln(25)
    
    # --- Module Breakdown ---
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 10, 'Security Module Analysis', ln=True)
    pdf.ln(2)
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_fill_color(249, 250, 251)
    pdf.cell(100, 8, ' Module Name', border='B', fill=True)
    pdf.cell(45, 8, ' Score', border='B', fill=True, align='C')
    pdf.cell(45, 8, ' Final Weight', border='B', fill=True, align='C', ln=True)
    
    pdf.set_font('helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    
    modules = data.get("modules", [])
    for mod in modules:
        pdf.cell(100, 8, f' {sanitize_text(mod["module"])}', border='B')
        pdf.cell(45, 8, f' {int(mod.get("score", 0) * 100)}%', border='B', align='C')
        pdf.cell(45, 8, ' - ', border='B', align='C', ln=True)
        
    pdf.ln(10)
    
    # --- Detailed Findings ---
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 10, 'Key Forensic Findings', ln=True)
    pdf.ln(2)
    
    findings = data.get("findings", [])
    pdf.set_font('helvetica', '', 10)
    for finding in findings:
        # Simple bullet point
        pdf.set_x(15)
        text = str(finding).replace('✅', '[PASS] ').replace('⚠️', '[WARN] ').replace('🔴', '[FAIL] ').replace('🟡', '[INFO] ').replace('📌', '[NOTE] ')
        pdf.multi_cell(0, 6, sanitize_text(f"- {text}"))
        
    pdf.ln(10)
    
    # --- Embedded Image Placeholders ---
    # In a real app we'd save images to disk and embed them here.
    # For now we'll just add labels for the maps.
    
    # Check if we have annotated image
    # Note: fpdf requires a physical file or a specific format.
    # We will skip embedding base64 images here for simplicity to avoid temp file bloat, 
    # but we will provide the structure.
    
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Metadata Signature', ln=True)
    
    meta = data.get("metadata", {})
    pdf.set_font('helvetica', '', 9)
    file_info = meta.get("file_info", {})
    software = meta.get("software_analysis", {})
    
    pdf.cell(0, 5, f'Format: {file_info.get("format", "N/A")} ({file_info.get("size_kb", 0)} KB)', ln=True)
    pdf.cell(0, 5, f'Editing Detection: {"POTENTIAL TAMPERING DETECTED" if software.get("editing_detected") else "No direct evidence of editing tools"}', ln=True)
    
    return pdf.output()
