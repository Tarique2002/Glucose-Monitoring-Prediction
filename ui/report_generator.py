from fpdf import FPDF
import base64
import os
from datetime import datetime

class HealthReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI Health Diagnostics Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.line(10, 30, 200, 30)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(0, 10, 'Disclaimer: This is an AI-generated report and not medical advice.', 0, 0, 'R')

def create_pdf_report(results: dict, inputs: dict) -> bytes:
    """Generates a PDF report and returns it as bytes."""
    pdf = HealthReportPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Patient Inputs", 0, 1)
    
    pdf.set_font("Arial", '', 12)
    for key, value in inputs.items():
        pdf.cell(0, 8, f"{key.replace('_', ' ').title()}: {value}", 0, 1)
        
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "AI Predictions", 0, 1)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Predicted Glucose: {results['predicted_glucose']:.1f} mg/dL", 0, 1)
    
    risk_status = "High Risk (Diabetic)" if results['diabetes_risk'] == 1 else "Normal Range"
    pdf.cell(0, 8, f"Diabetes Risk Status: {risk_status}", 0, 1)
    
    conf = (results['risk_proba'][1] if results['diabetes_risk'] == 1 else results['risk_proba'][0]) * 100
    pdf.cell(0, 8, f"AI Confidence Score: {conf:.1f}%", 0, 1)
    pdf.cell(0, 8, f"Overall Health Score: {results['health_score']}/100", 0, 1)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Smart Recommendations", 0, 1)
    
    pdf.set_font("Arial", '', 12)
    if results['diabetes_risk'] == 1:
        pdf.multi_cell(0, 8, "- Diet: Strict low GI foods, avoid refined sugars.\n- Activity: Daily cardiovascular exercise.\n- Medical: Consult a physician immediately for formal diagnosis.")
    else:
        pdf.multi_cell(0, 8, "- Diet: Maintain a balanced diet rich in fiber.\n- Activity: Regular exercise (150 mins/week).\n- Medical: Routine annual checkup.")
        
    return pdf.output(dest='S').encode('latin-1')

def get_pdf_download_link(pdf_bytes: bytes, filename: str = "health_report.pdf") -> str:
    """Generates a clickable download link for the PDF."""
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="stButton" style="text-decoration:none; display:inline-block; padding:10px 20px; background:var(--primary); color:white; border-radius:30px; font-weight:bold;">📥 Download Full PDF Report</a>'
