import os
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import requests
from datetime import datetime
from io import BytesIO

# Import ReportLab modules for clinical document compilation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Set page configuration with a modern healthcare icon
st.set_page_config(page_title="Bio-Inspired AI Airway Diagnostics", page_icon="🫁", layout="wide")

# =========================================================================
# PREMIUM CSS STYLING INJECTION
# =========================================================================
st.markdown("""
    <style>
    /* Main App Background Tuning */
    .stApp {
        background-color: #F8FAFC;
    }
    /* Custom Card Containers */
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 14px;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        color: #1E293B;
        font-weight: 700;
        margin-top: 5px;
    }
    /* Section headers */
    h1, h2, h3 {
        color: #0F172A !important;
        font-family: 'Inter', sans-serif;
    }
    /* Sidebar styling tweak */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
        color: #F1F5F9 !important;
    }
    /* Clean custom badge tags */
    .badge-normal { background-color: #DCFCE7; color: #14532D; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .badge-warning { background-color: #FEF9C3; color: #713F12; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .badge-error { background-color: #FEE2E2; color: #7F1D1D; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    </style>
""", unsafe_allow_html=True)

# API Configuration - Pointing directly to our active FastAPI microservice
API_URL = "http://127.0.0.1:8001/api/v1/diagnose"

# =========================================================================
# LONGITUDINAL LOGGING CORE
# =========================================================================
def init_longitudinal_db():
    conn = sqlite3.connect("airway_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnostics_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age INTEGER,
            gender TEXT,
            risk_score REAL,
            baais_coefficient REAL,
            triage_decision TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_diagnostic_run(age, gender, risk_score, baais_coeff, triage):
    conn = sqlite3.connect("airway_history.db")
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute("""
        INSERT INTO diagnostics_log (timestamp, age, gender, risk_score, baais_coefficient, triage_decision)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (now, age, gender, risk_score, baais_coeff, triage))
    conn.commit()
    conn.close()

def fetch_patient_history():
    conn = sqlite3.connect("airway_history.db")
    df = pd.read_sql_query("SELECT timestamp, risk_score, baais_coefficient FROM diagnostics_log ORDER BY id ASC", conn)
    conn.close()
    return df

init_longitudinal_db()

# =========================================================================
# USER INTERFACE RENDERING
# =========================================================================
st.title("🫁 AI-Based Airway Obstruction Diagnostics Platform")
st.markdown("<p style='color: #64748B; font-size: 16px; margin-top: -15px;'>Decoupled Microservice Architecture Layer Frontend</p>", unsafe_allow_html=True)
st.markdown("---")

# Patient Metadata Inputs (Sidebar Custom Styled Layout)
st.sidebar.markdown("<h2 style='margin-top:0;'>📋 Clinical Metadata</h2>", unsafe_allow_html=True)
age = st.sidebar.slider("Patient Age", 1, 100, 55)
gender = st.sidebar.selectbox("Biological Sex", ["Male", "Female", "Other"])
ethnicity = st.sidebar.selectbox("Patient Registry Cohort Group", ["Group A", "Group B", "Group C", "Group D"])
asthma = st.sidebar.selectbox("History of Asthma?", ["No", "Yes"], index=1)
copd = st.sidebar.selectbox("History of COPD?", ["No", "Yes"], index=0)
smoking = st.sidebar.selectbox("Current Smoking Status?", ["Non-Smoker", "Active Smoker"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("<h2>🌤️ Environmental Metrics</h2>", unsafe_allow_html=True)
aqi = st.sidebar.slider("Air Quality Index (AQI)", 0, 300, 48)
humidity = st.sidebar.slider("Ambient Humidity Level (%)", 0, 100, 55)

# Main Application Body Grid
banner_col1, banner_col2 = st.columns([2, 1])
with banner_col1:
    st.markdown("### 🎵 Respiratory Signal Stream Ingestion")
    uploaded_file = st.file_uploader("Upload a patient breathing recording (.wav file)", type=["wav"], label_visibility="collapsed")

if uploaded_file is not None:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.audio(uploaded_file, format="audio/wav")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Execute Multimodal Diagnostic Matrix", use_container_width=True):
        with st.spinner("Streaming data vectors to external AI Microservice endpoint..."):
            try:
                # 1. Package fields matching the exact validation schema requirements of api.py
                payload = {
                    "age": age,
                    "gender": gender,
                    "asthma": asthma,
                    "copd": copd
                }
                
                # Extract file binary buffer bytes to pipe over HTTP form fields
                file_bytes = uploaded_file.getvalue()
                files = {"file": (uploaded_file.name, file_bytes, "audio/wav")}
                
                # 2. Dispatch network pipeline transaction across standard localhost sockets
                response = requests.post(API_URL, data=payload, files=files)
                
                if response.status_code != 200:
                    st.error(f"❌ API Microservice Core Error: {response.json().get('detail', 'Unknown fault')}")
                else:
                    # Parse JSON output package safely
                    api_data = response.json()
                    
                    computed = api_data["computed_diagnostics"]
                    results_found = api_data["timeline_analysis"]
                    
                    baais_coeff = computed["baais_coefficient"]
                    base_compliance = computed["base_compliance"]
                    overall_risk_percentage = computed["overall_risk_percentage"]
                    risk_val_str = f"{overall_risk_percentage}%"
                    
                    # Deduce final structural metrics tags from data arrays
                    all_conditions = [r["classification"] for r in results_found]
                    if any("Both" in c for c in all_conditions): final_triage = "Both Crackles & Wheezes Present"
                    elif any("Crackles" in c for c in all_conditions) and any("Wheezes" in c for c in all_conditions): final_triage = "Mixed Multi-symptomatic Responses Flagged"
                    elif any("Crackles" in c for c in all_conditions): final_triage = "Crackles Detected"
                    elif any("Wheezes" in c for c in all_conditions): final_triage = "Wheezes Detected"
                    else: final_triage = "Normal Vesicular Breathing"
                    
                    if baais_coeff >= 0.85: adaptation_state = "Resilient Homeostatic Equilibrium"
                    elif baais_coeff >= 0.60: adaptation_state = "Compensated Respiratory Strain Profile"
                    else: adaptation_state = "Decompensated Airway Structural Failure State"

                    # 3. Handle data analysis displays locally
                    st.markdown("---")
                    st.header("📊 Visual Frequency Spectrogram Analysis")
                    temp_filename = "temp_frontend_spec.wav"
                    with open(temp_filename, "wb") as f:
                        f.write(file_bytes)
                    signal, sr = librosa.load(temp_filename, sr=None)
                    if os.path.exists(temp_filename): os.remove(temp_filename)
                    
                    # Modern Styled Plot Canvas
                    fig, ax = plt.subplots(figsize=(10, 3.2))
                    fig.patch.set_facecolor('#F8FAFC')
                    ax.set_facecolor('#F8FAFC')
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
                    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', ax=ax, cmap='magma')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    fig.colorbar(img, ax=ax, format="%+2.0f dB")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    metric_col, report_col = st.columns(2)
                    
                    with metric_col:
                        st.subheader("⏱️ Snapshot Timeline Results")
                        for idx, item in enumerate(results_found):
                            ts, te, cond = item["time_start"], item["time_end"], item["classification"]
                            r_cf, c_cf = item["raw_confidence"], item["baais_optimized_confidence"]
                            
                            # Elegant custom HTML layout row structures
                            if "Normal" in cond: badge_html = f"<span class='badge-normal'>{cond}</span>"
                            elif "Both" in cond: badge_html = f"<span class='badge-error'>{cond}</span>"
                            else: badge_html = f"<span class='badge-warning'>{cond}</span>"
                                
                            st.markdown(f"""
                                <div class='metric-card' style='display:flex; justify-content:space-between; align-items:center;'>
                                    <div>
                                        <b style='color:#1E293B; font-size:15px;'>Slice #{idx+1} ({ts:.1f}s - {te:.1f}s)</b><br>
                                        <span style='color:#64748B; font-size:12px;'>Raw: {r_cf:.1f}% | BAAIS Tuned: {c_cf:.1f}%</span>
                                    </div>
                                    <div>{badge_html}</div>
                                </div>
                            """, unsafe_allow_html=True)
                                
                    with report_col:
                        st.subheader("💡 Explainable AI Clinical Summary Report")
                        st.markdown(f"""
                        <div class='metric-card' style='background-color:#0F172A; color:#F8FAFC;'>
                            <b style='color:#94A3B8; font-size:13px; text-transform:uppercase;'>Automated Diagnostic Notes</b><br>
                            <p style='font-size:16px; margin-top:8px; line-height:1.6;'>
                            • Overall Triage Decision: <b style='color:#38BDF8;'>{final_triage}</b><br>
                            • Computed Composite Pathological Risk: <b style='color:#FB7185;'>{risk_val_str}</b><br>
                            • Patient Context Baseline: Age <b>{age}</b> | Biological Sex <b>{gender}</b>.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        log_diagnostic_run(age, gender, overall_risk_percentage, baais_coeff, final_triage)
                        
                        st.markdown("<h4 style='font-size:16px; color:#475569;'>🧬 BAAIS Adaptation Layer Matrix Diagnostics</h4>", unsafe_allow_html=True)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f"<div class='metric-card'><div class='metric-title'>Airway Risk</div><div class='metric-value' style='color:#EF4444;'>{risk_val_str}</div></div>", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"<div class='metric-card'><div class='metric-title'>BAAIS Coeff</div><div class='metric-value' style='color:#3B82F6;'>{baais_coeff:.2f}</div></div>", unsafe_allow_html=True)
                        with c3:
                            st.markdown(f"<div class='metric-card'><div class='metric-title'>Compliance</div><div class='metric-value' style='color:#10B981;'>{base_compliance:.2f}</div></div>", unsafe_allow_html=True)

                    # Render trend graphs from the logging engine tracking files using updated 2026 syntax
                    st.markdown("---")
                    st.header("📈 Longitudinal Patient Trend Detection Tracking")
                    history_df = fetch_patient_history()
                    if len(history_df) > 1:
                        t1, t2 = st.columns(2)
                        with t1: 
                            st.markdown("<p style='font-weight:600; color:#475569; font-size:14px;'>Historical Risk Trajectory</p>", unsafe_allow_html=True)
                            st.line_chart(data=history_df, x="timestamp", y="risk_score", width="stretch")
                        with t2: 
                            st.markdown("<p style='font-weight:600; color:#475569; font-size:14px;'>BAAIS Compliance Index Elasticity Trend</p>", unsafe_allow_html=True)
                            st.line_chart(data=history_df, x="timestamp", y="baais_coefficient", width="stretch")

                    # Fully Optimized PDF compiler routine with fixed column alignment grid
                    def generate_pdf():
                        buffer = BytesIO()
                        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                        styles = getSampleStyleSheet()
                        
                        title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#1A365D'), spaceAfter=4)
                        subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#4A5568'), spaceAfter=15)
                        section_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2C5282'), spaceBefore=14, spaceAfter=8)
                        
                        body_style = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9, leading=14)
                        bold_body_style = ParagraphStyle('BoldBody', parent=body_style, fontName='Helvetica-Bold')
                        alert_style = ParagraphStyle('AlertText', parent=body_style, textColor=colors.HexColor('#C53030'), fontName='Helvetica-Bold')
                        white_header_style = ParagraphStyle('WhiteHeader', parent=body_style, fontName='Helvetica-Bold', textColor=colors.white)
                        dark_header_style = ParagraphStyle('DarkHeader', parent=body_style, fontName='Helvetica-Bold', textColor=colors.HexColor('#1A365D'))
                        
                        story = [
                            Paragraph("🫁 AI AIRWAY DIAGNOSTICS & BIOMETRIC ASSESSMENT REPORT", title_style),
                            Paragraph(f"<b>GENERATED ON:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <b>SYSTEM PROTOCOL:</b> BAAIS CORE V1.1.0", subtitle_style),
                            Paragraph("<b>CONFIDENTIAL MEDICAL DATA RECORD — FOR CLINICAL INTERPRETATION ONLY</b>", alert_style),
                            Spacer(1, 10)
                        ]
                        
                        # 1. Patient Profile Table
                        story.append(Paragraph("1. Patient Demographic & Environmental Profile", section_style))
                        profile_data = [
                            [Paragraph("Clinical Parameter", white_header_style), Paragraph("Patient Value", white_header_style), Paragraph("Environmental Stressor", white_header_style), Paragraph("Value Logged", white_header_style)],
                            [Paragraph("Chronological Age", body_style), Paragraph(f"{age} Years", body_style), Paragraph("Air Quality Index (AQI)", body_style), Paragraph(f"{aqi} (PM2.5 Matrix)", body_style)],
                            [Paragraph("Biological Sex", body_style), Paragraph(gender, body_style), Paragraph("Ambient Humidity", body_style), Paragraph(f"{humidity}% RH", body_style)],
                            [Paragraph("Asthma History", body_style), Paragraph(asthma, bold_body_style), Paragraph("COPD History", body_style), Paragraph(copd, bold_body_style)],
                            [Paragraph("Smoking Status", body_style), Paragraph(smoking, body_style), Paragraph("Registry Cohort Group", body_style), Paragraph(ethnicity, body_style)]
                        ]
                        t1 = Table(profile_data, colWidths=[135, 135, 145, 125])
                        t1.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
                            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
                            ('PADDING', (0,0), (-1,-1), 6),
                            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F7FAFC'))
                        ]))
                        story.append(t1)
                        
                        story.append(Spacer(1, 12))
                        
                        # 2. BAAIS Core Metrics Table
                        story.append(Paragraph("2. Bio-Inspired Adaptive Airway Intelligence (BAAIS) Summary", section_style))
                        baais_data = [
                            [Paragraph("Diagnostic Metric", white_header_style), Paragraph("Value / Output", white_header_style), Paragraph("Clinical Interpretation & System Bounds", white_header_style)],
                            [Paragraph("Overall Calculated Risk Score", alert_style if overall_risk_percentage > 50 else bold_body_style), Paragraph(risk_val_str, bold_body_style), Paragraph("Aggregated probabilistic index tracking structural airway restriction vectors.", body_style)],
                            [Paragraph("BAAIS Scaling Coefficient", body_style), Paragraph(f"{baais_coeff:.2f}", body_style), Paragraph("Mathematical compliance modifier. Value below 0.65 suggests respiratory duress.", body_style)],
                            [Paragraph("Estimated Elastic Compliance", body_style), Paragraph(f"{base_compliance:.2f}", body_style), Paragraph("Tracking volumetric lung recoil elasticity properties based on demographic limits.", body_style)],
                            [Paragraph("Homeostatic Adaptation State", body_style), Paragraph(adaptation_state, bold_body_style), Paragraph("Categorized physiological stress buffer status determined by core matrix models.", body_style)]
                        ]
                        t2 = Table(baais_data, colWidths=[160, 90, 290])
                        t2.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4A5568')),
                            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
                            ('PADDING', (0,0), (-1,-1), 6),
                            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                            ('BACKGROUND', (0,1), (-1,-1), colors.white)
                        ]))
                        story.append(t2)
                        
                        story.append(Spacer(1, 12))
                        
                        # 3. Acoustic Slicing Analysis Table
                        story.append(Paragraph("3. Acoustic Audio Slicing Matrix Timeline Analysis", section_style))
                        story.append(Paragraph("The ingested respiratory signal was processed through a short-time windowing algorithm (3.0s bounds, 2.5s window steps) to yield continuous overlapping segment inferences via the deep convolutional network core:", body_style))
                        story.append(Spacer(1, 6))
                        
                        slice_data = [[
                            Paragraph("Segment", dark_header_style), 
                            Paragraph("Time Interval", dark_header_style), 
                            Paragraph("Classification Result", dark_header_style), 
                            Paragraph("Raw CNN Conf.", dark_header_style), 
                            Paragraph("BAAIS Tuned Conf.", dark_header_style)
                        ]]
                        for idx, item in enumerate(results_found):
                            slice_data.append([
                                Paragraph(f"Slice #{idx+1}", body_style), 
                                Paragraph(f"{item['time_start']:.1f}s - {item['time_end']:.1f}s", body_style), 
                                Paragraph(item['classification'], bold_body_style if item['classification'] != "Normal" else body_style), 
                                Paragraph(f"{item['raw_confidence']:.1f}%", body_style), 
                                Paragraph(f"{item['baais_optimized_confidence']:.1f}%", body_style)
                            ])
                        t3 = Table(slice_data, colWidths=[70, 100, 150, 110, 110])
                        t3.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EDF2F7')),
                            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
                            ('PADDING', (0,0), (-1,-1), 6),
                            ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
                        ]))
                        story.append(t3)
                        
                        story.append(Spacer(1, 12))
                        
                        # 4. Clinical Reference Guidelines & Definitions
                        story.append(Paragraph("4. Clinical Reference Guidelines & Classification Definitions", section_style))
                        guideline_text = """
                        • <b>Normal Vesicular Breathing:</b> Soft, low-pitched sounds heard throughout inspiration and fading out during expiration. Indicates uncompromised laminar airflow.<br/>
                        • <b>Crackles (Rales):</b> Discontinuous, adventitious lung sounds characterized by explosive popping. Associated with sudden opening of closed airways or fluid presence (e.g., Pneumonia, Fibrosis).<br/>
                        • <b>Wheezes:</b> Continuous, high-pitched whistling musical sounds produced by narrowed airways during expiration. Strongly correlated with bronchospasms and airway narrowing (e.g., Acute Asthma flare-ups, COPD exacerbation).
                        """
                        story.append(Paragraph(guideline_text, body_style))
                        
                        story.append(Spacer(1, 12))
                        
                        # 5. Triage Recommendations
                        story.append(Paragraph("5. AI-Generated Triage Recommendations & Action Plan", section_style))
                        
                        if overall_risk_percentage >= 65:
                            triage_action = "<b>🔴 HIGH RISK CRITERIA MET:</b> Immediate clinical review advised. Correlate findings with spirometry metrics (FEV1/FVC ratios) and evaluate for active bronchoconstriction. Consider adjustments to bronchodilator or anti-inflammatory therapeutic regimens."
                        elif overall_risk_percentage >= 35:
                            triage_action = "<b>🟡 MODERATE RISK CRITERIA MET:</b> Routine follow-up evaluation scheduled. Monitor patient symptoms longitudinally using the active trend dashboard tracking layers. Advise avoidance of high AQI ambient zones."
                        else:
                            triage_action = "<b>🟢 LOW RISK CRITERIA MET:</b> Maintain current prophylactic healthcare baseline management. No immediate clinical interventions requested based on digital acoustic markers."
                            
                        story.append(Paragraph(triage_action, body_style))
                        story.append(Spacer(1, 20))
                        
                        # Sign-off Blocks
                        story.append(Paragraph("<b>Report Authorization Sign-off:</b>", body_style))
                        story.append(Spacer(1, 15))
                        story.append(Paragraph("_______________________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ________________________", body_style))
                        story.append(Paragraph("Attending Physician / Lead Reviewer Signature &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date Evaluated", body_style))
                        
                        doc.build(story)
                        buffer.seek(0)
                        return buffer

                    st.markdown("<br>", unsafe_allow_html=True)
                    pdf_data = generate_pdf()
                    st.download_button(
                        label="📥 Download Comprehensive Clinical BAAIS PDF Report",
                        data=pdf_data,
                        file_name=f"Microservice_BAAIS_Airway_Report_{age}_{gender}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Failed to communicate with runtime microservice core pipeline: {str(e)}")