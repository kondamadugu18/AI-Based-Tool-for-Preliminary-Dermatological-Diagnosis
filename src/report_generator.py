"""
Clinical Diagnostic Summary Report Generator for DermaAI.
Produces clean, structured patient reports in HTML and Markdown formats
ready for viewing, printing, and medical export.
"""

from datetime import datetime
from typing import Any, Dict, Optional


def generate_html_report(
    patient_data: Dict[str, Any],
    tabular_results: Optional[Dict[str, Any]] = None,
    image_results: Optional[Dict[str, Any]] = None,
    condition_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates a professional, print-friendly HTML medical triage summary report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_id = patient_data.get("patient_id", f"PT-{datetime.now().strftime('%m%d%H%M')}")
    age = patient_data.get("age", "N/A")
    gender = patient_data.get("gender", "N/A")
    skin_type = patient_data.get("fitzpatrick_skin_type", "N/A")
    body_site = patient_data.get("body_site", "N/A")

    # Tabular results
    primary_diag = "Inconclusive / Pending"
    confidence_pct = 0.0
    urgency = "Routine Care / Monitoring"
    differential_rows = ""

    if tabular_results:
        primary_diag = tabular_results.get("primary_diagnosis", "N/A")
        confidence_pct = tabular_results.get("confidence_percentage", 0.0)
        urgency = tabular_results.get("urgency", urgency)
        for diff in tabular_results.get("differential_diagnoses", []):
            differential_rows += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">{diff['condition']}</td>
                <td style="padding: 8px; border-bottom: 1px solid #e2e8f0; text-align: right;"><b>{diff['percentage']}%</b></td>
            </tr>
            """

    # Image / ABCDE results
    abcde_rows = ""
    tds_score = "N/A"
    img_risk = "N/A"
    if image_results and "metrics" in image_results:
        m = image_results["metrics"]
        tds_score = f"{m.get('total_dermoscopy_score', 'N/A')}"
        img_risk = m.get("risk_level", "N/A")
        abcde_rows = f"""
        <tr><td><b>A - Asymmetry Index</b></td><td>{m.get('asymmetry_score', 'N/A')} / 2.0</td></tr>
        <tr><td><b>B - Border Irregularity</b></td><td>{m.get('border_score', 'N/A')} / 8.0</td></tr>
        <tr><td><b>C - Color Variegation</b></td><td>{m.get('color_score', 'N/A')} / 6.0</td></tr>
        <tr><td><b>D - Lesion Diameter</b></td><td>{m.get('diameter_mm', 'N/A')} mm</td></tr>
        <tr><td><b>E - Texture Entropy</b></td><td>{m.get('evolution_texture_score', 'N/A')} / 10.0</td></tr>
        <tr><td><b>Total Dermoscopy Score (TDS)</b></td><td><b>{tds_score}</b> ({img_risk})</td></tr>
        """

    # Condition knowledge guidance
    guidance_section = ""
    if condition_info:
        rec_list = "".join([f"<li>{t}</li>" for t in condition_info.get("management_tips", [])])
        red_flags = "".join([f"<li>{r}</li>" for r in condition_info.get("red_flags", [])])
        guidance_section = f"""
        <div style="margin-top: 20px; padding: 15px; background: #f8fafc; border-left: 4px solid #0284c7; border-radius: 4px;">
            <h3 style="margin-top: 0; color: #0f172a;">Condition Clinical Overview: {condition_info.get('name', primary_diag)}</h3>
            <p style="color: #334155; font-size: 14px;">{condition_info.get('description', '')}</p>
            <h4 style="margin-bottom: 5px; color: #0369a1;">Standard Clinical Management & Self-Care:</h4>
            <ul style="color: #334155; font-size: 14px; margin-top: 5px;">{rec_list}</ul>
            <h4 style="margin-bottom: 5px; color: #b91c1c;">Red Flags (Seek Immediate Medical Attention If):</h4>
            <ul style="color: #b91c1c; font-size: 14px; margin-top: 5px;">{red_flags}</ul>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>DermaAI Preliminary Diagnostic Summary Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
                background-color: #ffffff;
                color: #0f172a;
                margin: 0;
                padding: 30px;
                line-height: 1.5;
            }}
            .report-card {{
                max-width: 800px;
                margin: 0 auto;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 24px;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            }}
            .header {{
                border-bottom: 2px solid #0284c7;
                padding-bottom: 12px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .badge-urgent {{
                background-color: #fee2e2;
                color: #991b1b;
                padding: 6px 14px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}
            th {{
                text-align: left;
                background-color: #f1f5f9;
                padding: 8px;
            }}
            td {{
                padding: 8px;
                border-bottom: 1px solid #e2e8f0;
            }}
            .disclaimer {{
                margin-top: 30px;
                padding: 12px;
                background-color: #fffbeb;
                border: 1px solid #fde68a;
                border-radius: 6px;
                color: #92400e;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="report-card">
            <div class="header">
                <div>
                    <h2 style="margin: 0; color: #0369a1;">🩺 DermaAI Clinical Summary Report</h2>
                    <p style="margin: 4px 0 0 0; color: #64748b; font-size: 13px;">AI-Based Preliminary Dermatological Decision Support</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #64748b;">Report Date</div>
                    <div style="font-weight: 600; font-size: 14px;">{timestamp}</div>
                </div>
            </div>

            <!-- Patient Demographics -->
            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; font-size: 14px;">
                    <span><b>Patient Reference:</b> {patient_id}</span>
                    <span><b>Age/Gender:</b> {age} yrs / {gender}</span>
                    <span><b>Skin Phototype:</b> Fitzpatrick Type {skin_type}</span>
                    <span><b>Location:</b> {body_site}</span>
                </div>
            </div>

            <!-- Primary Finding -->
            <div style="border: 1px solid #e2e8f0; border-radius: 6px; padding: 16px; margin-bottom: 20px; background: #fafafa;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Primary Preliminary Diagnosis</div>
                        <div style="font-size: 24px; font-weight: 700; color: #0f172a; margin-top: 4px;">{primary_diag}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 12px; color: #64748b;">Model Confidence</div>
                        <div style="font-size: 22px; font-weight: 700; color: #0284c7;">{confidence_pct}%</div>
                    </div>
                </div>
                <div style="margin-top: 12px; padding: 8px 12px; background: #e0f2fe; border-radius: 4px; color: #0369a1; font-size: 14px; font-weight: 600;">
                    Triage Urgency: {urgency}
                </div>
            </div>

            <div class="grid-2">
                <!-- Differential Diagnosis -->
                <div>
                    <h4 style="margin-top: 0; margin-bottom: 8px; color: #334155;">Differential Diagnosis (Top Likelihoods)</h4>
                    <table>
                        <thead>
                            <tr><th>Condition</th><th style="text-align: right;">Probability</th></tr>
                        </thead>
                        <tbody>
                            {differential_rows}
                        </tbody>
                    </table>
                </div>

                <!-- Dermoscopy Metrics if present -->
                <div>
                    <h4 style="margin-top: 0; margin-bottom: 8px; color: #334155;">Dermoscopy / ABCDE Lesion Metrics</h4>
                    <table>
                        <tbody>
                            {abcde_rows if abcde_rows else "<tr><td colspan='2' style='color:#64748b;'>No lesion image submitted for this session.</td></tr>"}
                        </tbody>
                    </table>
                </div>
            </div>

            {guidance_section}

            <div class="disclaimer">
                <b>IMPORTANT CLINICAL DISCLAIMER:</b> This computer-generated report is generated by an artificial intelligence diagnostic decision support algorithm for preliminary educational and triage prioritization purposes only. It does <b>NOT</b> constitute a definitive medical diagnosis, pathology report, or prescription. A qualified, licensed dermatologist or physician must perform clinical dermoscopy, physical examination, and histopathological biopsy where indicated.
            </div>
        </div>
    </body>
    </html>
    """
    return html


def generate_markdown_report(
    patient_data: Dict[str, Any],
    tabular_results: Optional[Dict[str, Any]] = None,
    image_results: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates a clean Markdown representation of the patient diagnostic summary."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_id = patient_data.get("patient_id", "PT-ANON")

    md = f"""# 🩺 DermaAI Diagnostic Summary Report
**Date/Time:** {timestamp}  
**Patient Identifier:** `{patient_id}`  

---

### 👤 Patient Clinical Profile
- **Age / Gender:** {patient_data.get('age', 'N/A')} years / {patient_data.get('gender', 'N/A')}
- **Fitzpatrick Phototype:** Type {patient_data.get('fitzpatrick_skin_type', 'N/A')}
- **Lesion Anatomical Site:** {patient_data.get('body_site', 'N/A')}
- **Lesion Size:** {patient_data.get('lesion_size_mm', 'N/A')} mm
- **Symptoms:** Itching: {patient_data.get('itching', 'N/A')}/5 | Redness: {patient_data.get('redness', 'N/A')}/5 | Scaling: {patient_data.get('scaling_peeling', 'N/A')}/5

---

### 🔍 AI Diagnostic Findings
"""
    if tabular_results:
        md += f"""- **Primary Suspected Condition:** **{tabular_results.get('primary_diagnosis')}**
- **Confidence Level:** **{tabular_results.get('confidence_percentage')}%**
- **Recommended Triage Urgency:** {tabular_results.get('urgency')}

#### Top Differential Diagnoses:
"""
        for d in tabular_results.get("differential_diagnoses", []):
            md += f"- **{d['condition']}**: {d['percentage']}%\n"

    if image_results and "metrics" in image_results:
        m = image_results["metrics"]
        md += f"""
---

### 🔬 Dermoscopy ABCDE Morphometric Assessment
- **Asymmetry Score:** {m.get('asymmetry_score')} / 2.0
- **Border Irregularity Score:** {m.get('border_score')} / 8.0
- **Color Variegation Score:** {m.get('color_score')} / 6.0
- **Equivalent Diameter:** {m.get('diameter_mm')} mm
- **Texture Entropy:** {m.get('evolution_texture_score')} / 10.0
- **Total Dermoscopy Score (TDS):** `{m.get('total_dermoscopy_score')}` ({m.get('risk_level')})
"""

    md += """
---
> **Clinical Disclaimer:** This tool provides preliminary algorithmic decision support. A formal clinical evaluation and biopsy by a board-certified dermatologist is required for definitive diagnostic confirmation.
"""
    return md
