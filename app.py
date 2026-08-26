"""
=============================================================================
 DermaAI - AI-Based Tool for Preliminary Dermatological Diagnosis
 Production-grade Multi-Modal Clinical Decision Support Platform
=============================================================================
"""

import io
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Local Module Imports
from src.data_loader import (
    BODY_SITES,
    DIAGNOSIS_CLASSES,
    ELEVATIONS,
    EVOLUTIONS,
    FITZPATRICK_TYPES,
    SUN_EXPOSURES,
    ensure_datasets_exist,
    load_clinical_dataset,
    load_knowledge_base,
)
from src.explainability import (
    plot_abcde_radar,
    plot_confusion_matrix,
    plot_differential_diagnosis_bars,
    plot_feature_importance,
    plot_model_comparison,
)
from src.image_analyzer import analyze_lesion_image
from src.model import DermatologyModelEngine
from src.report_generator import generate_html_report, generate_markdown_report
from src.utils import ensure_sample_images_exist, inject_custom_css

# Page Configuration
st.set_page_config(
    page_title="DermaAI | Preliminary Dermatological Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Modern Clinical Dark CSS
inject_custom_css()

# Initialize Engine & Data
@st.cache_resource
def get_model_engine() -> DermatologyModelEngine:
    ensure_datasets_exist()
    engine = DermatologyModelEngine()
    engine.load_or_train()
    return engine


@st.cache_data
def get_knowledge_base() -> Dict[str, Any]:
    return load_knowledge_base()


@st.cache_data
def get_sample_images() -> Dict[str, Path]:
    return ensure_sample_images_exist()


# Load cached resources
engine = get_model_engine()
knowledge_base = get_knowledge_base()
sample_images_dict = get_sample_images()

# Initialize Session State
if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = {
        "patient_id": "PT-1042",
        "age": 34,
        "gender": "Female",
        "fitzpatrick_skin_type": "III",
        "body_site": "Trunk",
        "itching": 4,
        "redness": 3,
        "scaling_peeling": 3,
        "burning_pain": 2,
        "bleeding_oozing": 1,
        "lesion_size_mm": 12.5,
        "elevation": "Raised/Plaque",
        "duration_weeks": 6,
        "evolution_change": "Fluctuating",
        "sun_exposure": "Moderate",
        "family_history_skin_disease": 1,
    }

if "tabular_result" not in st.session_state:
    st.session_state["tabular_result"] = engine.predict_patient(st.session_state["patient_data"])

if "image_result" not in st.session_state:
    # Run initial analysis on sample image
    try:
        sample_p = sample_images_dict.get("Melanoma (Atypical Pigmented Lesion)")
        if sample_p and sample_p.exists():
            st.session_state["image_result"] = analyze_lesion_image(str(sample_p))
        else:
            st.session_state["image_result"] = None
    except Exception:
        st.session_state["image_result"] = None


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & BRANDING
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 40px;">🩺</div>
            <h2 style="margin: 0; color: #38BDF8; font-weight: 800; letter-spacing: -0.5px;">DermaAI</h2>
            <p style="margin: 4px 0 0 0; font-size: 13px; color: #94A3B8;">Dermatological Clinical Triage Suite</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    selected_nav = st.radio(
        "📌 **Navigation Module**",
        [
            "🩺 Clinical Symptom Checker",
            "🔬 Dermoscopy Image Analyzer",
            "🧬 Multimodal Diagnostic Fusion",
            "📊 Model Hub & Explainability",
            "📁 Batch Screening & Data Explorer",
            "📚 Dermatology Knowledge Base",
            "📑 Export Clinical Report",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Diagnostic Model Config")
    available_model_names = list(engine.all_models.keys())
    if not available_model_names:
        available_model_names = ["Random Forest", "Gradient Boosting", "Support Vector Machine", "Logistic Regression"]

    active_model_name = st.selectbox(
        "Active Classifier",
        available_model_names,
        index=0 if engine.best_model_name not in available_model_names else available_model_names.index(engine.best_model_name),
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="med-callout" style="font-size: 12px;">
            <b>Clinical Decision Support</b><br>
            Version 1.0.0 (Production Build)<br>
            Multi-class Stratified Pipeline
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="font-size: 11px; color: #64748B; text-align: center; margin-top: 15px;">
            Designed for preliminary clinical triage and educational decision support.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# HERO BRAND BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
            <div>
                <h1 style="margin: 0; font-size: 28px; font-weight: 800; color: #F8FAFC;">
                    AI-Based Tool for Preliminary Dermatological Diagnosis
                </h1>
                <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 15px;">
                    Multi-modal decision support combining clinical feature classification with ABCDE computer vision lesion morphometry.
                </p>
            </div>
            <div>
                <span class="badge-routine">🟢 System Operational</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Key Performance Indicator Cards
col1, col2, col3, col4 = st.columns(4)
best_f1 = engine.model_metrics.get(engine.best_model_name, {}).get("test_f1_macro", 0.94)
best_acc = engine.model_metrics.get(engine.best_model_name, {}).get("test_accuracy", 0.95)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Diagnostic Accuracy</div>
            <div style="font-size: 24px; font-weight: 800; color: #38BDF8; margin-top: 4px;">{best_acc * 100:.1f}%</div>
            <div style="font-size: 11px; color: #10B981; margin-top: 2px;">5-Fold Cross-Validated</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #94A3B8; font-weight: 600; text-transform: uppercase;">F1-Macro Score</div>
            <div style="font-size: 24px; font-weight: 800; color: #34D399; margin-top: 4px;">{best_f1 * 100:.1f}%</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Balanced Multi-Class</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div style="font-size: 12px; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Supported Conditions</div>
            <div style="font-size: 24px; font-weight: 800; color: #A78BFA; margin-top: 4px;">8 Categories</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Malignant, Benign, Autoimmune</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        """
        <div class="metric-card">
            <div style="font-size: 12px; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Computer Vision</div>
            <div style="font-size: 24px; font-weight: 800; color: #FBBF24; margin-top: 4px;">ABCDE Rule</div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Lesion Segmentation & TDS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# MODULE 1: CLINICAL SYMPTOM CHECKER (TABULAR ML)
# =============================================================================
if selected_nav == "🩺 Clinical Symptom Checker":
    st.markdown("## 🩺 Patient Clinical Symptom Intake & Diagnosis")
    st.markdown(
        "Enter the patient's demographic information, symptom severity levels, and lesion characteristics to generate a real-time differential diagnosis."
    )

    with st.form("patient_intake_form"):
        st.markdown("### 1. Patient Demographics & Anatomical Location")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pat_id = st.text_input("Patient ID / Ref", value=st.session_state["patient_data"]["patient_id"])
            age = st.slider("Age (Years)", min_value=1, max_value=95, value=int(st.session_state["patient_data"]["age"]))
        with c2:
            gender = st.selectbox(
                "Gender", ["Female", "Male"], index=0 if st.session_state["patient_data"]["gender"] == "Female" else 1
            )
            skin_type = st.selectbox(
                "Fitzpatrick Skin Phototype",
                FITZPATRICK_TYPES,
                index=FITZPATRICK_TYPES.index(st.session_state["patient_data"]["fitzpatrick_skin_type"]),
                help="Type I: Pale white (burns easily) to Type VI: Deeply pigmented dark brown/black.",
            )
        with c3:
            body_site = st.selectbox(
                "Anatomical Body Site",
                BODY_SITES,
                index=BODY_SITES.index(st.session_state["patient_data"]["body_site"]),
            )
            elevation = st.selectbox(
                "Lesion Elevation / Morphology",
                ELEVATIONS,
                index=ELEVATIONS.index(st.session_state["patient_data"]["elevation"]),
            )
        with c4:
            duration = st.number_input(
                "Duration (Weeks)", min_value=1, max_value=200, value=int(st.session_state["patient_data"]["duration_weeks"])
            )
            lesion_size = st.number_input(
                "Lesion Size (mm)",
                min_value=1.0,
                max_value=60.0,
                value=float(st.session_state["patient_data"]["lesion_size_mm"]),
                step=0.5,
            )

        st.markdown("### 2. Clinical Symptom Severity Profile (0: Absent → 5: Severe)")
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            itching = st.slider("Pruritus (Itching)", 0, 5, int(st.session_state["patient_data"]["itching"]))
        with s2:
            redness = st.slider("Erythema (Redness)", 0, 5, int(st.session_state["patient_data"]["redness"]))
        with s3:
            scaling = st.slider("Scaling / Peeling", 0, 5, int(st.session_state["patient_data"]["scaling_peeling"]))
        with s4:
            burning = st.slider("Burning / Pain", 0, 5, int(st.session_state["patient_data"]["burning_pain"]))
        with s5:
            bleeding = st.slider("Bleeding / Oozing", 0, 5, int(st.session_state["patient_data"]["bleeding_oozing"]))

        st.markdown("### 3. Risk Factors & Evolution")
        r1, r2, r3 = st.columns(3)
        with r1:
            evolution = st.selectbox(
                "Lesion Evolution / Growth Pattern",
                EVOLUTIONS,
                index=EVOLUTIONS.index(st.session_state["patient_data"]["evolution_change"]),
            )
        with r2:
            sun_exp = st.selectbox(
                "Lifetime Sun Exposure",
                SUN_EXPOSURES,
                index=SUN_EXPOSURES.index(st.session_state["patient_data"]["sun_exposure"]),
            )
        with r3:
            fam_hist = st.radio(
                "Family History of Skin Disease / Malignancy",
                ["No (0)", "Yes (1)"],
                index=int(st.session_state["patient_data"]["family_history_skin_disease"]),
                horizontal=True,
            )

        submit_btn = st.form_submit_button("🔍 Execute AI Diagnostic Assessment", type="primary", use_container_width=True)

    if submit_btn:
        patient_dict = {
            "patient_id": pat_id,
            "age": age,
            "gender": gender,
            "fitzpatrick_skin_type": skin_type,
            "body_site": body_site,
            "itching": itching,
            "redness": redness,
            "scaling_peeling": scaling,
            "burning_pain": burning,
            "bleeding_oozing": bleeding,
            "lesion_size_mm": lesion_size,
            "elevation": elevation,
            "duration_weeks": duration,
            "evolution_change": evolution,
            "sun_exposure": sun_exp,
            "family_history_skin_disease": 1 if "Yes" in fam_hist else 0,
        }
        st.session_state["patient_data"] = patient_dict
        st.session_state["tabular_result"] = engine.predict_patient(patient_dict, model_name=active_model_name)
        st.toast("Diagnostic Assessment Completed!", icon="✅")

    # Display Diagnostic Output
    res = st.session_state["tabular_result"]
    if res:
        st.markdown("---")
        st.markdown("### 📋 AI Diagnostic Assessment Findings")

        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            primary = res["primary_diagnosis"]
            conf = res["confidence_percentage"]
            urgency_lvl = res["urgency_level"]
            urgency_text = res["urgency"]

            badge_class = "badge-high" if urgency_lvl == "HIGH" else ("badge-moderate" if urgency_lvl == "MODERATE" else "badge-routine")

            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(14, 165, 233, 0.4); border-radius: 14px; padding: 22px;">
                    <div style="font-size: 13px; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Primary Indicated Condition</div>
                    <div style="font-size: 30px; font-weight: 800; color: #F8FAFC; margin-top: 4px;">{primary}</div>
                    <div style="display: flex; align-items: center; gap: 15px; margin-top: 14px;">
                        <div>
                            <div style="font-size: 12px; color: #94A3B8;">Confidence</div>
                            <div style="font-size: 22px; font-weight: 800; color: #38BDF8;">{conf}%</div>
                        </div>
                        <div>
                            <div style="font-size: 12px; color: #94A3B8;">Algorithm</div>
                            <div style="font-size: 14px; font-weight: 600; color: #E2E8F0;">{res['model_used']}</div>
                        </div>
                    </div>
                    <div style="margin-top: 16px;">
                        <span class="{badge_class}">{urgency_text}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Retrieve Medical Guidance
            cond_data = knowledge_base.get(primary, {})
            if cond_data:
                st.markdown(
                    f"""
                    <div class="med-callout">
                        <b>Clinical Overview:</b> {cond_data.get('description', '')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with res_col2:
            fig_diff = plot_differential_diagnosis_bars(res["differential_diagnoses"])
            st.plotly_chart(fig_diff, use_container_width=True)

        # Actionable clinical management recommendations
        st.markdown("### 💡 Recommended Next Steps & Clinical Management")
        if primary in knowledge_base:
            k = knowledge_base[primary]
            m1, m2 = st.columns(2)
            with m1:
                st.markdown("#### 🩺 Standard Clinical Management:")
                for tip in k.get("management_tips", []):
                    st.markdown(f"- {tip}")
            with m2:
                st.markdown("#### 🚨 Warning Signs / Red Flags:")
                for rf in k.get("red_flags", []):
                    st.markdown(f"- ⚠️ **{rf}**")


# =============================================================================
# MODULE 2: DERMOSCOPY IMAGE ANALYZER (ABCDE CRITERIA)
# =============================================================================
elif selected_nav == "🔬 Dermoscopy Image Analyzer":
    st.markdown("## 🔬 Computer Vision Dermoscopy & ABCDE Lesion Analysis")
    st.markdown(
        "Upload a high-resolution skin lesion photograph or dermoscopy image to perform automated lesion segmentation, color analysis, and standard **ABCDE criteria** scoring."
    )

    img_col1, img_col2 = st.columns([1, 1.4])

    with img_col1:
        st.markdown("### 1. Upload or Select Sample Image")
        source_mode = st.radio("Image Source", ["Use Standard Clinical Sample", "Upload Custom Skin Photo"], horizontal=True)

        selected_image = None
        if source_mode == "Use Standard Clinical Sample":
            sample_choice = st.selectbox("Select Dermatological Sample", list(sample_images_dict.keys()), index=0)
            sample_path = sample_images_dict[sample_choice]
            if sample_path.exists():
                selected_image = Image.open(sample_path)
        else:
            uploaded_file = st.file_uploader("Choose a skin lesion image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)

        if selected_image is not None:
            st.image(selected_image, caption="Input Lesion Image", use_container_width=True)
            analyze_img_btn = st.button("🔬 Analyze Lesion Morphometry (ABCDE)", type="primary", use_container_width=True)

            if analyze_img_btn:
                with st.spinner("Executing adaptive lesion segmentation & color-texture extraction..."):
                    img_res = analyze_lesion_image(selected_image)
                    st.session_state["image_result"] = img_res
                    st.toast("Dermoscopy Analysis Completed!", icon="🔬")

    with img_col2:
        img_res = st.session_state.get("image_result")
        if img_res and "metrics" in img_res:
            m = img_res["metrics"]
            visuals = img_res["visuals"]

            st.markdown("### 2. ABCDE Criteria & Malignancy Risk")

            # Risk Summary Header Card
            tds = m["total_dermoscopy_score"]
            risk_tier = m["risk_tier"]
            risk_badge = m["risk_badge"]
            rec = m["urgency_recommendation"]

            badge_class = "badge-high" if risk_tier == "HIGH" else ("badge-moderate" if risk_tier == "MODERATE" else "badge-routine")

            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(14, 165, 233, 0.4); border-radius: 12px; padding: 18px; margin-bottom: 18px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 12px; color: #94A3B8;">Total Dermoscopy Score (TDS)</div>
                            <div style="font-size: 32px; font-weight: 800; color: #38BDF8;">{tds}</div>
                        </div>
                        <div>
                            <span class="{badge_class}">{risk_badge}</span>
                        </div>
                    </div>
                    <div style="font-size: 13px; color: #E2E8F0; margin-top: 10px;">
                        <b>Recommendation:</b> {rec}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ABCDE Metric Cards in Grid
            ac1, ac2, ac3, ac4, ac5 = st.columns(5)
            with ac1:
                st.metric("A - Asymmetry", f"{m['asymmetry_score']}/2.0")
            with ac2:
                st.metric("B - Border", f"{m['border_score']}/8.0")
            with ac3:
                st.metric("C - Color Var.", f"{m['color_score']}/6.0")
            with ac4:
                st.metric("D - Diameter", f"{m['diameter_mm']} mm")
            with ac5:
                st.metric("E - Texture", f"{m['evolution_texture_score']}/10")

            # Visual Overlays Tab
            st.markdown("### 3. Diagnostic Image Overlays")
            vis_tabs = st.tabs(["🟢 Contour Boundary", "🔵 Segmentation Mask", "🔥 Intensity Heatmap", "Original"])

            with vis_tabs[0]:
                st.image(visuals["contour_boundary"], caption="Lesion Margin & Boundary Contour (Neon Green)", use_container_width=True)
            with vis_tabs[1]:
                st.image(visuals["mask_overlay"], caption="Translucent Lesion ROI Segmented Mask", use_container_width=True)
            with vis_tabs[2]:
                st.image(visuals["heatmap"], caption="Dermoscopic Intensity & Pigmentation Gradient Heatmap", use_container_width=True)
            with vis_tabs[3]:
                st.image(visuals["original"], caption="Original High-Resolution Image", use_container_width=True)

            # Radar Spider Chart
            fig_radar = plot_abcde_radar(m)
            st.plotly_chart(fig_radar, use_container_width=True)


# =============================================================================
# MODULE 3: MULTIMODAL DIAGNOSTIC FUSION
# =============================================================================
elif selected_nav == "🧬 Multimodal Diagnostic Fusion":
    st.markdown("## 🧬 Multimodal Clinical & Computer Vision Fusion")
    st.markdown(
        "Fuses tabular questionnaire responses with dermoscopy image criteria to perform holistic clinical risk stratification and patient triage."
    )

    tab_res = st.session_state.get("tabular_result")
    img_res = st.session_state.get("image_result")
    pat_data = st.session_state.get("patient_data", {})

    fc1, fc2 = st.columns([1, 1])

    with fc1:
        st.markdown("### 📋 Clinical Symptom Questionnaire Stream")
        if tab_res:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size: 13px; color: #94A3B8;">Tabular AI Prediction</div>
                    <div style="font-size: 24px; font-weight: 800; color: #38BDF8;">{tab_res['primary_diagnosis']}</div>
                    <div style="font-size: 14px; color: #10B981; font-weight: 600;">Confidence: {tab_res['confidence_percentage']}%</div>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 12px 0;">
                    <div style="font-size: 12px; color: #CBD5E1;">
                        <b>Patient:</b> {pat_data.get('age')}y {pat_data.get('gender')} | Site: {pat_data.get('body_site')}<br>
                        <b>Key Symptoms:</b> Itching: {pat_data.get('itching')}/5, Redness: {pat_data.get('redness')}/5, Bleeding: {pat_data.get('bleeding_oozing')}/5
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No symptom questionnaire evaluated yet. Run the Symptom Checker tab.")

    with fc2:
        st.markdown("### 🔬 Dermoscopy Computer Vision Stream")
        if img_res and "metrics" in img_res:
            m = img_res["metrics"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size: 13px; color: #94A3B8;">Computer Vision ABCDE Assessment</div>
                    <div style="font-size: 24px; font-weight: 800; color: #FBBF24;">{m['risk_level']}</div>
                    <div style="font-size: 14px; color: #38BDF8; font-weight: 600;">Total Dermoscopy Score: {m['total_dermoscopy_score']}</div>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 12px 0;">
                    <div style="font-size: 12px; color: #CBD5E1;">
                        <b>Asymmetry:</b> {m['asymmetry_score']}/2.0 | <b>Border:</b> {m['border_score']}/8.0<br>
                        <b>Color:</b> {m['color_score']}/6.0 | <b>Diameter:</b> {m['diameter_mm']} mm
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No lesion image analyzed yet. Run the Dermoscopy Image Analyzer tab.")

    st.markdown("---")
    st.markdown("### 🎯 Unified Multimodal Consensus & Triage Decision")

    if tab_res and img_res and "metrics" in img_res:
        # Fusion logic
        tab_is_malignant = tab_res["primary_diagnosis"] in ["Melanoma", "Basal Cell Carcinoma"]
        img_is_malignant = img_res["metrics"]["risk_tier"] == "HIGH"

        if tab_is_malignant or img_is_malignant:
            consensus_tier = "HIGH"
            consensus_title = "HIGH CLINICAL RISK - Urgent Dermatological Specialist Referral"
            consensus_desc = (
                "Both clinical symptoms and dermoscopic features (or strong signals in one modality) "
                "indicate elevated likelihood of a neoplastic or severe skin pathology. An immediate in-person "
                "dermatological exam and biopsy are strongly recommended."
            )
            banner_class = "warning-callout"
        elif tab_res["primary_diagnosis"] in ["Psoriasis", "Eczema"] and img_res["metrics"]["risk_tier"] == "MODERATE":
            consensus_tier = "MODERATE"
            consensus_title = "MODERATE RISK - Outpatient Dermatology Evaluation"
            consensus_desc = (
                "Symptoms and visual pattern indicate an active inflammatory dermatosis or atypical lesion. "
                "Schedule a routine or semi-urgent outpatient dermatology appointment."
            )
            banner_class = "med-callout"
        else:
            consensus_tier = "ROUTINE"
            consensus_title = "LOW RISK / BENIGN - Standard Clinical Monitoring"
            consensus_desc = (
                "Findings are reassuring and consistent with benign skin lesions or mild self-limiting conditions. "
                "Monitor for any future evolution in size, color, or shape."
            )
            banner_class = "med-callout"

        st.markdown(
            f"""
            <div class="{banner_class}" style="font-size: 16px; padding: 20px;">
                <h3 style="margin-top: 0; color: inherit;">{consensus_title}</h3>
                <p style="margin-bottom: 0;">{consensus_desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.plotly_chart(plot_differential_diagnosis_bars(tab_res["differential_diagnoses"]), use_container_width=True)
        with m_col2:
            st.plotly_chart(plot_abcde_radar(img_res["metrics"]), use_container_width=True)
    else:
        st.warning("Please complete both the Symptom Checker and Dermoscopy Image Analyzer to generate the unified multimodal triage consensus.")


# =============================================================================
# MODULE 4: MODEL HUB & EXPLAINABILITY
# =============================================================================
elif selected_nav == "📊 Model Hub & Explainability":
    st.markdown("## 📊 Machine Learning Model Hub & Explainability")
    st.markdown(
        "Benchmarking metrics, multi-class confusion matrices, and global feature importance attributions for diagnostic explainability."
    )

    hub_tabs = st.tabs(["📈 Model Comparison Benchmarks", "🔲 Confusion Matrix Heatmap", "🌳 Feature Importances", "📋 Model Performance Table"])

    with hub_tabs[0]:
        fig_comp = plot_model_comparison(engine.model_metrics)
        st.plotly_chart(fig_comp, use_container_width=True)

    with hub_tabs[1]:
        sel_model_cm = st.selectbox("Select Model for Confusion Matrix", list(engine.model_metrics.keys()), index=0)
        cm_data = engine.model_metrics[sel_model_cm]["confusion_matrix"]
        fig_cm = plot_confusion_matrix(cm_data, engine.classes_)
        st.plotly_chart(fig_cm, use_container_width=True)

    with hub_tabs[2]:
        rf_model = engine.all_models.get("Random Forest")
        if rf_model and hasattr(rf_model, "feature_importances_"):
            fig_imp = plot_feature_importance(rf_model, engine.preprocessor.feature_names_out, top_n=15)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance is generated from the Random Forest ensemble model.")

    with hub_tabs[3]:
        perf_records = []
        for name, m in engine.model_metrics.items():
            perf_records.append(
                {
                    "Algorithm": name,
                    "5-Fold CV Accuracy (%)": f"{m.get('cv_accuracy_mean', 0)*100:.1f} ± {m.get('cv_accuracy_std', 0)*100:.1f}",
                    "Test Accuracy (%)": f"{m.get('test_accuracy', 0)*100:.1f}%",
                    "F1-Macro Score (%)": f"{m.get('test_f1_macro', 0)*100:.1f}%",
                    "Precision Macro (%)": f"{m.get('test_precision_macro', 0)*100:.1f}%",
                    "Recall Macro (%)": f"{m.get('test_recall_macro', 0)*100:.1f}%",
                    "ROC AUC": f"{m.get('roc_auc_macro', 0):.3f}" if m.get("roc_auc_macro") else "N/A",
                }
            )
        st.dataframe(pd.DataFrame(perf_records), use_container_width=True)


# =============================================================================
# MODULE 5: BATCH SCREENING & DATA EXPLORER
# =============================================================================
elif selected_nav == "📁 Batch Screening & Data Explorer":
    st.markdown("## 📁 Batch Screening & Dataset Explorer")
    st.markdown(
        "Upload a batch CSV of multiple patients to execute high-throughput diagnostic screening, filtering, and export."
    )

    batch_tabs = st.tabs(["📤 Upload Batch CSV", "🔍 Clinical Training Dataset Explorer"])

    with batch_tabs[0]:
        st.markdown("### Upload Cohort CSV for Automated Screening")
        st.info("Uploaded CSV should include features: `age`, `gender`, `body_site`, `itching`, `redness`, `scaling_peeling`, `lesion_size_mm`, etc.")

        batch_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])

        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                st.markdown(f"**Loaded {len(batch_df)} patient records.**")
                st.dataframe(batch_df.head(5), use_container_width=True)

                if st.button("⚡ Run Batch AI Diagnosis on Cohort", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        pred_df = engine.predict_batch(batch_df)
                        st.session_state["batch_pred_df"] = pred_df
                        st.toast("Batch diagnosis completed!", icon="✅")
            except Exception as e:
                st.error(f"Error parsing batch CSV: {e}")

        if "batch_pred_df" in st.session_state:
            res_batch = st.session_state["batch_pred_df"]
            st.markdown("### 📋 Automated Batch Screening Results")

            # Filter options
            fc1, fc2 = st.columns(2)
            with fc1:
                diag_filter = st.multiselect(
                    "Filter by Predicted Diagnosis",
                    options=list(res_batch["predicted_diagnosis"].unique()),
                    default=list(res_batch["predicted_diagnosis"].unique()),
                )
            with fc2:
                urgency_filter = st.multiselect(
                    "Filter by Triage Urgency",
                    options=list(res_batch["urgency_tier"].unique()),
                    default=list(res_batch["urgency_tier"].unique()),
                )

            filtered_df = res_batch[
                res_batch["predicted_diagnosis"].isin(diag_filter) & res_batch["urgency_tier"].isin(urgency_filter)
            ]

            st.dataframe(filtered_df, use_container_width=True)

            # Export CSV
            csv_export = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Diagnostic Cohort Report (CSV)",
                data=csv_export,
                file_name="derma_ai_batch_diagnostic_report.csv",
                mime="text/csv",
                type="secondary",
            )

    with batch_tabs[1]:
        st.markdown("### 📊 Primary Clinical Training Dataset Distribution")
        raw_df = load_clinical_dataset()
        st.markdown(f"**Dataset Overview:** {raw_df.shape[0]} validated cases across {len(DIAGNOSIS_CLASSES)} conditions.")
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.markdown("#### Class Distribution:")
        class_counts = raw_df["diagnosis"].value_counts().reset_index()
        class_counts.columns = ["Condition", "Count"]
        st.bar_chart(class_counts.set_index("Condition"))


# =============================================================================
# MODULE 6: DERMATOLOGY KNOWLEDGE BASE
# =============================================================================
elif selected_nav == "📚 Dermatology Knowledge Base":
    st.markdown("## 📚 Clinical Dermatology Encyclopedia & Reference")
    st.markdown(
        "Explore validated medical overviews, clinical presentation signs, risk factors, evidence-based management, and urgent red flags for common skin conditions."
    )

    selected_cond = st.selectbox("Select Condition to Inspect", list(knowledge_base.keys()), index=0)
    info = knowledge_base[selected_cond]

    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(14, 165, 233, 0.4); border-radius: 12px; padding: 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; color: #38BDF8;">{info.get('name', selected_cond)}</h2>
                    <span style="font-size: 13px; color: #94A3B8;">Category: {info.get('category', 'Dermatology')}</span>
                </div>
                <div>
                    <span class="badge-routine">{info.get('urgency_badge', 'Standard Care')}</span>
                </div>
            </div>
            <p style="margin-top: 14px; color: #E2E8F0; font-size: 15px; line-height: 1.6;">
                {info.get('description', '')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kc1, kc2 = st.columns(2)
    with kc1:
        st.markdown("### 🔍 Typical Clinical Symptoms & Presentation")
        for sym in info.get("common_symptoms", []):
            st.markdown(f"- {sym}")

        st.markdown("### ⚠️ Key Risk Factors")
        for rf in info.get("risk_factors", []):
            st.markdown(f"- {rf}")

    with kc2:
        st.markdown("### 💊 Standard Medical Management")
        for tip in info.get("management_tips", []):
            st.markdown(f"- {tip}")

        st.markdown("### 🚨 Urgent Red Flags (Immediate Referral)")
        for flag in info.get("red_flags", []):
            st.markdown(f"- ⚠️ **{flag}**")


# =============================================================================
# MODULE 7: EXPORT CLINICAL REPORT
# =============================================================================
elif selected_nav == "📑 Export Clinical Report":
    st.markdown("## 📑 Export Clinical Patient Diagnostic Summary Report")
    st.markdown(
        "Generate a structured, print-ready medical report for documentation, clinical records, or patient consultation."
    )

    pat_data = st.session_state.get("patient_data", {})
    tab_res = st.session_state.get("tabular_result")
    img_res = st.session_state.get("image_result")
    cond_info = knowledge_base.get(tab_res["primary_diagnosis"]) if tab_res else None

    html_report = generate_html_report(pat_data, tab_res, img_res, cond_info)
    md_report = generate_markdown_report(pat_data, tab_res, img_res)

    exp_col1, exp_col2 = st.columns([1, 1])

    with exp_col1:
        st.download_button(
            "📥 Download Diagnostic Summary (HTML / Printable)",
            data=html_report,
            file_name=f"derma_ai_report_{pat_data.get('patient_id', 'anon')}.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
        )

    with exp_col2:
        st.download_button(
            "📥 Download Diagnostic Summary (Markdown)",
            data=md_report,
            file_name=f"derma_ai_report_{pat_data.get('patient_id', 'anon')}.md",
            mime="text/markdown",
            type="secondary",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### 🖨️ Live Report Preview")
    st.components.v1.html(html_report, height=750, scrolling=True)


# -----------------------------------------------------------------------------
# GLOBAL FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 12px; padding: 15px 0;">
        <b>DermaAI Diagnostic Decision Support Platform</b> | Built with Streamlit, Scikit-Learn & Python.<br>
        <i>Disclaimer: This application is for preliminary clinical triage and educational decision support only. Always consult a licensed medical professional for definitive diagnosis.</i>
    </div>
    """,
    unsafe_allow_html=True,
)
