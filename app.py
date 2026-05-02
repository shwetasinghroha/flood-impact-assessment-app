from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from inference import FloodAssessmentPredictor


# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Flood Impact Assessment",
    page_icon="🌊",
    layout="wide",
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        padding-top: 1.0rem;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }

    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.92;
        line-height: 1.5;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.65rem;
    }

    .soft-card {
        padding: 1rem 1rem 0.85rem 1rem;
        border-radius: 18px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(15,23,42,0.05);
        margin-bottom: 1rem;
    }

    .upload-helper {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.75rem;
        color: #475569;
        font-size: 0.95rem;
    }

    .legend-card {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(15,23,42,0.05);
        height: 100%;
    }

    .legend-title {
        font-size: 0.98rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
        color: #0f172a;
    }

    .legend-row {
        margin-bottom: 0.4rem;
        color: #475569;
        font-size: 0.92rem;
    }

    .legend-box {
        display: inline-block;
        width: 16px;
        height: 16px;
        border-radius: 4px;
        margin-right: 8px;
        vertical-align: middle;
        border: 1px solid rgba(0,0,0,0.08);
    }

    .severity-card {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        color: white;
        font-weight: 800;
        text-align: center;
        font-size: 1.35rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
        margin-bottom: 1rem;
    }

    .severity-low {
        background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
    }

    .severity-moderate {
        background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
    }

    .severity-severe {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    }

    .severity-unknown {
        background: linear-gradient(135deg, #64748b 0%, #94a3b8 100%);
    }

    .small-note {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    div[data-testid="column"] .metric-card {
        height: 100%;
    }

    .metric-card {
        padding: 0.95rem 0.8rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 14px rgba(15,23,42,0.06);
        text-align: center;
        min-height: 118px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }

    .metric-label {
        font-size: 0.83rem;
        color: #475569;
        margin-bottom: 0.35rem;
        line-height: 1.25;
        min-height: 34px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }

    .footer-note {
        color: #64748b;
        font-size: 0.84rem;
        margin-top: 1rem;
    }

    .disclaimer-box {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: #fff7ed;
        border: 1px solid #fdba74;
        color: #7c2d12;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Paths / assets
# ---------------------------------------------------------
ASSETS_DIR = Path("assets")
DEFAULT_SAMPLE_IMAGE = ASSETS_DIR / "sample_image.jpg"


# ---------------------------------------------------------
# Cached predictor
# ---------------------------------------------------------
@st.cache_resource
def load_predictor() -> FloodAssessmentPredictor:
    return FloodAssessmentPredictor()


predictor = load_predictor()


# ---------------------------------------------------------
# Hero section
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🌊 Evidence-Grounded Flood Impact Assessment</div>
        <div class="hero-subtitle">
            Upload a post-flood UAV image to generate a flood mask, overlay visualisation,
            quantitative flood indicators, and a preliminary infrastructure-impact severity label.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        This tool provides a preliminary image-based flood impact assessment from visible UAV imagery.
        It is intended for prototype and decision-support use only, and should not be treated as a
        structural engineering, emergency response, or insurance claims decision system.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------
st.markdown('<div class="section-title">Input</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="upload-helper">
        Upload a UAV image below, or use the sample image to quickly preview the full assessment workflow.
    </div>
    """,
    unsafe_allow_html=True,
)

u1, u2 = st.columns([3.8, 1.2])

with u1:
    uploaded_file = st.file_uploader(
        "Upload UAV image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

with u2:
    use_sample = False
    if DEFAULT_SAMPLE_IMAGE.exists():
        st.write("")
        st.write("")
        use_sample = st.button("Use Sample Image", use_container_width=True)
    else:
        st.caption("Add `assets/sample_image.jpg` to enable the sample-image button.")


# ---------------------------------------------------------
# Input selection
# ---------------------------------------------------------
selected_image = None
selected_image_name = None

if use_sample and DEFAULT_SAMPLE_IMAGE.exists():
    selected_image = Image.open(DEFAULT_SAMPLE_IMAGE).convert("RGB")
    selected_image_name = DEFAULT_SAMPLE_IMAGE.name
elif uploaded_file is not None:
    selected_image = Image.open(uploaded_file).convert("RGB")
    selected_image_name = uploaded_file.name


# ---------------------------------------------------------
# Helper to render severity card
# ---------------------------------------------------------
def render_severity_card(severity: str) -> None:
    severity_class = {
        "Low": "severity-low",
        "Moderate": "severity-moderate",
        "Severe": "severity-severe",
    }.get(severity, "severity-unknown")

    emoji = {
        "Low": "🟢",
        "Moderate": "🟠",
        "Severe": "🔴",
    }.get(severity, "⚪")

    st.markdown(
        f"""
        <div class="severity-card {severity_class}">
            {emoji} {severity}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Main prediction flow
# ---------------------------------------------------------
if selected_image is not None:
    with st.spinner("Running flood assessment..."):
        result = predictor.predict(selected_image, image_name=selected_image_name)

    # -----------------------
    # Visual outputs + legend
    # -----------------------
    st.markdown('<div class="section-title">Visual Outputs</div>', unsafe_allow_html=True)

    v1, v2, v3, v4 = st.columns([1, 1, 1, 0.85])

    with v1:
        st.markdown("**Input Image**")
        st.image(result.resized_input_image, use_column_width=True)

    with v2:
        st.markdown("**Predicted Mask**")
        st.image(result.predicted_mask_rgb, use_column_width=True)

    with v3:
        st.markdown("**Overlay**")
        st.image(result.overlay_image, use_column_width=True)

    with v4:
        st.markdown(
            """
            <div class="legend-card">
                <div class="legend-title">Class Legend</div>
                <div class="legend-row"><span class="legend-box" style="background:#ff0000;"></span>Flooded Building</div>
                <div class="legend-row"><span class="legend-box" style="background:#a09614;"></span>Flooded Road</div>
                <div class="legend-row"><span class="legend-box" style="background:#3de6fa;"></span>Water</div>
                <div class="legend-row"><span class="legend-box" style="background:#000000;"></span>Background / Other</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------
    # Severity + indicators
    # -----------------------
    left, right = st.columns([1.1, 2.3])

    with left:
        st.markdown('<div class="section-title">Preliminary Severity</div>', unsafe_allow_html=True)
        render_severity_card(result.severity)

        st.markdown(
            """
            <div class="soft-card">
                <div class="small-note">
                    Severity is based on visible flood indicators extracted from the predicted segmentation output,
                    with emphasis on likely infrastructure impact.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-title">Flood Indicators</div>', unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)

        metrics = [
            ("Water %", result.indicators["water_pct"]),
            ("Flooded Building %", result.indicators["flooded_building_pct"]),
            ("Flooded Road %", result.indicators["flooded_road_pct"]),
            ("Flooded Infra %", result.indicators["combined_flooded_infra_pct"]),
            ("Flood-Relevant %", result.indicators["combined_flood_relevant_pct"]),
        ]

        for col, (label, value) in zip([m1, m2, m3, m4, m5], metrics):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -----------------------
    # Evidence summary
    # -----------------------
    st.markdown('<div class="section-title">Evidence Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="small-note">{result.evidence_summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------
    # Structured summary table
    # -----------------------
    st.markdown('<div class="section-title">Structured Output</div>', unsafe_allow_html=True)

    summary_df = pd.DataFrame([{
        "image_name": selected_image_name,
        "severity": result.severity,
        "water_pct": round(result.indicators["water_pct"], 2),
        "flooded_building_pct": round(result.indicators["flooded_building_pct"], 2),
        "flooded_road_pct": round(result.indicators["flooded_road_pct"], 2),
        "combined_flooded_infra_pct": round(result.indicators["combined_flooded_infra_pct"], 2),
        "combined_flood_relevant_pct": round(result.indicators["combined_flood_relevant_pct"], 2),
        "evidence_summary": result.evidence_summary,
    }])

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # -----------------------
    # Downloads
    # -----------------------
    st.markdown('<div class="section-title">Download Outputs</div>', unsafe_allow_html=True)

    mask_pil = Image.fromarray(result.predicted_mask_rgb)
    overlay_pil = Image.fromarray(result.overlay_image)

    mask_buf = io.BytesIO()
    overlay_buf = io.BytesIO()
    csv_buf = io.StringIO()

    mask_pil.save(mask_buf, format="PNG")
    overlay_pil.save(overlay_buf, format="PNG")
    summary_df.to_csv(csv_buf, index=False)

    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            label="Download Predicted Mask",
            data=mask_buf.getvalue(),
            file_name="predicted_mask.png",
            mime="image/png",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            label="Download Overlay",
            data=overlay_buf.getvalue(),
            file_name="overlay.png",
            mime="image/png",
            use_container_width=True,
        )

    with d3:
        st.download_button(
            label="Download CSV Summary",
            data=csv_buf.getvalue(),
            file_name="flood_assessment_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(
        '<div class="footer-note">Prototype output based on visible flood evidence from UAV imagery.</div>',
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <div class="soft-card">
            <div class="small-note">
                Upload a UAV image or use the sample image button above to preview the full flood assessment workflow.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )