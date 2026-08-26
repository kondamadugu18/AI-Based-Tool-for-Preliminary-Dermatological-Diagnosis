"""
Styling tokens, custom CSS injector, sample assets generator,
and session management utilities for DermaAI.
"""

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"


def inject_custom_css() -> None:
    """Injects high-end medical dark / clinical teal CSS styles into the Streamlit app."""
    custom_css = """
    <style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header Brand Styling */
    .brand-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(14, 165, 233, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }

    /* Metric Glass Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(6, 182, 212, 0.5);
    }

    /* Clinical Urgency Badges */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-routine {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Medical Callout Box */
    .med-callout {
        background: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0EA5E9;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 15px 0;
        color: #E2E8F0;
        font-size: 14px;
    }

    /* Red Flag Warning Box */
    .warning-callout {
        background: rgba(239, 68, 68, 0.08);
        border-left: 4px solid #EF4444;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 15px 0;
        color: #FCA5A5;
        font-size: 14px;
    }

    /* Button Enhancements */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.4);
        border-color: #06B6D4;
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def ensure_sample_images_exist() -> Dict[str, Path]:
    """Generates realistic dermatological sample images for 1-click UI demos."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    samples = {
        "Melanoma (Atypical Pigmented Lesion)": SAMPLES_DIR / "melanoma_sample.png",
        "Benign Melanocytic Nevus (Mole)": SAMPLES_DIR / "benign_nevus.png",
        "Psoriasis (Erythematous Plaque)": SAMPLES_DIR / "psoriasis_plaque.png",
        "Eczema (Atopic Dermatitis Flare)": SAMPLES_DIR / "eczema_flare.png",
    }

    # 1. Melanoma Sample (Asymmetric, irregular border, variegated dark/brown colors)
    if not samples["Melanoma (Atypical Pigmented Lesion)"].exists():
        img = Image.new("RGB", (256, 256), color=(225, 190, 165))
        draw = ImageDraw.Draw(img)
        # Background skin texture
        for _ in range(300):
            x, y = np.random.randint(0, 256), np.random.randint(0, 256)
            draw.point((x, y), fill=(215, 180, 155))
        # Asymmetric multi-tone lesion
        draw.polygon([(70, 80), (140, 60), (195, 110), (180, 175), (120, 190), (60, 140)], fill=(75, 40, 30))
        draw.ellipse([90, 85, 170, 160], fill=(45, 25, 20))
        draw.ellipse([110, 110, 155, 150], fill=(20, 15, 15))
        draw.ellipse([140, 70, 180, 110], fill=(95, 55, 45))
        draw.ellipse([75, 120, 110, 160], fill=(130, 45, 45))  # Erythema focus
        img = img.filter(ImageFilter.GaussianBlur(radius=1.8))
        img.save(samples["Melanoma (Atypical Pigmented Lesion)"])

    # 2. Benign Nevus (Symmetric, regular border, uniform color)
    if not samples["Benign Melanocytic Nevus (Mole)"].exists():
        img = Image.new("RGB", (256, 256), color=(230, 195, 170))
        draw = ImageDraw.Draw(img)
        # Regular round oval mole
        draw.ellipse([95, 95, 160, 160], fill=(70, 45, 35))
        draw.ellipse([105, 105, 150, 150], fill=(60, 38, 28))
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        img.save(samples["Benign Melanocytic Nevus (Mole)"])

    # 3. Psoriasis Plaque (Red, silvery scaling center)
    if not samples["Psoriasis (Erythematous Plaque)"].exists():
        img = Image.new("RGB", (256, 256), color=(220, 185, 160))
        draw = ImageDraw.Draw(img)
        # Raised red plaque
        draw.ellipse([70, 70, 185, 185], fill=(195, 60, 60))
        # Silvery scales
        for _ in range(80):
            x = np.random.randint(90, 165)
            y = np.random.randint(90, 165)
            draw.ellipse([x, y, x + 12, y + 8], fill=(235, 230, 225))
        img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
        img.save(samples["Psoriasis (Erythematous Plaque)"])

    # 4. Eczema Flare (Diffuse redness, excoriation)
    if not samples["Eczema (Atopic Dermatitis Flare)"].exists():
        img = Image.new("RGB", (256, 256), color=(225, 188, 162))
        draw = ImageDraw.Draw(img)
        # Patchy erythema
        draw.ellipse([60, 80, 200, 175], fill=(190, 85, 80))
        draw.ellipse([90, 60, 170, 195], fill=(180, 75, 70))
        # Scratch lines
        draw.line([(85, 110), (140, 130)], fill=(130, 30, 30), width=2)
        draw.line([(100, 140), (165, 155)], fill=(130, 30, 30), width=2)
        img = img.filter(ImageFilter.GaussianBlur(radius=2.2))
        img.save(samples["Eczema (Atopic Dermatitis Flare)"])

    return samples


if __name__ == "__main__":
    s = ensure_sample_images_exist()
    print("Sample images generated successfully:", s)
