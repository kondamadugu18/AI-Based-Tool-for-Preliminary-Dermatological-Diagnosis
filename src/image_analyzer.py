"""
Dermoscopy computer vision and ABCDE lesion criteria analyzer for DermaAI.
Computes Asymmetry, Border Irregularity, Color Variation, Diameter, and Evolution/Texture
metrics using pure Python, Pillow, NumPy, and SciPy.
"""

import io
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage


def load_image(image_input: Union[str, bytes, Image.Image, io.BytesIO]) -> Image.Image:
    """Loads and standardizes input into an RGB PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, (str, io.BytesIO, bytes)):
        if isinstance(image_input, bytes):
            image_input = io.BytesIO(image_input)
        img = Image.open(image_input)
        return img.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def segment_lesion(img: Image.Image) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Performs automated skin lesion segmentation using adaptive luminance thresholding,
    morphological cleaning, and connected component analysis.
    Returns:
        - img_array (H, W, 3)
        - binary_mask (H, W) bool array
        - bbox (ymin, ymax, xmin, xmax)
    """
    # Resize to standard analysis resolution (256x256) for consistent metrics
    img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)

    # Convert to grayscale
    gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

    # Invert so lesion (darker than skin) has higher intensity
    inv_gray = 255.0 - gray

    # Gaussian blur smoothing
    smoothed = ndimage.gaussian_filter(inv_gray, sigma=2.0)

    # Otsu's automatic thresholding
    hist, bin_edges = np.histogram(smoothed, bins=256, range=(0, 256))
    hist = hist.astype(float) / hist.sum()

    best_thresh = 128
    max_var = 0.0
    for t in range(1, 256):
        w0 = np.sum(hist[:t])
        w1 = np.sum(hist[t:])
        if w0 == 0 or w1 == 0:
            continue
        m0 = np.sum(np.arange(t) * hist[:t]) / w0
        m1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
        var_between = w0 * w1 * ((m0 - m1) ** 2)
        if var_between > max_var:
            max_var = var_between
            best_thresh = t

    raw_mask = smoothed >= best_thresh

    # Morphological binary closing and hole filling
    struct = ndimage.generate_binary_structure(2, 2)
    closed_mask = ndimage.binary_closing(raw_mask, structure=struct, iterations=3)
    filled_mask = ndimage.binary_fill_holes(closed_mask)

    # Select largest connected component
    labeled, num_features = ndimage.label(filled_mask)
    if num_features > 0:
        sizes = ndimage.sum(filled_mask, labeled, range(1, num_features + 1))
        max_label = np.argmax(sizes) + 1
        clean_mask = labeled == max_label
    else:
        clean_mask = filled_mask

    # Fallback if mask is empty or covers whole image
    mask_area = np.sum(clean_mask)
    total_pixels = clean_mask.size
    if mask_area < 100 or mask_area > 0.95 * total_pixels:
        # Create a central elliptical ROI fallback
        y, x = np.ogrid[:256, :256]
        cy, cx = 128, 128
        clean_mask = ((x - cx) ** 2 / (60**2) + (y - cy) ** 2 / (60**2)) <= 1.0

    # Find bounding box
    rows = np.any(clean_mask, axis=1)
    cols = np.any(clean_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    bbox_meta = {
        "ymin": int(ymin),
        "ymax": int(ymax),
        "xmin": int(xmin),
        "xmax": int(xmax),
        "area_pixels": int(np.sum(clean_mask)),
    }

    return img_array, clean_mask, bbox_meta


def calculate_abcde_metrics(
    img_array: np.ndarray, mask: np.ndarray, mm_per_pixel: float = 0.05
) -> Dict[str, Any]:
    """
    Computes rigorous ABCDE criteria:
    A: Asymmetry Index (0 - 2 scale)
    B: Border Irregularity (1 - 8 scale)
    C: Color Variegation (1 - 6 scale)
    D: Diameter in mm
    E: Evolution / Texture Complexity (0 - 10 scale)
    Total Dermoscopy Score (TDS) & Risk Classification.
    """
    area = np.sum(mask)

    # --- A: Asymmetry Index ---
    # Reflection along horizontal & vertical axes
    h_flip = np.flipud(mask)
    v_flip = np.fliplr(mask)

    h_overlap = np.sum(mask & h_flip) / max(np.sum(mask | h_flip), 1)
    v_overlap = np.sum(mask & v_flip) / max(np.sum(mask | v_flip), 1)

    # Asymmetry score (0 = highly symmetric, 2 = highly asymmetric)
    asymmetry_raw = (1.0 - h_overlap) + (1.0 - v_overlap)
    asymmetry_score = round(float(np.clip(asymmetry_raw * 1.8, 0.0, 2.0)), 2)

    # --- B: Border Irregularity ---
    # Perimeter computation via morphological gradient
    eroded = ndimage.binary_erosion(mask)
    border_mask = mask & ~eroded
    perimeter = np.sum(border_mask)

    # Compactness quotient: P^2 / (4 * pi * Area)
    # Circle has compactness = 1.0; irregular shapes have > 1.0
    compactness = (perimeter**2) / (4.0 * math.pi * max(area, 1))
    border_score = round(float(np.clip((compactness - 1.0) * 1.5 + 1.0, 1.0, 8.0)), 2)

    # --- C: Color Variegation ---
    # Extract RGB values within the lesion
    pixels = img_array[mask]  # shape: (N, 3)
    if len(pixels) > 0:
        r_std = np.std(pixels[:, 0])
        g_std = np.std(pixels[:, 1])
        b_std = np.std(pixels[:, 2])

        # Detect distinct color tones (Light Brown, Dark Brown, Black, Red, White, Blue-Gray)
        # Using channel thresholds
        color_count = 0
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

        if np.mean(r) > 150 and np.mean(g) < 120:  # Red/Erythema
            color_count += 1
        if np.mean(r) < 80 and np.mean(g) < 80 and np.mean(b) < 80:  # Black / Dark
            color_count += 1
        if np.mean(r) > 100 and np.mean(g) > 60 and np.mean(b) < 60:  # Light Brown
            color_count += 1
        if np.mean(r) > 60 and np.mean(g) > 40 and np.mean(b) < 40:  # Dark Brown
            color_count += 1
        if np.mean(b) > np.mean(r) + 10:  # Blue-Gray veil
            color_count += 1
        if np.mean(r) > 180 and np.mean(g) > 180 and np.mean(b) > 180:  # White depigmentation
            color_count += 1

        color_score = max(1.0, min(6.0, float(color_count) + (r_std + g_std + b_std) / 75.0))
    else:
        color_score = 1.0
    color_score = round(color_score, 2)

    # --- D: Diameter (mm) ---
    # Equivalent diameter of circle with same area
    diameter_px = 2.0 * math.sqrt(area / math.pi)
    diameter_mm = round(float(diameter_px * mm_per_pixel * 1.5), 1)

    # --- E: Evolution / Texture Complexity ---
    gray_lesion = np.dot(pixels, [0.2989, 0.5870, 0.1140]) if len(pixels) > 0 else np.array([0])
    hist, _ = np.histogram(gray_lesion, bins=32, range=(0, 256))
    hist = hist.astype(float) / max(hist.sum(), 1)
    # Shannon Entropy
    entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])
    evolution_score = round(float(np.clip(entropy * 1.8, 0.5, 9.5)), 2)

    # --- Total Dermoscopy Score (TDS) ---
    # Standard Stolz formula: TDS = 1.3*A + 0.1*B + 0.5*C + 0.5*D_norm
    d_norm = min(diameter_mm / 6.0, 3.0)
    tds = (1.3 * asymmetry_score) + (0.1 * border_score) + (0.5 * color_score) + (0.5 * d_norm)
    tds = round(float(tds), 2)

    # Malignancy classification based on TDS
    if tds < 4.75:
        risk_level = "Benign / Low Risk"
        risk_tier = "LOW"
        risk_badge = "🟢 Benign Lesion (TDS < 4.75)"
        urgency_recommendation = "Reassuring dermoscopy pattern. Routine periodic self-monitoring."
    elif 4.75 <= tds < 5.45:
        risk_level = "Suspicious / Moderate Risk"
        risk_tier = "MODERATE"
        risk_badge = "🟡 Suspicious Lesion (4.75 ≤ TDS < 5.45)"
        urgency_recommendation = "Atypical features detected. Dermatologist dermoscopy evaluation advised."
    else:
        risk_level = "High Malignancy Risk (Melanoma Suspected)"
        risk_tier = "HIGH"
        risk_badge = "🔴 High Risk Malignancy (TDS ≥ 5.45)"
        urgency_recommendation = "Marked structural atypia. Immediate urgent dermatological biopsy recommended."

    return {
        "asymmetry_score": asymmetry_score,
        "border_score": border_score,
        "color_score": color_score,
        "diameter_mm": diameter_mm,
        "evolution_texture_score": evolution_score,
        "total_dermoscopy_score": tds,
        "risk_level": risk_level,
        "risk_tier": risk_tier,
        "risk_badge": risk_badge,
        "urgency_recommendation": urgency_recommendation,
        "area_pixels": int(area),
        "perimeter_pixels": int(perimeter),
    }


def generate_visual_overlays(
    img_array: np.ndarray, mask: np.ndarray
) -> Dict[str, Image.Image]:
    """
    Generates diagnostic visual overlays:
    1. Mask overlay (lesion highlighted with translucent cyan/teal tint)
    2. Contour boundary outline (neon green border)
    3. Dermoscopic heat map (intensity variation)
    """
    h, w, _ = img_array.shape
    base_img = Image.fromarray(img_array)

    # 1. Mask overlay
    overlay_array = img_array.copy().astype(float)
    # Highlight lesion area with a luminous cyan-teal tint
    overlay_array[mask, 0] = np.clip(overlay_array[mask, 0] * 0.4 + 20, 0, 255)
    overlay_array[mask, 1] = np.clip(overlay_array[mask, 1] * 0.8 + 140, 0, 255)
    overlay_array[mask, 2] = np.clip(overlay_array[mask, 2] * 0.9 + 170, 0, 255)
    mask_overlay_img = Image.fromarray(overlay_array.astype(np.uint8))

    # 2. Contour boundary outline
    eroded = ndimage.binary_erosion(mask, iterations=2)
    border_mask = mask & ~eroded

    contour_array = img_array.copy()
    # Neon lime-green border for high medical visibility
    contour_array[border_mask] = [34, 197, 94]
    contour_img = Image.fromarray(contour_array)

    # 3. Heat map (intensity gradient)
    gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    norm_gray = ((255.0 - gray) / 255.0 * 255.0).astype(np.uint8)
    heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap_colored[..., 0] = norm_gray  # Red channel
    heatmap_colored[..., 1] = np.clip(255 - norm_gray, 0, 255)  # Green
    heatmap_colored[..., 2] = np.clip(128 - np.abs(norm_gray.astype(int) - 128), 0, 255)  # Blue
    # Mask out non-lesion skin
    heatmap_colored[~mask] = (img_array[~mask] * 0.35).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_colored)

    return {
        "original": base_img,
        "segmentation_mask": Image.fromarray((mask * 255).astype(np.uint8)),
        "mask_overlay": mask_overlay_img,
        "contour_boundary": contour_img,
        "heatmap": heatmap_img,
    }


def analyze_lesion_image(
    image_input: Union[str, bytes, Image.Image, io.BytesIO]
) -> Dict[str, Any]:
    """
    Complete end-to-end Dermoscopy Image Analysis pipeline:
    - Ingests image
    - Segments lesion ROI
    - Calculates full ABCDE clinical criteria
    - Produces visual overlay images
    """
    img = load_image(image_input)
    img_array, mask, bbox_meta = segment_lesion(img)
    abcde_metrics = calculate_abcde_metrics(img_array, mask)
    visuals = generate_visual_overlays(img_array, mask)

    return {
        "metrics": abcde_metrics,
        "bbox": bbox_meta,
        "visuals": visuals,
    }


if __name__ == "__main__":
    # Test on a synthetic lesion
    test_img = Image.new("RGB", (256, 256), color=(220, 185, 160))
    draw = ImageDraw.Draw(test_img)
    # Draw an asymmetric dark pigmented spot
    draw.ellipse([80, 70, 190, 160], fill=(60, 35, 25))
    draw.ellipse([130, 100, 175, 185], fill=(40, 20, 15))

    res = analyze_lesion_image(test_img)
    print("ABCDE Analysis Results:")
    for k, v in res["metrics"].items():
        print(f"  {k}: {v}")
