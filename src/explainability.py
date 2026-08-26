"""
Explainability and interactive visualization suite for DermaAI.
Produces Plotly interactive charts for model diagnostics, feature importance,
differential diagnoses, and ABCDE radar metrics.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_model_comparison(metrics_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Creates an interactive grouped bar chart comparing benchmark metrics across models."""
    models = list(metrics_dict.keys())
    accuracies = [round(metrics_dict[m]["test_accuracy"] * 100, 1) for m in models]
    f1_scores = [round(metrics_dict[m]["test_f1_macro"] * 100, 1) for m in models]
    precisions = [round(metrics_dict[m]["test_precision_macro"] * 100, 1) for m in models]
    recalls = [round(metrics_dict[m]["test_recall_macro"] * 100, 1) for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Test Accuracy (%)", x=models, y=accuracies, marker_color="#06B6D4"))
    fig.add_trace(go.Bar(name="F1-Score Macro (%)", x=models, y=f1_scores, marker_color="#10B981"))
    fig.add_trace(go.Bar(name="Precision (%)", x=models, y=precisions, marker_color="#8B5CF6"))
    fig.add_trace(go.Bar(name="Recall (%)", x=models, y=recalls, marker_color="#F59E0B"))

    fig.update_layout(
        title="<b>Dermatology Diagnostic Model Benchmark Comparison</b>",
        barmode="group",
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(15, 23, 42, 0)",
        yaxis=dict(title="Score (%)", range=[50, 105], gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(title="Algorithms", gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str]) -> go.Figure:
    """Creates a high-contrast clinical confusion matrix heatmap."""
    cm_arr = np.array(cm)

    # Normalize by row (True class totals)
    row_sums = cm_arr.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_norm = np.round((cm_arr / row_sums) * 100, 1)

    text_matrix = [
        [f"<b>{cm_arr[i][j]}</b><br>({cm_norm[i][j]}%)" for j in range(len(class_names))]
        for i in range(len(class_names))
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=class_names,
            y=class_names,
            text=text_matrix,
            texttemplate="%{text}",
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sample Count"),
        )
    )

    fig.update_layout(
        title="<b>Multiclass Confusion Matrix Heatmap</b>",
        xaxis=dict(title="<b>Predicted Clinical Diagnosis</b>", tickangle=-30),
        yaxis=dict(title="<b>Actual Ground-Truth Diagnosis</b>", autorange="reversed"),
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(15, 23, 42, 0)",
        margin=dict(l=40, r=40, t=60, b=60),
    )
    return fig


def plot_feature_importance(model: Any, feature_names: List[str], top_n: int = 12) -> go.Figure:
    """Extracts and plots top global feature importances for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For linear models, take mean absolute coefficient
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        # Fallback uniform
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Format feature names cleanly
    clean_names = [f.replace("num__", "").replace("cat__", "").replace("_", " ").title() for f in feature_names]

    df_imp = pd.DataFrame({"Feature": clean_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=True).tail(top_n)

    fig = go.Figure(
        go.Bar(
            x=df_imp["Importance"],
            y=df_imp["Feature"],
            orientation="h",
            marker=dict(
                color=df_imp["Importance"],
                colorscale="Tealgrn",
                showscale=False,
            ),
        )
    )

    fig.update_layout(
        title=f"<b>Top {top_n} Clinical Diagnostic Drivers (Feature Importance)</b>",
        xaxis=dict(title="Relative Gini Importance / Weight", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="", tickfont=dict(size=12)),
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(15, 23, 42, 0)",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_differential_diagnosis_bars(differential: List[Dict[str, Any]]) -> go.Figure:
    """Plots a horizontal probability bar chart for differential diagnosis ranking."""
    conditions = [d["condition"] for d in reversed(differential)]
    percentages = [d["percentage"] for d in reversed(differential)]

    # Dynamic colors: red for Melanoma/High risk, amber for Psoriasis/BCC, cyan for others
    colors = []
    for c in conditions:
        if "Melanoma" in c:
            colors.append("#EF4444")
        elif "Basal Cell" in c or "Psoriasis" in c:
            colors.append("#F59E0B")
        elif "Healthy" in c:
            colors.append("#10B981")
        else:
            colors.append("#06B6D4")

    fig = go.Figure(
        go.Bar(
            x=percentages,
            y=conditions,
            orientation="h",
            text=[f"<b>{p}%</b>" for p in percentages],
            textposition="inside",
            marker_color=colors,
        )
    )

    fig.update_layout(
        title="<b>Differential Diagnosis Probabilities (Top Likelihoods)</b>",
        xaxis=dict(title="Model Confidence Probability (%)", range=[0, 105], gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title=""),
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(15, 23, 42, 0)",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_abcde_radar(metrics: Dict[str, Any]) -> go.Figure:
    """Creates an ABCDE Dermoscopy spider radar chart against clinical thresholds."""
    categories = [
        "A - Asymmetry",
        "B - Border Irreg.",
        "C - Color Variegation",
        "D - Diameter (Norm)",
        "E - Texture Entropy",
    ]

    # Normalize metrics to 0-10 scale for visual comparability
    a_norm = min((metrics["asymmetry_score"] / 2.0) * 10, 10.0)
    b_norm = min((metrics["border_score"] / 8.0) * 10, 10.0)
    c_norm = min((metrics["color_score"] / 6.0) * 10, 10.0)
    d_norm = min((metrics["diameter_mm"] / 12.0) * 10, 10.0)
    e_norm = min((metrics["evolution_texture_score"] / 10.0) * 10, 10.0)

    patient_values = [a_norm, b_norm, c_norm, d_norm, e_norm]
    # Close polygon
    patient_values.append(patient_values[0])
    categories_closed = categories + [categories[0]]

    # Malignancy threshold line (e.g. 5.5 on 10-scale)
    threshold_values = [5.5, 5.5, 5.5, 5.5, 5.5, 5.5]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=patient_values,
            theta=categories_closed,
            fill="toself",
            name="Analyzed Lesion",
            fillcolor="rgba(6, 182, 212, 0.4)",
            line=dict(color="#06B6D4", width=3),
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=threshold_values,
            theta=categories_closed,
            name="Atypical / High Risk Threshold",
            line=dict(color="#EF4444", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                ticktext=["2 (Low)", "4", "6 (Elevated)", "8", "10 (High)"],
                gridcolor="rgba(255,255,255,0.15)",
            ),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
            bgcolor="rgba(15, 23, 42, 0.6)",
        ),
        title="<b>ABCDE Lesion Morphometric Radar Profile</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0)",
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig
