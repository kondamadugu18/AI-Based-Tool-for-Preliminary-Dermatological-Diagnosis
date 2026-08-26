"""
Data loading and dataset generation utilities for DermaAI.
Provides clinical dataset generation, validation, and loading routines.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_PATH = DATA_DIR / "skin_conditions_sample.csv"
EXTENDED_DATA_PATH = DATA_DIR / "dermatology_extended.csv"
KNOWLEDGE_PATH = DATA_DIR / "condition_knowledge.json"

DIAGNOSIS_CLASSES = [
    "Eczema",
    "Psoriasis",
    "Melanoma",
    "Basal Cell Carcinoma",
    "Seborrheic Keratosis",
    "Acne Vulgaris",
    "Fungal Infection",
    "Healthy Skin",
]

BODY_SITES = [
    "Face/Neck",
    "Trunk",
    "Arms/Hands",
    "Legs/Feet",
    "Scalp",
    "Flexural Folds",
]

FITZPATRICK_TYPES = ["I", "II", "III", "IV", "V", "VI"]
ELEVATIONS = ["Flat", "Raised/Plaque", "Nodular/Cystic"]
EVOLUTIONS = ["Stable", "Slow Growth", "Rapid Growth", "Fluctuating"]
SUN_EXPOSURES = ["Low", "Moderate", "High"]


def generate_synthetic_clinical_dataset(
    n_samples: int = 600, random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic synthetic dermatological clinical dataset based on
    clinical dermatology diagnostic distributions and lesion morphology.
    """
    np.random.seed(random_state)
    records = []

    samples_per_class = n_samples // len(DIAGNOSIS_CLASSES)

    for cls in DIAGNOSIS_CLASSES:
        for i in range(samples_per_class):
            pid = f"PT-{len(records) + 1:04d}"
            gender = np.random.choice(["Male", "Female"], p=[0.48, 0.52])

            if cls == "Eczema":
                age = int(np.clip(np.random.normal(24, 14), 2, 75))
                skin_type = np.random.choice(FITZPATRICK_TYPES, p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05])
                body_site = np.random.choice(
                    BODY_SITES, p=[0.15, 0.1, 0.25, 0.1, 0.05, 0.35]
                )
                itching = int(np.clip(np.random.normal(4.2, 0.8), 2, 5))
                redness = int(np.clip(np.random.normal(3.8, 0.9), 2, 5))
                scaling = int(np.clip(np.random.normal(3.2, 1.0), 1, 5))
                burning = int(np.clip(np.random.normal(2.1, 1.0), 0, 4))
                bleeding = int(np.clip(np.random.normal(1.8, 1.1), 0, 4))
                lesion_size = round(float(np.clip(np.random.normal(14.0, 6.0), 3.0, 45.0)), 1)
                elevation = np.random.choice(ELEVATIONS, p=[0.3, 0.65, 0.05])
                duration = int(np.clip(np.random.exponential(16), 1, 120))
                evolution = np.random.choice(EVOLUTIONS, p=[0.1, 0.15, 0.1, 0.65])
                sun = np.random.choice(SUN_EXPOSURES, p=[0.4, 0.4, 0.2])
                fam_history = np.random.choice([0, 1], p=[0.4, 0.6])

            elif cls == "Psoriasis":
                age = int(np.clip(np.random.normal(42, 15), 12, 82))
                skin_type = np.random.choice(FITZPATRICK_TYPES, p=[0.25, 0.3, 0.25, 0.1, 0.06, 0.04])
                body_site = np.random.choice(
                    BODY_SITES, p=[0.05, 0.25, 0.3, 0.25, 0.15, 0.0]
                )
                itching = int(np.clip(np.random.normal(2.9, 1.1), 1, 5))
                redness = int(np.clip(np.random.normal(4.4, 0.7), 3, 5))
                scaling = int(np.clip(np.random.normal(4.6, 0.6), 3, 5))
                burning = int(np.clip(np.random.normal(2.4, 1.1), 0, 4))
                bleeding = int(np.clip(np.random.normal(1.9, 1.0), 0, 4))
                lesion_size = round(float(np.clip(np.random.normal(18.0, 8.0), 4.0, 55.0)), 1)
                elevation = np.random.choice(ELEVATIONS, p=[0.05, 0.9, 0.05])
                duration = int(np.clip(np.random.exponential(35), 4, 200))
                evolution = np.random.choice(EVOLUTIONS, p=[0.2, 0.4, 0.1, 0.3])
                sun = np.random.choice(SUN_EXPOSURES, p=[0.35, 0.45, 0.2])
                fam_history = np.random.choice([0, 1], p=[0.45, 0.55])

            elif cls == "Melanoma":
                age = int(np.clip(np.random.normal(56, 13), 22, 88))
                skin_type = np.random.choice(FITZPATRICK_TYPES, p=[0.45, 0.35, 0.12, 0.05, 0.02, 0.01])
                body_site = np.random.choice(
                    BODY_SITES, p=[0.25, 0.35, 0.2, 0.15, 0.05, 0.0]
                )
                itching = int(np.clip(np.random.normal(1.8, 1.2), 0, 4))
                redness = int(np.clip(np.random.normal(2.1, 1.2), 0, 4))
                scaling = int(np.clip(np.random.normal(1.4, 1.1), 0, 3))
                burning = int(np.clip(np.random.normal(1.2, 1.0), 0, 3))
                bleeding = int(np.clip(np.random.normal(2.9, 1.3), 0, 5))
                lesion_size = round(float(np.clip(np.random.normal(9.5, 4.0), 4.0, 30.0)), 1)
                elevation = np.random.choice(ELEVATIONS, p=[0.25, 0.45, 0.3])
                duration = int(np.clip(np.random.exponential(24), 2, 80))
                evolution = np.random.choice(EVOLUTIONS, p=[0.05, 0.35, 0.6, 0.0])
                sun = np.random.choice(SUN_EXPOSURES, p=[0.15, 0.35, 0.5])
                fam_history = np.random.choice([0, 1], p=[0.6, 0.4])

            elif cls == "Basal Cell Carcinoma":
                age = int(np.clip(np.random.normal(63, 11), 35, 90))
                skin_type = np.random.choice(FITZPATRICK_TYPES, p=[0.4, 0.4, 0.15, 0.03, 0.01, 0.01])
                body_site = np.random.choice(
                    BODY_SITES, p=[0.65, 0.15, 0.1, 0.05, 0.05, 0.0]
                )
                itching = int(np.clip(np.random.normal(1.2, 1.0), 0, 3))
                redness = int(np.clip(np.random.normal(2.5, 1.0), 1, 4))
                scaling = int(np.clip(np.random.normal(1.6, 1.0), 0, 4))
                burning = int(np.clip(np.random.normal(1.0, 0.9), 0, 3))
                bleeding = int(np.clip(np.random.normal(3.4, 1.1), 1, 5))
                lesion_size = round(float(np.clip(np.random.normal(8.0, 3.5), 3.0, 24.0)), 1)
                elevation = np.random.choice(ELEVATIONS, p=[0.1, 0.35, 0.55])
                duration = int(np.clip(np.random.exponential(36), 6, 150))
                evolution = np.random.choice(EVOLUTIONS, p=[0.1, 0.75, 0.15, 0.0])
                sun = np.random.choice(SUN_EXPOSURES, p=[0.1, 0.3, 0.6])
                fam_history = np.random.choice([0, 1], p=[0.75, 0.25])

            elif cls == "Seborrheic Keratosis":
                age = int(np.clip(np.random.normal(61, 12), 38, 92))
                skin_type = np.random.choice(FITZPATRICK_TYPES, p=[0.25, 0.3, 0.25, 0.1, 0.06, 0.04])
                body_site = np.random.choice(
                    BODY_SITES, p=[0.2, 0.5, 0.15, 0.1, 0.05, 0.0]
                )
                itching = int(np.clip(np.random.normal(1.4, 1.1), 0, 3))
                redness = int(np.clip(np.random.normal(1.1, 0.9), 0, 3))
                scaling = int(np.clip(np.random.normal(2.8, 1.2), 0, 5))
                burning = int(np.clip(np.random.normal(0.4, 0.6), 0, 2))
                bleeding = int(np.clip(np.random.normal(0.5, 0.7), 0, 2))
                lesion_size = round(float(np.clip(np.random.normal(9.0, 4.5), 2.5, 28.0)), 1)
                elevation = np.random.choice(ELEVATIONS, p=[0.05, 0.85, 0.1])
                duration = int(np.clip(np.random.exponential(60), 10, 250))
                evolution = np.random.choice(EVOLUTIONS, p=[0.65, 0.3, 0.05, 0.0])
                sun = np.random.choice(SUN_EXPOSURES, p=[0.3, 0.45, 0.25])
                fam_history = np.random.choice([0, 1], p=[0.6, 0.4])

            elif cls == "Acne Vulgaris":
                age = int(np.clip(np.random.normal(19, 5), 11, 40))
                skin_type = np.random.choice(FITZPATRICK_TYPES, p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05])
                body_site = np.random.choice(
                    BODY_SITES, p=[0.75, 0.15, 0.05, 0.0, 0.05, 0.0]
                )
                itching = int(np.clip(np.random.normal(1.6, 1.0), 0, 4))
                redness = int(np.clip(np.random.normal(3.7, 0.9), 2, 5))
                scaling = int(np.clip(np.random.normal(1.2, 0.9), 0, 3))
                burning = int(np.clip(np.random.normal(2.2, 1.1), 0, 4))
                bleeding = int(np.clip(np.random.normal(1.5, 1.1), 0, 4))
                lesion_size = round(float(np.clip(np.random.normal(4.2, 1.8), 1.5, 12.0)), 1)
                elevation = np.random.choice(ELEVATIONS, p=[0.1, 0.45, 0.45])
                duration = int(np.clip(np.random.exponential(14), 1, 60))
                evolution = np.random.choice(EVOLUTIONS, p=[0.1, 0.2, 0.15, 0.55])
                sun = np.random.choice(SUN_EXPOSURES, p=[0.35, 0.45, 0.2])
                fam_history = np.random.choice([0, 1], p=[0.35, 0.65])

            elif cls == "Fungal Infection":
                age = int(np.clip(np.random.normal(33, 14), 8, 70))
                skin_type = np.random.choice(FITZPATRICK_TYPES, p=[0.2, 0.2, 0.25, 0.2, 0.1, 0.05])
                body_site = np.random.choice(
                    BODY_SITES, p=[0.05, 0.3, 0.2, 0.25, 0.05, 0.15]
                )
                itching = int(np.clip(np.random.normal(3.8, 0.9), 2, 5))
                redness = int(np.clip(np.random.normal(3.2, 0.9), 1, 5))
                scaling = int(np.clip(np.random.normal(3.9, 0.8), 2, 5))
                burning = int(np.clip(np.random.normal(1.8, 1.0), 0, 3))
                bleeding = int(np.clip(np.random.normal(0.6, 0.7), 0, 2))
                lesion_size = round(float(np.clip(np.random.normal(15.0, 7.0), 4.0, 40.0)), 1)
                elevation = np.random.choice(ELEVATIONS, p=[0.2, 0.75, 0.05])
                duration = int(np.clip(np.random.exponential(6), 1, 30))
                evolution = np.random.choice(EVOLUTIONS, p=[0.05, 0.7, 0.2, 0.05])
                sun = np.random.choice(SUN_EXPOSURES, p=[0.4, 0.4, 0.2])
                fam_history = np.random.choice([0, 1], p=[0.8, 0.2])

            else:  # Healthy Skin / Benign Nevus
                age = int(np.clip(np.random.normal(36, 16), 6, 85))
                skin_type = np.random.choice(FITZPATRICK_TYPES, p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05])
                body_site = np.random.choice(
                    BODY_SITES, p=[0.2, 0.3, 0.25, 0.2, 0.03, 0.02]
                )
                itching = int(np.clip(np.random.normal(0.2, 0.4), 0, 1))
                redness = int(np.clip(np.random.normal(0.3, 0.5), 0, 1))
                scaling = int(np.clip(np.random.normal(0.2, 0.4), 0, 1))
                burning = int(np.clip(np.random.normal(0.1, 0.3), 0, 1))
                bleeding = 0
                lesion_size = round(float(np.clip(np.random.normal(3.2, 1.4), 1.0, 6.0)), 1)
                elevation = np.random.choice(ELEVATIONS, p=[0.7, 0.28, 0.02])
                duration = int(np.clip(np.random.exponential(90), 20, 300))
                evolution = np.random.choice(EVOLUTIONS, p=[0.92, 0.08, 0.0, 0.0])
                sun = np.random.choice(SUN_EXPOSURES, p=[0.35, 0.45, 0.2])
                fam_history = np.random.choice([0, 1], p=[0.75, 0.25])

            records.append(
                {
                    "patient_id": pid,
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
                    "sun_exposure": sun,
                    "family_history_skin_disease": fam_history,
                    "diagnosis": cls,
                }
            )

    df = pd.DataFrame(records)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def ensure_datasets_exist() -> None:
    """Ensures data directory and necessary dataset files exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Extended dataset
    if not EXTENDED_DATA_PATH.exists():
        df_ext = generate_synthetic_clinical_dataset(n_samples=640, random_state=42)
        df_ext.to_csv(EXTENDED_DATA_PATH, index=False)

    # 2. Sample dataset check & sync
    root_sample = BASE_DIR / "skin_conditions_sample.csv"
    if not SAMPLE_DATA_PATH.exists():
        if root_sample.exists():
            SAMPLE_DATA_PATH.write_bytes(root_sample.read_bytes())
        else:
            # Create standard sample dataset
            sample_df = pd.DataFrame(
                [
                    {"age": 22, "itching": 1, "redness": 1, "scaling": 0, "lesion_size_mm": 3.2, "diagnosis": "Eczema"},
                    {"age": 35, "itching": 0, "redness": 1, "scaling": 1, "lesion_size_mm": 5.1, "diagnosis": "Psoriasis"},
                    {"age": 41, "itching": 1, "redness": 1, "scaling": 1, "lesion_size_mm": 4.0, "diagnosis": "Psoriasis"},
                    {"age": 29, "itching": 1, "redness": 0, "scaling": 0, "lesion_size_mm": 2.5, "diagnosis": "Healthy"},
                    {"age": 50, "itching": 0, "redness": 0, "scaling": 1, "lesion_size_mm": 6.3, "diagnosis": "Melanoma"},
                    {"age": 33, "itching": 1, "redness": 1, "scaling": 1, "lesion_size_mm": 3.9, "diagnosis": "Eczema"},
                    {"age": 27, "itching": 0, "redness": 0, "scaling": 0, "lesion_size_mm": 2.2, "diagnosis": "Healthy"},
                    {"age": 46, "itching": 1, "redness": 1, "scaling": 1, "lesion_size_mm": 5.8, "diagnosis": "Psoriasis"},
                    {"age": 38, "itching": 1, "redness": 0, "scaling": 0, "lesion_size_mm": 3.6, "diagnosis": "Eczema"},
                    {"age": 31, "itching": 0, "redness": 1, "scaling": 1, "lesion_size_mm": 4.4, "diagnosis": "Dermatitis"},
                ]
            )
            sample_df.to_csv(SAMPLE_DATA_PATH, index=False)


def load_clinical_dataset() -> pd.DataFrame:
    """Loads the main extended clinical dataset."""
    ensure_datasets_exist()
    return pd.read_csv(EXTENDED_DATA_PATH)


def load_sample_dataset() -> pd.DataFrame:
    """Loads the original sample dataset."""
    ensure_datasets_exist()
    return pd.read_csv(SAMPLE_DATA_PATH)


def load_knowledge_base() -> Dict:
    """Loads the clinical condition medical reference knowledge base."""
    if not KNOWLEDGE_PATH.exists():
        return {}
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    ensure_datasets_exist()
    df = load_clinical_dataset()
    print(f"Extended dataset loaded successfully with shape: {df.shape}")
    print("Class distribution:")
    print(df["diagnosis"].value_counts())
