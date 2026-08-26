"""
Feature preprocessing, scaling, and encoding pipeline for DermaAI.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Numerical features to standard-scale
NUMERICAL_FEATURES = [
    "age",
    "itching",
    "redness",
    "scaling_peeling",
    "burning_pain",
    "bleeding_oozing",
    "lesion_size_mm",
    "duration_weeks",
]

# Categorical features to one-hot encode
CATEGORICAL_FEATURES = [
    "gender",
    "fitzpatrick_skin_type",
    "body_site",
    "elevation",
    "evolution_change",
    "sun_exposure",
    "family_history_skin_disease",
]

TARGET_COLUMN = "diagnosis"


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer that scales numerical variables
    and applies One-Hot Encoding with handle_unknown='ignore' to categorical variables.
    """
    num_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


class DataPreprocessor:
    """
    Complete data preprocessing management class that handles:
    - Preprocessing pipeline fitting & transformation
    - Target label encoding & inverse decoding
    - Single patient dictionary conversion into a model-ready feature vector
    """

    def __init__(self):
        self.preprocessor = build_preprocessor()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names_out: List[str] = []

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fits the transformer on input training DataFrame."""
        X = df.drop(columns=[TARGET_COLUMN, "patient_id"], errors="ignore")
        self.preprocessor.fit(X)

        if TARGET_COLUMN in df.columns:
            self.label_encoder.fit(df[TARGET_COLUMN])

        # Extract output feature names
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names_out = NUMERICAL_FEATURES + encoded_cat_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """Transforms input DataFrame into processed X array and y array (if target exists)."""
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted before transforming data.")

        X = df.drop(columns=[TARGET_COLUMN, "patient_id"], errors="ignore")
        X_trans = self.preprocessor.transform(X)

        y_trans = None
        if TARGET_COLUMN in df.columns:
            y_trans = self.label_encoder.transform(df[TARGET_COLUMN])

        return X_trans, y_trans

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fits and transforms input DataFrame."""
        self.fit(df)
        return self.transform(df)

    def transform_single_patient(self, patient_dict: Dict) -> np.ndarray:
        """
        Converts a single patient intake dictionary into a 2D numpy feature vector
        ready for model inference.
        """
        if not self.is_fitted:
            raise ValueError("DataPreprocessor is not fitted.")

        df_single = pd.DataFrame([patient_dict])
        X_trans = self.preprocessor.transform(df_single)
        return X_trans

    def decode_predictions(self, encoded_preds: np.ndarray) -> List[str]:
        """Decodes numeric class IDs back into clinical diagnosis strings."""
        return list(self.label_encoder.inverse_transform(encoded_preds))

    @property
    def classes(self) -> List[str]:
        """Returns the list of string class labels."""
        return list(self.label_encoder.classes_)
