from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from funcs import _explain_sensor_status


# ============================================================
# MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "water_quality_frozen_clusters.pkl"
)

bundle = joblib.load(MODEL_PATH)


# ============================================================
# LOAD FROZEN MODEL INFORMATION
# ============================================================

feature_scalers = bundle["feature_scalers"]
sensor_feature_sets = bundle["sensor_feature_sets"]
clustering_weights = bundle["clustering_weights"]

sensor_kmeans_models = bundle["sensor_kmeans_models"]

cluster_labels = bundle["cluster_labels"]
cluster_contributions = bundle["cluster_contributions"]

turbidity_transform = bundle["turbidity_transform"]


# ============================================================
# COLUMN NAMES
# ============================================================

SENSOR_COL = "sensor_id"

CONDUCTIVITY_COL = "conductivity_of_solution_ mS_cm"
ORP_COL = "oxidation_reduction_potential_ mV"
OXYGEN_COL = "oxygen_saturation_%"
TURBIDITY_COL = "turbidity_NTU"


# ============================================================
# PREPROCESS ONE FEATURE
# ============================================================

def preprocess_feature(feature, value):
    """
    Apply exactly the same transformation, scaling and weighting
    that were used during training.
    """

    value = float(value)

    # --------------------------------------------------------
    # Turbidity transformation
    # --------------------------------------------------------

    if feature == TURBIDITY_COL:

        transform_type = turbidity_transform["type"]

        if transform_type == "log1p_divide":

            divisor = turbidity_transform["divisor"]

            value = np.log1p(
                value / divisor
            )

        else:
            raise ValueError(
                f"Unknown turbidity transformation: "
                f"{transform_type}"
            )

    # --------------------------------------------------------
    # Frozen RobustScaler
    # --------------------------------------------------------

    scaler = feature_scalers[feature]

    # This matters for turbidity because its scaler was fitted
    # using the transformed column name.
    fitted_column_name = scaler.feature_names_in_[0]

    value_df = pd.DataFrame(
        {
            fitted_column_name: [value]
        }
    )

    scaled_value = scaler.transform(
        value_df
    )[0, 0]

    # --------------------------------------------------------
    # Same feature weight as training
    # --------------------------------------------------------

    weighted_value = (
        scaled_value
        * clustering_weights[feature]
    )

    return weighted_value


# ============================================================
# PREDICT WATER QUALITY
# ============================================================

def predict_water_quality(reading):
    """
    Predict the frozen water-quality state for one live reading.

    Example input:

    {
        "sensor_id": 984,
        "conductivity_of_solution_ mS_cm": 51500,
        "oxidation_reduction_potential_ mV": 280,
        "oxygen_saturation_%": 72,
        "turbidity_NTU": 35
    }
    """

    # --------------------------------------------------------
    # Sensor
    # --------------------------------------------------------

    if SENSOR_COL not in reading:
        raise ValueError(
            "The reading does not contain sensor_id."
        )

    sensor_id = int(
        reading[SENSOR_COL]
    )

    if sensor_id not in sensor_kmeans_models:
        raise ValueError(
            f"No frozen model exists for sensor {sensor_id}."
        )

    # --------------------------------------------------------
    # Find which features this sensor uses
    # --------------------------------------------------------

    active_features = sensor_feature_sets[
        sensor_id
    ]

    processed_values = []

    # --------------------------------------------------------
    # Preprocess each required feature
    # --------------------------------------------------------

    for feature in active_features:

        if feature not in reading:
            raise ValueError(
                f"Sensor {sensor_id} requires "
                f"'{feature}', but it was not provided."
            )

        value = reading[feature]

        if pd.isna(value):
            raise ValueError(
                f"Sensor {sensor_id} requires "
                f"'{feature}', but its value is missing."
            )

        processed_value = preprocess_feature(
            feature,
            value
        )

        processed_values.append(
            processed_value
        )

    # --------------------------------------------------------
    # Create the exact feature vector expected by KMeans
    # --------------------------------------------------------

    X = pd.DataFrame(
        [processed_values],
        columns=active_features
    )

    # --------------------------------------------------------
    # Frozen KMeans prediction
    # --------------------------------------------------------

    kmeans = sensor_kmeans_models[
        sensor_id
    ]

    cluster = int(
        kmeans.predict(X)[0]
    )

    # --------------------------------------------------------
    # Find the frozen label for this sensor + cluster
    # --------------------------------------------------------

    label_row = cluster_labels[
        (
            cluster_labels[SENSOR_COL]
            == sensor_id
        )
        &
        (
            cluster_labels["cluster"]
            == cluster
        )
    ]

    if label_row.empty:
        raise ValueError(
            f"No frozen label found for "
            f"sensor {sensor_id}, cluster {cluster}."
        )

    label_row = label_row.iloc[0]

    label = label_row[
        "Water_Quality_Label"
    ]

    quality_score = float(
        label_row[
            "cluster_quality_score"
        ]
    )

    # --------------------------------------------------------
    # Find the frozen contribution information
    # --------------------------------------------------------

    contribution_row = cluster_contributions[
        (
            cluster_contributions[SENSOR_COL]
            == sensor_id
        )
        &
        (
            cluster_contributions["cluster"]
            == cluster
        )
    ]

    if contribution_row.empty:
        raise ValueError(
            f"No contribution information found for "
            f"sensor {sensor_id}, cluster {cluster}."
        )

    contribution_row = (
        contribution_row
        .iloc[0]
        .copy()
    )

    # The explanation function needs the final label.
    contribution_row[
        "Water_Quality_Label"
    ] = label

    # --------------------------------------------------------
    # Human-readable explanation
    # --------------------------------------------------------

    main_driver, explanation = (
        _explain_sensor_status(
            contribution_row,
            label_col="Water_Quality_Label"
        )
    )

    # --------------------------------------------------------
    # Return API-friendly result
    # --------------------------------------------------------

    return {
        "sensor_id": sensor_id,
        "cluster": cluster,
        "Water_Quality_Label": label,
        "cluster_quality_score": quality_score,
        "main_driver": main_driver,
        "explanation": explanation,
    }