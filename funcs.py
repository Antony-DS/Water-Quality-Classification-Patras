from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


try:
    from IPython.display import display
except ImportError:
    display = print

def plot_sensors(
    sensor_ids,
    metrics,
    df,
    date_col="date_insert",
    sensor_col="sensor_id",
    location_col="location",
    max_gap="24h",
    show_points=False,
):
    """
    Plot time series for one or more sensors.

    Lines are broken when the time between consecutive observations
    exceeds max_gap, preventing missing periods from appearing as
    continuous trends.

    Parameters
    ----------
    sensor_ids:
        One sensor ID or a collection of sensor IDs.

    metrics:
        Column names to plot.

    df:
        Dataframe containing the sensor observations.

    date_col:
        Datetime column.

    sensor_col:
        Sensor identifier column.

    location_col:
        Optional location-name column.

    max_gap:
        Maximum time difference for connecting consecutive points.
        Examples: "12h", "24h", "3D". Use None to connect every point.

    show_points:
        Whether to display individual measurements as markers.
    """

    if not isinstance(sensor_ids, (list, tuple, set)):
        sensor_ids = [sensor_ids]

    plotting_df = df.copy()

    plotting_df[date_col] = pd.to_datetime(
        plotting_df[date_col],
        errors="coerce",
    )

    n_sensors = len(sensor_ids)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        n_sensors,
        n_metrics,
        figsize=(5 * n_metrics, 4 * n_sensors),
        squeeze=False,
    )

    gap_threshold = (
        pd.Timedelta(max_gap)
        if max_gap is not None
        else None
    )

    for row, sensor_id in enumerate(sensor_ids):

        sensor_df = (
            plotting_df[
                plotting_df[sensor_col] == sensor_id
            ]
            .dropna(subset=[date_col])
            .sort_values(date_col)
            .copy()
        )

        if sensor_df.empty:
            print(
                f"Warning: no data found for sensor {sensor_id}"
            )

            for axis in axes[row]:
                axis.set_visible(False)

            continue

        if (
            location_col in sensor_df.columns
            and sensor_df[location_col].notna().any()
        ):
            location = sensor_df[location_col].dropna().iloc[0]
        else:
            location = "Unknown location"

        time_gaps = sensor_df[date_col].diff()

        for col, metric in enumerate(metrics):

            axis = axes[row][col]

            plot_data = sensor_df[
                [date_col, metric]
            ].copy()

            if gap_threshold is not None:
                gap_mask = time_gaps > gap_threshold
                plot_data.loc[gap_mask, metric] = pd.NA

            axis.plot(
                plot_data[date_col],
                plot_data[metric],
                marker="o" if show_points else None,
                markersize=2,
                linewidth=1,
            )

            axis.set_title(
                f"{metric} — Sensor {sensor_id} - {location}"
            )
            axis.set_xlabel("Date")
            axis.set_ylabel(metric)
            axis.tick_params(
                axis="x",
                rotation=45,
            )

    plt.tight_layout()
    plt.show()



DEFAULT_FEATURES = [
    "conductivity_of_solution_ mS_cm",
    "oxidation_reduction_potential_ mV",
    "oxygen_saturation_%",
    "turbidity_NTU",
]

CONTRIBUTION_COLUMNS = [
    "turbidity_contribution",
    "orp_contribution",
    "oxygen_contribution",
    "conductivity_contribution",
]


def _convert_to_sensor_list(
    sensor_ids: Any | Iterable[Any],
) -> list[Any]:
    """
    Convert one sensor ID or an iterable of sensor IDs into a list.
    """

    if isinstance(sensor_ids, (str, int, float, np.integer, np.floating)):
        return [sensor_ids]

    try:
        return list(sensor_ids)
    except TypeError as error:
        raise TypeError(
            "sensor_ids must be one sensor ID or an iterable of sensor IDs."
        ) from error


def _show_or_save_figure(
    fig: plt.Figure,
    *,
    save_dir: Path | None,
    filename: str,
    show_plots: bool,
) -> None:
    """
    Display a figure and optionally save it.
    """

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def inspect_sensors(

        
    sensor_ids: Any | Iterable[Any],
    final_training_df: pd.DataFrame,
    *,
    cluster_contributions: pd.DataFrame | None = None,
    silhouette_df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    sensor_id_col: str = "sensor_id",
    cluster_col: str = "cluster",
    label_col: str = "Water_Quality_Label",
    score_col: str = "cluster_quality_score",
    label_order: tuple[str, ...] = ("Ok", "Medium", "Danger"),
    show_tables: bool = True,
    plot_by_label: bool = True,
    plot_by_cluster: bool = True,
    plot_contributions: bool = True,
    show_plots: bool = True,
    save_dir: str | Path | None = None,
) -> dict[Any, dict[str, Any]]:
    """
    Inspect one sensor or multiple sensors from the water-quality project.

    The function produces:
        1. Label counts.
        2. Feature statistics by quality label.
        3. Feature statistics by K-means cluster.
        4. Cluster contribution information.
        5. The sensor's silhouette score, when provided.
        6. Boxplots grouped by quality label.
        7. Boxplots grouped by the original K-means cluster.
        8. A contribution bar chart for each cluster.

    Parameters
    ----------
    sensor_ids:
        A single sensor ID, such as 976, or a list such as
        [976, 981, 983].

    final_training_df:
        The final observation-level dataframe containing clusters and labels.

    cluster_contributions:
        Optional cluster-level dataframe containing the contribution columns.

    silhouette_df:
        Optional dataframe containing sensor-level silhouette scores.
        The sensor column can be named either "sensor" or match sensor_id_col.

    features:
        Feature columns to inspect. The four project features are used
        by default.

    save_dir:
        Optional folder in which the figures will be saved.

    Returns
    -------
    dict
        Dictionary keyed by sensor ID. Each sensor contains its summary
        tables, contribution table, and silhouette score.
    """

    if features is None:
        features = DEFAULT_FEATURES.copy()

    sensor_ids = _convert_to_sensor_list(sensor_ids)

    required_columns = [
        sensor_id_col,
        cluster_col,
        label_col,
        *features,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in final_training_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "The following columns are missing from final_training_df: "
            f"{missing_columns}"
        )

    output_directory = (
        Path(save_dir)
        if save_dir is not None
        else None
    )

    available_sensors = set(
        final_training_df[sensor_id_col].unique()
    )

    results: dict[Any, dict[str, Any]] = {}

    for sensor_id in sensor_ids:

        if sensor_id not in available_sensors:
            print(
                f"Sensor {sensor_id} was not found in "
                "final_training_df."
            )
            continue

        sensor_df = (
            final_training_df.loc[
                final_training_df[sensor_id_col] == sensor_id
            ]
            .copy()
        )

        present_labels = sensor_df[label_col].dropna().unique()

        present_label_order = [
            label
            for label in label_order
            if label in present_labels
        ]

        # Preserve unexpected labels rather than silently excluding them.
        present_label_order.extend(
            label
            for label in present_labels
            if label not in present_label_order
        )

        # -------------------------------------------------------------
        # Label distribution
        # -------------------------------------------------------------

        label_counts = (
            sensor_df[label_col]
            .value_counts()
            .reindex(present_label_order, fill_value=0)
            .rename("observations")
            .to_frame()
        )

        label_counts["percentage"] = (
            100
            * label_counts["observations"]
            / len(sensor_df)
        )

        # -------------------------------------------------------------
        # Feature statistics by environmental label
        # -------------------------------------------------------------

        statistics_by_label = (
            sensor_df
            .groupby(label_col)[features]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reindex(present_label_order)
        )

        # -------------------------------------------------------------
        # Feature statistics by original K-means cluster
        # -------------------------------------------------------------

        cluster_aggregation: dict[str, tuple[str, str]] = {
            "observations": (cluster_col, "size"),
        }

        if score_col in sensor_df.columns:
            cluster_aggregation[score_col] = (
                score_col,
                "first",
            )

        for feature in features:
            cluster_aggregation[f"{feature}_mean"] = (
                feature,
                "mean",
            )
            cluster_aggregation[f"{feature}_median"] = (
                feature,
                "median",
            )

        cluster_summary = (
            sensor_df
            .groupby(
                [cluster_col, label_col],
                as_index=False,
            )
            .agg(**cluster_aggregation)
        )

        if score_col in cluster_summary.columns:
            cluster_summary = cluster_summary.sort_values(
                score_col
            )
        else:
            cluster_summary = cluster_summary.sort_values(
                cluster_col
            )

        # -------------------------------------------------------------
        # Contribution table
        # -------------------------------------------------------------

        sensor_contributions = None

        if cluster_contributions is not None:

            if sensor_id_col not in cluster_contributions.columns:
                raise KeyError(
                    f"'{sensor_id_col}' is missing from "
                    "cluster_contributions."
                )

            sensor_contributions = (
                cluster_contributions.loc[
                    cluster_contributions[sensor_id_col]
                    == sensor_id
                ]
                .copy()
            )

            if score_col in sensor_contributions.columns:
                sensor_contributions = (
                    sensor_contributions.sort_values(score_col)
                )

        elif all(
            column in sensor_df.columns
            for column in CONTRIBUTION_COLUMNS
        ):
            aggregation_dictionary = {
                column: "median"
                for column in CONTRIBUTION_COLUMNS
            }

            if score_col in sensor_df.columns:
                aggregation_dictionary[score_col] = "first"

            sensor_contributions = (
                sensor_df
                .groupby(
                    [sensor_id_col, cluster_col, label_col],
                    as_index=False,
                )
                .agg(aggregation_dictionary)
            )

        # -------------------------------------------------------------
        # Silhouette score
        # -------------------------------------------------------------

        sensor_silhouette = None

        if silhouette_df is not None:

            if sensor_id_col in silhouette_df.columns:
                silhouette_sensor_col = sensor_id_col
            elif "sensor" in silhouette_df.columns:
                silhouette_sensor_col = "sensor"
            else:
                raise KeyError(
                    "silhouette_df must contain either "
                    f"'{sensor_id_col}' or 'sensor'."
                )

            silhouette_rows = silhouette_df.loc[
                silhouette_df[silhouette_sensor_col] == sensor_id
            ]

            if not silhouette_rows.empty:
                sensor_silhouette = float(
                    silhouette_rows.iloc[0]["silhouette"]
                )

        # -------------------------------------------------------------
        # Display tables
        # -------------------------------------------------------------

        if show_tables:

            print("=" * 80)
            print(f"SENSOR {sensor_id}")
            print("=" * 80)

            print(f"Observations: {len(sensor_df)}")

            if sensor_silhouette is not None:
                print(
                    "Silhouette score: "
                    f"{sensor_silhouette:.4f}"
                )

            print("\nLabel distribution:")
            display(label_counts.round(3))

            print("\nCluster summary:")
            display(cluster_summary.round(3))

            print("\nStatistics by quality label:")
            display(statistics_by_label.round(3))

            if (
                sensor_contributions is not None
                and not sensor_contributions.empty
            ):
                print("\nCluster score contributions:")
                display(sensor_contributions.round(3))

        # -------------------------------------------------------------
        # Boxplots grouped by final quality label
        # -------------------------------------------------------------

        if plot_by_label:

            fig, axes = plt.subplots(
                nrows=2,
                ncols=2,
                figsize=(15, 10),
            )

            axes = axes.flatten()

            for axis, feature in zip(axes, features):

                sns.boxplot(
                    data=sensor_df,
                    x=label_col,
                    y=feature,
                    order=present_label_order,
                    ax=axis,
                )

                axis.set_title(
                    f"Sensor {sensor_id}: {feature}"
                )
                axis.set_xlabel("Quality label")

            fig.suptitle(
                f"Sensor {sensor_id}: distributions by quality label",
                fontsize=15,
                y=1.02,
            )

            fig.tight_layout()

            _show_or_save_figure(
                fig,
                save_dir=output_directory,
                filename=(
                    f"sensor_{sensor_id}_by_quality_label.png"
                ),
                show_plots=show_plots,
            )

        # -------------------------------------------------------------
        # Boxplots grouped by original K-means cluster
        # -------------------------------------------------------------

        if plot_by_cluster:

            fig, axes = plt.subplots(
                nrows=2,
                ncols=2,
                figsize=(15, 10),
            )

            axes = axes.flatten()

            cluster_order = sorted(
                sensor_df[cluster_col].unique()
            )

            for axis, feature in zip(axes, features):

                sns.boxplot(
                    data=sensor_df,
                    x=cluster_col,
                    y=feature,
                    order=cluster_order,
                    ax=axis,
                )

                axis.set_title(
                    f"Sensor {sensor_id}: {feature}"
                )
                axis.set_xlabel("K-means cluster")

            fig.suptitle(
                f"Sensor {sensor_id}: distributions by K-means cluster",
                fontsize=15,
                y=1.02,
            )

            fig.tight_layout()

            _show_or_save_figure(
                fig,
                save_dir=output_directory,
                filename=(
                    f"sensor_{sensor_id}_by_cluster.png"
                ),
                show_plots=show_plots,
            )

        # -------------------------------------------------------------
        # Contribution chart
        # -------------------------------------------------------------

        valid_contribution_columns = []

        if sensor_contributions is not None:
            valid_contribution_columns = [
                column
                for column in CONTRIBUTION_COLUMNS
                if column in sensor_contributions.columns
            ]

        if (
            plot_contributions
            and sensor_contributions is not None
            and not sensor_contributions.empty
            and valid_contribution_columns
        ):

            contribution_plot_df = (
                sensor_contributions.copy()
            )

            if label_col not in contribution_plot_df.columns:
                label_mapping = (
                    sensor_df[
                        [cluster_col, label_col]
                    ]
                    .drop_duplicates()
                )

                contribution_plot_df = (
                    contribution_plot_df.merge(
                        label_mapping,
                        on=cluster_col,
                        how="left",
                    )
                )

            contribution_plot_df["cluster_display"] = (
                "Cluster "
                + contribution_plot_df[cluster_col].astype(str)
                + " ("
                + contribution_plot_df[label_col].astype(str)
                + ")"
            )

            contribution_plot_df = (
                contribution_plot_df.set_index(
                    "cluster_display"
                )
            )

            fig, axis = plt.subplots(
                figsize=(11, 6)
            )

            contribution_plot_df[
                valid_contribution_columns
            ].plot(
                kind="bar",
                ax=axis,
            )

            axis.axhline(
                0,
                linewidth=1,
            )

            axis.set_title(
                f"Sensor {sensor_id}: median score contributions"
            )
            axis.set_xlabel("Cluster and final label")
            axis.set_ylabel(
                "Contribution to concern score"
            )
            axis.tick_params(
                axis="x",
                rotation=0,
            )
            axis.legend(
                title="Score component",
            )

            fig.tight_layout()

            _show_or_save_figure(
                fig,
                save_dir=output_directory,
                filename=(
                    f"sensor_{sensor_id}_contributions.png"
                ),
                show_plots=show_plots,
            )

        results[sensor_id] = {
            "data": sensor_df,
            "label_counts": label_counts,
            "statistics_by_label": statistics_by_label,
            "cluster_summary": cluster_summary,
            "contributions": sensor_contributions,
            "silhouette": sensor_silhouette,
        }

    return results






CONTRIBUTION_COLUMNS = [
    "turbidity_contribution",
    "orp_contribution",
    "oxygen_contribution",
    "conductivity_contribution",
]


def _as_sensor_list(
    sensor_ids: Any | Iterable[Any] | None,
) -> list[Any] | None:
    """
    Convert one sensor ID or an iterable of IDs into a list.
    """

    if sensor_ids is None:
        return None

    if isinstance(
        sensor_ids,
        (str, int, float, np.integer, np.floating),
    ):
        return [sensor_ids]

    return list(sensor_ids)


def _join_phrases(phrases: list[str]) -> str:
    """
    Join one or two explanation phrases naturally.
    """

    if not phrases:
        return ""

    if len(phrases) == 1:
        return phrases[0]

    return f"{phrases[0]} and {phrases[1]}"


def _explain_sensor_status(
    row: pd.Series,
    *,
    label_col: str,
) -> tuple[str, str]:
    """
    Create the main driver and explanation for one sensor status.

    Positive contributions increase concern.
    Negative contributions decrease concern.
    """

    contribution_information = {
        "turbidity_contribution": {
            "name": "Turbidity",
            "worse": "elevated turbidity",
            "better": "low turbidity",
        },
        "orp_contribution": {
            "name": "ORP",
            "worse": "low ORP",
            "better": "favourable ORP",
        },
        "oxygen_contribution": {
            "name": "Oxygen saturation",
            "worse": "low oxygen saturation",
            "better": "good oxygen saturation",
        },
        "conductivity_contribution": {
            "name": "Conductivity",
            "worse": (
                "conductivity outside this sensor's typical range"
            ),
            "better": "stable conductivity",
        },
    }

    contributions = {
        column: float(row[column])
        for column in CONTRIBUTION_COLUMNS
        if column in row.index and pd.notna(row[column])
    }

    label = str(row[label_col])

    positive_contributions = sorted(
        [
            (column, value)
            for column, value in contributions.items()
            if value > 0
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    negative_contributions = sorted(
        [
            (column, value)
            for column, value in contributions.items()
            if value < 0
        ],
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    # For Ok states, explain which measurements reduce concern.
    if label == "Ok":

        helpful_factors = [
            contribution_information[column]["better"]
            for column, _ in negative_contributions[:2]
        ]

        if helpful_factors:
            main_driver = contribution_information[
                negative_contributions[0][0]
            ]["name"]

            explanation = (
                "Overall concern is low, mainly because of "
                f"{_join_phrases(helpful_factors)}."
            )

            return main_driver, explanation

        return (
            "Combined score",
            "The combined measurements indicate a low level of concern.",
        )

    # For Medium and Danger states, explain the largest positive drivers.
    if positive_contributions:

        largest_column, largest_value = positive_contributions[0]

        selected_drivers = [largest_column]

        # Mention the second factor only when it is substantial.
        if len(positive_contributions) > 1:
            second_column, second_value = positive_contributions[1]

            if second_value >= 0.5 * largest_value:
                selected_drivers.append(second_column)

        worsening_factors = [
            contribution_information[column]["worse"]
            for column in selected_drivers
        ]

        main_driver = contribution_information[
            largest_column
        ]["name"]

        if label == "Danger":
            prefix = "High concern is mainly associated with "
        else:
            prefix = "Moderate concern is mainly associated with "

        explanation = (
            prefix
            + _join_phrases(worsening_factors)
            + "."
        )

        return main_driver, explanation

    return (
        "Combined score",
        "The status is based on the combined behaviour of all measurements.",
    )


def build_latest_sensor_statuses(
    final_training_df: pd.DataFrame,
    cluster_contributions: pd.DataFrame,
    *,
    timestamp_col: str,
    sensor_ids: Any | Iterable[Any] | None = None,
    sensor_id_col: str = "sensor_id",
    cluster_col: str = "cluster",
    label_col: str = "Water_Quality_Label",
    score_col: str = "cluster_quality_score",
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Create dashboard-ready status information for the latest reading
    from each sensor.

    Parameters
    ----------
    final_training_df:
        Observation-level dataframe containing sensor IDs, timestamps,
        clusters, quality labels and original measurements.

    cluster_contributions:
        Cluster-level dataframe containing the four contribution
        columns and the cluster quality score.

    timestamp_col:
        Name of the datetime column.

    sensor_ids:
        Optional sensor ID or list of sensor IDs. When omitted,
        all sensors are returned.

    Returns
    -------
    pandas.DataFrame
        One row per sensor containing the latest status, measurements,
        main driver and explanation.
    """

    if feature_columns is None:
        feature_columns = [
            "conductivity_of_solution_ mS_cm",
            "oxidation_reduction_potential_ mV",
            "oxygen_saturation_%",
            "turbidity_NTU",
        ]

    required_status_columns = [
        sensor_id_col,
        timestamp_col,
        cluster_col,
        label_col,
        *feature_columns,
    ]

    missing_status_columns = [
        column
        for column in required_status_columns
        if column not in final_training_df.columns
    ]

    if missing_status_columns:
        raise KeyError(
            "The following columns are missing from "
            f"final_training_df: {missing_status_columns}"
        )

    required_contribution_columns = [
        sensor_id_col,
        cluster_col,
        score_col,
        *CONTRIBUTION_COLUMNS,
    ]

    missing_contribution_columns = [
        column
        for column in required_contribution_columns
        if column not in cluster_contributions.columns
    ]

    if missing_contribution_columns:
        raise KeyError(
            "The following columns are missing from "
            f"cluster_contributions: {missing_contribution_columns}"
        )

    data = final_training_df.copy()

    data[timestamp_col] = pd.to_datetime(
        data[timestamp_col],
        errors="coerce",
    )

    invalid_timestamp_count = data[timestamp_col].isna().sum()

    if invalid_timestamp_count > 0:
        raise ValueError(
            f"{invalid_timestamp_count} rows contain invalid or missing "
            f"values in '{timestamp_col}'."
        )

    selected_sensors = _as_sensor_list(sensor_ids)

    if selected_sensors is not None:

        unavailable_sensors = sorted(
            set(selected_sensors)
            - set(data[sensor_id_col].unique())
        )

        if unavailable_sensors:
            raise ValueError(
                "These sensor IDs were not found: "
                f"{unavailable_sensors}"
            )

        data = data[
            data[sensor_id_col].isin(selected_sensors)
        ].copy()

    # Use _row_id as a tie-breaker when two readings have the same time.
    sort_columns = [
        sensor_id_col,
        timestamp_col,
    ]

    if "_row_id" in data.columns:
        sort_columns.append("_row_id")

    latest_rows = (
        data.sort_values(sort_columns)
        .groupby(sensor_id_col, as_index=False)
        .tail(1)
        .copy()
    )

    # The cluster-level contribution table is treated as authoritative.
    columns_to_remove = [
        column
        for column in [score_col, *CONTRIBUTION_COLUMNS]
        if column in latest_rows.columns
    ]

    latest_rows = latest_rows.drop(
        columns=columns_to_remove,
        errors="ignore",
    )

    contribution_table = cluster_contributions[
        required_contribution_columns
    ].copy()

    duplicate_clusters = contribution_table.duplicated(
        subset=[sensor_id_col, cluster_col],
        keep=False,
    )

    if duplicate_clusters.any():
        duplicated_keys = (
            contribution_table.loc[
                duplicate_clusters,
                [sensor_id_col, cluster_col],
            ]
            .drop_duplicates()
            .to_dict(orient="records")
        )

        raise ValueError(
            "cluster_contributions contains duplicate sensor-cluster "
            f"combinations: {duplicated_keys}"
        )

    latest_status = latest_rows.merge(
        contribution_table,
        on=[sensor_id_col, cluster_col],
        how="left",
        validate="many_to_one",
    )

    missing_scores = latest_status[score_col].isna()

    if missing_scores.any():
        missing_keys = (
            latest_status.loc[
                missing_scores,
                [sensor_id_col, cluster_col],
            ]
            .to_dict(orient="records")
        )

        raise ValueError(
            "No contribution information was found for: "
            f"{missing_keys}"
        )

    explanation_results = latest_status.apply(
        lambda row: _explain_sensor_status(
            row,
            label_col=label_col,
        ),
        axis=1,
    )

    latest_status["main_driver"] = [
        result[0]
        for result in explanation_results
    ]

    latest_status["explanation"] = [
        result[1]
        for result in explanation_results
    ]

    status_rank_mapping = {
        "Ok": 0,
        "Medium": 1,
        "Danger": 2,
    }

    latest_status["status_rank"] = (
        latest_status[label_col]
        .map(status_rank_mapping)
        .fillna(99)
        .astype(int)
    )

    output_columns = [
        sensor_id_col,
        timestamp_col,
        label_col,
        "status_rank",
        cluster_col,
        score_col,
        "main_driver",
        "explanation",
        *feature_columns,
        *CONTRIBUTION_COLUMNS,
    ]

    return (
        latest_status[output_columns]
        .sort_values(sensor_id_col)
        .reset_index(drop=True)
    )