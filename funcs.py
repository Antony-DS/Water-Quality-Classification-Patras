import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

def plot_sensors(sensor_ids, metrics, df, date_col="date_insert"):
    """
    Plots time series for one or more sensors, one row per sensor,
    one column per metric.
    
    Parameters
    ----------
    sensor_ids : single sensor id or list/array of sensor ids
    metrics : list of column names to plot
    df : the dataframe containing the data
    date_col : name of the date/time column (default "date_insert")
    """
    # allow a single sensor id to be passed instead of a list
    if not isinstance(sensor_ids, (list, tuple, set)):
        sensor_ids = [sensor_ids]

    n_sensors = len(sensor_ids)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        n_sensors, n_metrics,
        figsize=(5 * n_metrics, 4 * n_sensors),
        squeeze=False
    )

    for row, id in enumerate(sensor_ids):
        sensor_df = df[df['sensor_id'] == id]

        if sensor_df.empty:
            print(f"Warning: no data found for sensor {id}")

        for col, metric in enumerate(metrics):
            ax = axes[row][col]
            ax.plot(sensor_df[date_col], sensor_df[metric])
            ax.set_title(f"{metric} — Sensor {id} - {df[df['sensor_id'] == id]['location'].iloc[0]}")
            ax.set_xlabel("Date")
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()



def plot_sensors2(sensor_ids, metrics, df, date_col="date_insert"):
    """Plots time series for one or more sensors, one row per sensor,

    one column per metric. Shows true gaps without artificial lines.

    Parameters
    ----------
    sensor_ids : single sensor id or list/array of sensor ids
    metrics : list of column names to plot
    df : the dataframe containing the data
    date_col : name of the date/time column (default "date_insert")
    """
    # ensure date_col is datetime type for correct axis scaling
    df[date_col] = pd.to_datetime(df[date_col])

    # allow a single sensor id to be passed instead of a list
    if not isinstance(sensor_ids, (list, tuple, set)):
        sensor_ids = [sensor_ids]

    n_sensors = len(sensor_ids)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        n_sensors,
        n_metrics,
        figsize=(5 * n_metrics, 4 * n_sensors),
        squeeze=False,
    )

    for row, id in enumerate(sensor_ids):
        sensor_df = df[df["sensor_id"] == id]

        if sensor_df.empty:
            print(f"Warning: no data found for sensor {id}")
            continue

        # Get location safely
        loc_name = (
            sensor_df["location"].iloc[0]
            if "location" in sensor_df.columns
            else "Unknown"
        )

        for col, metric in enumerate(metrics):
            ax = axes[row][col]

            # --- ΑΛΛΑΓΗ ΕΔΩ: Χρήση scatter αντί για plot ---
            # s=1: πολύ μικρό μέγεθος τελείας για να φαίνεται σαν λεπτή γραμμή όταν έχει πυκνά δεδομένα
            # edgecolor=None: αφαιρεί το περίγραμμα για να μην μπερδεύεται το γράφημα
            ax.scatter(
                sensor_df[date_col],
                sensor_df[metric],
                s=5,
                color="#1f77b4",
                edgecolor="none",
            )

            ax.set_title(f"{metric} — Sensor {id} - {loc_name}")
            ax.set_xlabel("Date")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=45)

            # Ρύθμιση των ορίων του άξονα Χ ώστε να δείχνει όλο το εύρος του αρχικού DataFrame
            ax.set_xlim(df[date_col].min(), df[date_col].max())

    plt.tight_layout()
    plt.show()