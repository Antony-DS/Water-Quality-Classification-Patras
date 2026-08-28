# Water Quality Classification – Patras

An unsupervised machine learning project for analyzing and classifying water-quality measurements collected from IoT sensors in Patras, Greece.

The project processes raw sensor measurements, identifies characteristic water-quality states using sensor-specific clustering, and assigns interpretable **Ok**, **Medium**, or **Danger** labels based on the environmental characteristics of each cluster.

## Overview

Environmental sensor data can be difficult to analyze directly due to measurement errors, different sensor baselines, skewed distributions, missing values, and differences in scale between variables.

This project implements an end-to-end workflow that includes:

* exploratory analysis of raw sensor measurements
* data cleaning and preprocessing
* feature selection
* feature transformation and scaling
* sensor-specific K-Means clustering
* cluster evaluation
* environmental scoring and cluster labeling
* identification of the measurements driving each classification
* inference logic for classifying new sensor readings

The goal is not simply to separate the observations into clusters, but to transform those clusters into interpretable water-quality states that can be used in monitoring and visualization systems.

---

## Measurements Used

The final analysis focuses on four water-quality measurements:

| Feature                   | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| **Turbidity (NTU)**       | Indicates the amount of suspended material in the water      |
| **ORP (mV)**              | Oxidation-Reduction Potential                                |
| **Oxygen Saturation (%)** | Percentage of dissolved oxygen saturation                    |
| **Conductivity**          | Indicator of the concentration of dissolved ionic substances |

Additional measurements were explored during the initial data analysis but were excluded when they were redundant, strongly related to other variables, or not considered useful for the final clustering process.

---

## Methodology

### 1. Exploratory Data Analysis

The raw sensor dataset is first inspected to understand:

* measurement distributions
* missing values
* extreme or invalid measurements
* differences between sensors
* relationships and correlations between variables
* temporal behaviour of the measurements

This stage is contained primarily in:

`exploration-nonclean.ipynb`

---

### 2. Data Cleaning

Real-world IoT measurements contained anomalous and invalid observations that could significantly affect the clustering process.

The cleaning workflow removes or handles problematic observations and produces a dataset suitable for subsequent analysis.

The cleaned data is generated in:

`cleaning_cleaned_final.ipynb`

The corresponding datasets are stored under:

`data/`

---

### 3. Feature Transformation and Scaling

The selected variables have very different numerical ranges and distributions, so they cannot be used directly with a distance-based algorithm such as K-Means.

The preprocessing pipeline therefore applies feature-specific transformations and robust scaling.

Turbidity receives additional transformation because of its strongly right-skewed distribution and the presence of large extreme values. This reduces the influence of very large observations while preserving meaningful differences at lower turbidity levels.

Feature weighting is also applied so that the clustering representation reflects the relative importance of the different measurements.

---

### 4. Sensor-Specific Clustering

Instead of fitting a single global clustering model to all sensors, the project fits **separate K-Means models for individual sensors**.

This is important because different sensors may operate under different baseline environmental conditions.

Each sensor's observations are therefore compared primarily against the historical behaviour of that sensor rather than against a single global reference distribution.

The resulting clusters represent different recurring environmental states for each sensor.

---

### 5. Cluster Evaluation

The quality and interpretability of the resulting clusters are evaluated using:

* cluster-level feature statistics
* median and mean measurements
* feature distributions
* boxplots
* silhouette scores
* comparison of environmental characteristics between clusters

This allows the clusters to be examined before assigning environmental labels.

---

### 6. Environmental Scoring and Labeling

K-Means produces cluster identifiers, but the numerical cluster numbers themselves have no environmental meaning.

A separate scoring stage therefore evaluates the characteristics of each cluster using the behaviour of the four selected measurements.

The scoring logic considers factors such as:

* elevated turbidity
* low ORP
* low oxygen saturation
* conductivity outside the sensor's typical range

Each cluster receives a **water-quality concern score** and is subsequently classified as:

* **Ok**
* **Medium**
* **Danger**

This separation between **clustering** and **labeling** is intentional.

The clustering algorithm is responsible only for identifying patterns in the data. Environmental interpretation is performed afterwards rather than forcing the clustering process itself to generate predetermined quality categories.

---

## Explainable Output

In addition to the final classification, the system identifies which measurement contributed most strongly to the assigned status.

Example output:

```python
{
    "sensor_id": 978,
    "Water_Quality_Label": "Medium",
    "main_driver": "Turbidity",
    "explanation": "Moderate concern is mainly associated with elevated turbidity and low ORP.",
    "cluster_quality_score": 0.43
}
```

For **Medium** and **Danger** states, the explanation highlights the measurements that increase environmental concern.

For **Ok** states, the explanation identifies the measurements that contribute to the lower concern level.

This makes the output more interpretable than returning a cluster number or classification alone.

---

## Inference on New Measurements

`water_quality_ML_deployment.py` contains the inference workflow for new sensor readings.

For every new observation, the pipeline:

1. identifies the corresponding sensor
2. loads the feature set used by that sensor
3. applies the same transformations used during training
4. applies the stored feature scalers and weights
5. assigns the observation to the nearest frozen K-Means cluster
6. retrieves the environmental label associated with that cluster
7. returns the quality score, main driver, and human-readable explanation

This allows the clustering models learned from historical measurements to be reused without retraining them every time a new observation arrives.

### Example Input

```python
reading = {
    "sensor_id": 984,
    "conductivity_of_solution_ mS_cm": 51500,
    "oxidation_reduction_potential_ mV": 280,
    "oxygen_saturation_%": 72,
    "turbidity_NTU": 35
}
```

---

## Repository Structure

```text
Water-Quality-Classification-Patras/
│
├── data/
│   ├── water_stats.csv
│   ├── cleaned_stats.csv
│   ├── water_model_ready.csv
│   └── labeled_water_stats.csv
│
├── exploration-nonclean.ipynb
│   └── Initial exploration of the raw sensor data
│
├── cleaning_cleaned_final.ipynb
│   └── Data cleaning and preprocessing
│
├── Classification-labeling_log_turbidity.ipynb
│   └── Clustering, environmental scoring and final labeling
│
├── funcs.py
│   └── Reusable preprocessing, evaluation, visualization and explanation functions
│
└── water_quality_ML_deployment.py
    └── Inference pipeline for new sensor measurements
```

---

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* joblib

Main machine-learning techniques:

* K-Means Clustering
* Robust Feature Scaling
* Feature Transformation
* Feature Weighting
* Silhouette Analysis
* Cluster Profiling
* Unsupervised Classification

---

## Project Workflow

```text
Raw IoT Sensor Data
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Feature Selection
        ↓
Transformation & Scaling
        ↓
Sensor-Specific K-Means
        ↓
Cluster Evaluation
        ↓
Environmental Concern Scoring
        ↓
Ok / Medium / Danger Labels
        ↓
Main Driver + Explanation
        ↓
Inference on New Measurements
```

---

## Important Note

The labels produced by this project are **analytical classifications derived from sensor behaviour and the scoring methodology used in this project**.

They should not be interpreted as official regulatory assessments of drinking-water safety or as a substitute for laboratory testing and environmental expertise.

---

## Context

This project was developed using real-world IoT water-quality sensor data from the city of Patras, Greece.

It focuses on the practical challenges of working with environmental sensor data, including noisy measurements, sensor-specific behaviour, preprocessing, unsupervised learning, interpretability, and the transition from exploratory analysis to reusable inference logic.
