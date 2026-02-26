import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Ensure the current directory is in the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.clustering import SelfConstructingClustering
from config import (
    ADAPTIVE_THRESHOLD,
    ANOMALY_BUFFER,
    CLUSTER_THRESHOLD,
    DISTANCE_METRIC,
    FEATURE_NAMES,
    LEARNING_RATE,
    MINKOWSKI_P,
)


st.set_page_config(
    page_title="Federated DDoS IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "dummy_data.csv")
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)


def prepare_features(df, configured_features):
    available = [f for f in configured_features if f in df.columns]
    missing = [f for f in configured_features if f not in df.columns]

    if not available:
        raise ValueError("None of the configured FEATURE_NAMES columns exist in dataset.")

    feature_df = df[available].copy()
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    non_constant_cols = feature_df.columns[feature_df.var() > 0].tolist()
    feature_df = feature_df[non_constant_cols]

    if feature_df.shape[1] == 0:
        raise ValueError("No non-constant numeric features remain after preprocessing.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values)
    return feature_df, X_scaled, scaler, missing


def compute_silhouette(X_scaled, cluster_labels, metric_name, minkowski_p):
    n_clusters = len(np.unique(cluster_labels))
    if n_clusters <= 1 or n_clusters >= len(X_scaled):
        return None

    metric = {
        "euclidean": "euclidean",
        "manhattan": "manhattan",
        "cosine": "cosine",
        "minkowski": "minkowski",
    }.get(metric_name, "euclidean")

    kwargs = {"metric": metric}
    if metric == "minkowski":
        kwargs["p"] = minkowski_p

    try:
        return silhouette_score(X_scaled, cluster_labels, **kwargs)
    except Exception:
        return None


# --- Sidebar Configuration ---
st.sidebar.title("Configuration")

distance_options = ["euclidean", "manhattan", "minkowski", "cosine"]
default_metric_idx = distance_options.index(DISTANCE_METRIC) if DISTANCE_METRIC in distance_options else 0
distance_metric = st.sidebar.selectbox(
    "Distance Metric",
    distance_options,
    index=default_metric_idx,
    help="Metric used in SCC distance comparisons.",
)

st.sidebar.subheader("Model Hyperparameters")

if distance_metric == "cosine":
    threshold = st.sidebar.slider(
        "Cluster Threshold",
        min_value=0.01,
        max_value=1.0,
        value=float(min(max(CLUSTER_THRESHOLD, 0.01), 1.0)),
        step=0.01,
        help="For cosine distance, values are typically in [0, 1]. Lower = stricter cluster creation.",
    )
else:
    threshold = st.sidebar.slider(
        "Cluster Threshold",
        min_value=0.05,
        max_value=3.0,
        value=float(min(max(CLUSTER_THRESHOLD, 0.05), 3.0)),
        step=0.05,
        help="Distance threshold for assigning to an existing cluster.",
    )

anomaly_buffer = st.sidebar.slider(
    "Anomaly Buffer (lambda)",
    min_value=0.1,
    max_value=5.0,
    value=float(min(max(ANOMALY_BUFFER, 0.1), 5.0)),
    step=0.1,
    help="Anomaly threshold = mean(min_distance) + lambda * std(min_distance).",
)

adaptive_threshold = st.sidebar.checkbox(
    "Use Adaptive Threshold (per cluster)",
    value=ADAPTIVE_THRESHOLD,
    help="If enabled, each cluster gets its own anomaly threshold.",
)

minkowski_p = MINKOWSKI_P
if distance_metric == "minkowski":
    minkowski_p = st.sidebar.slider(
        "Minkowski p",
        min_value=1,
        max_value=10,
        value=max(1, int(MINKOWSKI_P)),
        step=1,
        help="Minkowski order parameter p.",
    )

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.01,
    max_value=1.0,
    value=float(min(max(LEARNING_RATE, 0.01), 1.0)),
    step=0.01,
    help="Kept for API compatibility; centroid update currently uses exact running mean.",
)

st.sidebar.caption(
    "Note: `learning_rate` is currently not used in centroid update logic in "
    "`client/clustering.py`."
)


# --- Main Content ---
st.title("🛡️ Federated DDoS IDS Dashboard")
st.markdown(
    f"**Metric:** `{distance_metric}` | "
    f"**Cluster Threshold:** `{threshold}` | "
    f"**Anomaly Buffer (lambda):** `{anomaly_buffer}` | "
    f"**Adaptive Threshold:** `{adaptive_threshold}`"
)

df = load_data()
if df is None:
    st.error("Data file not found at `data/dummy_data.csv`.")
    st.stop()

with st.expander("View Raw Data Snippet"):
    st.dataframe(df.head())

try:
    feature_df, X_scaled, scaler, missing_features = prepare_features(df, FEATURE_NAMES)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if missing_features:
    st.warning(
        "Some configured features are missing in the dataset and were ignored: "
        + ", ".join(missing_features)
    )

st.subheader("1. Model Training & Clustering")

model = SelfConstructingClustering(
    threshold=threshold,
    learning_rate=learning_rate,
    distance_metric=distance_metric,
    anomaly_buffer=anomaly_buffer,
    minkowski_p=minkowski_p,
    adaptive_threshold=adaptive_threshold,
)

with st.spinner(f"Training model with {distance_metric} metric..."):
    model.fit(X_scaled)

num_clusters = len(model.centroids_)
if num_clusters == 0:
    st.error("No clusters formed. Try increasing threshold or checking data quality.")
    st.stop()

all_dists = model.transform(X_scaled)
min_distances = np.min(all_dists, axis=1)
cluster_labels = np.argmin(all_dists, axis=1)
predictions = model.predict(X_scaled)
anomaly_mask = predictions.astype(bool)

sil_score = compute_silhouette(X_scaled, cluster_labels, distance_metric, minkowski_p)
anomaly_rate = 100.0 * np.mean(anomaly_mask)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Samples", f"{X_scaled.shape[0]}")
kpi2.metric("Features Used", f"{X_scaled.shape[1]}")
kpi3.metric("Clusters", f"{num_clusters}")
kpi4.metric("Anomaly Threshold", f"{model.anomaly_threshold_:.4f}")
kpi5.metric("Predicted Anomaly Rate", f"{anomaly_rate:.1f}%")

if sil_score is not None:
    st.caption(f"Silhouette Score ({distance_metric}): `{sil_score:.4f}`")
else:
    st.caption("Silhouette Score: not available for current cluster configuration.")

if adaptive_threshold and model.cluster_thresholds_:
    st.info("Adaptive thresholding is enabled: each cluster uses its own anomaly threshold.")
else:
    st.info("Global thresholding is enabled: one anomaly threshold is used for all clusters.")


# --- Visualizations ---
tab_centroids, tab_dist, tab_pca, tab_summary = st.tabs(
    ["2. Cluster Centroids", "3. Distance Distribution", "4. 2D Projection (PCA)", "5. Cluster Summary"]
)

with tab_centroids:
    centroids_df = pd.DataFrame(model.centroids_, columns=feature_df.columns)
    centroids_df.index = [f"Cluster {i}" for i in range(num_clusters)]

    st.caption("Index: each row is a centroid (`Cluster 0`, `Cluster 1`, ...).")
    st.dataframe(centroids_df)

    fig_centroid, ax_centroid = plt.subplots(
        figsize=(max(8, len(feature_df.columns) * 0.9), max(3.0, num_clusters * 0.8))
    )
    im = ax_centroid.imshow(centroids_df.values, cmap="viridis", aspect="auto")
    ax_centroid.set_title("Centroid Values Heatmap")
    ax_centroid.set_xticks(np.arange(len(feature_df.columns)))
    ax_centroid.set_xticklabels(feature_df.columns, rotation=45, ha="right")
    ax_centroid.set_yticks(np.arange(num_clusters))
    ax_centroid.set_yticklabels(centroids_df.index.tolist())

    if num_clusters <= 8 and len(feature_df.columns) <= 12:
        for r in range(num_clusters):
            for c in range(len(feature_df.columns)):
                val = centroids_df.values[r, c]
                txt_color = "white" if val < np.mean(centroids_df.values) else "black"
                ax_centroid.text(c, r, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=8)

    cbar = fig_centroid.colorbar(im, ax=ax_centroid)
    cbar.set_label("Standardized Feature Value (z-score)")
    st.pyplot(fig_centroid)

    st.caption(
        "Color index: purple/blue = lower z-score values, green = medium, yellow = higher z-score values."
    )

with tab_dist:
    normal_dists = min_distances[~anomaly_mask]
    attack_dists = min_distances[anomaly_mask]

    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    ax_hist.hist(
        [normal_dists, attack_dists],
        bins=20,
        stacked=True,
        color=["#60a5fa", "#ef4444"],
        edgecolor="black",
        label=["Predicted Normal", "Predicted Attack"],
    )
    ax_hist.axvline(
        model.anomaly_threshold_,
        color="black",
        linestyle="dashed",
        linewidth=2,
    )
    ax_hist.set_title("Distance to Nearest Cluster")
    ax_hist.set_xlabel("Distance")
    ax_hist.set_ylabel("Frequency")

    legend_handles = [
        Patch(facecolor="#60a5fa", edgecolor="black", label="Predicted Normal samples"),
        Patch(facecolor="#ef4444", edgecolor="black", label="Predicted Attack samples"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="dashed",
            linewidth=2,
            label=f"Anomaly Threshold = {model.anomaly_threshold_:.4f}",
        ),
    ]
    ax_hist.legend(handles=legend_handles)
    st.pyplot(fig_hist)

    st.caption(
        "Index: blue bars = normal predictions, red bars = attack predictions, black dashed line = active threshold."
    )

    distance_stats = pd.DataFrame(
        {
            "Metric": ["Min Distance", "Mean Distance", "Max Distance", "Anomaly Threshold"],
            "Value": [
                float(np.min(min_distances)),
                float(np.mean(min_distances)),
                float(np.max(min_distances)),
                float(model.anomaly_threshold_),
            ],
        }
    )
    st.dataframe(distance_stats, hide_index=True)

with tab_pca:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    cluster_cmap = plt.get_cmap("tab10", num_clusters)
    fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
    scatter = ax_pca.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        cmap=cluster_cmap,
        vmin=-0.5,
        vmax=num_clusters - 0.5,
        alpha=0.8,
        s=50,
    )

    # Highlight predicted anomalies with ring markers
    if np.any(anomaly_mask):
        ax_pca.scatter(
            X_pca[anomaly_mask, 0],
            X_pca[anomaly_mask, 1],
            facecolors="none",
            edgecolors="red",
            s=120,
            linewidths=1.5,
            label="Predicted Attack (ring)",
        )

    centroids_pca = pca.transform(np.array(model.centroids_))
    ax_pca.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c="black",
        marker="X",
        s=240,
        label="Centroids",
        edgecolors="white",
        linewidths=1.0,
    )
    for i, (cx, cy) in enumerate(centroids_pca):
        ax_pca.annotate(f"C{i}", (cx, cy), textcoords="offset points", xytext=(6, 6), fontsize=10)

    ax_pca.set_title("Data Projected to 2D PCA (Color = Cluster ID)")
    ax_pca.set_xlabel(f"Principal Component 1 ({explained[0] * 100:.1f}% var)")
    ax_pca.set_ylabel(f"Principal Component 2 ({explained[1] * 100:.1f}% var)")
    ax_pca.legend(loc="best")

    cbar = plt.colorbar(scatter, ax=ax_pca, ticks=np.arange(num_clusters), label="Cluster ID")
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(num_clusters)])
    st.pyplot(fig_pca)

    cluster_counts = np.bincount(cluster_labels, minlength=num_clusters)
    cluster_anomalies = np.bincount(cluster_labels[anomaly_mask], minlength=num_clusters)
    cluster_color_index_df = pd.DataFrame(
        {
            "Cluster ID": [f"Cluster {i}" for i in range(num_clusters)],
            "Color (hex)": [mcolors.to_hex(cluster_cmap(i)) for i in range(num_clusters)],
            "Samples": cluster_counts,
            "Predicted Attacks": cluster_anomalies,
        }
    )
    st.caption("Cluster color index for PCA plot:")
    st.dataframe(cluster_color_index_df, hide_index=True)

with tab_summary:
    cluster_rows = []
    for k in range(num_clusters):
        mask = cluster_labels == k
        dists_k = all_dists[mask, k]
        attacks_k = int(np.sum(anomaly_mask[mask]))

        if adaptive_threshold and model.cluster_thresholds_:
            threshold_k = float(model.cluster_thresholds_[k])
        else:
            threshold_k = float(model.anomaly_threshold_)

        cluster_rows.append(
            {
                "Cluster ID": f"Cluster {k}",
                "Samples": int(np.sum(mask)),
                "Mean Distance to Own Centroid": float(np.mean(dists_k)) if np.any(mask) else np.nan,
                "Std Distance": float(np.std(dists_k)) if np.any(mask) else np.nan,
                "Threshold Used": threshold_k,
                "Predicted Attacks": attacks_k,
            }
        )

    summary_df = pd.DataFrame(cluster_rows)
    st.dataframe(summary_df, hide_index=True)


# --- Inference Section ---
st.divider()
st.subheader("6. Real-time Inference Playground")
st.write(
    "Adjust feature values and run prediction. The app will show nearest cluster, "
    "distance values, and threshold used."
)

input_features = []
input_cols = st.columns(3)
for i, feature in enumerate(feature_df.columns):
    with input_cols[i % 3]:
        col_min = float(feature_df[feature].min())
        col_max = float(feature_df[feature].max())
        col_median = float(feature_df[feature].median())
        range_pad = max((col_max - col_min) * 0.3, 1e-6)
        val = st.slider(
            feature,
            min_value=float(col_min - range_pad),
            max_value=float(col_max + range_pad),
            value=col_median,
        )
        input_features.append(val)

if st.button("Predict"):
    input_arr = np.array([input_features], dtype=float)
    input_scaled = scaler.transform(input_arr)

    dists = model.transform(input_scaled)[0]
    nearest_cluster = int(np.argmin(dists))
    min_dist = float(np.min(dists))

    if adaptive_threshold and model.cluster_thresholds_:
        threshold_used = float(model.cluster_thresholds_[nearest_cluster])
    else:
        threshold_used = float(model.anomaly_threshold_)

    prediction = int(min_dist > threshold_used)
    result = "ATTACK" if prediction == 1 else "NORMAL"
    ratio = min_dist / threshold_used if threshold_used > 0 else np.inf

    if prediction == 1:
        st.error(
            f"Prediction: **{result}** | Nearest Cluster: `Cluster {nearest_cluster}` | "
            f"Distance: `{min_dist:.4f}` > Threshold: `{threshold_used:.4f}`"
        )
    else:
        st.success(
            f"Prediction: **{result}** | Nearest Cluster: `Cluster {nearest_cluster}` | "
            f"Distance: `{min_dist:.4f}` <= Threshold: `{threshold_used:.4f}`"
        )

    st.caption(
        f"Distance-to-threshold ratio: `{ratio:.3f}` "
        "(> 1.0 usually indicates anomaly)."
    )

    dist_df = pd.DataFrame(
        {
            "Cluster ID": [f"Cluster {i}" for i in range(num_clusters)],
            "Distance": dists,
            "Threshold Used for Decision": [threshold_used] * num_clusters,
        }
    ).sort_values("Distance")
    st.dataframe(dist_df, hide_index=True)

    fig_input, ax_input = plt.subplots(figsize=(8, 4))
    ax_input.bar(dist_df["Cluster ID"], dist_df["Distance"], color="#60a5fa", edgecolor="black")
    ax_input.axhline(threshold_used, color="black", linestyle="dashed", linewidth=2, label="Decision Threshold")
    ax_input.set_title("Input Sample: Distance to Each Cluster")
    ax_input.set_ylabel("Distance")
    ax_input.set_xlabel("Cluster")
    ax_input.legend()
    st.pyplot(fig_input)
