import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import sys

# Ensure the current directory is in the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.clustering import SelfConstructingClustering
from config import FEATURE_NAMES

st.set_page_config(
    page_title="Federated DDoS IDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")

# 1. Distance Metric Selection
distance_metric = st.sidebar.selectbox(
    "Select Distance Metric",
    ["euclidean", "manhattan", "minkowski", "cosine"],
    index=0,
    help="Choose the distance metric for clustering."
)

# 2. Hyperparameters
st.sidebar.subheader("Model Hyperparameters")
threshold = st.sidebar.slider(
    "Cluster Threshold",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Distance threshold for creating a new cluster."
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="Rate at which centroids update."
)

# --- Main Content ---
st.title("🛡️ Federated DDoS IDS Dashboard")
st.markdown(f"**Current Metric:** `{distance_metric}` | **Threshold:** `{threshold}` | **Learning Rate:** `{learning_rate}`")

# --- Data Loading ---
@st.cache_data
def load_data():
    # Load dummy data
    data_path = os.path.join(os.path.dirname(__file__), "data", "dummy_data.csv")
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}. Please ensure 'data/dummy_data.csv' exists.")
        return None
    return pd.read_csv(data_path)

df = load_data()

if df is not None:
    # Display Data Snippet
    with st.expander("View Raw Data Snippet"):
        st.dataframe(df.head())
    
    # Preprocessing (Standardization is crucial for distance metrics)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES].values)
    
    # --- Model Training ---
    st.subheader("1. Model Training & Clustering")
    
    # Initialize and Train Model
    model = SelfConstructingClustering(
        threshold=threshold,
        distance_metric=distance_metric,
        learning_rate=learning_rate
    )
    
    with st.spinner(f"Training model with {distance_metric} metric..."):
        model.fit(X_scaled)
    
    # Display Clustering Results
    # Use standard sklearn attribute syntax (ending with underscore)
    num_clusters = len(model.centroids_)
    st.success(f"Training Complete! Formed **{num_clusters} clusters**.")
    st.info(f"Anomaly Threshold (calculated): **{model.anomaly_threshold_:.4f}**")
    
    # --- Visualizations ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2. Cluster Centroids")
        if num_clusters > 0:
            centroids_df = pd.DataFrame(model.centroids_, columns=FEATURE_NAMES)
            centroids_df.index = [f"Cluster {i}" for i in range(num_clusters)]

            st.caption("Index: each row is a centroid (`Cluster 0`, `Cluster 1`, ...).")
            st.dataframe(centroids_df)

            # Heatmap with explicit colorbar so color meaning is clear
            fig_centroid, ax_centroid = plt.subplots(
                figsize=(max(8, len(FEATURE_NAMES) * 0.9), max(2.8, num_clusters * 0.6))
            )
            im = ax_centroid.imshow(centroids_df.values, cmap="viridis", aspect="auto")
            ax_centroid.set_title("Centroid Values Heatmap")
            ax_centroid.set_xticks(np.arange(len(FEATURE_NAMES)))
            ax_centroid.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
            ax_centroid.set_yticks(np.arange(num_clusters))
            ax_centroid.set_yticklabels(centroids_df.index.tolist())
            cbar = fig_centroid.colorbar(im, ax=ax_centroid)
            cbar.set_label("Standardized Feature Value (z-score)")
            st.pyplot(fig_centroid)

            st.caption(
                "Color index for centroids: purple/blue = lower value, green = medium, "
                "yellow = higher value (based on z-score)."
            )

            cluster_sizes_df = pd.DataFrame(
                {
                    "Cluster ID": [f"Cluster {i}" for i in range(num_clusters)],
                    "Samples Assigned": model.cluster_counts_,
                }
            )
            st.dataframe(cluster_sizes_df, hide_index=True)
        else:
            st.warning("No clusters formed.")

    with col2:
        st.subheader("3. Distance Distribution")
        # Calculate distances of all points to their nearest centroid
        if num_clusters > 0:
            # Transform returns (n_samples, n_clusters) distances
            all_dists = model.transform(X_scaled)
            min_distances = np.min(all_dists, axis=1)
            
            hist_color = "skyblue"
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(min_distances, bins=20, color=hist_color, edgecolor='black')
            ax_hist.axvline(
                model.anomaly_threshold_,
                color='red',
                linestyle='dashed',
                linewidth=2
            )
            ax_hist.set_title("Distance to Nearest Cluster")
            ax_hist.set_xlabel("Distance")
            ax_hist.set_ylabel("Frequency")
            legend_handles = [
                Patch(facecolor=hist_color, edgecolor="black", label="Samples per distance bin"),
                Line2D(
                    [0], [0],
                    color="red",
                    linestyle="dashed",
                    linewidth=2,
                    label=f"Anomaly Threshold = {model.anomaly_threshold_:.4f}"
                ),
            ]
            ax_hist.legend(handles=legend_handles)
            st.pyplot(fig_hist)
            st.caption(
                "Index: blue bars show number of samples in each distance range; "
                "red dashed line is the anomaly cutoff used for prediction."
            )
        else:
            st.warning("No clusters formed.")

    st.subheader("4. 2D Projection (PCA)")
    if num_clusters > 0:
        # Assign each point to a cluster for coloring
        all_dists = model.transform(X_scaled)
        cluster_labels = np.argmin(all_dists, axis=1)
            
        # PCA Projection
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot with explicit cluster color index
        cluster_cmap = plt.get_cmap("tab10", num_clusters)
        fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
        scatter = ax_pca.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=cluster_labels,
            cmap=cluster_cmap,
            vmin=-0.5,
            vmax=num_clusters - 0.5,
            alpha=0.7
        )
        
        # Plot Centroids on PCA (Need to transform them too)
        centroids_pca = pca.transform(np.array(model.centroids_))
        ax_pca.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroids')
        
        ax_pca.set_title(f"Data Projected on 2D (PCA) - Colored by Cluster")
        ax_pca.set_xlabel("Principal Component 1")
        ax_pca.set_ylabel("Principal Component 2")
        ax_pca.legend()
        cbar = plt.colorbar(scatter, ax=ax_pca, ticks=np.arange(num_clusters), label="Cluster ID")
        cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(num_clusters)])
        st.pyplot(fig_pca)

        cluster_counts = np.bincount(cluster_labels, minlength=num_clusters)
        cluster_color_index_df = pd.DataFrame(
            {
                "Cluster ID": [f"Cluster {i}" for i in range(num_clusters)],
                "Color (hex)": [mcolors.to_hex(cluster_cmap(i)) for i in range(num_clusters)],
                "Points in PCA Plot": cluster_counts
            }
        )
        st.caption("Cluster color index for PCA plot:")
        st.dataframe(cluster_color_index_df, hide_index=True)

    # --- Inference Section ---
    st.divider()
    st.subheader("5. Real-time Inference Playground")
    st.write("Adjust the slider to simulate a network packet and see if it's classified as Normal or Attack.")
    
    # Create sliders for a single input
    input_features = []
    cols = st.columns(3)
    for i, feature in enumerate(FEATURE_NAMES):
        with cols[i % 3]:
            # Use mean/std from data to set reasonable ranges
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            
            # Allow sliders to go a bit beyond observed range to simulate anomalies
            val = st.slider(f"{feature}", min_value=0.0, max_value=max_val * 2, value=mean_val)
            input_features.append(val)
    
    if st.button("Predict"):
        if num_clusters > 0:
            # Scale the input
            input_arr = np.array([input_features])
            input_scaled = scaler.transform(input_arr)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            result = "ATTACK" if prediction == 1 else "NORMAL"
            
            # Get distance for visualization
            dists = model.transform(input_scaled)[0]
            min_dist = np.min(dists)
            
            if result == "ATTACK":
                st.error(f"Prediction: **{result}** (Distance: {min_dist:.4f} > Threshold: {model.anomaly_threshold_:.4f})")
            else:
                st.success(f"Prediction: **{result}** (Distance: {min_dist:.4f} <= Threshold: {model.anomaly_threshold_:.4f})")
        else:
            st.error("Model not trained (no clusters).")

else:
    st.warning("Please ensure data is available.")
