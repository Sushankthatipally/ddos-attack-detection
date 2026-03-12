import pandas as pd
import numpy as np
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    FEATURE_NAMES,
    CLUSTER_THRESHOLD,
    LEARNING_RATE,
    ANOMALY_BUFFER,
    DISTANCE_METRIC,
    MINKOWSKI_P,
    ADAPTIVE_THRESHOLD,
)
from client.preprocess import preprocess_data
from client.clustering import SelfConstructingClustering


def load_data(filepath):
    """Loads CSV data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return pd.read_csv(filepath)

def train_and_evaluate():
    print("--- Federated DDoS IDS Training Pipeline ---")

    # 1. Load Data
    data_path = os.path.join("data", "dummy_data.csv")
    print(f"Loading data from {data_path}...")
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Preprocess
    print("Preprocessing data...")
    # Note: preprocess_data returns scaled numpy array
    # It drops non-numeric columns and handles NaNs
    try:
        X_train_scaled = preprocess_data(df)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return

    print(f"Data Shape after preprocessing: {X_train_scaled.shape}")

    # 3. Initialize Model
    print(
        "Initializing SelfConstructingClustering "
        f"(metric={DISTANCE_METRIC}, threshold={CLUSTER_THRESHOLD})..."
    )
    model = SelfConstructingClustering(
        threshold=CLUSTER_THRESHOLD,
        learning_rate=LEARNING_RATE,
        distance_metric=DISTANCE_METRIC,
        anomaly_buffer=ANOMALY_BUFFER,
        minkowski_p=MINKOWSKI_P,
        adaptive_threshold=ADAPTIVE_THRESHOLD,
    )

    # 4. Train
    print("Training model...")
    model.fit(X_train_scaled)
    print(f"Training Complete. Clusters formed: {len(model.centroids_)}")
    print(f"Anomaly Threshold: {model.anomaly_threshold_:.4f}")

    # 5. Inference / Evaluation (Self-Test on Training Data for now)
    # Ideally we'd split train/test but this is a demo script
    print("\n--- Evaluation on Training Set (Sanity Check) ---")
    predictions = model.predict(X_train_scaled)
    n_anomalies = np.sum(predictions)
    print(f"Predicted Anomalies in Training Set: {n_anomalies} / {len(predictions)}")

    # 6. Test on specific scenarios
    print("\n--- Manual Scenarios ---")

    def score_sample(sample):
        """Return prediction and distance diagnostics for one sample."""
        dists = model.transform(sample)[0]
        nearest_cluster = int(np.argmin(dists))
        min_dist = float(np.min(dists))
        if model.adaptive_threshold and model.cluster_thresholds_:
            threshold_used = float(model.cluster_thresholds_[nearest_cluster])
        else:
            threshold_used = float(model.anomaly_threshold_)
        pred = int(min_dist > threshold_used)
        return pred, min_dist, threshold_used, nearest_cluster

    # Typical seen sample: choose the closest training point to any centroid.
    train_dists = model.transform(X_train_scaled)
    train_min_dists = np.min(train_dists, axis=1)
    typical_idx = int(np.argmin(train_min_dists))
    normal_sample = X_train_scaled[typical_idx].reshape(1, -1)
    pred_normal, dist_normal, thr_normal, cluster_normal = score_sample(normal_sample)
    print(
        "Scenario 1 (Typical Seen Traffic): "
        f"{'Attack' if pred_normal else 'Normal'} | "
        f"cluster={cluster_normal}, distance={dist_normal:.4f}, threshold={thr_normal:.4f}"
    )

    # Stress sample: evaluate multiple synthetic candidates and keep the farthest one.
    attack_candidates = [
        -normal_sample,
        normal_sample + 8.0,
        normal_sample - 8.0,
    ]
    single_feature_spike = normal_sample.copy()
    single_feature_spike[0, 0] += 12.0
    attack_candidates.append(single_feature_spike)

    candidate_scores = []
    for candidate in attack_candidates:
        _, min_dist, threshold_used, _ = score_sample(candidate)
        ratio = min_dist / threshold_used if threshold_used > 0 else np.inf
        candidate_scores.append((candidate, min_dist, threshold_used, ratio))

    attack_sample, attack_dist, attack_thr, attack_ratio = max(
        candidate_scores, key=lambda item: item[3]
    )
    pred_attack, _, _, cluster_attack = score_sample(attack_sample)
    print(
        "Scenario 2 (Synthetic Stress Traffic): "
        f"{'Attack' if pred_attack else 'Normal'} | "
        f"cluster={cluster_attack}, distance={attack_dist:.4f}, "
        f"threshold={attack_thr:.4f}, ratio={attack_ratio:.3f}"
    )

if __name__ == "__main__":
    train_and_evaluate()
