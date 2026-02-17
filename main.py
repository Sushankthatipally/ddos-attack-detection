import pandas as pd
import numpy as np
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FEATURE_NAMES, CLUSTER_THRESHOLD, LEARNING_RATE, ANOMALY_BUFFER, DISTANCE_METRIC, MINKOWSKI_P, ADAPTIVE_THRESHOLD
from client.preprocess import preprocess_data
from client.clustering import SelfConstructingClustering
from sklearn.metrics import accuracy_score, classification_report

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
    print(f"Initializing SelfConstructingClustering (metric={DISTANCE_METRIC}, threshold={CLUSTER_THRESHOLD})...")
    model = SelfConstructingClustering(
        threshold=CLUSTER_THRESHOLD,
        learning_rate=LEARNING_RATE,
        distance_metric=DISTANCE_METRIC,
        anomaly_buffer=ANOMALY_BUFFER,
        minkowski_p=MINKOWSKI_P,
        adaptive_threshold=ADAPTIVE_THRESHOLD
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
    
    # Normal-like sample (taken from mean of training data)
    normal_sample = np.mean(X_train_scaled, axis=0).reshape(1, -1)
    pred_normal = model.predict(normal_sample)[0]
    print(f"Scenario 1 (Average Traffic): {'Attack' if pred_normal else 'Normal'}")

    # Attack-like sample (very high values)
    # create a sample that is 10 standard deviations away
    attack_sample = normal_sample + 10.0 
    pred_attack = model.predict(attack_sample)[0]
    print(f"Scenario 2 (Extreme Traffic): {'Attack' if pred_attack else 'Normal'}")

if __name__ == "__main__":
    train_and_evaluate()
