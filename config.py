# config.py

NUM_CLIENTS = 5
PCA_VARIANCE = 0.95

DISTANCE_METRIC = "cosine"  
# options: euclidean, manhattan, minkowski, cosine

ALPHA_SEARCH_SPACE = {
    "euclidean": [1.5, 2.0, 2.5, 3.0, 3.5],
    "manhattan": [1.2, 1.6, 2.0, 2.4, 2.8],
    "minkowski": [1.5, 2.0, 2.5, 3.0],
    "cosine": [1.5, 2.0, 2.5, 3.0, 3.5]
}

FEATURE_NAMES = [
    'Packet_Count', 'Byte_Count', 'Duration', 'Source_Bytes', 'Dest_Bytes', 
    'Same_Srv_Rate', 'Diff_Srv_Rate', 'SYN_Flag_Count', 'ACK_Flag_Count'
]

# Model Hyperparameters
CLUSTER_THRESHOLD = 0.3
LEARNING_RATE = 0.1
ANOMALY_BUFFER = 1.1  # Lambda for statistical threshold (mean + lambda * std)
MINKOWSKI_P = 3        # p parameter for Minkowski distance (p>=1)
ADAPTIVE_THRESHOLD = False  # True = per-cluster thresholds (Option 3), False = global (Option 2)

