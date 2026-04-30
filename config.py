"""
Configuration for the CLAPP paper-reproduction pipeline.

Defaults are set for the real NSL-KDD dataset:
    data/NSL-KDD-full.csv
"""

# Dataset paths
DATASET_PATH = "data/NSL-KDD-full.csv"
LABEL_COL = "label"
N_SAMPLES = 5000
RANDOM_STATE = 42
BINARY_LABEL = False

# Compatibility values used by the existing Streamlit app.
NUM_CLIENTS = 5
PCA_VARIANCE = 0.95

FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Paper settings
CLAPP_THRESHOLD = 0.9999
SIGMA_C = 0.5
MINKOWSKI_P = 3
ANOMALY_BUFFER = 1.5
ADAPTIVE_THRESHOLD = False

# Run only the paper metric for reproduction.
METRICS = [
    {
        "name": "Fuzzy Gaussian (Paper)",
        "similarity_metric": "fuzzy_gaussian",
        "threshold": CLAPP_THRESHOLD,
        "sigma": SIGMA_C,
        "scale_input": False,
    },
]

KNN_K_VALUES = [1, 3, 5]
TEST_SIZE = 0.30
