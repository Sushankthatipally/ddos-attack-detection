"""
generate_dataset.py

Generates a synthetic network traffic dataset in the style of
NSL-KDD / DARPA that is compatible with the CLAPP pipeline.

The dataset has:
  - A Process–System Call matrix (each row = one network connection / process,
    each column = a feature / "system call" count)
  - A 'label' column with integer values: 0 = Normal, 1 = Attack

Attack sub-types mirror NSL-KDD categories:
  0 → Normal
  1 → DoS   (Denial of Service)
  2 → Probe
  3 → R2L   (Remote-to-Local)   ← rare, hard to detect
  4 → U2R   (User-to-Root)      ← rarest, hardest to detect

The label column is KEPT in the CSV so the CLAPP posterior-probability
step (Eq. 4) can use it during feature reduction.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions (mirrors NSL-KDD 19-feature subset + a few extras)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "duration",
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
    "protocol_type",          # 0=tcp, 1=udp, 2=icmp
    "service",                # 0-65 encoded service port
    "flag",                   # 0-10 encoded TCP flag
]

N_FEATURES = len(FEATURE_NAMES)

# Class distribution (approximate NSL-KDD proportions)
CLASS_NAMES   = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_WEIGHTS = [0.53,      0.36,   0.09,   0.016, 0.004]   # sums to 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-class feature generation (mean / std for each feature per class)
# ─────────────────────────────────────────────────────────────────────────────

# Each class has a characteristic "signature" in feature space.
# Values are (mean, std) for a Gaussian draw; clipped to ≥ 0.

_SIGNATURES: dict = {
    0: {  # Normal
        "duration":              (10,  20),
        "src_bytes":             (500, 800),
        "dst_bytes":             (300, 600),
        "land":                  (0,   0.01),
        "wrong_fragment":        (0,   0.1),
        "urgent":                (0,   0.02),
        "hot":                   (1,   2),
        "num_failed_logins":     (0,   0.3),
        "logged_in":             (0.8, 0.2),
        "num_compromised":       (0,   0.5),
        "root_shell":            (0,   0.05),
        "su_attempted":          (0,   0.05),
        "num_root":              (0,   0.2),
        "num_file_creations":    (0,   0.5),
        "num_shells":            (0,   0.1),
        "num_access_files":      (0,   0.5),
        "num_outbound_cmds":     (0,   0.1),
        "is_host_login":         (0,   0.05),
        "is_guest_login":        (0,   0.1),
        "count":                 (50,  30),
        "srv_count":             (40,  25),
        "serror_rate":           (0.02, 0.05),
        "srv_serror_rate":       (0.02, 0.05),
        "rerror_rate":           (0.01, 0.03),
        "srv_rerror_rate":       (0.01, 0.03),
        "same_srv_rate":         (0.85, 0.15),
        "diff_srv_rate":         (0.10, 0.10),
        "srv_diff_host_rate":    (0.08, 0.10),
        "dst_host_count":        (200, 60),
        "dst_host_srv_count":    (180, 55),
        "dst_host_same_srv_rate":(0.8,  0.2),
        "dst_host_diff_srv_rate":(0.08, 0.08),
        "dst_host_same_src_port_rate": (0.15, 0.15),
        "dst_host_srv_diff_host_rate": (0.05, 0.05),
        "dst_host_serror_rate":  (0.02, 0.04),
        "dst_host_srv_serror_rate":(0.02, 0.04),
        "dst_host_rerror_rate":  (0.01, 0.03),
        "dst_host_srv_rerror_rate":(0.01, 0.03),
        "protocol_type":         (0,    1),
        "service":               (20,   15),
        "flag":                  (2,    2),
    },
    1: {  # DoS — high traffic volume, high error rates
        "duration":              (0,    1),
        "src_bytes":             (5000, 3000),
        "dst_bytes":             (0,    100),
        "land":                  (0,    0.05),
        "wrong_fragment":        (1,    1),
        "urgent":                (0,    0.1),
        "hot":                   (0,    1),
        "num_failed_logins":     (0,    0.1),
        "logged_in":             (0,    0.1),
        "num_compromised":       (0,    0.2),
        "root_shell":            (0,    0.01),
        "su_attempted":          (0,    0.01),
        "num_root":              (0,    0.1),
        "num_file_creations":    (0,    0.1),
        "num_shells":            (0,    0.05),
        "num_access_files":      (0,    0.2),
        "num_outbound_cmds":     (0,    0.05),
        "is_host_login":         (0,    0.01),
        "is_guest_login":        (0,    0.05),
        "count":                 (480,  50),
        "srv_count":             (470,  55),
        "serror_rate":           (0.9,  0.1),
        "srv_serror_rate":       (0.9,  0.1),
        "rerror_rate":           (0.05, 0.05),
        "srv_rerror_rate":       (0.05, 0.05),
        "same_srv_rate":         (0.95, 0.05),
        "diff_srv_rate":         (0.05, 0.05),
        "srv_diff_host_rate":    (0.02, 0.03),
        "dst_host_count":        (255,  5),
        "dst_host_srv_count":    (250,  10),
        "dst_host_same_srv_rate":(0.95, 0.05),
        "dst_host_diff_srv_rate":(0.05, 0.05),
        "dst_host_same_src_port_rate":(0.9, 0.1),
        "dst_host_srv_diff_host_rate":(0.01,0.02),
        "dst_host_serror_rate":  (0.9,  0.1),
        "dst_host_srv_serror_rate":(0.9, 0.1),
        "dst_host_rerror_rate":  (0.05, 0.05),
        "dst_host_srv_rerror_rate":(0.05,0.05),
        "protocol_type":         (0,    0.3),
        "service":               (5,    3),
        "flag":                  (8,    2),
    },
    2: {  # Probe — port scanning, moderate traffic
        "duration":              (0,    0.5),
        "src_bytes":             (100,  200),
        "dst_bytes":             (50,   100),
        "land":                  (0,    0.01),
        "wrong_fragment":        (0,    0.2),
        "urgent":                (0,    0.05),
        "hot":                   (0,    0.5),
        "num_failed_logins":     (0,    0.2),
        "logged_in":             (0.1,  0.2),
        "num_compromised":       (0,    0.3),
        "root_shell":            (0,    0.02),
        "su_attempted":          (0,    0.02),
        "num_root":              (0,    0.1),
        "num_file_creations":    (0,    0.1),
        "num_shells":            (0,    0.05),
        "num_access_files":      (0,    0.3),
        "num_outbound_cmds":     (0,    0.05),
        "is_host_login":         (0,    0.02),
        "is_guest_login":        (0,    0.05),
        "count":                 (150,  80),
        "srv_count":             (10,   15),
        "serror_rate":           (0.3,  0.3),
        "srv_serror_rate":       (0.3,  0.3),
        "rerror_rate":           (0.3,  0.3),
        "srv_rerror_rate":       (0.3,  0.3),
        "same_srv_rate":         (0.2,  0.3),
        "diff_srv_rate":         (0.8,  0.2),
        "srv_diff_host_rate":    (0.3,  0.3),
        "dst_host_count":        (50,   60),
        "dst_host_srv_count":    (20,   30),
        "dst_host_same_srv_rate":(0.2,  0.3),
        "dst_host_diff_srv_rate":(0.6,  0.3),
        "dst_host_same_src_port_rate":(0.1,0.2),
        "dst_host_srv_diff_host_rate":(0.4,0.3),
        "dst_host_serror_rate":  (0.3,  0.3),
        "dst_host_srv_serror_rate":(0.3,0.3),
        "dst_host_rerror_rate":  (0.3,  0.3),
        "dst_host_srv_rerror_rate":(0.3,0.3),
        "protocol_type":         (1,    0.5),
        "service":               (30,   20),
        "flag":                  (1,    2),
    },
    3: {  # R2L — login attempts, low byte counts
        "duration":              (20,   30),
        "src_bytes":             (200,  300),
        "dst_bytes":             (1500, 2000),
        "land":                  (0,    0.01),
        "wrong_fragment":        (0,    0.1),
        "urgent":                (0,    0.05),
        "hot":                   (3,    4),
        "num_failed_logins":     (3,    2),
        "logged_in":             (0.3,  0.3),
        "num_compromised":       (1,    2),
        "root_shell":            (0,    0.1),
        "su_attempted":          (0,    0.1),
        "num_root":              (0,    0.5),
        "num_file_creations":    (0,    0.5),
        "num_shells":            (0,    0.2),
        "num_access_files":      (1,    1),
        "num_outbound_cmds":     (0,    0.1),
        "is_host_login":         (0,    0.05),
        "is_guest_login":        (0.3,  0.4),
        "count":                 (5,    8),
        "srv_count":             (4,    7),
        "serror_rate":           (0.01, 0.05),
        "srv_serror_rate":       (0.01, 0.05),
        "rerror_rate":           (0.2,  0.3),
        "srv_rerror_rate":       (0.2,  0.3),
        "same_srv_rate":         (0.4,  0.4),
        "diff_srv_rate":         (0.4,  0.4),
        "srv_diff_host_rate":    (0.1,  0.2),
        "dst_host_count":        (20,   30),
        "dst_host_srv_count":    (15,   20),
        "dst_host_same_srv_rate":(0.3,  0.3),
        "dst_host_diff_srv_rate":(0.3,  0.3),
        "dst_host_same_src_port_rate":(0.05,0.1),
        "dst_host_srv_diff_host_rate":(0.1,0.2),
        "dst_host_serror_rate":  (0.01, 0.05),
        "dst_host_srv_serror_rate":(0.01,0.05),
        "dst_host_rerror_rate":  (0.15, 0.2),
        "dst_host_srv_rerror_rate":(0.15,0.2),
        "protocol_type":         (0,    0.3),
        "service":               (40,   10),
        "flag":                  (2,    2),
    },
    4: {  # U2R — privilege escalation; subtle differences from normal
        "duration":              (15,   20),
        "src_bytes":             (400,  500),
        "dst_bytes":             (2000, 3000),
        "land":                  (0,    0.01),
        "wrong_fragment":        (0,    0.1),
        "urgent":                (0,    0.05),
        "hot":                   (8,    5),
        "num_failed_logins":     (0,    0.5),
        "logged_in":             (0.9,  0.1),
        "num_compromised":       (5,    8),
        "root_shell":            (0.8,  0.3),
        "su_attempted":          (0.5,  0.5),
        "num_root":              (3,    4),
        "num_file_creations":    (2,    3),
        "num_shells":            (1,    1),
        "num_access_files":      (2,    2),
        "num_outbound_cmds":     (0,    0.1),
        "is_host_login":         (0,    0.1),
        "is_guest_login":        (0,    0.1),
        "count":                 (8,    10),
        "srv_count":             (6,    8),
        "serror_rate":           (0.01, 0.05),
        "srv_serror_rate":       (0.01, 0.05),
        "rerror_rate":           (0.01, 0.05),
        "srv_rerror_rate":       (0.01, 0.05),
        "same_srv_rate":         (0.7,  0.3),
        "diff_srv_rate":         (0.2,  0.2),
        "srv_diff_host_rate":    (0.1,  0.15),
        "dst_host_count":        (30,   30),
        "dst_host_srv_count":    (20,   20),
        "dst_host_same_srv_rate":(0.6,  0.3),
        "dst_host_diff_srv_rate":(0.2,  0.2),
        "dst_host_same_src_port_rate":(0.1, 0.15),
        "dst_host_srv_diff_host_rate":(0.1,0.15),
        "dst_host_serror_rate":  (0.01, 0.05),
        "dst_host_srv_serror_rate":(0.01,0.05),
        "dst_host_rerror_rate":  (0.01, 0.05),
        "dst_host_srv_rerror_rate":(0.01,0.05),
        "protocol_type":         (0,    0.2),
        "service":               (22,   5),
        "flag":                  (2,    1),
    },
}


def _generate_class_rows(
    class_id: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate `n` rows for a given class using its Gaussian signature."""
    sig = _SIGNATURES[class_id]
    rows = np.zeros((n, N_FEATURES))
    for j, feat in enumerate(FEATURE_NAMES):
        mean, std = sig[feat]
        vals = rng.normal(mean, std, n)
        # Clip to [0, ∞) and round to 2 dp
        rows[:, j] = np.clip(vals, 0, None).round(2)
    return rows


def generate_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
    output_path: Optional[str] = None,
    binary_label: bool = False,
) -> pd.DataFrame:
    """
    Generate the synthetic dataset.

    Parameters
    ----------
    n_samples    : Total number of samples.
    random_state : Random seed for reproducibility.
    output_path  : If given, save CSV to this path.
    binary_label : If True, collapse all attack classes to label=1 (Normal=0).
                   If False, keep multi-class labels 0-4.

    Returns
    -------
    pd.DataFrame with columns = FEATURE_NAMES + ['label', 'label_name']
    """
    rng = np.random.default_rng(random_state)

    # Compute class counts from weights
    counts = (np.array(CLASS_WEIGHTS) * n_samples).astype(int)
    # Fix rounding so total == n_samples
    counts[-1] += n_samples - counts.sum()

    all_rows = []
    all_labels = []
    all_names = []

    for class_id, (name, count) in enumerate(zip(CLASS_NAMES, counts)):
        rows = _generate_class_rows(class_id, count, rng)
        all_rows.append(rows)
        all_labels.extend([class_id] * count)
        all_names.extend([name] * count)

    data = np.vstack(all_rows)

    # Shuffle
    idx = rng.permutation(len(data))
    data = data[idx]
    all_labels = np.array(all_labels)[idx]
    all_names  = np.array(all_names)[idx]

    df = pd.DataFrame(data, columns=FEATURE_NAMES)

    if binary_label:
        # 0 = Normal, 1 = Attack (any type)
        df["label"] = (all_labels > 0).astype(int)
        df["label_name"] = df["label"].map({0: "Normal", 1: "Attack"})
    else:
        df["label"] = all_labels
        df["label_name"] = all_names

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[generate_dataset] Saved {len(df)} rows → {output_path}")
        print(df["label_name"].value_counts().to_string())

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLAPP-compatible synthetic dataset")
    parser.add_argument("--n_samples",    type=int,   default=5000,            help="Total samples")
    parser.add_argument("--output",       type=str,   default="data/dataset.csv", help="Output CSV path")
    parser.add_argument("--random_state", type=int,   default=42)
    parser.add_argument("--binary",       action="store_true",                 help="Binary label (Normal vs Attack)")
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        random_state=args.random_state,
        output_path=args.output,
        binary_label=args.binary,
    )
