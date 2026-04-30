"""
main.py

CLAPP Anomaly Detection Pipeline
─────────────────────────────────
Full end-to-end pipeline:

  1. Load / generate dataset
  2. For each metric in config.METRICS:
       a. Preprocess (raw PS matrix for fuzzy_gaussian, scaled for others)
       b. Fit CLAPPClustering → build pattern vectors → reduce dimensions
       c. Run kNN (k=1,3,5) on the reduced [P×G] matrix
       d. Print confusion matrix + classification report
  3. Print a final benchmark comparison table

Run:
    python main.py
    python main.py --n_samples 2000 --threshold 0.85 --sigma 0.5
"""

from __future__ import annotations

import argparse
import os
import textwrap
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler

import config
from client.preprocess import preprocess_data, train_test_split_stratified
from client.clustering import CLAPPClustering, SUPPORTED_METRICS
from generate_dataset import generate_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEP = "─" * 72


def _banner(title: str):
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def _section(title: str):
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _print_confusion_matrix(cm: np.ndarray, class_names: list):
    """Pretty-print a confusion matrix with row/col labels."""
    col_w = max(max(len(n) for n in class_names), 6)
    header = " " * (col_w + 2) + "  ".join(f"{n:>{col_w}}" for n in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = f"{row_name:>{col_w}}  " + "  ".join(f"{v:>{col_w}}" for v in cm[i])
        print(row_str)


# ─────────────────────────────────────────────────────────────────────────────
# Single metric evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_metric(
    metric_cfg: dict,
    PS_raw: np.ndarray,
    PS_scaled: np.ndarray,
    labels: np.ndarray,
    k_values: list,
    binary: bool,
    anomaly_buffer: float,
    adaptive_threshold: bool,
) -> dict:
    """
    Fit CLAPP with a single metric, then evaluate kNN for each k.
    Returns a result dict for the benchmark table.
    """
    name         = metric_cfg["name"]
    metric       = metric_cfg["similarity_metric"]
    threshold    = metric_cfg.get("threshold",   config.CLAPP_THRESHOLD)
    sigma        = metric_cfg.get("sigma",        config.SIGMA_C)
    p            = metric_cfg.get("minkowski_p",  config.MINKOWSKI_P)
    scale_input  = metric_cfg.get("scale_input",  True)

    PS = PS_scaled if scale_input else PS_raw

    _section(f"Metric: {name}")

    # ── Fit CLAPP ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    clapp = CLAPPClustering(
        threshold=threshold,
        sigma=sigma,
        similarity_metric=metric,
        minkowski_p=p,
        anomaly_buffer=anomaly_buffer,
        adaptive_threshold=adaptive_threshold,
    )
    clapp.fit(PS, labels)
    fit_time = time.perf_counter() - t0

    info = clapp.get_cluster_info()
    orig_dim = PS.shape[1]
    reduced_dim = info["reduced_dims"] or orig_dim

    print(f"  Clusters formed      : {info['n_clusters']}")
    print(f"  Original dims        : {orig_dim}")
    print(f"  Reduced dims         : {reduced_dim}  "
          f"(reduction: {(1 - reduced_dim/orig_dim)*100:.1f}%)")
    print(f"  Anomaly threshold    : {info['anomaly_threshold']:.4f}")
    print(f"  Fit time             : {fit_time:.3f}s")

    # ── Get reduced [P×G] matrix ─────────────────────────────────────
    X_reduced = clapp.transform(PS)

    # ── Train/test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X_reduced, labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    class_names = (
        ["Normal", "Attack"] if binary
        else ["Normal", "DoS", "Probe", "R2L", "U2R"]
    )

    # ── Also evaluate built-in predict() for comparison ──────────────
    results = {
        "metric": name,
        "n_clusters": info["n_clusters"],
        "orig_dim": orig_dim,
        "reduced_dim": reduced_dim,
        "knn_results": {},
    }

    # ── kNN for k = 1, 3, 5 ─────────────────────────────────────────
    for k in k_values:
        print(f"\n  ── kNN  k = {k} {'─'*50}")
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

        print(f"\n  Confusion Matrix:")
        _print_confusion_matrix(cm, class_names)

        print(f"\n  Classification Report:")
        report = classification_report(
            y_test, y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
        # Indent the report
        for line in report.splitlines():
            print("    " + line)

        results["knn_results"][k] = {
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark comparison table
# ─────────────────────────────────────────────────────────────────────────────

def _print_benchmark(all_results: list, k_values: list):
    _banner("BENCHMARK COMPARISON — All Metrics")
    col_w = 30

    # Header
    header_parts = [f"{'Metric':<{col_w}}", f"{'Clusters':>10}", f"{'Dims':>10}"]
    for k in k_values:
        header_parts.append(f"{'kNN-'+str(k)+' Acc%':>14}")
    print("  " + "  ".join(header_parts))
    print("  " + "─" * (col_w + 10 + 10 + 14 * len(k_values) + 4 * (2 + len(k_values))))

    for r in all_results:
        row_parts = [
            f"{r['metric']:<{col_w}}",
            f"{r['n_clusters']:>10}",
            f"{r['orig_dim']:>5} → {r['reduced_dim']:>3}",
        ]
        for k in k_values:
            acc = r["knn_results"].get(k, {}).get("accuracy", 0.0)
            row_parts.append(f"{acc:>13.2f}%")
        print("  " + "  ".join(row_parts))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    _banner("CLAPP — Anomaly Detection Pipeline")

    # ── 1. Load or generate dataset ───────────────────────────────────
    if os.path.exists(args.dataset):
        print(f"\n[data] Loading existing dataset: {args.dataset}")
        df = pd.read_csv(args.dataset)
    else:
        print(f"\n[data] Generating synthetic dataset ({args.n_samples} samples) …")
        df = generate_dataset(
            n_samples=args.n_samples,
            random_state=args.random_state,
            output_path=args.dataset,
            binary_label=args.binary,
        )

    print(f"[data] Shape: {df.shape}")
    print(f"[data] Class distribution:\n{df['label'].value_counts().sort_index().to_string()}")

    # ── 2. Preprocess — raw (for fuzzy_gaussian) ──────────────────────
    PS_raw, labels, feature_names = preprocess_data(
        df, label_col=config.LABEL_COL, scale=False
    )
    print(f"\n[preprocess] PS matrix shape : {PS_raw.shape}")
    print(f"[preprocess] n_classes        : {len(np.unique(labels))}")

    # Scaled version for distance-based metrics
    scaler   = StandardScaler()
    PS_scaled = scaler.fit_transform(PS_raw)

    # ── 3. Run each metric ────────────────────────────────────────────
    all_results = []
    for metric_cfg in config.METRICS:
        # Allow CLI override of threshold / sigma
        metric_cfg = dict(metric_cfg)   # copy so we don't mutate config
        if args.threshold is not None:
            metric_cfg["threshold"] = args.threshold
        if args.sigma is not None:
            metric_cfg["sigma"] = args.sigma

        try:
            result = run_metric(
                metric_cfg   = metric_cfg,
                PS_raw       = PS_raw,
                PS_scaled    = PS_scaled,
                labels       = labels,
                k_values     = config.KNN_K_VALUES,
                binary       = args.binary,
                anomaly_buffer       = config.ANOMALY_BUFFER,
                adaptive_threshold   = config.ADAPTIVE_THRESHOLD,
            )
            all_results.append(result)
        except Exception as exc:
            print(f"\n  [ERROR] Metric '{metric_cfg['name']}' failed: {exc}")

    # ── 4. Benchmark table ────────────────────────────────────────────
    if all_results:
        _print_benchmark(all_results, config.KNN_K_VALUES)

    print(f"\n{'═'*72}")
    print("  Done.")
    print(f"{'═'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLAPP Anomaly Detection — all similarity metrics",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",    type=str,   default=config.DATASET_PATH,
        help="Path to CSV dataset (generated if missing)"
    )
    parser.add_argument(
        "--n_samples",  type=int,   default=config.N_SAMPLES,
        help="Samples to generate if dataset is missing"
    )
    parser.add_argument(
        "--threshold",  type=float, default=None,
        help=f"Override CLAPP similarity threshold (default: {config.CLAPP_THRESHOLD})"
    )
    parser.add_argument(
        "--sigma",      type=float, default=None,
        help=f"Override σ_c (default: {config.SIGMA_C})"
    )
    parser.add_argument(
        "--random_state", type=int, default=config.RANDOM_STATE
    )
    parser.add_argument(
        "--binary",     action="store_true", default=config.BINARY_LABEL,
        help="Binary classification: Normal vs Attack"
    )

    main(parser.parse_args())
