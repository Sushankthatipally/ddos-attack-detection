# Federated DDoS IDS - Complete Project Documentation (Q&A)

This document explains the project in detail, using simple Q&A format so you can use it for learning, demos, viva, and reports.

## 1) Project Basics

### Q1. What is this project?

This is a prototype Intrusion Detection System (IDS) focused on DDoS-like traffic patterns. It uses clustering-based anomaly detection to separate normal-looking traffic from suspicious traffic.

### Q2. What does the project name mean?

- Federated: A machine learning approach where clients train locally and share model updates, not raw data.
- DDoS: Distributed Denial of Service, where many sources flood a target.
- IDS: Intrusion Detection System, which monitors behavior and flags attacks.

### Q3. Is this project fully federated right now?

Not fully. The repository includes federated pieces (`server/aggregate.py`, `NUM_CLIENTS` in config), but the main working pipeline (`main.py`, `app.py`) currently runs on one dataset locally. So it is federated-inspired, not a complete distributed federated deployment.

### Q4. Why did we build this project?

- To detect abnormal network behavior without needing strict attack signatures.
- To use an unsupervised-style approach that can work even when labels are limited.
- To provide an interactive dashboard for understanding how anomaly thresholds behave.

## 2) Core Security and ML Terms

### Q5. What is DDoS in simple words?

DDoS is when many machines send huge traffic to a victim server so real users cannot access it.

### Q6. What is an IDS in simple words?

An IDS is a monitoring system that watches traffic and raises alerts if activity looks malicious.

### Q7. What is anomaly detection?

Anomaly detection means learning what "normal" looks like and flagging points that are too far from normal.

### Q8. What is clustering?

Clustering groups similar data points together. Here, each group has a centroid (center point).

### Q9. What is a centroid?

A centroid is the center representation of one cluster.

### Q10. What is federated learning in simple words?

Federated learning trains models on multiple clients locally. Clients keep raw data private and only send updates (like weights/centroids) to a server, which aggregates them.

## 3) What Technology Is Used and Why

### Q11. Which technologies are used?

- Python: core language.
- NumPy: fast vector/matrix math.
- Pandas: CSV loading and dataframe processing.
- SciPy: distance calculations (`cdist`) for clustering.
- scikit-learn: scaler, PCA, silhouette score, estimator-style model structure.
- Matplotlib: charts in dashboard.
- Streamlit: interactive UI dashboard.
- Pytest: automated tests.

### Q12. Where are dependencies listed?

`requirements.txt`

## 4) Repository Structure and Purpose

### Q13. What does each main file/folder do?

- `app.py`: Streamlit dashboard (main UI).
- `main.py`: CLI training + simple scenario checks.
- `config.py`: all key hyperparameters and feature names.
- `client/clustering.py`: Self-Constructing Clustering model.
- `client/preprocess.py`: preprocessing helper for pipeline.
- `server/aggregate.py`: federated centroid averaging helper.
- `generate_dataset.py`: synthetic data generator.
- `evaluation/`: helper metrics/plots.
- `tests/`: unit and integration tests.
- `data/dummy_data.csv`: primary data file used by app/pipeline.

## 5) Data Details

### Q14. What data is currently used by default?

The app reads `data/dummy_data.csv`.

Current file profile:

- Rows: 40
- Feature columns: 9
- Data type: all numeric (`float64`)

### Q15. Which features are used?

From `config.py`:

- `Packet_Count`
- `Byte_Count`
- `Duration`
- `Source_Bytes`
- `Dest_Bytes`
- `Same_Srv_Rate`
- `Diff_Srv_Rate`
- `SYN_Flag_Count`
- `ACK_Flag_Count`

### Q16. What does each feature mean?

- `Packet_Count`: number of packets in a flow/window.
- `Byte_Count`: total bytes transferred.
- `Duration`: connection/session duration.
- `Source_Bytes`: bytes sent by source side.
- `Dest_Bytes`: bytes sent by destination side.
- `Same_Srv_Rate`: fraction of recent connections to same service.
- `Diff_Srv_Rate`: fraction of recent connections to different services.
- `SYN_Flag_Count`: count of SYN TCP flags (often spikes in SYN floods).
- `ACK_Flag_Count`: count of ACK TCP flags.

### Q17. How is data generated?

`generate_dataset.py` creates synthetic normal and attack samples using random ranges:

- Normal: lower packet rates, longer durations, balanced behavior.
- Attack: much higher packets/bytes, shorter durations, high SYN patterns.

It writes:

- `data/dummy_data.csv` (primary)
- `data/dummy_dataset.csv` (legacy copy)

### Q18. Is there a label column?

- Existing `data/dummy_data.csv` in the repo currently has only feature columns.
- Generated synthetic files include a `Label` column (`Normal` or `Attack`), but the anomaly model itself works without labels.

## 6) Preprocessing Pipeline

### Q19. How is data preprocessed in the dashboard?

In `app.py`:

1. Use only configured feature names.
2. Convert to numeric.
3. Replace inf with NaN.
4. Fill missing values with median.
5. Drop constant columns.
6. Standardize features with `StandardScaler`.

### Q20. Why standardization is used?

Features have different scales. Standardization ensures one large-scale feature does not dominate distance computations.

## 7) Model Details (Self-Constructing Clustering)

### Q21. What model is used?

`SelfConstructingClustering` in `client/clustering.py`.

### Q22. How does it train?

Online clustering logic:

1. Start first centroid from first sample.
2. For each new sample, compute distance to all centroids.
3. If nearest distance <= cluster threshold, assign to that cluster and update centroid by exact running mean.
4. Else create a new cluster.

### Q23. How is centroid updated?

Running mean formula:
`new_centroid = (n_k * old_centroid + x) / (n_k + 1)`

### Q24. How is anomaly threshold computed?

Global threshold (default):
`threshold = mean(min_distances) + anomaly_buffer * std(min_distances)`

Adaptive threshold mode:
For each cluster `k`:
`threshold_k = mean(distances_to_cluster_k) + anomaly_buffer * std(distances_to_cluster_k)`

### Q25. How prediction works?

For each sample:

1. Compute distances to centroids.
2. Take nearest cluster distance.
3. If distance > threshold (global or per-cluster), predict attack (`1`), else normal (`0`).

## 8) Dashboard Settings and Term-by-Term Explanation

You asked about terms such as:
`Federated Node · Active | Distance Metric | Cluster Threshold | Anomaly Buffer (lambda) | Learning Rate | Model Active | Samples | Features | Clusters | Anomaly Threshold | Anomaly Rate | Silhouette Score | PCA | Cluster Summary`

Current repository defaults from `config.py`:

- `distance_metric = cosine`
- `cluster_threshold = 0.3`
- `anomaly_buffer = 1.1`
- `learning_rate = 0.1`
- `adaptive_threshold = False`
- `minkowski_p = 3` only if metric is `minkowski`

With the current `data/dummy_data.csv` and default settings, the dashboard typically shows:

- `40` samples
- `9` usable features
- `2` clusters
- global anomaly threshold about `0.0080`
- anomaly rate about `12.5%`
- silhouette score about `0.9942`

Training latency depends on your machine, so that number can change from run to run.

### Q26. What do the sidebar controls and top badges mean?

| Term | Definition | How it is used in this project | How the value affects results |
|---|---|---|---|
| `Federated Node · Active` | A dashboard status label shown in the sidebar. | It tells the user the dashboard session is active. In this repository, `app.py` still trains locally on one dataset, so this should be read as a UI state badge, not proof of a live multi-client federated round. | No direct algorithmic effect. |
| `Distance Metric` | The distance rule used to compare each sample to each centroid. Options are `euclidean`, `manhattan`, `minkowski`, and `cosine`. | Used inside `SelfConstructingClustering.fit()`, `predict()`, `transform()`, and also in the silhouette score calculation. | Changing it changes the meaning and scale of every distance, which can change cluster count, anomaly threshold, anomaly rate, and silhouette score. Threshold values are not directly comparable across different metrics. |
| `Cluster Threshold` | The cluster formation boundary `theta`. If the nearest centroid distance is below this value, the sample joins that cluster; otherwise a new cluster is created. | Used during training only, when SCC decides whether to update an existing centroid or create a new one. | Increase it -> fewer clusters and broader clusters. Decrease it -> more clusters and tighter clusters. This indirectly changes centroids, distance distribution, anomaly threshold, and anomaly rate. |
| `Anomaly Buffer (lambda)` | The multiplier used in the statistical anomaly threshold formula: `mean(min_distances) + lambda * std(min_distances)`. | Used after training to compute `anomaly_threshold_` and, if adaptive mode is on, `cluster_thresholds_`. | Increase it -> higher anomaly cutoff and fewer predicted attacks. Decrease it -> lower cutoff and more predicted attacks. |
| `Adaptive Threshold` | A switch between one global anomaly threshold and separate thresholds for each cluster. | If `False`, prediction uses one global `anomaly_threshold_`. If `True`, prediction uses the threshold of the nearest cluster. | `False` is simpler and uniform. `True` is better when clusters have different natural spread. It changes detection behavior, but not centroid creation itself. |
| `Minkowski p` | The order `p` used only when the metric is `minkowski`. | Passed to `scipy.spatial.distance.cdist(..., metric="minkowski", p=p)`. | `p=1` behaves like Manhattan, `p=2` like Euclidean, and larger `p` makes large per-feature differences matter more strongly. |
| `Learning Rate` | A model parameter exposed in the UI for compatibility. | It is passed into the model constructor, but the current SCC implementation updates centroids with the exact running mean formula, so the learning rate is not used in the actual centroid math. | Right now, changing it does not materially change training or prediction results. |
| `Model Active` | A system-status badge shown in the sidebar. | Indicates the dashboard has an active model session and is ready to train/predict. | No direct mathematical effect. |
| `metric`, `threshold`, `buffer (lambda)`, `adaptive` badges | A compact summary of the currently selected hyperparameters. | These badges mirror the current UI settings so the user can see the active configuration at a glance. | Informational only, but they describe the settings that drive training and prediction. |
| `SCC-V1.0-COS` | A display identifier for the current SCC model variant. `COS` is derived from the selected metric name. | Helps identify the configuration being shown in the dashboard. | No direct mathematical effect. |

### Q27. What do the training KPIs mean?

| Dashboard term | Definition | How it is calculated in this project | What changes it |
|---|---|---|---|
| `Samples` | Number of rows used for training. | `X_scaled.shape[0]` after loading and preprocessing the dataset. | Changes when the dataset changes. |
| `Features` | Number of usable numeric features after preprocessing. | Count of selected feature columns after numeric conversion, missing-value filling, and constant-column removal. | Changes if the dataset schema changes or if any feature becomes constant and gets dropped. |
| `Clusters` | Number of centroids finally created by SCC. | `len(model.centroids_)` after training. | Depends strongly on distance metric, cluster threshold, data distribution, and sample order. |
| `Anomaly Threshold` | Final cutoff used to flag anomalies. | Global mode: `mean(min_distances) + lambda * std(min_distances)`. Adaptive mode: one such value per cluster. | Depends on the distance metric, cluster structure, training data, and anomaly buffer. It is different from `Cluster Threshold`, which is used earlier during cluster formation. |
| `Anomaly Rate` | Percentage of training samples predicted as anomalous by the fitted model. | `100 * mean(predictions)` where `predictions` are `0` for normal and `1` for attack. | Goes up when the anomaly threshold goes down. Goes down when the anomaly threshold goes up. |
| `Silhouette Score (cosine)` | A clustering-quality measure, not an attack-accuracy measure. Values near `1` mean clusters are well separated. | Computed with `sklearn.metrics.silhouette_score` using the currently selected metric. | Depends on cluster assignments and the chosen metric. It can become undefined if there is only one cluster or if every point becomes its own cluster. |
| `Training Latency` | Time taken to fit the SCC model on the current data. | Measured in milliseconds with `time.perf_counter()` around `model.fit(X_scaled)`. | Depends on sample count, feature count, number of clusters, metric cost, and hardware speed. |
| `Threshold Mode` | Indicates whether anomaly detection is global or cluster-specific. | Shows `GLOBAL (UNIFIED)` when `Adaptive Threshold` is off and `ADAPTIVE (PER-CLUSTER)` when it is on. | Changes only when the adaptive-threshold toggle changes. |

Important practical note:

- `Cluster Threshold = 0.3` is the rule for creating clusters during training.
- `Anomaly Threshold = 0.0080` is the final detection cutoff learned after training from the distance distribution.
- They serve different purposes, so they do not need to be numerically close.

### Q28. What do the visualization tabs mean?

| Dashboard term | Definition | How it is used in this project | What the shown values depend on |
|---|---|---|---|
| `Cluster Centroids` | A table and heatmap of the learned centroid vectors. | Each centroid represents the center of one SCC cluster. The dashboard shows them after standardization. | Depends on training data, standardization, distance metric, and cluster assignments. |
| `z-score` centroid values | Standardized feature values inside each centroid. `0` means around the dataset mean, positive values mean above average, and negative values mean below average. | Helps explain what kind of traffic pattern each cluster represents. | Depends on `StandardScaler` and the samples assigned to that cluster. |
| `Distance Distribution` | Histogram of each sample's minimum distance to its nearest centroid. | Used to visually separate predicted normal samples from predicted anomalies and to show the threshold line. | Depends on centroids, distance metric, and anomaly threshold. |
| `Predicted Normal / Predicted Attack` | Colors used in the histogram and PCA view. | Based on `model.predict(X_scaled)`. | Change when the anomaly threshold or the cluster structure changes. |
| `2D Projection (PCA)` | A 2D visualization of the standardized data using Principal Component Analysis. | Used only for visualization so humans can inspect cluster separation in two dimensions. | Depends on feature variance structure and cluster labels. It does not affect training or prediction. |
| `PC1` / `PC2` | The first two principal components shown on the PCA chart. | They summarize as much variance as possible in two axes. | Their percentages depend on the dataset and scaling. |
| `Cluster Summary` | Per-cluster summary table. | Shows sample count, mean distance, standard deviation of distance, threshold used, and predicted attacks for each cluster. | Depends on cluster assignments, whether threshold mode is global or adaptive, and the anomaly predictions. |

### Q29. What do the Real-time Inference Playground terms mean?

| Term | Definition | How it is used in this project | What the value depends on |
|---|---|---|---|
| Feature sliders such as `Packet_Count`, `Source_Bytes`, `Diff_Srv_Rate`, etc. | Manual inputs for a new traffic sample. | The dashboard collects these raw feature values, then applies the same `StandardScaler` used during training before prediction. | Each slider range is built from the dataset column's `min`, `max`, and `median`, with extra padding of `30%` of the observed range. |
| `Predict Vector` | Runs one-sample inference against the trained model. | Calls `scaler.transform()`, `model.transform()`, finds the nearest cluster, and compares the distance with the active threshold. | Output depends on the entered feature values and the current model settings. |
| `Nearest Cluster` | The centroid with the smallest distance to the input sample. | Used to decide which learned traffic pattern the new sample is closest to. | Depends on the input sample after scaling and on the current centroids. |
| `Distance` | The smallest distance from the input sample to any centroid. | This is the core anomaly score used for the final normal/attack decision. | Depends on the metric, the input values, and the centroids. |
| `Threshold` | The cutoff used for this specific prediction. | In global mode it is `model.anomaly_threshold_`. In adaptive mode it is the threshold of the nearest cluster. | Depends on training data, anomaly buffer, distance metric, and adaptive mode. |
| `Prediction: NORMAL / ATTACK` | The final inference result. | `ATTACK` if `distance > threshold`, else `NORMAL`. | Depends on the relationship between the distance and threshold. |
| `Distance-to-threshold ratio` | How close the input is to the decision boundary. | Helps the user understand margin. Above `1` means beyond the anomaly threshold. | Depends on `distance / threshold`. |
| `Confidence Score` | A UI heuristic shown in the result card. | Computed from the distance-threshold ratio for display only. | It is not a calibrated probability and should be explained as a rough confidence indicator, not a statistical certainty. |
| `Inference Time` | Time taken to score one input vector. | Measured with `time.perf_counter()` around the prediction path. | Depends on hardware and current model size. |

### Q30. How are the nine network features used during prediction?

The nine configured features are the variables that define each traffic sample. After preprocessing and standardization, they become the coordinates used in clustering and anomaly detection.

| Feature | Meaning | How it helps in this IDS |
|---|---|---|
| `Packet_Count` | Number of packets in the observed flow or window. | High values can push a sample toward heavy-traffic or flood-like behavior. |
| `Byte_Count` | Total bytes transferred. | Helps distinguish low-volume from high-volume traffic bursts. |
| `Duration` | Flow or session duration. | Very short repetitive activity can look suspicious when combined with high packet or SYN counts. |
| `Source_Bytes` | Bytes sent from the source side. | Helps capture one-sided traffic surges from the sender. |
| `Dest_Bytes` | Bytes sent from the destination side. | Helps measure traffic balance between source and destination. |
| `Same_Srv_Rate` | Fraction of recent traffic aimed at the same service. | High values can indicate concentrated targeting of one service. |
| `Diff_Srv_Rate` | Fraction of recent traffic aimed at different services. | Can help separate focused flooding from broader probing or mixed behavior. |
| `SYN_Flag_Count` | Count of SYN TCP flags. | Very useful for spotting SYN-flood-like behavior. |
| `ACK_Flag_Count` | Count of ACK TCP flags. | Helps contrast normal handshake/established traffic against SYN-heavy abnormal traffic. |

### Q31. What is the exact meaning of `Metric: cosine` in this project?

Cosine distance measures direction similarity between feature vectors:

`cosine_distance = 1 - cosine_similarity`

In our project, that means the model compares the shape of the traffic pattern more than its raw magnitude. Two samples can have different total scale but still be considered similar if their standardized feature directions are close.

Why it is useful here:

- DDoS-like traffic often has a repeated shape across several features.
- Cosine can group samples that behave similarly even if absolute volume changes.
- It often works well when features are standardized and the pattern shape matters more than raw size.

Important note:

- In theory, cosine distance can go above `1` in opposite-direction cases.
- In this dashboard, the `Cluster Threshold` slider for cosine is intentionally limited to `0.01` to `1.00` for practical tuning.

### Q31A. What are the exact slider ranges, and what do value changes do?

In the dashboard (`app.py` sidebar), you change these values:

- `Cluster Threshold`:
  - if metric is `cosine`: `0.01` to `1.00` with step `0.01`
  - otherwise: `0.05` to `3.00` with step `0.05`
- `Anomaly Buffer (lambda)`: `0.10` to `5.00` with step `0.10`
- `Learning Rate`: `0.01` to `1.00` with step `0.01`
- `Minkowski p`: `1` to `10` with step `1`, only when metric is `minkowski`

Effect summary:

- Increase `Cluster Threshold` -> samples join existing clusters more easily -> fewer clusters.
- Decrease `Cluster Threshold` -> new clusters are created more easily -> more clusters.
- Increase `Anomaly Buffer (lambda)` -> anomaly threshold rises -> fewer predicted attacks.
- Decrease `Anomaly Buffer (lambda)` -> anomaly threshold falls -> more predicted attacks.
- Turn `Adaptive Threshold` on -> each cluster gets its own anomaly cutoff.
- Turn `Adaptive Threshold` off -> one global cutoff is used everywhere.
- Change `Distance Metric` -> all distances change, so thresholds usually need retuning.
- Change `Learning Rate` -> currently no meaningful effect in this codebase.

### Q31B. How do these values depend on each other in the full pipeline?

This is the most important dependency chain in the dashboard:

1. Raw feature values are cleaned and standardized.
2. The chosen `Distance Metric` and `Cluster Threshold` determine how many clusters are formed.
3. Those clusters determine the centroid locations and all minimum sample-to-centroid distances.
4. `Anomaly Buffer (lambda)` uses those distances to set the final anomaly threshold.
5. `Adaptive Threshold` decides whether that cutoff is global or per cluster.
6. The final threshold then determines `Anomaly Rate`, histogram coloring, cluster summary attack counts, and real-time prediction output.

Short dependency summary:

- `Distance Metric` changes the geometry of the whole model.
- `Cluster Threshold` changes model structure.
- `Anomaly Buffer` changes sensitivity.
- `Adaptive Threshold` changes decision style.
- `Learning Rate` is currently a no-op in this implementation.

## 9) How to Run the Project

### Q32. How do I start the dashboard?

From PowerShell:

```powershell
cd C:\Users\nani\Desktop\minor2\federated_ddos_ids
..\.venv\Scripts\python -m pip install -r requirements.txt
..\.venv\Scripts\python -m streamlit run app.py
```

Open:
`http://localhost:8501`

### Q33. How do I run CLI pipeline?

```powershell
cd C:\Users\nani\Desktop\minor2\federated_ddos_ids
python main.py
```

### Q34. How do I run tests?

```powershell
cd C:\Users\nani\Desktop\minor2\federated_ddos_ids
pytest -q
```

## 10) Current State of the Project (Practical Status)

### Q35. What is implemented and working?

- End-to-end local training pipeline.
- Interactive Streamlit dashboard with visualizations and manual inference.
- Multiple distance metrics (euclidean, manhattan, minkowski, cosine).
- Global and adaptive anomaly threshold options.
- Tests passing (`6 passed`).

### Q36. What is not fully implemented yet?

- Full multi-client federated training orchestration (client updates + server rounds + communication loop).
- Production-grade dataset/versioning and real network ingestion pipeline.
- Strong supervised benchmark reporting on a labeled real-world dataset.

### Q37. Is this useful for learning/demo?

Yes. It is strong for concept demonstration: clustering-based IDS, threshold tuning, metric effects, and explainable visualization.

## 11) Important Interview / Viva Questions and Answers

### Q38. Why unsupervised anomaly detection instead of pure classification?

Because attack labels are often limited or noisy in real security data. Unsupervised methods can detect unknown/novel attacks.

### Q39. Why cosine metric for network features?

Cosine focuses on pattern direction rather than absolute magnitude, which can help when total volume varies but behavior shape is similar.

### Q40. What is tradeoff between false positives and false negatives?

- Too sensitive threshold -> catches more attacks but may flag normal traffic.
- Too lenient threshold -> fewer false alarms but may miss real attacks.

### Q41. Why adaptive threshold can help?

Different clusters can have different natural spread. Per-cluster thresholds can reduce over-flagging for broad clusters and under-flagging for tight clusters.

### Q42. What is one key limitation of this prototype?

It currently trains and evaluates mostly on local/synthetic style data rather than a full federated, real-world, continuously updated traffic stream.

## 12) Quick Glossary

- Cluster: group of similar samples.
- Centroid: center point of a cluster.
- Distance: how different two vectors are.
- Threshold: boundary for decision.
- Anomaly: sample outside expected normal behavior.
- Standardization: feature scaling to zero mean and unit variance.
- PCA: dimension reduction for 2D plotting.
- Silhouette score: cluster quality indicator.
- Federated averaging: combine client model parameters at server side.

---

If you want, this file can be converted into:

- a short viva-ready one-page summary,
- a formal final report chapter format,
- or a PPT slide script with speaker notes.
