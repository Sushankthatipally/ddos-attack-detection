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

## 8) Dashboard Settings Explained (Your Specific Question)

You asked about:
`Metric: cosine | Cluster Threshold: 0.3 | Anomaly Buffer (lambda): 1.1 | Adaptive Threshold: False`

### Q26. What does `Metric: cosine` mean?
Cosine distance measures angle difference between vectors, not raw magnitude difference.
- 0 means very similar direction.
- Larger values mean less similar direction.

In this IDS context: traffic profiles with similar behavior shape stay close under cosine distance.

### Q27. What does `Cluster Threshold: 0.3` mean?
During training:
- If nearest centroid distance <= 0.3, sample joins existing cluster.
- If > 0.3, new cluster is created.

Interpretation:
- Smaller threshold -> stricter matching -> more clusters.
- Larger threshold -> looser matching -> fewer clusters.

### Q28. What does `Anomaly Buffer (lambda): 1.1` mean?
`lambda` controls how far above average distance a point must be to be flagged.

Formula:
`anomaly_threshold = mean(min_distances) + lambda * std(min_distances)`

If lambda is:
- Low -> more sensitive -> more anomalies flagged.
- High -> less sensitive -> fewer anomalies flagged.

### Q29. What does `Adaptive Threshold: False` mean?
`False` means one global threshold is used for all clusters.
`True` means each cluster gets its own threshold (better when clusters have very different spread).

### Q30. What about `Learning Rate`?
The UI exposes it, but current centroid update uses exact running mean and does not use learning rate directly.

### Q31. What is `Minkowski p`?
Used only when metric is `minkowski`.
It controls the distance family:
- `p=1` -> Manhattan-like
- `p=2` -> Euclidean-like

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
