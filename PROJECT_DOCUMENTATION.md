# Federated DDoS IDS - Project Flow, File Responsibilities, Formulas, Dataset, and Results

## 1) End-to-End Project Flow

This project has two runnable paths:

1. `main.py` (CLI training/evaluation pipeline)
2. `app.py` (Streamlit dashboard with interactive controls)

### Data-to-Detection Flow Diagram

```mermaid
flowchart TD
    A[Raw Data: data/dummy_data.csv] --> B[Preprocessing: client/preprocess.py]
    B --> B1[Drop ip/id columns]
    B1 --> B2[Convert to numeric + handle NaN/Inf]
    B2 --> B3[Remove constant columns]
    B3 --> B4[StandardScaler z-score]
    B4 --> C[Feature Matrix X]

    C --> D[Train SCC: client/clustering.py fit()]
    D --> D1[Initialize first centroid]
    D1 --> D2[For each sample: compute distances to centroids]
    D2 --> D3{Min distance <= cluster threshold?}
    D3 -- Yes --> D4[Assign to nearest cluster]
    D4 --> D5[Update centroid by exact running mean]
    D3 -- No --> D6[Create new cluster]
    D5 --> E[After training: compute min distances]
    D6 --> E
    E --> F[Compute anomaly threshold: mu + lambda*sigma]
    E --> G[Optional per-cluster threshold: mu_k + lambda*sigma_k]

    H[New Incoming Sample] --> H1[Apply same preprocessing/scaling]
    H1 --> I[Predict: distances to trained centroids]
    I --> J[Take nearest-centroid distance]
    J --> K{Distance > anomaly threshold?}
    K -- Yes --> L[Output: Attack (1)]
    K -- No --> M[Output: Normal (0)]
```

### A. CLI flow (`main.py`)

1. Load dataset from `data/dummy_data.csv`.
2. Preprocess with `client/preprocess.py`:
   - Drop identifier/IP columns.
   - Convert to numeric.
   - Handle NaN/Inf.
   - Remove constant columns.
   - Standardize features (z-score).
3. Initialize `SelfConstructingClustering` from `client/clustering.py` using values from `config.py`.
4. Train model online (`fit`):
   - Create/update clusters sample-by-sample.
   - Compute anomaly threshold(s) after clustering.
5. Predict anomalies on training set (`predict`).
6. Run two manual scenarios:
   - mean sample
   - extreme shifted sample

### B. Dashboard flow (`app.py`)

1. Load same dataset.
2. Standardize selected features.
3. User chooses metric/threshold/learning-rate in sidebar.
4. Train SCC model.
5. Show:
   - number of clusters,
   - anomaly threshold,
   - centroid table,
   - distance histogram with threshold line,
   - PCA 2D projection.
6. Real-time inference:
   - user sets synthetic feature values,
   - app scales input,
   - app predicts Normal/Attack and shows distance vs threshold.

## 2) Which File Does What

### Root files

- `main.py`: script pipeline (load -> preprocess -> train -> predict -> manual checks).
- `app.py`: Streamlit dashboard for training visualization and inference playground.
- `config.py`: global settings (metric, thresholds, feature list, alpha search space, etc.).
- `generate_dataset.py`: creates a separate synthetic labeled dataset (`data/dummy_dataset.csv`).
- `requirements.txt`: dependencies.

### Client modules

- `client/preprocess.py`: numeric cleaning + scaling.
- `client/clustering.py`: core SCC algorithm, distance handling, cluster updates, anomaly detection.
- `client/threshold.py`: helper threshold formulas (`mu + alpha*sigma`, cosine percentile).
- `client/alpha_tuning.py`: sweeps alpha and computes precision/recall/F1.
- `client/reducer.py`: PCA dimensionality reduction utility.

### Server module

- `server/aggregate.py`: simple federated averaging over client centroids.

### Evaluation modules

- `evaluation/metrics.py`: accuracy/precision/recall/F1 wrapper.
- `evaluation/plots.py`: alpha-vs-F1 and ROC plotting utilities.

### Tests

- `tests/test_clustering.py`: SCC unit tests (init, fitting, prediction, cosine behavior).
- `tests/test_pipeline.py`: integration tests for preprocess + train + predict flow.

## 3) All Formulas Used (Distance + Thresholding + Updates)

Let:
- sample: \(x \in \mathbb{R}^d\)
- centroid \(k\): \(\mu_k\)
- number of clusters: \(C\)
- distance to centroid \(k\): \(D_k = D(x,\mu_k)\)

### A. Distance metrics

1. Euclidean:
\[
D_{\text{euclidean}}(x,\mu)=\sqrt{\sum_{j=1}^{d}(x_j-\mu_j)^2}
\]

2. Manhattan (cityblock in SciPy):
\[
D_{\text{manhattan}}(x,\mu)=\sum_{j=1}^{d}|x_j-\mu_j|
\]

3. Minkowski (parameter \(p\), `minkowski_p` in config):
\[
D_{\text{minkowski}}(x,\mu)=\left(\sum_{j=1}^{d}|x_j-\mu_j|^p\right)^{1/p}
\]

4. Cosine distance:
\[
D_{\text{cosine}}(x,\mu)=1-\frac{x\cdot \mu}{\|x\|_2\|\mu\|_2}
\]

### B. Cluster assignment and creation

Nearest cluster index:
\[
k^*=\arg\min_k D_k,\quad D_{\min}=D_{k^*}
\]

If \(D_{\min}\le \theta\) (`threshold`), assign to cluster \(k^*\).
If \(D_{\min}>\theta\), create a new cluster with centroid \(x\).

### C. Centroid update rule (exact mean)

If assigned to cluster \(k\) with old count \(n_k\):
\[
\mu_k^{new}=\frac{n_k\mu_k^{old}+x}{n_k+1}
\]

### D. Threshold conversion logic (how threshold is converted/derived)

There are two threshold concepts:

1. **Cluster formation threshold** \(\theta\):
   - User/config provided directly.
   - Used during `fit` to decide update-vs-new-cluster.

2. **Anomaly decision threshold** (derived after fitting from distance statistics):

Global (default, Option 2):
\[
\theta_{anomaly}=\mu_D+\lambda\sigma_D
\]
where \(D\) are each sample's minimum centroid distances, \(\mu_D\)=mean, \(\sigma_D\)=std, \(\lambda\)=`anomaly_buffer`.

Cluster-adaptive (Option 3, if `adaptive_threshold=True`):
\[
\theta_k=\mu_{D_k}+\lambda\sigma_{D_k}
\]
for distances of samples assigned to cluster \(k\).

Prediction rule:
- Global: Attack if \(D_{\min} > \theta_{anomaly}\)
- Adaptive: Attack if \(D_{\min} > \theta_{k^*}\)

### E. Additional threshold formulas in helper modules

From `client/threshold.py`:
\[
T=\mu+\alpha\sigma
\]
and cosine helper:
\[
T_{cosine}=\text{percentile}_{5\%}(\text{similarities})
\]

From `client/alpha_tuning.py`:
- For each \(\alpha\), compute \(T_\alpha=\mu+\alpha\sigma\),
- Predict \(y=1\) if \(d>T_\alpha\), else \(0\),
- Evaluate precision, recall, F1.

## 4) Dataset Explanation (Detailed)

## Active training dataset in pipeline

- File used by `main.py` and `app.py`: `data/dummy_data.csv`
- Shape: **40 rows x 9 features**
- Columns:
  - `Packet_Count`
  - `Byte_Count`
  - `Duration`
  - `Source_Bytes`
  - `Dest_Bytes`
  - `Same_Srv_Rate`
  - `Diff_Srv_Rate`
  - `SYN_Flag_Count`
  - `ACK_Flag_Count`

### Observed structure of this dataset

- No explicit label column in `dummy_data.csv`.
- It is strongly bimodal by value range:
  - first 20 rows: low-value group (overall mean ~0.046)
  - last 20 rows: higher-value group (overall mean ~0.548)
- This naturally produces two clusters after standardization and SCC training.

### Preprocessing impact

`client/preprocess.py` applies:

1. Drop columns containing `ip`/`id`.
2. Convert all values to numeric; non-numeric -> NaN.
3. Drop all-NaN columns.
4. Replace Inf/-Inf with NaN.
5. Fill NaN with column mean.
6. Remove zero-variance columns.
7. Standardize with:
\[
z=\frac{x-\mu}{\sigma}
\]

This makes feature scales comparable before distance computation.

## Additional synthetic generator dataset

`generate_dataset.py` creates a different file (`data/dummy_dataset.csv`) with labels (`Normal`/`Attack`) and 10 rows (5 normal + 5 attack), but this generated file is not the one used by `main.py` currently.

## 5) How Clustering Is Happening (Step-by-Step)

Inside `SelfConstructingClustering.fit`:

1. Start with no clusters.
2. First sample becomes first centroid.
3. For each next sample:
   - compute distances to all centroids,
   - pick nearest centroid,
   - if nearest distance <= cluster threshold: update that centroid using exact running mean,
   - else: create a new centroid from this sample.
4. After all samples:
   - compute all sample-to-centroid distances,
   - compute each sample's minimum distance,
   - derive global anomaly threshold (and optional per-cluster thresholds).

So this is an **online, threshold-driven, self-growing clustering** method.

## 6) How Detection Is Happening (Normal vs Attack)

Inside `predict`:

1. For each incoming sample, compute distances to all trained centroids.
2. Find minimum distance to nearest cluster.
3. Compare that distance to threshold:
   - default global threshold: `anomaly_threshold_`
   - optional adaptive threshold: threshold of nearest cluster
4. Output:
   - `0` => Normal
   - `1` => Attack

Interpretation: samples far from learned normal structure are flagged as attacks.

## 7) Results Obtained from This Project (Current Workspace Run)

These were obtained by running current code:

- Command: `python main.py` (from `federated_ddos_ids`)
  - Preprocessed shape: `40 x 9`
  - Distance metric: `cosine`
  - Cluster threshold: `0.3`
  - Clusters formed: **2**
  - Global anomaly threshold: **0.0080**
  - Predicted anomalies on training set: **5 / 40**
  - Manual scenario (average sample): **Attack**
  - Manual scenario (extreme shifted sample): **Normal**

Additional runtime checks:

- Cluster assignment split: **20 samples in cluster 0, 20 in cluster 1**
- Cluster counts tracked during fit: **[20, 20]**
- Predicted anomalies by row-block:
  - first 20 rows: 2 anomalies
  - last 20 rows: 3 anomalies

Test results:

- Command: `$env:PYTHONPATH='.'; pytest -q tests -p no:cacheprovider`
- Outcome: **6 passed**

## 8) Important Notes

1. `learning_rate` is currently an exposed parameter but not used in centroid updates (exact mean update is used instead).
2. The project contains federated utilities (`server/aggregate.py`), but `main.py` currently runs a single-node training flow.
3. With cosine distance, vector direction dominates magnitude; this can produce unintuitive manual-scenario outcomes after scaling.
