import os
import sys
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.clustering import CLAPPClustering
from client.preprocess import preprocess_data
from config import (
    ADAPTIVE_THRESHOLD,
    ANOMALY_BUFFER,
    CLAPP_THRESHOLD,
    DATASET_PATH,
    FEATURE_NAMES,
    KNN_K_VALUES,
    LABEL_COL,
    RANDOM_STATE,
    SIGMA_C,
    TEST_SIZE,
)


CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

EXPECTED = {
    "dataset_shape": (149227, 43),
    "labels": {0: 77054, 1: 54343, 2: 13987, 3: 2885, 4: 958},
    "ps_shape": (149227, 41),
    "clusters": 35,
    "reduced_dims": 35,
}

PAPER_K1 = np.array(
    [
        [66004, 86, 139, 57, 19],
        [43, 44912, 222, 0, 0],
        [102, 307, 11057, 0, 1],
        [44, 1, 0, 927, 0],
        [26, 0, 0, 1, 25],
    ]
)

PAPER_K5 = np.array(
    [
        [65831, 148, 284, 41, 1],
        [100, 44811, 266, 0, 0],
        [163, 450, 10854, 0, 0],
        [54, 1, 1, 916, 0],
        [39, 0, 0, 1, 12],
    ]
)


st.set_page_config(
    page_title="SENTINEL - CLAPP DDoS IDS",
    page_icon="shield",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #0f1419; color: #dee3ea; }
    [data-testid="stHeader"] { background: rgba(15, 20, 25, 0.92); }
    [data-testid="stSidebar"] { background: #171c21; border-right: 1px solid #27313a; }
    .block-container { padding-top: 2rem; max-width: 1440px; }
    h1, h2, h3 { letter-spacing: 0 !important; color: #e5edf5 !important; }
    div[data-testid="stMetric"] {
        background: #171c21;
        border: 1px solid #27313a;
        padding: 1rem;
        border-radius: 6px;
    }
    div[data-testid="stMetric"] label { color: #8b9daf !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #2dd4a8 !important; }
    .status-ok { color: #34d399; font-weight: 700; }
    .status-bad { color: #ef4444; font-weight: 700; }
    .note {
        background: #171c21;
        border-left: 4px solid #f59e0b;
        padding: 0.85rem 1rem;
        border-radius: 4px;
        color: #dbe4ed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def resolve_dataset_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(__file__), path)


def file_mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_data(show_spinner=False)
def load_dataset(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_matrix(df: pd.DataFrame):
    ps, labels, feature_names = preprocess_data(df, label_col=LABEL_COL, scale=False)

    configured = [name for name in FEATURE_NAMES if name in feature_names]
    if not configured:
        raise ValueError("None of the configured NSL-KDD feature columns were found.")

    indices = [feature_names.index(name) for name in configured]
    ps = ps[:, indices]
    feature_df = pd.DataFrame(ps, columns=configured)
    return feature_df, ps, labels, configured


@st.cache_resource(show_spinner=False)
def fit_clapp_model(path: str, mtime: float, threshold: float, sigma: float):
    df = load_dataset(path, mtime)
    feature_df, ps, labels, feature_names = prepare_matrix(df)

    start = time.perf_counter()
    model = CLAPPClustering(
        threshold=threshold,
        sigma=sigma,
        similarity_metric="fuzzy_gaussian",
        anomaly_buffer=ANOMALY_BUFFER,
        adaptive_threshold=ADAPTIVE_THRESHOLD,
    )
    model.fit(ps, labels)
    elapsed = time.perf_counter() - start
    reduced = model.transform(ps)
    return df, feature_df, ps, labels, feature_names, model, reduced, elapsed


def status_text(matches: bool) -> str:
    cls = "status-ok" if matches else "status-bad"
    text = "MATCH" if matches else "DIFFERENT"
    return f'<span class="{cls}">{text}</span>'


def comparison_table(df: pd.DataFrame, ps: np.ndarray, model: CLAPPClustering) -> pd.DataFrame:
    actual_labels = df["label"].value_counts().sort_index().to_dict()
    actual_shape = tuple(df.shape)
    actual_ps_shape = tuple(ps.shape)
    actual_clusters = len(model.cluster_means_)
    actual_reduced_dims = model.transformation_matrix_.shape[1]

    rows = [
        ("Dataset shape", EXPECTED["dataset_shape"], actual_shape, actual_shape == EXPECTED["dataset_shape"]),
        ("Label counts", EXPECTED["labels"], actual_labels, actual_labels == EXPECTED["labels"]),
        ("PS matrix shape", EXPECTED["ps_shape"], actual_ps_shape, actual_ps_shape == EXPECTED["ps_shape"]),
        ("Clusters formed", EXPECTED["clusters"], actual_clusters, actual_clusters == EXPECTED["clusters"]),
        ("Reduced dims", EXPECTED["reduced_dims"], actual_reduced_dims, actual_reduced_dims == EXPECTED["reduced_dims"]),
    ]
    return pd.DataFrame(rows, columns=["Check", "Expected", "Actual", "Matches"])


def render_comparison(df: pd.DataFrame, ps: np.ndarray, model: CLAPPClustering):
    comp = comparison_table(df, ps, model)
    display = comp.copy()
    display["Matches"] = display["Matches"].map(lambda value: "MATCH" if value else "DIFFERENT")
    st.dataframe(display, hide_index=True, use_container_width=True)

    if not bool(comp["Matches"].all()):
        st.markdown(
            """
            <div class="note">
            The local run does not match the exact paper checklist. The main blocker is the
            dataset currently available locally: it has different row counts and class counts,
            especially U2R. With different data distribution, the 35-cluster paper result should
            not be expected.
            </div>
            """,
            unsafe_allow_html=True,
        )


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label"].value_counts().sort_index()
    rows = []
    for label, count in counts.items():
        name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else str(label)
        rows.append({"Label": int(label), "Class": name, "Rows": int(count)})
    return pd.DataFrame(rows)


def cluster_feature_table(model: CLAPPClustering, feature_names: list[str], reduced: np.ndarray) -> pd.DataFrame:
    cluster_labels = np.argmax(reduced, axis=1)
    rows = []
    for idx, mean in enumerate(model.cluster_means_):
        feature_indices = model.cluster_feature_indices_[idx] if idx < len(model.cluster_feature_indices_) else []
        features = [feature_names[i] for i in feature_indices]
        rows.append(
            {
                "Cluster": idx,
                "Feature Count": len(features),
                "Dominant Samples": int(np.sum(cluster_labels == idx)),
                "Features": ", ".join(features),
                "Pattern Mean": np.round(mean, 4).tolist(),
            }
        )
    return pd.DataFrame(rows)


def run_knn(reduced: np.ndarray, labels: np.ndarray, k: int):
    x_train, x_test, y_train, y_test = train_test_split(
        reduced,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    start = time.perf_counter()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    elapsed = time.perf_counter() - start
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    return cm, report, accuracy_score(y_test, y_pred), elapsed


def cm_dataframe(cm: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)


def plot_reduced_projection(reduced: np.ndarray, labels: np.ndarray, max_points: int = 8000):
    rng = np.random.default_rng(RANDOM_STATE)
    if len(reduced) > max_points:
        idx = rng.choice(len(reduced), size=max_points, replace=False)
        reduced_plot = reduced[idx]
        labels_plot = labels[idx]
    else:
        reduced_plot = reduced
        labels_plot = labels

    pca = PCA(n_components=2)
    coords = pca.fit_transform(reduced_plot)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#2dd4a8", "#ef4444", "#f59e0b", "#60a5fa", "#c084fc"]
    for label in sorted(np.unique(labels_plot)):
        mask = labels_plot == label
        label_int = int(label)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.65,
            color=colors[label_int % len(colors)],
            label=CLASS_NAMES[label_int] if label_int < len(CLASS_NAMES) else str(label_int),
        )
    ax.set_title("Reduced CLAPP Space")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.grid(alpha=0.18)
    ax.legend(markerscale=2)
    fig.tight_layout()
    return fig


st.sidebar.caption("CLAPP paper settings")
threshold = st.sidebar.slider(
    "Similarity threshold",
    min_value=0.50,
    max_value=0.9999,
    value=float(min(max(CLAPP_THRESHOLD, 0.50), 0.9999)),
    step=0.0001,
    format="%.4f",
)
sigma = st.sidebar.slider(
    "Sigma C",
    min_value=0.05,
    max_value=2.0,
    value=float(min(max(SIGMA_C, 0.05), 2.0)),
    step=0.05,
)
selected_k = st.sidebar.selectbox("kNN k", KNN_K_VALUES, index=0)
st.sidebar.caption(f"Dataset: {DATASET_PATH}")

st.title("SENTINEL - CLAPP DDoS IDS")
st.caption("NSL-KDD feature clustering with paper-configuration checks")

dataset_path = resolve_dataset_path(DATASET_PATH)
if not os.path.exists(dataset_path):
    st.error(f"Dataset not found: {DATASET_PATH}")
    st.code("python download_nslkdd.py --output data", language="bash")
    st.stop()

try:
    with st.spinner("Loading dataset and fitting CLAPP..."):
        df, feature_df, ps, labels, feature_names, model, reduced, fit_seconds = fit_clapp_model(
            dataset_path,
            file_mtime(dataset_path),
            threshold,
            sigma,
        )
except Exception as exc:
    st.exception(exc)
    st.stop()

reduction = (1.0 - reduced.shape[1] / ps.shape[1]) * 100.0
summary_cols = st.columns(5)
summary_cols[0].metric("Dataset Rows", f"{df.shape[0]:,}")
summary_cols[1].metric("PS Features", f"{ps.shape[1]}")
summary_cols[2].metric("CLAPP Clusters", f"{len(model.cluster_means_)}")
summary_cols[3].metric("Reduced Dims", f"{reduced.shape[1]}", f"{reduction:.1f}%")
summary_cols[4].metric("Fit Time", f"{fit_seconds:.2f}s")

tab_check, tab_data, tab_clusters, tab_projection, tab_knn = st.tabs(
    ["Paper Check", "Dataset", "Feature Clusters", "Projection", "kNN Evaluation"]
)

with tab_check:
    st.subheader("Expected vs Actual")
    render_comparison(df, ps, model)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Paper Table 12, kNN k=1")
        st.dataframe(cm_dataframe(PAPER_K1), use_container_width=True)
    with c2:
        st.caption("Paper Table 14, kNN k=5")
        st.dataframe(cm_dataframe(PAPER_K5), use_container_width=True)

with tab_data:
    st.subheader("Loaded Dataset")
    st.write(f"Resolved path: `{dataset_path}`")
    st.dataframe(label_distribution(df), hide_index=True, use_container_width=True)
    if "label_name" in df.columns:
        st.caption("Attack type distribution")
        attack_counts = df["label_name"].value_counts().rename_axis("Attack Type").reset_index(name="Rows")
        st.dataframe(attack_counts, hide_index=True, use_container_width=True)
    with st.expander("Raw data sample"):
        st.dataframe(df.head(50), use_container_width=True)

with tab_clusters:
    st.subheader("CLAPP Feature Clusters")
    cluster_df = cluster_feature_table(model, feature_names, reduced)
    st.dataframe(cluster_df, hide_index=True, use_container_width=True)

    class_cols = [CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else str(c) for c in model.classes_]
    pattern_df = pd.DataFrame(model.cluster_means_, columns=class_cols)
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(pattern_df) * 0.28)))
    im = ax.imshow(pattern_df.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(class_cols)))
    ax.set_xticklabels(class_cols)
    ax.set_yticks(np.arange(len(pattern_df)))
    ax.set_yticklabels([f"C{i}" for i in range(len(pattern_df))])
    ax.set_title("Posterior Probability Pattern Means")
    fig.colorbar(im, ax=ax, label="probability")
    fig.tight_layout()
    st.pyplot(fig)

with tab_projection:
    st.subheader("Reduced Space Preview")
    st.pyplot(plot_reduced_projection(reduced, labels))

with tab_knn:
    st.subheader("kNN Evaluation")
    st.write(
        "This uses the configured stratified split. Run it only when you want to compare "
        "classification results; full kNN can take time on the complete dataset."
    )
    if st.button(f"Run kNN k={selected_k}"):
        with st.spinner(f"Running kNN k={selected_k} on reduced CLAPP matrix..."):
            cm, report, accuracy, elapsed = run_knn(reduced, labels, int(selected_k))
        st.metric("Accuracy", f"{accuracy * 100:.2f}%", f"{elapsed:.2f}s")
        st.caption("Confusion matrix")
        st.dataframe(cm_dataframe(cm), use_container_width=True)

        report_df = pd.DataFrame(report).T
        st.caption("Classification report")
        st.dataframe(report_df, use_container_width=True)
