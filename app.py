import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Ensure the current directory is in the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.clustering import SelfConstructingClustering
from config import (
    ADAPTIVE_THRESHOLD,
    ANOMALY_BUFFER,
    CLUSTER_THRESHOLD,
    DISTANCE_METRIC,
    FEATURE_NAMES,
    LEARNING_RATE,
    MINKOWSKI_P,
)

# ═════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM: Obsidian Sentinel
# ═════════════════════════════════════════════════════════════════════
_BG          = "#0F1419"
_SURFACE_LOW = "#171C21"
_SURFACE     = "#1B2025"
_SURFACE_HI  = "#252A30"
_SURFACE_TOP = "#30353B"
_OUTLINE     = "#3C4A44"
_TEXT        = "#DEE3EA"
_TEXT_SEC    = "#8B9DAF"
_PRIMARY     = "#2DD4A8"
_PRIMARY_DIM = "#3EDEB2"
_SECONDARY   = "#F59E0B"
_DANGER      = "#EF4444"
_SUCCESS     = "#34D399"
_SLATE       = "#64748B"

st.set_page_config(
    page_title="SENTINEL · Federated DDoS Detection",
    page_icon="🛡️",
    layout="wide",
)

# ═════════════════════════════════════════════════════════════════════
# FULL CSS — Obsidian Sentinel Theme
# ═════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global Reset ──────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    [data-testid="stAppViewContainer"] {
        background: #0F1419 !important;
    }
    [data-testid="stHeader"] {
        background: rgba(15, 20, 25, 0.88) !important;
        backdrop-filter: blur(16px) saturate(1.3);
        -webkit-backdrop-filter: blur(16px) saturate(1.3);
        border-bottom: 1px solid rgba(60, 74, 68, 0.12);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #0F1419; }
    ::-webkit-scrollbar-thumb { background: #30353B; border-radius: 0; }

    /* ── Sidebar ────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #171C21 !important;
        border-right: 1px solid rgba(60, 74, 68, 0.15);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #DEE3EA !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #64748B !important;
        font-size: 0.6875rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #475569 !important;
        font-size: 0.625rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(60, 74, 68, 0.15) !important;
        margin: 1rem 0 !important;
    }

    /* ── Typography ─────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        color: #DEE3EA !important;
    }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1rem !important; }

    p, span, li, td, th, label, div {
        font-family: 'Inter', sans-serif !important;
    }
    .stMarkdown p, .stText, small {
        color: #DEE3EA !important;
    }

    /* ── Buttons ────────────────────────────────────── */
    .stButton > button {
        background: #2DD4A8 !important;
        color: #00382a !important;
        border: none !important;
        border-radius: 0 !important;
        font-weight: 800 !important;
        font-size: 0.75rem !important;
        padding: 0.75rem 1.75rem !important;
        letter-spacing: 0.15em !important;
        transition: all 0.15s ease !important;
        text-transform: uppercase !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button:hover {
        background: #3EDEB2 !important;
        color: #00382a !important;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.99);
    }

    /* ── Input Fields ──────────────────────────────── */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div {
        background: #0F1419 !important;
        border: none !important;
        border-bottom: 1px solid #3C4A44 !important;
        border-radius: 0 !important;
        color: #DEE3EA !important;
    }
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stNumberInput > div > div:focus-within {
        border-bottom-color: #2DD4A8 !important;
        box-shadow: none !important;
    }

    /* ── Slider ─────────────────────────────────────── */
    .stSlider [data-testid="stThumbValue"] {
        color: #2DD4A8 !important;
        font-variant-numeric: tabular-nums !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.6875rem !important;
        font-weight: 600 !important;
    }

    /* ── Tabs ───────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: transparent !important;
        border-bottom: 1px solid rgba(60, 74, 68, 0.2) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #64748B !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        font-weight: 600 !important;
        font-size: 0.6875rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #DEE3EA !important;
        background: rgba(45, 212, 168, 0.04) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2DD4A8 !important;
        border-bottom: 2px solid #2DD4A8 !important;
        background: transparent !important;
    }

    /* ── Metrics ────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        color: #DEE3EA !important;
        font-variant-numeric: tabular-nums !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.4rem !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        text-transform: uppercase !important;
        font-size: 0.5625rem !important;
        letter-spacing: 0.12em !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] > div {
        color: #2DD4A8 !important;
    }
    [data-testid="stMetric"] {
        background: #1B2025 !important;
        border: 1px solid rgba(60, 74, 68, 0.12) !important;
        border-radius: 0 !important;
        padding: 1rem 1.25rem !important;
    }

    /* ── DataFrames ─────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(60, 74, 68, 0.12) !important;
        border-radius: 0 !important;
        overflow: hidden !important;
    }

    /* ── Expander ───────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid rgba(60, 74, 68, 0.12) !important;
        border-radius: 0 !important;
    }
    [data-testid="stExpander"] details {
        border: none !important;
        border-radius: 0 !important;
    }
    [data-testid="stExpander"] summary {
        background: #1B2025 !important;
        color: #DEE3EA !important;
        border-radius: 0 !important;
    }
    [data-testid="stExpander"] summary p {
        font-weight: 600 !important;
        font-size: 0.8125rem !important;
        color: #DEE3EA !important;
    }
    [data-testid="stExpander"] summary svg {
        color: #64748B !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: #171C21 !important;
        border-top: 1px solid rgba(60, 74, 68, 0.08) !important;
    }

    /* ── Alerts ─────────────────────────────────────── */
    div[data-testid="stAlert"] {
        border-radius: 0 !important;
        border: none !important;
        border-left: 3px solid !important;
    }

    /* ── Code ───────────────────────────────────────── */
    code {
        color: #2DD4A8 !important;
        background: rgba(45, 212, 168, 0.06) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        border-radius: 0 !important;
        padding: 0.125rem 0.375rem !important;
        font-weight: 500 !important;
    }

    /* ── Divider ────────────────────────────────────── */
    hr { border-color: rgba(60, 74, 68, 0.15) !important; }

    /* ── Caption ────────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748B !important;
        font-size: 0.6875rem !important;
    }

    /* ── Spinner ────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: #2DD4A8 !important;
    }

    /* ═══════════════════════════════════════════════ */
    /* CUSTOM DESIGN COMPONENTS                       */
    /* ═══════════════════════════════════════════════ */
    
    .sentinel-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.25rem;
    }
    .sentinel-brand-icon {
        width: 36px;
        height: 36px;
        background: #2DD4A8;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        color: #00382a;
        font-weight: 900;
    }
    .sentinel-brand-text {
        font-size: 1.35rem;
        font-weight: 900;
        color: #2DD4A8;
        letter-spacing: -0.03em;
        text-transform: uppercase;
        line-height: 1;
    }
    .sentinel-brand-sub {
        font-size: 0.5625rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: #475569;
        font-weight: 600;
        margin-top: 2px;
    }
    .sentinel-subtitle {
        color: #64748B;
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: -0.25rem;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    /* Config badges */
    .config-bar {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.25rem 0 1.25rem 0;
    }
    .sentinel-badge {
        display: inline-block;
        padding: 0.25rem 0.625rem;
        background: rgba(45, 212, 168, 0.06);
        color: #2DD4A8;
        font-size: 0.625rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid rgba(45, 212, 168, 0.12);
        text-transform: uppercase;
    }
    .sentinel-badge-amber {
        background: rgba(245, 158, 11, 0.06);
        color: #F59E0B;
        border-color: rgba(245, 158, 11, 0.12);
    }
    .sentinel-badge-red {
        background: rgba(239, 68, 68, 0.06);
        color: #EF4444;
        border-color: rgba(239, 68, 68, 0.12);
    }
    .sentinel-badge-neutral {
        background: rgba(100, 116, 139, 0.06);
        color: #94A3B8;
        border-color: rgba(100, 116, 139, 0.12);
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        border-left: 2px solid #2DD4A8;
        padding-left: 1rem;
        margin-bottom: 0rem;
    }
    .section-header-icon {
        font-size: 1.5rem;
        line-height: 1;
    }
    .section-header h2 {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 1.15rem !important;
        letter-spacing: -0.02em !important;
    }
    .section-header-desc {
        color: #64748B;
        font-size: 0.6875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.125rem;
    }
    
    /* Info notes */
    .sentinel-note {
        background: #1B2025;
        border-left: 3px solid #2DD4A8;
        padding: 0.75rem 1rem;
        color: #DEE3EA;
        font-size: 0.75rem;
        margin: 0.75rem 0;
        font-family: 'Inter', sans-serif;
    }
    .sentinel-note-amber {
        border-left-color: #F59E0B;
    }
    
    /* Result cards */
    .result-card {
        position: relative;
        overflow: hidden;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
    }
    .result-card-attack {
        background: rgba(239, 68, 68, 0.04);
        border-left: 4px solid #EF4444;
        border-top: 1px solid rgba(239, 68, 68, 0.08);
        border-right: 1px solid rgba(239, 68, 68, 0.08);
        border-bottom: 1px solid rgba(239, 68, 68, 0.08);
    }
    .result-card-normal {
        background: rgba(45, 212, 168, 0.04);
        border-left: 4px solid #2DD4A8;
        border-top: 1px solid rgba(45, 212, 168, 0.08);
        border-right: 1px solid rgba(45, 212, 168, 0.08);
        border-bottom: 1px solid rgba(45, 212, 168, 0.08);
    }
    .result-label {
        font-size: 0.5625rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .result-title {
        font-size: 1.75rem;
        font-weight: 900;
        color: #DEE3EA;
        letter-spacing: -0.03em;
        text-transform: uppercase;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .result-details {
        font-size: 0.6875rem;
        font-family: 'JetBrains Mono', monospace;
        color: #64748B;
        line-height: 1.8;
    }
    .result-details span.val { color: #DEE3EA; font-weight: 600; }
    .result-details span.danger { color: #EF4444; }
    .result-details span.warn { color: #F59E0B; }
    .result-details span.safe { color: #2DD4A8; }
    .result-ratio {
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        font-size: 0.5625rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.375rem;
    }
    
    /* Model ID badge */
    .model-id-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.625rem;
        font-weight: 500;
        background: #252A30;
        color: #2DD4A8;
        padding: 0.25rem 0.625rem;
        border: 1px solid rgba(60, 74, 68, 0.15);
        letter-spacing: 0.02em;
    }
    
    /* Mini stat cards */
    .mini-stat {
        background: #171C21;
        border: 1px solid rgba(60, 74, 68, 0.1);
        padding: 0.75rem 1rem;
    }
    .mini-stat-label {
        font-size: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: #475569;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .mini-stat-value {
        font-size: 1.25rem;
        font-weight: 900;
        color: #DEE3EA;
        font-variant-numeric: tabular-nums;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# Matplotlib Sentinel Theme
# ═════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': _BG,
    'axes.facecolor': _SURFACE,
    'axes.edgecolor': _OUTLINE,
    'axes.labelcolor': _TEXT_SEC,
    'axes.titlecolor': _TEXT,
    'xtick.color': _TEXT_SEC,
    'ytick.color': _TEXT_SEC,
    'text.color': _TEXT,
    'grid.color': _OUTLINE,
    'grid.alpha': 0.2,
    'legend.facecolor': _SURFACE_HI,
    'legend.edgecolor': _OUTLINE,
    'legend.labelcolor': _TEXT,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica', 'Arial'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 700,
    'axes.labelsize': 10,
})

# Sentinel colormaps
sentinel_cmap = LinearSegmentedColormap.from_list(
    "sentinel", [_BG, _SURFACE, _PRIMARY, "#56F1C3"], N=256
)


# ═════════════════════════════════════════════════════════════════════
# DATA & HELPERS
# ═════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "dummy_data.csv")
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)


def prepare_features(df, configured_features):
    available = [f for f in configured_features if f in df.columns]
    missing = [f for f in configured_features if f not in df.columns]
    if not available:
        raise ValueError("None of the configured FEATURE_NAMES columns exist in dataset.")
    feature_df = df[available].copy()
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    non_constant_cols = feature_df.columns[feature_df.var() > 0].tolist()
    feature_df = feature_df[non_constant_cols]
    if feature_df.shape[1] == 0:
        raise ValueError("No non-constant numeric features remain after preprocessing.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values)
    return feature_df, X_scaled, scaler, missing


def compute_silhouette(X_scaled, cluster_labels, metric_name, minkowski_p):
    n_clusters = len(np.unique(cluster_labels))
    if n_clusters <= 1 or n_clusters >= len(X_scaled):
        return None
    metric = {"euclidean": "euclidean", "manhattan": "manhattan",
              "cosine": "cosine", "minkowski": "minkowski"}.get(metric_name, "euclidean")
    kwargs = {"metric": metric}
    if metric == "minkowski":
        kwargs["p"] = minkowski_p
    try:
        return silhouette_score(X_scaled, cluster_labels, **kwargs)
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR — Configuration Panel
# ═════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="margin-bottom: 1.5rem;">
    <div style="font-size: 1.1rem; font-weight: 900; letter-spacing: -0.03em; color: #2DD4A8; text-transform: uppercase;"></div>
    <div style="font-size: 0.5625rem; text-transform: uppercase; letter-spacing: 0.2em; color: #475569; font-weight: 600; margin-top: 2px;">Federated Node · Active</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.caption("DISTANCE METRIC")
distance_options = ["euclidean", "manhattan", "minkowski", "cosine"]
default_metric_idx = distance_options.index(DISTANCE_METRIC) if DISTANCE_METRIC in distance_options else 0
distance_metric = st.sidebar.selectbox(
    "Distance Metric",
    distance_options,
    index=default_metric_idx,
    help="Metric used in SCC distance comparisons.",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("MODEL HYPERPARAMETERS")

if distance_metric == "cosine":
    threshold = st.sidebar.slider(
        "Cluster Threshold",
        min_value=0.01, max_value=1.0,
        value=float(min(max(CLUSTER_THRESHOLD, 0.01), 1.0)), step=0.01,
        help="For cosine: values in [0,1]. Lower = stricter.",
    )
else:
    threshold = st.sidebar.slider(
        "Cluster Threshold",
        min_value=0.05, max_value=3.0,
        value=float(min(max(CLUSTER_THRESHOLD, 0.05), 3.0)), step=0.05,
        help="Distance threshold for cluster assignment.",
    )

anomaly_buffer = st.sidebar.slider(
    "Anomaly Buffer (λ)",
    min_value=0.1, max_value=5.0,
    value=float(min(max(ANOMALY_BUFFER, 0.1), 5.0)), step=0.1,
    help="threshold = mean + λ × std",
)

adaptive_threshold = st.sidebar.checkbox(
    "Adaptive Threshold",
    value=ADAPTIVE_THRESHOLD,
    help="Per-cluster adaptive anomaly thresholds.",
)

minkowski_p = MINKOWSKI_P
if distance_metric == "minkowski":
    minkowski_p = st.sidebar.slider(
        "Minkowski p", min_value=1, max_value=10,
        value=max(1, int(MINKOWSKI_P)), step=1,
    )

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.01, max_value=1.0,
    value=float(min(max(LEARNING_RATE, 0.01), 1.0)), step=0.01,
    help="API compat — centroid uses running mean.",
)

st.sidebar.markdown("---")
st.sidebar.caption("SYSTEM STATUS")
st.sidebar.markdown(f"""
<div style="display:flex; align-items:center; gap:0.5rem; margin-top:0.25rem;">
    <div style="width:6px; height:6px; background:#2DD4A8; animation: pulse 2s infinite;"></div>
    <span style="font-size:0.625rem; color:#64748B; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;">Model Active</span>
</div>
<style>
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="sentinel-brand">
    <div class="sentinel-brand-icon">🛡</div>
    <div>
        <div class="sentinel-brand-text">Federated DDoS Dashboard</div>
        <div class="sentinel-brand-sub">Self-Constructing Clustering · Anomaly Detection · Federated Learning</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Config summary badges
st.markdown(f"""
<div class="config-bar">
    <span class="sentinel-badge">metric: {distance_metric}</span>
    <span class="sentinel-badge">threshold: {threshold}</span>
    <span class="sentinel-badge sentinel-badge-amber">buffer (λ): {anomaly_buffer}</span>
    <span class="sentinel-badge sentinel-badge-neutral">adaptive: {'on' if adaptive_threshold else 'off'}</span>
    <span class="model-id-badge">SCC-V1.0-{distance_metric[:3].upper()}</span>
</div>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────
df = load_data()
if df is None:
    st.error("Data file not found at `data/dummy_data.csv`.")
    st.stop()

show_raw = st.checkbox("Show Raw Data Snippet", value=False)
if show_raw:
    st.dataframe(df.head())

try:
    feature_df, X_scaled, scaler, missing_features = prepare_features(df, FEATURE_NAMES)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if missing_features:
    st.warning(
        "Missing features (ignored): " + ", ".join(missing_features)
    )


# ═════════════════════════════════════════════════════════════════════
# SECTION 1: MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <span class="section-header-icon">🔬</span>
    <div>
        <h2>Model Training & Clustering</h2>
        <div class="section-header-desc">SCC model fitted on standardized feature vectors</div>
    </div>
</div>
""", unsafe_allow_html=True)

model = SelfConstructingClustering(
    threshold=threshold,
    learning_rate=learning_rate,
    distance_metric=distance_metric,
    anomaly_buffer=anomaly_buffer,
    minkowski_p=minkowski_p,
    adaptive_threshold=adaptive_threshold,
)

t_start = time.perf_counter()
with st.spinner(f"Training with {distance_metric} metric..."):
    model.fit(X_scaled)
t_train = (time.perf_counter() - t_start) * 1000  # ms

num_clusters = len(model.centroids_)
if num_clusters == 0:
    st.error("No clusters formed. Try increasing threshold or checking data quality.")
    st.stop()

all_dists = model.transform(X_scaled)
min_distances = np.min(all_dists, axis=1)
cluster_labels = np.argmin(all_dists, axis=1)
predictions = model.predict(X_scaled)
anomaly_mask = predictions.astype(bool)

sil_score = compute_silhouette(X_scaled, cluster_labels, distance_metric, minkowski_p)
anomaly_rate = 100.0 * np.mean(anomaly_mask)

# KPI row — split into 3 + 2 to avoid truncation
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Samples", f"{X_scaled.shape[0]}")
kpi2.metric("Features", f"{X_scaled.shape[1]}")
kpi3.metric("Clusters", f"{num_clusters}")

kpi4, kpi5 = st.columns(2)
kpi4.metric("Anomaly Threshold", f"{model.anomaly_threshold_:.4f}")
kpi5.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

# Secondary stats row
stat1, stat2, stat3 = st.columns(3)
with stat1:
    sil_text = f"{sil_score:.4f}" if sil_score is not None else "N/A"
    st.markdown(f"""
    <div class="mini-stat">
        <div class="mini-stat-label">Silhouette Score ({distance_metric})</div>
        <div class="mini-stat-value">{sil_text}</div>
    </div>
    """, unsafe_allow_html=True)
with stat2:
    st.markdown(f"""
    <div class="mini-stat">
        <div class="mini-stat-label">Training Latency</div>
        <div class="mini-stat-value">{t_train:.1f}ms</div>
    </div>
    """, unsafe_allow_html=True)
with stat3:
    mode_text = "ADAPTIVE (PER-CLUSTER)" if (adaptive_threshold and model.cluster_thresholds_) else "GLOBAL (UNIFIED)"
    st.markdown(f"""
    <div class="mini-stat">
        <div class="mini-stat-label">Threshold Mode</div>
        <div class="mini-stat-value" style="font-size:0.875rem;">{mode_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════
tab_centroids, tab_dist, tab_pca, tab_summary = st.tabs(
    ["Cluster Centroids", "Distance Distribution", "2D Projection (PCA)", "Cluster Summary"]
)

# ── Tab 1: Centroids ──────────────────────────────────────────────
with tab_centroids:
    centroids_df = pd.DataFrame(model.centroids_, columns=feature_df.columns)
    centroids_df.index = [f"Cluster {i}" for i in range(num_clusters)]

    st.caption("Each row is a centroid in standardized feature space (z-scores).")
    st.dataframe(centroids_df)

    fig_centroid, ax_centroid = plt.subplots(
        figsize=(max(8, len(feature_df.columns) * 0.9), max(3.0, num_clusters * 0.8))
    )
    im = ax_centroid.imshow(centroids_df.values, cmap=sentinel_cmap, aspect="auto")
    ax_centroid.set_title("Centroid Values Heatmap", pad=14)
    ax_centroid.set_xticks(np.arange(len(feature_df.columns)))
    ax_centroid.set_xticklabels(feature_df.columns, rotation=45, ha="right", fontsize=9)
    ax_centroid.set_yticks(np.arange(num_clusters))
    ax_centroid.set_yticklabels(centroids_df.index.tolist(), fontsize=9)

    if num_clusters <= 8 and len(feature_df.columns) <= 12:
        for r in range(num_clusters):
            for c in range(len(feature_df.columns)):
                val = centroids_df.values[r, c]
                txt_color = _TEXT if val < np.mean(centroids_df.values) else _BG
                ax_centroid.text(c, r, f"{val:.2f}", ha="center", va="center",
                                color=txt_color, fontsize=8, fontfamily='monospace')

    cbar = fig_centroid.colorbar(im, ax=ax_centroid)
    cbar.set_label("z-score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig_centroid.tight_layout()
    st.pyplot(fig_centroid)

    st.caption("Dark → low z-score · Teal → medium · Bright → high z-score")

# ── Tab 2: Distance Distribution ──────────────────────────────────
with tab_dist:
    normal_dists = min_distances[~anomaly_mask]
    attack_dists = min_distances[anomaly_mask]

    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    ax_hist.hist(
        [normal_dists, attack_dists],
        bins=20, stacked=True,
        color=[_PRIMARY, _DANGER],
        edgecolor=_SURFACE_TOP, linewidth=0.5, alpha=0.85,
        label=["Predicted Normal", "Predicted Attack"],
    )
    ax_hist.axvline(
        model.anomaly_threshold_,
        color=_SECONDARY, linestyle="dashed", linewidth=2,
    )
    ax_hist.set_title("Distance to Nearest Cluster", pad=14)
    ax_hist.set_xlabel("Distance")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(True, axis='y', alpha=0.12)

    legend_handles = [
        Patch(facecolor=_PRIMARY, edgecolor=_SURFACE_TOP, label="Predicted Normal"),
        Patch(facecolor=_DANGER, edgecolor=_SURFACE_TOP, label="Predicted Attack"),
        Line2D([0], [0], color=_SECONDARY, linestyle="dashed", linewidth=2,
               label=f"Threshold = {model.anomaly_threshold_:.4f}"),
    ]
    ax_hist.legend(handles=legend_handles, framealpha=0.85, fontsize=9)
    fig_hist.tight_layout()
    st.pyplot(fig_hist)

    st.caption("Teal = normal · Red = attack · Amber dashed = anomaly threshold")

    # Stats cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f'<div class="mini-stat"><div class="mini-stat-label">Min Distance</div>'
                    f'<div class="mini-stat-value">{float(np.min(min_distances)):.4f}</div></div>',
                    unsafe_allow_html=True)
    with sc2:
        st.markdown(f'<div class="mini-stat"><div class="mini-stat-label">Mean Distance</div>'
                    f'<div class="mini-stat-value">{float(np.mean(min_distances)):.4f}</div></div>',
                    unsafe_allow_html=True)
    with sc3:
        st.markdown(f'<div class="mini-stat"><div class="mini-stat-label">Max Distance</div>'
                    f'<div class="mini-stat-value">{float(np.max(min_distances)):.4f}</div></div>',
                    unsafe_allow_html=True)
    with sc4:
        st.markdown(f'<div class="mini-stat"><div class="mini-stat-label">Anomaly Threshold</div>'
                    f'<div class="mini-stat-value" style="color:#F59E0B;">{float(model.anomaly_threshold_):.4f}</div></div>',
                    unsafe_allow_html=True)

# ── Tab 3: PCA Projection ─────────────────────────────────────────
with tab_pca:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    cluster_palette = [_PRIMARY, _SUCCESS, _SLATE, "#78716C", _PRIMARY_DIM, "#94A3B8", "#A3E635", "#FBBF24"]
    cluster_colors_list = [cluster_palette[i % len(cluster_palette)] for i in range(num_clusters)]

    fig_pca, ax_pca = plt.subplots(figsize=(10, 6))

    for k in range(num_clusters):
        mask_k = cluster_labels == k
        ax_pca.scatter(
            X_pca[mask_k, 0], X_pca[mask_k, 1],
            c=cluster_colors_list[k], alpha=0.7, s=50,
            label=f"Cluster {k}", edgecolors='none',
        )

    if np.any(anomaly_mask):
        ax_pca.scatter(
            X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
            facecolors="none", edgecolors=_DANGER,
            s=120, linewidths=1.5, label="Predicted Attack",
        )

    centroids_pca = pca.transform(np.array(model.centroids_))
    ax_pca.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        c="white", marker="X", s=200,
        label="Centroids", edgecolors=_BG,
        linewidths=1.0, zorder=10,
    )
    for i, (cx, cy) in enumerate(centroids_pca):
        ax_pca.annotate(f"C{i}", (cx, cy), textcoords="offset points",
                        xytext=(6, 6), fontsize=10, color=_TEXT, fontweight=700)

    ax_pca.set_title("Data Projected to 2D PCA (Color = Cluster ID)", pad=14)
    ax_pca.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax_pca.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    ax_pca.legend(loc="best", framealpha=0.85, fontsize=9)
    ax_pca.grid(True, alpha=0.08)
    fig_pca.tight_layout()
    st.pyplot(fig_pca)

    cluster_counts = np.bincount(cluster_labels, minlength=num_clusters)
    cluster_anomalies = np.bincount(cluster_labels[anomaly_mask], minlength=num_clusters)
    cluster_color_index_df = pd.DataFrame({
        "Cluster ID": [f"Cluster {i}" for i in range(num_clusters)],
        "Color": [cluster_colors_list[i] for i in range(num_clusters)],
        "Samples": cluster_counts,
        "Predicted Attacks": cluster_anomalies,
    })
    st.caption("Cluster legend:")
    st.dataframe(cluster_color_index_df, hide_index=True)

# ── Tab 4: Cluster Summary ────────────────────────────────────────
with tab_summary:
    cluster_rows = []
    for k in range(num_clusters):
        mask = cluster_labels == k
        dists_k = all_dists[mask, k]
        attacks_k = int(np.sum(anomaly_mask[mask]))
        threshold_k = float(model.cluster_thresholds_[k]) if (adaptive_threshold and model.cluster_thresholds_) else float(model.anomaly_threshold_)
        cluster_rows.append({
            "Cluster ID": f"Cluster {k}",
            "Samples": int(np.sum(mask)),
            "Mean Distance": float(np.mean(dists_k)) if np.any(mask) else np.nan,
            "Std Distance": float(np.std(dists_k)) if np.any(mask) else np.nan,
            "Threshold": threshold_k,
            "Predicted Attacks": attacks_k,
        })
    summary_df = pd.DataFrame(cluster_rows)
    st.dataframe(summary_df, hide_index=True)


# ═════════════════════════════════════════════════════════════════════
# REAL-TIME INFERENCE PLAYGROUND
# ═════════════════════════════════════════════════════════════════════
st.divider()

st.markdown("""
<div class="section-header" style="border-left-color: #F59E0B;">
    <span class="section-header-icon">⚡</span>
    <div>
        <h2>Real-time Inference Playground</h2>
        <div class="section-header-desc">Adjust feature values and run prediction against the current model</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")  # spacing

# Feature sliders in 3 columns
input_features = []
input_cols = st.columns(3)
for i, feature in enumerate(feature_df.columns):
    with input_cols[i % 3]:
        col_min = float(feature_df[feature].min())
        col_max = float(feature_df[feature].max())
        col_median = float(feature_df[feature].median())
        range_pad = max((col_max - col_min) * 0.3, 1e-6)
        val = st.slider(
            feature,
            min_value=float(col_min - range_pad),
            max_value=float(col_max + range_pad),
            value=col_median,
        )
        input_features.append(val)

if st.button("⚡  PREDICT VECTOR"):
    t_inf_start = time.perf_counter()
    input_arr = np.array([input_features], dtype=float)
    input_scaled = scaler.transform(input_arr)

    dists = model.transform(input_scaled)[0]
    nearest_cluster = int(np.argmin(dists))
    min_dist = float(np.min(dists))
    t_inf = (time.perf_counter() - t_inf_start) * 1000

    threshold_used = float(model.cluster_thresholds_[nearest_cluster]) if (adaptive_threshold and model.cluster_thresholds_) else float(model.anomaly_threshold_)

    prediction = int(min_dist > threshold_used)
    result = "ATTACK" if prediction == 1 else "NORMAL"
    ratio = min_dist / threshold_used if threshold_used > 0 else np.inf

    # Result + Chart side by side
    res_col, chart_col = st.columns([1, 1])

    with res_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-card-attack">
                <div class="result-label" style="color:#EF4444;">⚠ ANOMALY DETECTED</div>
                <div class="result-title">Prediction: ATTACK</div>
                <div class="result-details">
                    NEAREST CLUSTER: <span class="danger">Cluster {nearest_cluster}</span><br>
                    DISTANCE: <span class="val">{min_dist:.4f}</span> &gt; THRESHOLD: <span class="warn">{threshold_used:.4f}</span>
                </div>
                <div class="result-ratio" style="color:#EF4444;">
                    📊 Input is {ratio:.1f}× above the anomaly threshold
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-card-normal">
                <div class="result-label" style="color:#2DD4A8;">✓ TRAFFIC NORMAL</div>
                <div class="result-title">Prediction: NORMAL</div>
                <div class="result-details">
                    NEAREST CLUSTER: <span class="safe">Cluster {nearest_cluster}</span><br>
                    DISTANCE: <span class="val">{min_dist:.4f}</span> ≤ THRESHOLD: <span class="warn">{threshold_used:.4f}</span>
                </div>
                <div class="result-ratio" style="color:#2DD4A8;">
                    📊 Distance-to-threshold ratio: {ratio:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Mini stats under result
        ms1, ms2 = st.columns(2)
        with ms1:
            confidence = max(0, min(100, (1 - min_dist / (threshold_used * 2)) * 100)) if prediction == 0 else max(0, min(100, (ratio / (ratio + 1)) * 100))
            st.markdown(f"""
            <div class="mini-stat">
                <div class="mini-stat-label">Confidence Score</div>
                <div class="mini-stat-value">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with ms2:
            st.markdown(f"""
            <div class="mini-stat">
                <div class="mini-stat-label">Inference Time</div>
                <div class="mini-stat-value">{t_inf:.2f}ms</div>
            </div>
            """, unsafe_allow_html=True)

    with chart_col:
        fig_input, ax_input = plt.subplots(figsize=(8, 5))
        dist_df = pd.DataFrame({
            "Cluster ID": [f"C{i}" for i in range(num_clusters)],
            "Distance": dists,
        }).sort_values("Distance")

        bar_colors = [_PRIMARY if d <= threshold_used else _DANGER for d in dist_df["Distance"]]
        bars = ax_input.bar(
            dist_df["Cluster ID"], dist_df["Distance"],
            color=bar_colors, edgecolor=_SURFACE_TOP, linewidth=0.5, width=0.6,
        )
        ax_input.axhline(
            threshold_used, color=_SECONDARY,
            linestyle="dashed", linewidth=2, label=f"Threshold: {threshold_used:.4f}"
        )
        ax_input.set_title("Input Sample: Distance to Each Cluster", pad=14)
        ax_input.set_ylabel("Distance")
        ax_input.set_xlabel("Cluster")
        ax_input.legend(framealpha=0.85, fontsize=9)
        ax_input.grid(True, axis='y', alpha=0.12)
        fig_input.tight_layout()
        st.pyplot(fig_input)

    # Full distance table
    st.markdown("")
    full_dist_df = pd.DataFrame({
        "Cluster ID": [f"CLUSTER_{i:02d}" for i in range(num_clusters)],
        "Distance": [f"{d:.4f}" for d in dists],
        "Threshold": [f"{threshold_used:.4f}"] * num_clusters,
        "Status": ["MATCH_TARGET" if i == nearest_cluster else ("IN_BOUNDS" if dists[i] <= threshold_used else "OUT_OF_BOUNDS") for i in range(num_clusters)],
    })
    st.dataframe(full_dist_df, hide_index=True)
