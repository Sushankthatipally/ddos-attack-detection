"""
client/clustering.py

CLAPP: Self-Constructing Feature CLustering APProach for Anomaly Detection.
Implements the fuzzy Gaussian membership function from the paper (Eqs. 12–19)
AND supports alternative distance-based similarity metrics:
    - fuzzy_gaussian  (paper's native method, operates on posterior probability vectors)
    - euclidean       (Gaussian-kernel conversion: sim = exp(-d² / 2σ²))
    - manhattan       (Laplace-kernel conversion: sim = exp(-d / σ))
    - minkowski       (generalised Gaussian: sim = exp(-(d^p)^(2/p) / 2σ²))
    - cosine          (linear rescale: sim = (1 + cosine_similarity) / 2)

All metrics plug into the SAME self-constructing cluster loop so you can
benchmark them head-to-head on the same dataset.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fuzzy_gaussian_sim(vi: np.ndarray, vj: np.ndarray, sigma: float) -> float:
    """
    Generalised fuzzy membership between two pattern vectors (Eq. 13 / 17).
    Each dimension d contributes:
        μ_d = 0.5 * (1 + exp( -((vi[d]-vj[d]) / sigma)^2 ))
    Overall membership = product over all d  (Eq. 14/15).
    Result is always in (0, 1].
    """
    mu = 1.0
    for vi_d, vj_d in zip(vi, vj):
        mu *= 0.5 * (1.0 + np.exp(-((vi_d - vj_d) / sigma) ** 2))
    return float(mu)


def _euclidean_sim(vi: np.ndarray, vj: np.ndarray, sigma: float) -> float:
    """Gaussian-kernel similarity: sim = exp(-||vi-vj||² / (2σ²))."""
    d = np.linalg.norm(vi - vj)
    return float(np.exp(-(d ** 2) / (2 * sigma ** 2)))


def _manhattan_sim(vi: np.ndarray, vj: np.ndarray, sigma: float) -> float:
    """Laplace-kernel similarity: sim = exp(-||vi-vj||₁ / σ)."""
    d = np.sum(np.abs(vi - vj))
    return float(np.exp(-d / sigma))


def _minkowski_sim(vi: np.ndarray, vj: np.ndarray, sigma: float, p: int = 3) -> float:
    """
    Minkowski-kernel similarity:
        d_p = (Σ|vi-vj|^p)^(1/p)
        sim = exp(-d_p² / (2σ²))
    """
    d = np.sum(np.abs(vi - vj) ** p) ** (1.0 / p)
    return float(np.exp(-(d ** 2) / (2 * sigma ** 2)))


def _cosine_sim(vi: np.ndarray, vj: np.ndarray, sigma: float = 0.5, **_) -> float:
    """
    Cosine similarity rescaled to (0, 1]:
        sim = (1 + cosine_similarity(vi, vj)) / 2
    Pure cosine (vi·vj)/(||vi||·||vj||) ∈ [-1,1], rescaled to [0,1].
    sigma is accepted but unused (API consistency).
    """
    norm_i = np.linalg.norm(vi)
    norm_j = np.linalg.norm(vj)
    if norm_i == 0 or norm_j == 0:
        return 0.0
    raw = float(np.dot(vi, vj) / (norm_i * norm_j))
    return (1.0 + raw) / 2.0


_METRIC_FN = {
    "fuzzy_gaussian": _fuzzy_gaussian_sim,
    "euclidean":      _euclidean_sim,
    "manhattan":      _manhattan_sim,
    "minkowski":      _minkowski_sim,
    "cosine":         _cosine_sim,
}

SUPPORTED_METRICS = list(_METRIC_FN.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class CLAPPClustering:
    """
    CLAPP unified clustering with pluggable similarity metrics.

    Parameters
    ----------
    threshold : float
        Minimum similarity for a pattern to join an existing cluster.
        Paper uses ~0.9999 for NSL-KDD; 0.75-0.90 works for small datasets.
    sigma : float
        σ_c — spread parameter used by all kernel-based metrics.
        Paper Examples 3/4 use 0.5.
    similarity_metric : str
        One of: 'fuzzy_gaussian', 'euclidean', 'manhattan', 'minkowski', 'cosine'.
        'fuzzy_gaussian' is the paper's native method and requires labeled data.
        All others can be used on raw/scaled feature vectors (labels optional).
    minkowski_p : int
        Exponent for Minkowski metric (ignored for other metrics). Default 3.
    anomaly_buffer : float
        λ in anomaly_threshold = mean_dist + λ × std_dist.
    adaptive_threshold : bool
        If True, compute a per-cluster anomaly threshold instead of a global one.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        sigma: float = 0.5,
        similarity_metric: str = "fuzzy_gaussian",
        minkowski_p: int = 3,
        anomaly_buffer: float = 1.5,
        adaptive_threshold: bool = False,
    ):
        if similarity_metric not in _METRIC_FN:
            raise ValueError(
                f"Unknown metric '{similarity_metric}'. "
                f"Choose from: {SUPPORTED_METRICS}"
            )
        self.threshold = threshold
        self.sigma = sigma
        self.similarity_metric = similarity_metric
        self.minkowski_p = minkowski_p
        self.anomaly_buffer = anomaly_buffer
        self.adaptive_threshold = adaptive_threshold

        # Set at fit time
        self.pattern_vectors_: Optional[np.ndarray] = None   # (n_features, n_classes)
        self.cluster_means_: List[np.ndarray] = []           # (n_clusters, n_classes)
        self.clusters_: List[np.ndarray] = self.cluster_means_
        self.cluster_deviations_: List[float] = []           # per-cluster σ
        self.transformation_matrix_: Optional[np.ndarray] = None  # (n_features, n_clusters)
        self.reduced_PS_: Optional[np.ndarray] = None        # (n_samples, n_clusters)
        self.anomaly_threshold_: float = 0.0
        self.cluster_thresholds_: List[float] = []
        self._classes_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self._centroid_matrix_: Optional[np.ndarray] = None
        self.cluster_feature_indices_: List[List[int]] = []

    # ─────────────────────────────────────────────────────────────────
    # Posterior probability pattern vectors (Eq. 4 in paper)
    # ─────────────────────────────────────────────────────────────────

    def _compute_pattern_vectors(
        self, PS: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Build system-call pattern matrix C of shape (n_features, n_classes).

        C[j, d] = Σ_i PS[i,j] * Md(i) / Σ_i PS[i,j]          (Eq. 4)

        where Md(i) = 1 if sample i belongs to class d, else 0.  (Eq. 5)
        """
        n_samples, n_features = PS.shape
        classes = np.unique(labels)
        self._classes_ = classes
        self.classes_ = classes
        n_classes = len(classes)
        C = np.zeros((n_features, n_classes))

        for j in range(n_features):
            col_sum = float(np.sum(PS[:, j]))
            if col_sum == 0.0:
                # uniform if feature is all-zero → contributes equally to all classes
                C[j] = np.ones(n_classes) / n_classes
                continue
            for d_idx, d in enumerate(classes):
                Md = (labels == d).astype(float)          # Eq. 5
                C[j, d_idx] = np.sum(PS[:, j] * Md) / col_sum   # Eq. 4

        return C  # shape (n_features, n_classes)

    def compute_pattern_vectors(self, PS: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Backward-compatible public alias for the paper Eq. 4 computation."""
        return self._compute_pattern_vectors(PS, labels)

    # ─────────────────────────────────────────────────────────────────
    # Similarity dispatch
    # ─────────────────────────────────────────────────────────────────

    def _sim(self, vi: np.ndarray, vj: np.ndarray, sigma: Optional[float] = None) -> float:
        """Compute similarity between two vectors using the chosen metric."""
        s = sigma if sigma is not None else self.sigma
        fn = _METRIC_FN[self.similarity_metric]
        if self.similarity_metric == "minkowski":
            return fn(vi, vj, s, self.minkowski_p)
        return fn(vi, vj, s)

    def membership(self, ci: np.ndarray, cj: np.ndarray, sigma: Optional[float] = None) -> float:
        """Backward-compatible fuzzy Gaussian membership helper."""
        s = sigma if sigma is not None else self.sigma
        return _fuzzy_gaussian_sim(np.asarray(ci, dtype=float), np.asarray(cj, dtype=float), s)

    # ─────────────────────────────────────────────────────────────────
    # Self-constructing cluster loop (Steps 2-4 in paper)
    # ─────────────────────────────────────────────────────────────────

    def _build_clusters(
        self, vectors: np.ndarray
    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray], List[List[int]]]:
        """
        Iterate over every pattern/feature vector.
        Assign to best matching cluster if similarity ≥ threshold,
        else create a new cluster.

        Returns
        -------
        clusters      : list of lists — each inner list contains the vectors in that cluster
        cluster_means : list of running mean vectors (one per cluster)
        """
        clusters: List[List[np.ndarray]] = []
        cluster_means: List[np.ndarray] = []
        cluster_indices: List[List[int]] = []

        for idx, vec in enumerate(vectors):
            if not clusters:
                # Step 2: first cluster
                clusters.append([vec])
                cluster_means.append(vec.copy())
                cluster_indices.append([idx])
                continue

            # Compute similarity to every existing cluster mean
            sims = [self._sim(vec, mean) for mean in cluster_means]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]

            if best_sim >= self.threshold:
                # Step 3: add to best cluster
                clusters[best_idx].append(vec)
                cluster_indices[best_idx].append(idx)
                # Step 4: update running mean
                cluster_means[best_idx] = np.mean(clusters[best_idx], axis=0)
            else:
                # Step 3: new cluster
                clusters.append([vec])
                cluster_means.append(vec.copy())
                cluster_indices.append([idx])

        return clusters, cluster_means, cluster_indices

    # ─────────────────────────────────────────────────────────────────
    # Transformation matrix [S × G] (Steps 5-6 in paper)
    # ─────────────────────────────────────────────────────────────────

    def _build_transformation_matrix(
        self,
        vectors: np.ndarray,
        final_means: List[np.ndarray],
        final_stds: List[float],
    ) -> np.ndarray:
        """
        Compute soft [S × G] matrix where entry [j, g] is the fuzzy
        similarity of the j-th pattern vector to the g-th cluster (Eq. 18-19).
        Uses the per-cluster updated deviation (s_u) as the sigma.
        """
        n_features = len(vectors)
        n_clusters = len(final_means)
        T = np.zeros((n_features, n_clusters))
        for j, vec in enumerate(vectors):
            for g, (mean, std) in enumerate(zip(final_means, final_stds)):
                if np.allclose(vec, mean):
                    T[j, g] = 1.0                     # Eq. 18 exact match branch
                else:
                    T[j, g] = self._sim(vec, mean, sigma=std)
        return T  # shape (n_features, n_clusters)

    # ─────────────────────────────────────────────────────────────────
    # Anomaly threshold calibration
    # ─────────────────────────────────────────────────────────────────

    def _calibrate_anomaly_threshold(self, reduced: np.ndarray, cluster_labels: np.ndarray):
        """
        Global threshold: mean + λ × std of minimum distances to assigned cluster.
        Per-cluster threshold: same but computed within each cluster's members.
        We convert cluster similarity to 'distance' = 1 - similarity for thresholding.
        """
        # Similarity of each sample to its assigned cluster (diagonal of soft match)
        n_clusters = len(self.cluster_means_)
        sim_to_assigned = np.zeros(len(reduced))
        for i, label in enumerate(cluster_labels):
            # Recompute similarity of this reduced sample to its cluster mean
            # In reduced space the centroid is just the mean of reduced vectors
            pass  # done via min_distances below

        # Compute centroids in reduced [P×G] space (only for represented clusters)
        unique_gs = sorted(set(cluster_labels.tolist()))
        centroid_map = {
            g: np.mean(reduced[cluster_labels == g], axis=0)
            for g in unique_gs
        }
        # Map cluster_labels to a contiguous index for distance computation
        label_to_idx = {g: idx for idx, g in enumerate(unique_gs)}
        centroids_reduced = np.array([centroid_map[g] for g in unique_gs])
        remapped_labels = np.array([label_to_idx[g] for g in cluster_labels])

        # Distance = Euclidean in the reduced [P×G] space for threshold calibration
        dists = np.array([
            np.linalg.norm(reduced[i] - centroids_reduced[remapped_labels[i]])
            for i in range(len(reduced))
        ])

        mean_d = float(np.mean(dists))
        std_d = float(np.std(dists))
        self.anomaly_threshold_ = mean_d + self.anomaly_buffer * std_d

        if self.adaptive_threshold:
            self.cluster_thresholds_ = []
            for g in unique_gs:
                mask = cluster_labels == g
                if np.sum(mask) < 2:
                    self.cluster_thresholds_.append(self.anomaly_threshold_)
                    continue
                d_g = dists[mask]
                self.cluster_thresholds_.append(
                    float(np.mean(d_g) + self.anomaly_buffer * np.std(d_g))
                )

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def fit(self, PS: np.ndarray, labels: Optional[np.ndarray] = None) -> "CLAPPClustering":
        """
        Fit the CLAPP model.

        For 'fuzzy_gaussian' metric:
            - labels are REQUIRED (needed to compute posterior probabilities).
            - Clustering is done in posterior probability space (n_features × n_classes).

        For all other metrics:
            - labels are OPTIONAL (used only for anomaly threshold calibration).
            - Clustering is done directly in feature space (n_samples × n_features).
            - Each ROW (sample) is treated as the vector to cluster.

        Parameters
        ----------
        PS     : np.ndarray  shape (n_samples, n_features)
        labels : np.ndarray  shape (n_samples,)  — integer class labels
        """
        PS = np.asarray(PS, dtype=float)

        if self.similarity_metric == "fuzzy_gaussian":
            if labels is None:
                raise ValueError(
                    "'fuzzy_gaussian' requires labels to compute posterior probabilities."
                )
            # ── Paper path ──────────────────────────────────────────
            # Step 1: compute pattern vectors C[j] for each feature j
            self.pattern_vectors_ = self._compute_pattern_vectors(PS, labels)

            # Steps 2-4: cluster pattern vectors (n_features vectors of size n_classes)
            clusters, _, cluster_indices = self._build_clusters(self.pattern_vectors_)

            # Step 4 update: final mean and updated deviation per cluster
            final_means: List[np.ndarray] = []
            final_stds: List[float] = []
            for cluster_pvs in clusters:
                arr = np.array(cluster_pvs)
                final_means.append(np.mean(arr, axis=0))
                # updated deviation = initial sigma + std of cluster members
                final_stds.append(self.sigma + float(np.std(arr)))

            self.cluster_means_ = final_means
            self.clusters_ = self.cluster_means_
            self.cluster_feature_indices_ = cluster_indices
            self.cluster_deviations_ = final_stds

            # Steps 5-6: soft transformation matrix [S × G]
            self.transformation_matrix_ = self._build_transformation_matrix(
                self.pattern_vectors_, final_means, final_stds
            )

            # Step 7: reduced matrix [P × G] = PS @ T
            self.reduced_PS_ = PS @ self.transformation_matrix_

        else:
            # ── Distance-metric path ─────────────────────────────────
            # Cluster ROWS (samples) in feature space directly.
            # Each "vector" is a sample row from PS.
            clusters, _, cluster_indices = self._build_clusters(PS)

            final_means: List[np.ndarray] = []
            final_stds: List[float] = []
            for cluster_rows in clusters:
                arr = np.array(cluster_rows)
                final_means.append(np.mean(arr, axis=0))          # (n_features,)
                # scalar std across all elements of this cluster
                final_stds.append(self.sigma + float(np.mean(np.std(arr, axis=0))))

            self.cluster_means_ = final_means
            self.clusters_ = self.cluster_means_
            self.cluster_feature_indices_ = cluster_indices
            self.cluster_deviations_ = final_stds
            # Store raw centroid matrix (n_clusters, n_features) for transform()
            self._centroid_matrix_ = np.array(final_means)

            # Soft [n_samples × n_clusters] similarity matrix
            n_clusters = len(final_means)
            T = np.zeros((len(PS), n_clusters))
            for i, row in enumerate(PS):
                for g, (mean, std) in enumerate(zip(final_means, final_stds)):
                    T[i, g] = self._sim(row, mean, sigma=std)

            # For distance metrics transformation_matrix_ is NOT used for dot-product.
            # We store it only for shape inspection; reduced_PS_ is the actual output.
            self.transformation_matrix_ = T   # (n_samples, n_clusters)
            self.reduced_PS_ = T

        # ── Anomaly threshold ────────────────────────────────────────
        reduced = self.reduced_PS_   # (n_samples, n_clusters)
        # Cluster assignment = column with highest similarity in the soft matrix
        cluster_labels = np.argmax(reduced, axis=1).astype(int)
        self._calibrate_anomaly_threshold(reduced, cluster_labels)

        return self

    def transform(self, PS: np.ndarray) -> np.ndarray:
        """
        Project new samples into the reduced cluster space.

        For fuzzy_gaussian: uses pre-computed pattern vectors → [P×G] via matrix multiply.
        For distance metrics: computes similarity of each new sample to each cluster centroid.
        """
        PS = np.asarray(PS, dtype=float)

        if self.similarity_metric == "fuzzy_gaussian":
            # Paper path: [n_new × n_features] @ [n_features × n_clusters]
            return PS @ self.transformation_matrix_
        else:
            # Distance path: similarity of each new sample to each centroid (feature space)
            centroids = self._centroid_matrix_   # (n_clusters, n_features)
            n_clusters = len(centroids)
            T = np.zeros((len(PS), n_clusters))
            for i, row in enumerate(PS):
                for g in range(n_clusters):
                    T[i, g] = self._sim(row, centroids[g], sigma=self.cluster_deviations_[g])
            return T

    def fit_transform(self, PS: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit CLAPP and return the reduced process-cluster matrix."""
        return self.fit(PS, labels).transform(PS)

    def predict(self, PS: np.ndarray) -> np.ndarray:
        """
        Predict anomaly (1) or normal (0) for each sample.
        Uses distance in the reduced [P×G] space against the anomaly threshold.
        """
        reduced = self.transform(PS)
        n_clusters = len(self.cluster_means_)

        # Centroids in reduced space (mean of training reduced vectors per cluster)
        train_reduced = self.reduced_PS_
        train_cluster_labels = np.argmax(train_reduced, axis=1)
        unique_gs = sorted(set(train_cluster_labels.tolist()))
        centroids_reduced = np.array([
            np.mean(train_reduced[train_cluster_labels == g], axis=0)
            if np.any(train_cluster_labels == g)
            else np.zeros(train_reduced.shape[1])
            for g in unique_gs
        ])
        n_clusters = len(centroids_reduced)

        predictions = np.zeros(len(reduced), dtype=int)
        for i, row in enumerate(reduced):
            dists = np.linalg.norm(centroids_reduced - row, axis=1)
            nearest_g = int(np.argmin(dists))
            min_dist = float(dists[nearest_g])

            if self.adaptive_threshold and self.cluster_thresholds_:
                threshold = self.cluster_thresholds_[nearest_g]
            else:
                threshold = self.anomaly_threshold_

            predictions[i] = 1 if min_dist > threshold else 0

        return predictions

    def get_cluster_info(self) -> dict:
        """Return a summary dict of cluster statistics."""
        return {
            "n_clusters": len(self.cluster_means_),
            "cluster_deviations": self.cluster_deviations_,
            "anomaly_threshold": self.anomaly_threshold_,
            "cluster_thresholds": self.cluster_thresholds_ if self.adaptive_threshold else [],
            "similarity_metric": self.similarity_metric,
            "sigma": self.sigma,
            "threshold": self.threshold,
            "reduced_dims": (
                self.transformation_matrix_.shape[1]
                if self.transformation_matrix_ is not None
                else None
            ),
        }
