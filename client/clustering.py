import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import cdist

class SelfConstructingClustering(BaseEstimator, ClusterMixin):
    def __init__(self, threshold=0.5, learning_rate=0.1, distance_metric='euclidean', anomaly_buffer=3.0, minkowski_p=3, adaptive_threshold=False):
        """
        Self-Constructing Clustering (SCC) implementation compatible with Scikit-Learn.
        
        Parameters:
        - threshold: Distance threshold (theta) to create a new cluster.
        - learning_rate: Unused in exact mean update, kept for API compatibility.
        - distance_metric: 'euclidean', 'cosine', 'manhattan' (mapped to cityblock), etc.
        - anomaly_buffer: Lambda for statistical threshold (mean + lambda * std). Default 3.0.
        - minkowski_p: The p parameter for Minkowski distance (p>=1). Default 3.
        - adaptive_threshold: If True, use per-cluster thresholds (Option 3).
        """
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.distance_metric = distance_metric
        self.anomaly_buffer = anomaly_buffer
        self.minkowski_p = minkowski_p
        self.adaptive_threshold = adaptive_threshold
        
        self.centroids_ = [] 
        self.cluster_counts_ = [] # n_k: Number of samples in each cluster
        self.anomaly_threshold_ = None
        self.cluster_thresholds_ = [] # theta_k: Per-cluster thresholds (Option 3)
        self._metric_map = {
            'manhattan': 'cityblock',
            'euclidean': 'euclidean',
            'cosine': 'cosine',
            'minkowski': 'minkowski'
        }

    def _get_scipy_metric(self):
        return self._metric_map.get(self.distance_metric.lower(), self.distance_metric)

    def _cdist(self, XA, XB):
        """Wrapper around cdist that passes the p parameter for Minkowski."""
        metric = self._get_scipy_metric()
        if metric == 'minkowski':
            return cdist(XA, XB, metric=metric, p=self.minkowski_p)
        return cdist(XA, XB, metric=metric)

    def fit(self, X, y=None):
        """
        Fit the model to the data X using online learning with exact mean update.
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Reset centroids and counts
        self.centroids_ = []
        self.cluster_counts_ = []
        self.cluster_thresholds_ = []

        # Online Learning Phase
        for i in range(n_samples):
            x = X[i].reshape(1, -1)
            
            if not self.centroids_:
                self.centroids_.append(x[0].copy())
                self.cluster_counts_.append(1)
                continue
                
            # Step 1: Compute distances to all existing centroids
            # D_k = D(x, mu_k) for all k in {1, ..., C}
            dists = self._cdist(x, np.array(self.centroids_))[0]
            
            # Step 2: Find closest cluster
            # k* = argmin_k D_k,  D_min = D_{k*}
            min_dist_idx = np.argmin(dists)
            min_dist = dists[min_dist_idx]
            
            # Step 4: Threshold-based decision rule
            # If D_min <= theta => assign to cluster k*
            # If D_min >  theta => create new cluster
            if min_dist <= self.threshold:
                # Assign to nearest cluster & Update centroid (Exact Mean)
                # Formula: mu_k_new = (n_k * mu_k_old + x) / (n_k + 1)
                n_k = self.cluster_counts_[min_dist_idx]
                old_centroid = self.centroids_[min_dist_idx]
                
                new_centroid = (n_k * old_centroid + x[0]) / (n_k + 1)
                
                self.centroids_[min_dist_idx] = new_centroid
                self.cluster_counts_[min_dist_idx] += 1
            else:
                # Create new cluster
                self.centroids_.append(x[0].copy())
                self.cluster_counts_.append(1)
        
        # Anomaly Threshold Calculation
        if self.centroids_:
            centroids_arr = np.array(self.centroids_)
            all_dists = self._cdist(X, centroids_arr)
            min_dists = np.min(all_dists, axis=1)
            
            # Option 2: Global statistical threshold
            # theta = mu_D + lambda * sigma_D
            mu_D = np.mean(min_dists)
            sigma_D = np.std(min_dists)
            self.anomaly_threshold_ = mu_D + self.anomaly_buffer * sigma_D

            # Option 3: Cluster-adaptive thresholds
            # theta_k = mu_{D_k} + lambda * sigma_{D_k}  for each cluster k
            cluster_assignments = np.argmin(all_dists, axis=1)
            self.cluster_thresholds_ = []
            for k in range(len(self.centroids_)):
                mask = cluster_assignments == k
                if np.any(mask):
                    dists_k = all_dists[mask, k]
                    mu_Dk = np.mean(dists_k)
                    sigma_Dk = np.std(dists_k)
                    self.cluster_thresholds_.append(mu_Dk + self.anomaly_buffer * sigma_Dk)
                else:
                    self.cluster_thresholds_.append(self.anomaly_threshold_)
        else:
            self.anomaly_threshold_ = 0.0
            self.cluster_thresholds_ = []

        return self

    def predict(self, X):
        """
        Predict if samples are Normal (0) or Attack (1) based on anomaly threshold.
        Supports both global threshold (Option 2) and cluster-adaptive threshold (Option 3).
        """
        X = np.array(X)
        if not self.centroids_:
            return np.ones(X.shape[0]) # All attacks if no clusters

        centroids_arr = np.array(self.centroids_)
        dists = self._cdist(X, centroids_arr)

        if self.adaptive_threshold and self.cluster_thresholds_:
            # Option 3: Cluster-adaptive threshold
            # Decision: D_k <= theta_k  for nearest cluster k
            nearest_cluster = np.argmin(dists, axis=1)
            min_dists = np.min(dists, axis=1)
            thresholds = np.array(self.cluster_thresholds_)
            per_sample_threshold = thresholds[nearest_cluster]
            return (min_dists > per_sample_threshold).astype(int)
        else:
            # Option 2: Global statistical threshold
            min_dists = np.min(dists, axis=1)
            return (min_dists > self.anomaly_threshold_).astype(int)

    def transform(self, X):
        """
        Transform X to cluster-distance space.
        z_i = [D(x_i, mu_1), D(x_i, mu_2), ..., D(x_i, mu_C)]
        Result: Z in R^{N x C}
        """
        X = np.array(X)
        if not self.centroids_:
            return np.zeros((X.shape[0], 0))
            
        centroids_arr = np.array(self.centroids_)
        return self._cdist(X, centroids_arr)
