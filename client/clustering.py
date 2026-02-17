import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import cdist
from client.threshold import compute_distance_threshold

class SelfConstructingClustering(BaseEstimator, ClusterMixin):
    def __init__(self, threshold=0.5, learning_rate=0.1, distance_metric='euclidean', anomaly_buffer=1.1):
        """
        Self-Constructing Clustering (SCC) implementation compatible with Scikit-Learn.
        
        Parameters:
        - threshold: Distance threshold to create a new cluster.
        - learning_rate: Rate at which centroids update towards input.
        - distance_metric: 'euclidean', 'cosine', 'cityblock' (manhattan), etc.
        - anomaly_buffer: Multiplier for anomaly detection threshold (e.g. 1.1 = 10% buffer).
        """
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.distance_metric = distance_metric
        self.anomaly_buffer = anomaly_buffer
        
        self.centroids_ = [] 
        self.anomaly_threshold_ = None

    def fit(self, X, y=None):
        """
        Fit the model to the data X.
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Reset centroids
        self.centroids_ = []

        # Online Learning Phase
        for i in range(n_samples):
            x = X[i].reshape(1, -1)
            
            if not self.centroids_:
                self.centroids_.append(x[0].copy())
                continue
                
            # Compute distances to all existing centroids
            dists = cdist(x, np.array(self.centroids_), metric=self.distance_metric)[0]
            min_dist_idx = np.argmin(dists)
            min_dist = dists[min_dist_idx]
            
            if min_dist <= self.threshold:
                # Update winner
                self.centroids_[min_dist_idx] += self.learning_rate * (x[0] - self.centroids_[min_dist_idx])
            else:
                # Create new cluster
                self.centroids_.append(x[0].copy())
        
        # Anomaly Threshold Calculation
        # Compute distances of all training points to their nearest cluster
        if self.centroids_:
            centroids_arr = np.array(self.centroids_)
            all_dists = cdist(X, centroids_arr, metric=self.distance_metric)
            min_dists = np.min(all_dists, axis=1)
            
            # Simple approach: Max distance + buffer
            # Alternatively, use statistical approach: mean + 3*std
            # For now, sticking to max + buffer as per original logic but parameterized
            self.anomaly_threshold_ = np.max(min_dists) * self.anomaly_buffer
        else:
            self.anomaly_threshold_ = 0.0

        return self

    def predict(self, X):
        """
        Predict if samples are Normal (0) or Attack (1) based on anomaly threshold.
        """
        X = np.array(X)
        if not self.centroids_:
            return np.ones(X.shape[0]) # All attacks if no clusters

        centroids_arr = np.array(self.centroids_)
        dists = cdist(X, centroids_arr, metric=self.distance_metric)
        min_dists = np.min(dists, axis=1)
        
        return (min_dists > self.anomaly_threshold_).astype(int)

    def transform(self, X):
        """
        Transform X to cluster-distance space.
        """
        X = np.array(X)
        if not self.centroids_:
            return np.zeros((X.shape[0], 0))
            
        centroids_arr = np.array(self.centroids_)
        return cdist(X, centroids_arr, metric=self.distance_metric)
