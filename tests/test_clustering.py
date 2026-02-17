import pytest
import numpy as np
from client.clustering import SelfConstructingClustering

class TestSelfConstructingClustering:
    def test_initialization(self):
        """Test default parameters and initialization."""
        model = SelfConstructingClustering()
        assert model.threshold == 0.5
        assert model.learning_rate == 0.1
        assert model.distance_metric == 'euclidean'
        assert model.centroids_ == []

    def test_fit_simple_data(self):
        """Test fitting on simple synthetic data."""
        X = np.array([[1, 1], [1.1, 1.1], [10, 10], [10.1, 10.1]])
        model = SelfConstructingClustering(threshold=0.5)
        model.fit(X)
        
        # Expect 2 clusters: around (1,1) and (10,10)
        assert len(model.centroids_) == 2
        
        # Centroids should be close to the cluster centers
        c1 = model.centroids_[0]
        c2 = model.centroids_[1]
        
        # Check if one is near (1,1) and other near (10,10)
        # Order depends on input order
        assert (np.linalg.norm(c1 - [1, 1]) < 0.5) or (np.linalg.norm(c1 - [10, 10]) < 0.5)
        assert (np.linalg.norm(c2 - [1, 1]) < 0.5) or (np.linalg.norm(c2 - [10, 10]) < 0.5)

    def test_predict(self):
        """Test prediction logic (anomaly detection)."""
        X_train = np.array([[1, 1], [1.1, 1.1], [1, 1.2]])
        model = SelfConstructingClustering(threshold=0.5, anomaly_buffer=1.5)
        model.fit(X_train)
        
        # Normal sample (close to cluster)
        normal = np.array([[1.05, 1.05]])
        pred = model.predict(normal)
        assert pred[0] == 0 # Normal
        
        # Anomaly sample (far away)
        anomaly = np.array([[100, 100]])
        pred_anomaly = model.predict(anomaly)
        assert pred_anomaly[0] == 1 # Attack

    def test_cosine_metric(self):
        """Test with cosine distance metric."""
        # Cosine distance = 1 - cos_sim
        # Vectors [1, 0] and [0, 1] are orthogonal -> dist = 1.0
        # Vectors [1, 0] and [1, 0] are identical -> dist = 0.0
        X = np.array([[1, 0], [0, 1], [1, 0.1]]) # 3rd point close to 1st in angle
        
        model = SelfConstructingClustering(threshold=0.1, distance_metric='cosine')
        model.fit(X)
        
        # Should form 2 clusters: {[1,0], [1,0.1]} and {[0,1]}
        assert len(model.centroids_) == 2
