import pytest
import numpy as np
from client.clustering import CLAPPClustering

class TestCLAPPClustering:
    def test_initialization(self):
        """Test default parameters and initialization."""
        model = CLAPPClustering()
        assert model.threshold == 0.85
        assert model.sigma == 0.5
        assert model.clusters_ == []
        assert model.transformation_matrix_ is None

    def test_compute_pattern_vectors(self):
        """Test posterior probability vectors from the PS matrix."""
        PS = np.array([
            [10, 0],
            [10, 0],
            [0, 5],
            [0, 5],
        ])
        labels = np.array([0, 0, 1, 1])
        model = CLAPPClustering()
        patterns = model.compute_pattern_vectors(PS, labels)
        
        np.testing.assert_allclose(patterns[0], [1.0, 0.0])
        np.testing.assert_allclose(patterns[1], [0.0, 1.0])
        
    def test_fit_and_transform(self):
        """Test fitting CLAPP and reducing the PS matrix."""
        PS = np.array([
            [10, 9, 0, 0],
            [8, 7, 0, 0],
            [0, 0, 5, 6],
            [0, 0, 7, 8],
        ])
        labels = np.array([0, 0, 1, 1])
        model = CLAPPClustering(threshold=0.85, sigma=0.5)
        model.fit(PS, labels)

        assert len(model.clusters_) == 2
        assert model.transformation_matrix_.shape == (4, 2)

        reduced = model.transform(PS)
        assert reduced.shape == (4, 2)

    def test_membership_same_vector_is_one(self):
        """Identical pattern vectors should have full membership."""
        model = CLAPPClustering()
        assert model.membership([0.8, 0.2], [0.8, 0.2]) == pytest.approx(1.0)
