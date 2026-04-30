import pandas as pd
import numpy as np
import pytest
from client.preprocess import preprocess_data
from client.clustering import CLAPPClustering
from config import CLAPP_THRESHOLD, FEATURE_NAMES, SIGMA_C
from sklearn.neighbors import KNeighborsClassifier

class TestPipeline:
    def test_preprocess_integration(self):
        """Test data loading and preprocessing."""
        # Create a dummy DataFrame with required columns + noise
        data = {
            'Packet_Count': [1, 2, 3],
            'Byte_Count': [100, 200, 300],
            'Duration': [0.1, 0.2, 0.3],
            'Source_Bytes': [50, 60, 70],
            'Dest_Bytes': [40, 50, 60],
            'Same_Srv_Rate': [0.9, 0.9, 0.9],
            'Diff_Srv_Rate': [0.1, 0.1, 0.1],
            'SYN_Flag_Count': [0, 0, 0],
            'ACK_Flag_Count': [1, 1, 1],
            'Label': ['Normal', 'Attack', 'Normal'],
            'label': [0, 1, 0],
            'ip.src': ['1.1.1.1', '2.2.2.2', '3.3.3.3'], # Non-numeric values become 0
            'id': [1, 2, 3]
        }
        df = pd.DataFrame(data)
        
        PS, labels, feature_names = preprocess_data(df)
        
        assert PS.shape[0] == 3
        assert labels.tolist() == [0, 1, 0]
        assert "Label" not in feature_names
        assert "label" not in feature_names
        assert PS.shape[1] == len(feature_names)
        
    def test_full_pipeline(self):
        """Integration test: Load -> Preprocess -> CLAPP reduce -> kNN classify."""
        # 1. Create Data
        df = pd.DataFrame(np.random.rand(10, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
        df["label"] = [0, 1] * 5
        
        # 2. Preprocess
        PS, labels, _ = preprocess_data(df)
        
        # 3. CLAPP feature reduction
        model = CLAPPClustering(threshold=CLAPP_THRESHOLD, sigma=SIGMA_C)
        X_reduced = model.fit_transform(PS, labels)
        
        # 4. kNN classification on reduced matrix
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_reduced, labels)
        preds = knn.predict(X_reduced)
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})
