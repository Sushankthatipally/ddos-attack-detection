import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def tune_alpha(distances_normal, distances_all, y_true, alpha_range):
    """
    distances_normal : distances of NORMAL samples only
    distances_all    : distances of validation samples (normal + attack)
    y_true           : true labels (0 = Normal, 1 = DDoS)
    alpha_range      : list of alpha values to test
    """

    mu = np.mean(distances_normal)
    sigma = np.std(distances_normal)

    results = []

    for alpha in alpha_range:
        threshold = mu + alpha * sigma

        y_pred = [1 if d > threshold else 0 for d in distances_all]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        results.append({
            "alpha": alpha,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return results
