import numpy as np

def compute_distance_threshold(distances, alpha):
    mu = np.mean(distances)
    sigma = np.std(distances)
    return mu + alpha * sigma

def compute_cosine_threshold(similarities):
    # Typically we take a lower percentile of normal similarities
    return np.percentile(similarities, 5)
