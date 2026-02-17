import numpy as np

def federated_average(client_centroids):
    all_centroids = []
    for centroids in client_centroids:
        all_centroids.extend(centroids)

    return np.mean(all_centroids, axis=0)
