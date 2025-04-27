import numpy as np
from sklearn.metrics import pairwise_distances

class KMedoids:
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        
        # Initialisation aléatoire des médoides
        n_samples = X.shape[0]
        self.medoids_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids = X[self.medoids_indices]
        
        for _ in range(self.max_iter):
            # Assigner chaque point au médoid le plus proche
            distances = pairwise_distances(X, self.medoids)
            labels = np.argmin(distances, axis=1)
            
            # Mettre à jour les médoides
            new_medoids_indices = np.zeros(self.n_clusters, dtype=int)
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    cluster_distances = pairwise_distances(cluster_points)
                    total_distances = np.sum(cluster_distances, axis=1)
                    new_medoids_indices[i] = np.argmin(total_distances)
                    # On prend l'indice dans le cluster original
                    mask = (labels == i)
                    new_medoids_indices[i] = np.where(mask)[0][new_medoids_indices[i]]
            
            # Vérifier la convergence
            if np.all(self.medoids_indices == new_medoids_indices):
                break
                
            self.medoids_indices = new_medoids_indices
            self.medoids = X[self.medoids_indices]
            
        self.labels_ = np.argmin(pairwise_distances(X, self.medoids), axis=1)
        self.inertia_ = np.sum(np.min(pairwise_distances(X, self.medoids), axis=1)**2)
        return self
    
    def predict(self, X):
        return np.argmin(pairwise_distances(X, self.medoids), axis=1)
