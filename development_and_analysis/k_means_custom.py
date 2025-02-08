import numpy as np

class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)

        # **Use K-Means++ initialization instead of random selection**
        self.centroids = self._initialize_centroids_kmeans_plus_plus(X)

        for _ in range(self.max_iter):
            self.labels = self._assign_clusters(X)

            new_centroids = np.array([X[np.where(self.labels == k)].mean(axis=0) for k in range(self.n_clusters)])


            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

    def predict(self, X):
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _initialize_centroids_kmeans_plus_plus(self, X):
        """K-Means++ initialization for better centroid selection"""
        n_samples, _ = X.shape
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        # Pick first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]

        # Select remaining centroids
        for i in range(1, self.n_clusters):
            # Compute squared distances from the nearest centroid
            distances = np.min([np.linalg.norm(X - centroid, axis=1)**2 for centroid in centroids[:i]], axis=0)

            # Choose a new centroid with probability proportional to squared distance
            probs = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()

            # Pick the point corresponding to probability
            new_centroid_index = np.where(cumulative_probs >= r)[0][0]
            centroids[i] = X[new_centroid_index]

        return centroids
