import numpy as np


class KMeans:
    """Minimal KMeans clustering (NumPy).

    Attributes:
        cluster_centers_: array, shape (n_clusters, n_features)
        labels_: array, shape (n_samples,)
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def _init_centers(self, X):
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[idx].astype(float)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, _ = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        centers = self._init_centers(X)

        for i in range(self.max_iter):
            # assign
            d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(d, axis=1)

            # update
            new_centers = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                                    for k in range(self.n_clusters)])

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift <= self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        d = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(d, axis=1)
