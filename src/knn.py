import numpy as np


class KNearestNeighbors:
    """Simple K-Nearest Neighbors classifier (pure NumPy)."""

    def __init__(self, n_neighbors=3):
        self.n_neighbors = int(n_neighbors)
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y)
        return self

    def _distances(self, X):
        X = np.asarray(X, dtype=float)
        a = np.sum(X**2, axis=1)[:, None]
        b = np.sum(self._X**2, axis=1)[None, :]
        ab = X.dot(self._X.T)
        d2 = a + b - 2 * ab
        return np.sqrt(np.maximum(d2, 0.0))

    def predict(self, X):
        d = self._distances(X)
        idx = np.argsort(d, axis=1)[:, : self.n_neighbors]
        neigh = self._y[idx]
        # majority vote per row
        preds = []
        for row in neigh:
            vals, counts = np.unique(row, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)
