"""Small examples demonstrating the implemented algorithms.

Run as a module from the repo root:
    python -m src.example
"""
import numpy as np

try:
    from .linear_regression import LinearRegression
    from .logistic_regression import LogisticRegression
    from .knn import KNearestNeighbors
    from .kmeans import KMeans
except Exception:
    # allow running as script (directly in src/)
    from linear_regression import LinearRegression
    from logistic_regression import LogisticRegression
    from knn import KNearestNeighbors
    from kmeans import KMeans


def demo_linear():
    print("--- Linear Regression demo ---")
    rng = np.random.RandomState(0)
    X = 2 * rng.rand(100, 1)
    y = 4 + 3 * X[:, 0] + rng.randn(100)

    lr = LinearRegression(lr=0.1, n_iter=2000)
    lr.fit(X, y)
    print("Estimated coefficients:", lr.coef_.ravel())


def demo_logistic():
    print("--- Logistic Regression demo ---")
    rng = np.random.RandomState(1)
    X = rng.randn(200, 2)
    # separates by a linear boundary
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = LogisticRegression(lr=0.5, n_iter=2000)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = (preds == y).mean()
    print(f"Train accuracy: {acc:.3f}")


def demo_knn():
    print("--- KNN demo ---")
    rng = np.random.RandomState(2)
    X = np.vstack([rng.randn(20, 2) + np.array([2, 2]), rng.randn(20, 2) + np.array([-2, -2])])
    y = np.array([0] * 20 + [1] * 20)

    knn = KNearestNeighbors(n_neighbors=3).fit(X, y)
    preds = knn.predict(X)
    print("Train accuracy:", (preds == y).mean())


def demo_kmeans():
    print("--- KMeans demo ---")
    rng = np.random.RandomState(3)
    X = np.vstack([rng.randn(50, 2) + np.array([5, 0]), rng.randn(50, 2) + np.array([-5, 0])])

    km = KMeans(n_clusters=2, random_state=0).fit(X)
    print("Cluster centers:\n", km.cluster_centers_)


def main():
    demo_linear()
    demo_logistic()
    demo_knn()
    demo_kmeans()


if __name__ == "__main__":
    main()
