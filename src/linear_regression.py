import numpy as np


class LinearRegression:
    """Simple linear regression using gradient descent."""

    def __init__(self, lr=0.01, n_iter=1000, fit_intercept=True):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if self.fit_intercept:
            X = self._add_intercept(X)

        m, n = X.shape
        self.coef_ = np.zeros((n, 1))

        for _ in range(self.n_iter):
            preds = X.dot(self.coef_)
            grad = (2 / m) * X.T.dot(preds - y)
            self.coef_ -= self.lr * grad

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = self._add_intercept(X)
        return X.dot(self.coef_).ravel()
