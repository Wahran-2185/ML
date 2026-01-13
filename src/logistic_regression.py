import numpy as np


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    """Binary logistic regression with gradient descent."""

    def __init__(self, lr=0.1, n_iter=1000, fit_intercept=True):
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
            z = X.dot(self.coef_)
            preds = _sigmoid(z)
            grad = (1 / m) * X.T.dot(preds - y)
            self.coef_ -= self.lr * grad

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = self._add_intercept(X)
        return _sigmoid(X.dot(self.coef_)).ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
