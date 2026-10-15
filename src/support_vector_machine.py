import numpy as np

class SupportVectorMachine():
    def __init__(self, learning_rate: float = 0.001, iters: int = 1000, lambda_: float = 0.01):
        self.learning_rate = learning_rate
        self.iters = iters
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self._classes = np.unique(y)
        self.weights = np.zeros((self.n_features, len(self._classes)))
        self.bias = np.zeros((len(self._classes)))
        self.history = np.zeros((len(self._classes), self.iters))

        for i, _class in enumerate(self._classes):
            y_binary = np.where(y == _class, 1, -1)
            w, b, h = self.fit_binary(X, y_binary)

            self.weights[:, i] = w
            self.bias[i] = b
            self.history[i, :] = h 
            
        return self

    def predict(self, X):
        score = X @ self.weights + self.bias
        return self._classes[np.argmax(score, axis=1)]

    def fit_binary(self, X, y):
        n_samples, n_features = X.shape

        weights = np.zeros(n_features)
        bias = 0.0
        history = []

        for i in range(self.iters):
            grad_w, grad_b = self._cal_gradients(X, y, weights, bias, n_samples)

            weights -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b

            loss = self.loss_function(X, y, weights, bias)
            history.append(loss)

        return weights, bias, history

    def decision_function(self, X, w, b):
        return X @ w + b

    def predict_binary(self, X, w, b):
        scores = self.decision_function(X, w, b)
        return np.where(scores >= 0, 1, -1)

    def hinge_loss(self, X, y, w, b):
        scores = self.decision_function(X, w, b)

        margin = y * scores

        hinge_loss = np.maximum(0, 1 - margin)

        return np.mean(hinge_loss)

    def loss_function(self, X, y, w, b):
        hinge_loss = self.hinge_loss(X, y, w, b)

        regularization = 0.5 * self.lambda_ * np.sum(w**2)

        return regularization + hinge_loss

    def _cal_gradients(self, X, y, w, b, n_samples):
        scores = self.decision_function(X, w, b)
        margin = y * scores

        is_violate = margin < 1

        grad_w = self.lambda_ * w
        grad_b = 0.0

        if np.any(is_violate):
            X_violate = X[is_violate]
            y_violate = y[is_violate]

            grad_w -= X_violate.T @ y_violate / n_samples
            grad_b = -np.sum(y_violate) / n_samples

        return grad_w, grad_b

