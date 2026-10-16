import numpy as np


class SoftmaxRegression():
    def __init__(self, learning_rate: float = 0.1, iters: int = 1000):
        self.lr = learning_rate
        self.iters = iters
        self._classes = None
        self.weights = None
        self.bias = None
        self.loss_history = None

    def get_params(self):
        return {
            "learning_rate": self.lr,
            "iters": self.iters
        }

    def get_history(self):
        return self.loss_history
    
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self._classes, self.y_idx = np.unique(y, return_inverse=True)
        self.weights = np.zeros((self.n_features, len(self._classes)))
        self.bias = np.zeros(len(self._classes))
        self.loss_history = []

        for i in range(self.iters):
            loss, grad_w, grad_b = self._cal_loss_and_gradient(X, self.y_idx, self.weights, self.bias)
            self.loss_history.append(loss)
            self.weights = self.weights - self.lr * grad_w
            self.bias = self.bias - self.lr * grad_b
        
        return self

    def _cal_loss_and_gradient(self, X, y_idx, w, b):
        z = np.dot(X, w) + b

        # tính hàm mất mát
        losses = []
        for i, row in enumerate(z):
            m = max(row)
            log_sum = m + np.log(np.sum(np.exp(row-m)))
            losses.append(log_sum - row[y_idx[i]])
        loss = np.mean(losses)

        # gradient
        grad = self._softmax_function(z)
        for i in range(len(grad)):
            grad[i, y_idx[i]] -= 1
        
        grad = grad / self.n_samples
        grad_w = np.dot(X.T, grad)
        grad_b = np.sum(grad, axis=0)

        return loss, grad_w, grad_b

    def _softmax_function(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def predict_prob(self, X):
        return self._softmax_function(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        return self._classes[np.argmax(self.predict_prob(X), axis=1)]
