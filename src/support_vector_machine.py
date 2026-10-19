import numpy as np

class SupportVectorMachine():
    def __init__(self, iters: int = 1000,
                gamma: float = 1e-4,
                C: float = 100000):
        self.iters = iters
        self.gamma = gamma
        self.C = C

    def get_params(self):
        return {
            "iters": self.iters,
            "gamma": self.gamma
        }

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self._classes = np.unique(y)

        self.alphas = []
        self.bias = []
        self.ks = []
        self.y = []
        self.X_train = X.copy()
        for i, _class in enumerate(self._classes):
            y_binary = np.where(y == _class, 1, -1)
            a, b, k = self.fit_binary(X, y_binary)

            self.alphas.append(a)
            self.bias.append(b)  
            self.ks.append(k)
            self.y.append(y_binary)
            
        return self

    def predict(self, X):
        K = self._rbf_kernel(self.X_train, X)
        scores = []

        for i in range(len(self._classes)):

            score = (self.alphas[i] * self.y[i]) @ K + self.bias[i]

            scores.append(score)

        scores = np.array(scores).T
        return self._classes[np.argmax(scores, axis=1)]

    def _rbf_kernel(self, X, Z):
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Z_norm = np.sum(Z ** 2, axis=1).reshape(1, -1)

        squared_distance = X_norm + Z_norm - 2 * np.dot(X, Z.T)
        squared_distance = np.maximum(squared_distance, 0)

        return np.exp(-self.gamma * squared_distance)

    def fit_binary(self, X, y):
        n_samples, n_features = X.shape

        alphas = np.zeros(n_samples)
        bias = 0.0

        k_matrix = self._rbf_kernel(X, X)

        for k in range(self.iters):
            scores = self.score_func(alphas, k_matrix, bias, y)

            i = np.random.randint(n_samples)
            j = np.random.randint(n_samples - 1)

            if j >= i: j += 1   

            E_i = scores[i] - y[i]
            E_j = scores[j] - y[j]

            if y[i] != y[j]:
                L = max(0, alphas[j] - alphas[i])
                H = min(self.C, self.C + alphas[j] - alphas[i])
            else:
                L = max(0, alphas[i] + alphas[j] - self.C)
                H = min(self.C, alphas[i] + alphas[j])

            if L == H: continue

            eta = 2 * k_matrix[i, j] - k_matrix[i, i] - k_matrix[j, j]
            if eta >= 0: continue

            alpha_j_old = alphas[j]
            alpha_i_old = alphas[i]

            alphas[j] -= y[j] * (E_i - E_j) / eta
            alphas[j] = np.clip(alphas[j], L, H)


            alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alphas[j])

            b1 = (bias - E_i - y[i] * (alphas[i] - alpha_i_old) * k_matrix[i, i] - y[j] * (alphas[j] - alpha_j_old) * k_matrix[i, j])
            b2 = (bias - E_j - y[i] * (alphas[i] - alpha_i_old) * k_matrix[i, j] - y[j] * (alphas[j] - alpha_j_old) * k_matrix[j, j])

            if 0 < alphas[i] < self.C:
                bias = b1
            elif 0 < alphas[j] < self.C:
                bias = b2
            else:
                bias = (b1 + b2) / 2

        return alphas, bias, k_matrix

    def score_func(self, alphas, k, bias: float, y):
        return (alphas * y) @ k + bias