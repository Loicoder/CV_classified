import numpy as np


def euclid_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance


class KNN():
    def __init__(self, k: int = 3):
        self.name = "K-NearestNeighbour"
        self.k = k

    def get_params(self):
        return {"k": self.k}
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # calculate euclid distance
        distances = [euclid_distance(x, x_train) for x_train in self.X_train]

        # get the closest K neighbor
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # vote
        counter = {}

        for neighbor_label in k_nearest_labels:
            if neighbor_label not in counter:
                counter[neighbor_label] = 1
            else:
                counter[neighbor_label] += 1

        return max(counter, key=counter.get)