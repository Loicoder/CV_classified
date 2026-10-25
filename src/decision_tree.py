import numpy as np

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree():
    def __init__(self, min_sample_split=2, max_depth=1000):
        self.name = "DecisionTree"
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.root = None

    def get_params(self):
        return {
            "min_sample_split": self.min_sample_split,
            "max_depth": self.max_depth
        }
    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or \
            n_labels == 1 or \
            n_samples<self.min_sample_split):
                leaf_value = self._most_common_label(y)
                return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, n_features):
        best_gain = 0
        split_idx, split_threshold = None, None

        for feat_idx in range(n_features):
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)

            for thr in thresholds:
                gain = self._calculate_gain(y, X_col, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _calculate_gain(self, y, X_col, threshold):
        entropy_before = self._calculate_entropy(y)

        left_idxs, right_idxs = self._split(X_col, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = self._calculate_entropy(y[left_idxs]), self._calculate_entropy(y[right_idxs])

        entropy_after = (n_left/n) * entropy_left + (n_right/n) * entropy_right

        information_gain = entropy_before - entropy_after
        return information_gain

    def _split(self, X_col, threshold):
        left_idxs = np.argwhere(X_col <= threshold).flatten()
        right_idxs = np.argwhere(X_col > threshold).flatten()
        return left_idxs, right_idxs

    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def _most_common_label(self, y):
        # counting
        seq = {}
        for label in y:
            if label not in seq.keys():
                seq[label] = 1
            else:
                seq[label] += 1

        # max
        max_count = 0
        max_label = None

        for label, count in seq.items():
            if count > max_count:
                max_count = count
                max_label = label

        return max_label

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

            