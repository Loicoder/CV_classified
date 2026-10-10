"""
    Cài đặt thuật toán Bayes ngây thơ từ đầu.
"""
import numpy as np


class NaiveBayes():
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self.n_numeric_features = 4
        self.n_boolean_features = n_features - self.n_numeric_features

        # tính toán trung bình, độ lệch chuẩn và xác suất tiên nghiệm cho mỗi nhãn
        self._mean = np.zeros((n_classes, self.n_numeric_features), dtype=np.float64)
        self._varience = np.zeros((n_classes, self.n_numeric_features), dtype=np.float64)
        self._prior = np.zeros(n_classes, dtype=np.float64)
        self._boolean_prob = np.zeros((n_classes, self.n_boolean_features))

        for index, clss in enumerate(self._classes):
            X_class = X[ y==clss ]

            X_class_numeric = X_class[:, :self.n_numeric_features]
            X_class_boolean = X_class[:, self.n_numeric_features:]

            self._mean[index, :] = self._cal_mean(X_class_numeric)
            self._varience[index, :] = self._cal_var(X_class_numeric, index)
            self._prior[index] = X_class.shape[0] / float(n_samples)
            self._boolean_prob[index, :] = self._cal_boolean_prob(X_class_boolean)

    def _cal_mean(self, X_class):
        class_len, feature_len = X_class.shape
        mean_row = []
        for i in range(feature_len):
            sum = 0
            for j in range(class_len):
                sum += X_class[j][i]
            mean_row.append(sum/class_len)
        return mean_row

    def _cal_var(self, X_class_numeric, class_index):
        class_len, feature_len = X_class_numeric.shape
        var_row = []

        for i in range(feature_len):
            sum = 0
            for j in range(class_len):
                sum += (X_class_numeric[j][i] - self._mean[class_index][i])**2
            var = sum / (class_len-1)
            var_row.append(var)

        return var_row

    def _cal_boolean_prob(self, X_class_boolean):
        n_class = X_class_boolean.shape[0]
        probs = []

        for i in range(self.n_boolean_features):
            n_ones = 0
            for j in range(n_class):
                if (X_class_boolean[j][i] == 1):
                    n_ones += 1

            prob = n_ones / n_class
            probs.append(prob)

        return probs

    # pdf - probability density function
    def _cal_pdf(self, x, class_index):
        mean = self._mean[class_index]
        varience = self._varience[class_index]
        numerator = np.exp(-(x-mean)**2/(2*varience))
        denominator = np.sqrt(2*np.pi*varience)
        return numerator/denominator

    def predict(self, X):
        y_predict = []
        for x in X:
            y_predict.append(self._predict(x))

        return np.array(y_predict)

    def _predict(self, x):
        posteriors = []
        numeric = x[:self.n_numeric_features]
        boolean = x[self.n_numeric_features:]

        for class_index, clss in enumerate(self._classes):
            prior = np.log(self._prior[class_index])
            numeric_posterior = np.sum(np.log(self._cal_pdf(numeric, class_index)))
            boolean_posterior = 0
            for feature_index, feature in enumerate(boolean):
                if feature == 1:
                    boolean_posterior += np.log(self._boolean_prob[class_index][feature_index])
                else:
                    boolean_posterior += np.log(1 - self._boolean_prob[class_index][feature_index])
            posterior = prior + numeric_posterior + boolean_posterior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]