import joblib
import numpy as np
from pathlib import Path
from naive_bayes import NaiveBayes

class ModelEvaluator:
    def __init__(self,
                 model,
                 X_train,
                 X_evaluation,
                 X_test,
                 Y_train,
                 Y_evaluation,
                 Y_test,
                 classes):
        self.model = model
        self.X_train = X_train
        self.X_evaluation = X_evaluation
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_evaluation = Y_evaluation
        self.Y_test = Y_test
        self.classes = classes
        self.predicts = []

    def check_valid_predict(self, Y_label, Y_predict):
        if not np.isin(Y_label, classes).all():
            Y_wrong = np.unique(Y_label[~np.isin(Y_label, classes)])
            raise ValueError(f"[ERR] Label ({Y_wrong}) is not in {classes}")

        if not np.isin(Y_predict, classes).all():
            Y_wrong = np.unique(Y_predict[~np.isin(Y_predict, classes)])
            raise ValueError(f"[ERR] Label ({Y_wrong}) is not in {classes}")

    def create_confusion_matrix(self, Y_label, Y_predict):
        Y_label = np.asarray(Y_label)
        Y_predict = np.asarray(Y_predict)
        classes = self.classes

        self.check_valid_predict(Y_label, Y_predict)

        reflex_label = {c:i for i,c in enumerate(classes)}

        matrix = np.zeros((len(classes), len(classes)), dtype=int)

        for t, d in zip(Y_label, Y_predict):
            matrix[reflex_label[t], reflex_label[d]] += 1

        return matrix

    def cal_eva_value_by_class(self, confusion_matrix):
        classes = self.classes
        matrix = confusion_matrix
        class_len = len(classes)
        total = np.sum(matrix)
        accuracy = 0

        precision = np.zeros(class_len, dtype=float)
        recall = np.zeros(class_len, dtype=float)
        f1 = np.zeros(class_len, dtype=float)

        for i in range(class_len):
            tp = matrix[i][i]
            fp = np.sum(matrix, axis=0)[i] - tp
            fn = np.sum(matrix, axis=1)[i] - tp
            tn = np.sum(matrix) - tp - fp - fn

            precision[i] = float(tp) / (tp + fp)
            recall[i] = float(tp) / (tp + fn)
            f1[i] = float(2 * precision[i] * recall[i]) / (precision[i] + recall[i])
            accuracy += float(tp) / total
        

        return precision, recall, accuracy, f1


    
