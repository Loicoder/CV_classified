import joblib
import numpy as np
from pathlib import Path
from naive_bayes import NaiveBayes

class ModelEvaluator:
    def __init__(self,
                 model_name,
                 model,
                 X_train,
                 X_evaluation,
                 X_test,
                 Y_train,
                 Y_evaluation,
                 Y_test,
                 classes = ["Business_Analyst",
                    "Developer",
                    "Quality_Assurance",
                    "UIUX_Designer"]):

        self.model_name = model_name
        self.model = model
        self.X_train = X_train
        self.X_evaluation = X_evaluation
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_evaluation = Y_evaluation
        self.Y_test = Y_test
        self.classes = classes
        self.result = {}

    def check_valid_predict(self, Y_label, Y_predict):
        classes = self.classes

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

        precisions = np.zeros(class_len, dtype=float)
        recalls = np.zeros(class_len, dtype=float)
        f1s = np.zeros(class_len, dtype=float)

        
        for i in range(class_len):
            tp = matrix[i][i]
            fp = np.sum(matrix, axis=0)[i] - tp
            fn = np.sum(matrix, axis=1)[i] - tp

            precisions[i] = float(tp) / (tp + fp) if (tp+fp != 0) else 0.0
            recalls[i] = float(tp) / (tp + fn) if (tp+fn != 0) else 0.0
            f1s[i] = float(2 * precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) if (precisions[i] + recalls[i] != 0) else 0.0
            accuracy += float(tp) / total

        macro_prec = np.sum(precisions) / class_len
        macro_rec = np.sum(recalls) / class_len
        macro_f1 = np.sum(f1s) / class_len

        return precisions, recalls, accuracy, f1s, macro_prec, macro_rec, macro_f1

    def evaluate(self):
        self.model.fit(self.X_train, self.Y_train)
        predict = self.model.predict(self.X_evaluation)

        confusion_matrix = self.create_confusion_matrix(self.Y_evaluation, predict)
        precisions, recalls, accuracy, f1s, macro_prec, macro_rec, macro_f1 = self.cal_eva_value_by_class(confusion_matrix)


        self.result = {
            "evaluation model": self.model_name,
            "precision_per_class": precisions,
            "recall_per_class": recalls,
            "f1_per_class": f1s,
            "accuracy": accuracy,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1
        }

        return self.result