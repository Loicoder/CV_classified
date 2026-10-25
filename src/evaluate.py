import numpy as np
import time

class ModelEvaluator:
    def __init__(self,
                model_name,
                model,
                X_train,
                X_evaluation,
                Y_train,
                Y_evaluation,
                classes):
        self.model_name = model_name
        self.model = model
        self.X_train = X_train
        self.X_evaluation = X_evaluation
        self.Y_train = Y_train
        self.Y_evaluation = Y_evaluation
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
        fit_start_time = time.perf_counter()
        self.model.fit(self.X_train, self.Y_train)
        fit_end_time = time.perf_counter()
        training_time = fit_end_time - fit_start_time

        predict_start_time = time.perf_counter()
        predict = self.model.predict(self.X_evaluation)
        predict_end_time = time.perf_counter()
        predict_time = predict_end_time - predict_start_time

        confusion_matrix = self.create_confusion_matrix(self.Y_evaluation, predict)
        precisions, recalls, accuracy, f1s, macro_prec, macro_rec, macro_f1 = self.cal_eva_value_by_class(confusion_matrix)

        metrics = {
            "model_name": self.model_name,
            "precision_per_class": precisions,
            "recall_per_class": recalls,
            "f1_per_class": f1s,
            "accuracy": accuracy,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "training_time": training_time,
            "predict_time": predict_time,
            "confusion_matrix": confusion_matrix
        }

        params = self.model.get_params()

        self.result = {**metrics, **params}

        return self.result