
from naive_bayes import NaiveBayes
from evaluate import ModelEvaluator
from pathlib import Path
import joblib
import numpy as np


def load_package(file_path: Path):
    if file_path.exists():
        package = joblib.load(file_path)
        return package
    else:
        print(f"[ERR] Cannot find {file_path.name} in the directory")
        return None

if __name__ == "__main__":
    model = NaiveBayes()

    data_file_path = Path("./processed_data/features.pkl")
    package = load_package(data_file_path)

    X_train_LR_SVM = package["X_train_LR_SVM"].to_numpy(dtype=np.float64)
    X_eval_LR_SVM = package["X_eval_LR_SVM"].to_numpy(dtype=np.float64)
    X_test_LR_SVM = package["X_test_LR_SVM"].to_numpy(dtype=np.float64)
    X_train_NB = package["X_train_NB"].to_numpy(dtype=np.float64)
    X_eval_NB = package["X_eval_NB"].to_numpy(dtype=np.float64)
    X_test_NB = package["X_test_NB"].to_numpy(dtype=np.float64)
    Y_train = np.asarray(package["Y_train"])
    Y_eval = np.asarray(package["Y_eval"])
    Y_test = np.asarray(package["Y_test"])

    eval = ModelEvaluator("NaiveBayes",
                          model,
                          X_train_NB,
                          X_eval_NB,
                          X_test_NB,
                          Y_train,
                          Y_eval,
                          Y_test)
    
    eval.evaluate()
    
    