
from naive_bayes import NaiveBayes
from softmax_regression import SoftmaxRegression
from support_vector_machine import SupportVectorMachine
from k_nearest_neighbor import KNN
from decision_tree import DecisionTree

from evaluate import ModelEvaluator
from pathlib import Path
from visualize import visualize
import joblib
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = ROOT / "processed_data"

labels = [
    "Business_Analyst",
    "Developer",
    "Quality_Assurance",
    "UIUX_Designer"
]

def load_package(file_path: Path):
    if file_path.exists():
        package = joblib.load(file_path)
        return package
    else:
        print(f"[ERR] Cannot find {file_path.name} in the directory")
        return None

def save_json_file(data, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(convert_to_json(data), file, indent=4, ensure_ascii=False)

def convert_to_json(data):
    if isinstance(data, np.ndarray):
        return data.tolist()

    if isinstance(data, np.generic):
        return data.item()

    if isinstance(data, dict):
        return {key: convert_to_json(value) for key, value in data.items()}

    if isinstance(data, (list, tuple)):
        return [convert_to_json(item) for item in data]

    return data


if __name__ == "__main__":
    # file paths
    pre_optimize_file_path = PROCESSED_DATA / "pre_optimize.json"
    data_file_path = Path("./processed_data/features.pkl")

    package = load_package(data_file_path)
    X_train_scaled = package["X_train_scaled"].to_numpy(dtype=np.float64)
    X_eval_scaled = package["X_eval_scaled"].to_numpy(dtype=np.float64)
    X_test_scaled = package["X_test_scaled"].to_numpy(dtype=np.float64)
    X_train_unscaled = package["X_train_unscaled"].to_numpy(dtype=np.float64)
    X_eval_unscaled = package["X_eval_unscaled"].to_numpy(dtype=np.float64)
    X_test_unscaled = package["X_test_unscaled"].to_numpy(dtype=np.float64)
    Y_train = np.asarray(package["Y_train"])
    Y_eval = np.asarray(package["Y_eval"])
    Y_test = np.asarray(package["Y_test"])

    pre_optimize_eval_data = []

    models = [
        NaiveBayes(),
        SoftmaxRegression(),
        SupportVectorMachine(),
        KNN(),
        DecisionTree()
    ]

    for model in models:
        if isinstance(model, NaiveBayes):
            X_train = X_train_unscaled
            X_evaluation = X_eval_unscaled
            X_test = X_test_unscaled
        else:
            X_train = X_train_scaled
            X_evaluation = X_eval_scaled
            X_test = X_test_scaled

        evaluation = ModelEvaluator(model_name=model.name,
                                model=model,
                                X_train=X_train,
                                X_evaluation=X_evaluation,
                                X_test=X_test,
                                Y_train=Y_train,
                                Y_evaluation=Y_eval,
                                Y_test=Y_test,
                                classes=labels)
        pre_optimize_eval_data.append(evaluation.evaluate())

    save_json_file(pre_optimize_eval_data, pre_optimize_file_path)


    
    
    