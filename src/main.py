
from models.naive_bayes import NaiveBayes
from models.softmax_regression import SoftmaxRegression
from models.support_vector_machine import SupportVectorMachine
from models.k_nearest_neighbor import KNN
from models.decision_tree import DecisionTree

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

def split_folds(y, k=5, seed=0):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    class_indices = []

    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        class_indices.append(indices)

    folds = [[] for _ in range(k)]

    for indices in class_indices:
        parts = np.array_split(indices, k)

        for i in range(k):
            folds[i].extend(parts[i].tolist())

    all_indices = np.arange(len(y))
    result = []

    for i in range(k):
        val_idx = np.array(sorted(folds[i]), dtype=int)
        train_idx = np.setdiff1d(all_indices, val_idx)

        result.append((train_idx, val_idx))

    return result

def cross_validation(fold_idxs, X, y, model_factory):    
    fold_scores = []
    class_scores = []
    wrong_predictions = []

    for (train_idx, val_idx) in fold_idxs:
        model = model_factory()
        
        X_train = X[train_idx]
        y_train = y[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]  

        eval_model = ModelEvaluator(model_name=model.name,
                            model=model,
                            X_train=X_train,
                            X_evaluation=X_val,
                            Y_train=y_train,
                            Y_evaluation=y_val,
                            classes=labels)
        eval_res = eval_model.evaluate()

        fold_scores.append(eval_res["macro_f1"])
        class_scores.append(eval_res["f1_per_class"])

        y_pred = model.predict(X_val)
        for i, idx in enumerate(val_idx):
            if y_pred[i] != y_val[i]:
                wrong_predictions.append({
                    "sample_idx": int(idx),
                    "y_true": str(y_val[i]),
                    "y_pred": str(y_pred[i])
                })


    return {
        "model_name": model.name,
        "fold_score": fold_scores,
        "mean_macro_f1": float(np.mean(fold_scores)),
        "std_macro_f1": float(np.std(fold_scores)),
        "f1_per_class": class_scores,
        "wrong_predictions": wrong_predictions
    }

def baseline(Y_train, X_train_unscaled, X_train_scaled):
    # cross validation
    model_factorys = [
        lambda: NaiveBayes(),
        lambda: SoftmaxRegression(),
        lambda: SupportVectorMachine(),
        lambda: KNN(),
        lambda: DecisionTree()
    ]

    cross_valid_res = []
    folds = split_folds(Y_train, k=5, seed=301)
    for factory in model_factorys:
        model = factory()
        if isinstance(model, NaiveBayes):
            X_train = X_train_unscaled
        else:
            X_train = X_train_scaled

        res = cross_validation(folds, X_train, Y_train, factory)
        cross_valid_res.append(res)
    save_json_file(cross_valid_res, PROCESSED_DATA / "cv_baseline.json")

if __name__ == "__main__":
    # file paths
    file_path = PROCESSED_DATA / "optimized.json"
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
                                Y_train=Y_train,
                                Y_evaluation=Y_eval,
                                classes=labels)
        pre_optimize_eval_data.append(evaluation.evaluate())

    save_json_file(pre_optimize_eval_data, file_path)




 
        