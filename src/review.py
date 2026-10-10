import joblib
import numpy as np
from pathlib import Path
from naive_bayes import NaiveBayes


def load_package(file_path: Path):
    if file_path.exists():
        package_data = joblib.load(file_path)
        return package_data
    else:
        print(f"[ERR] Cannot find {file_path.name} in directory")
        return None

package = load_package(Path("./processed_data/features.pkl"))
X_train_NB = package["X_train_NB"].to_numpy(dtype=np.float64)
y = np.array(package["Y_train"])
model = NaiveBayes()

model.fit(X_train_NB, y)
model.predict(X_train_NB)


