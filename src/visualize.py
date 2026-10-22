import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpecFromSubplotSpec
import json
import numpy as np
import argparse as arg

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = ROOT / "processed_data"

def load_json(file_path: Path):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    else:
        print(f"[ERR] Cannot file {file_path.name} in the directory")
        return None

def draw_metric(data, axis):
    accuracies = [model["accuracy"] for model in data]
    macro_precisions = [model["macro_precision"] for model in data]
    macro_recalls = [model["macro_recall"] for model in data]
    macro_f1s = [model["macro_f1"] for model in data]
    model_names = [model["model_name"] for model in data]

    metrics = {
        "Accuracy": accuracies,
        "Macro Precision": macro_precisions,
        "Macro Recall": macro_recalls,
        "Macro F1": macro_f1s
    }

    x = np.arange(len(metrics))
    width = 0.15

    for i, model_name in enumerate(model_names):
        values = [metric_values[i] for metric_values in metrics.values()]

        bars = axis.bar(
            x + (i - (len(model_names) - 1) / 2) * width,
            values,
            width,
            label=model_name
        )

        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    axis.set_title("Model Performance")
    axis.set_ylabel("Score")
    axis.set_xticks(x)
    axis.set_xticklabels(metrics.keys())
    axis.set_ylim(0, 1.1)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

def draw_time(data, axis):
    training_times = [model["training_time"] for model in data]

    prediction_times = [model["predict_time"] for model in data]

    x = np.arange(len(data))
    width = 0.35

    bar1 = axis.bar(x - width / 2, training_times, width, label="Training")
    bar2 = axis.bar(x + width / 2, prediction_times, width, label="Prediction")

    for bars in [bar1, bar2]:
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    axis.set_title("Execution Time")
    axis.set_ylabel("Time (seconds)")
    axis.set_xticks(x)
    axis.set_xticklabels([model["model_name"] for model in data], rotation=10, fontsize=9)
    axis.set_ylim(0, max(max(training_times), max(prediction_times)) * 1.2)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

# main function
def visualize(data_file_path: Path):
    data = load_json(data_file_path)
    
    figure, axes = plt.subplots(1, 2, figsize=(14, 9))

    figure.suptitle("Models Evaluation Board", fontsize = 16)

    #axis 
    left_ax = axes[0]
    right_ax = axes[1]
 
    draw_metric(data, left_ax)
    draw_time(data, right_ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "--file",
        default=PROCESSED_DATA / "pre_optimize.json"
    )

    args = parser.parse_args()
    file_path = PROCESSED_DATA / args.file
    visualize(file_path)
