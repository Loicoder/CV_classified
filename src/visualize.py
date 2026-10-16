import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpecFromSubplotSpec
import json
import numpy as np

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

    x = np.arange(len(data))
    width = 0.2

    bar1 = axis.bar(x - width * 1.5, accuracies, width, label="Accuracy")
    bar2 = axis.bar(x - width / 2, macro_precisions, width, label="Macro Precision")
    bar3 = axis.bar(x + width / 2, macro_recalls, width, label="Macro Recall")
    bar4 = axis.bar(x + width * 1.5, macro_f1s, width, label="Macro F1")

    for bars in [bar1, bar2, bar3, bar4]:
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    axis.set_title("Model Performance")
    axis.set_ylabel("Score")
    axis.set_xticks(x)
    axis.set_xticklabels(model_names)
    axis.set_ylim(0, 1.5)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

def draw_loss_history(data, axis):
    for model_info in data:
        history = model_info["loss_history"]

        if history is None:
            continue

        history = np.asarray(history)

        if history.size == 0:
            continue

        if history.ndim == 1:
            axis.plot(history, label=model_info["model_name"])
        elif history.ndim == 2:
            axis.plot(history.mean(axis=0), label=f"{model_info['model_name']} (mean)")

    axis.set_title("Training Loss")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.3)

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
    axis.set_xticklabels([model["model_name"] for model in data])
    axis.set_ylim(0, max(max(training_times), max(prediction_times)) * 1.2)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

def draw_confusion_matrix(data, axis, figure):
    confusion_matrices = [np.asarray(model["confusion_matrix"]) for model in data]

    model_names = ["NaiveBayes", "Softmax", "SVM"]
    class_names = ["BA", "Dev", "QA", "UI/UX"]

    subplot_spec = axis.get_subplotspec()
    axis.remove()

    inner_grid = GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=subplot_spec,
        wspace=0.35
    )

    for i, (cm, name) in enumerate(zip(confusion_matrices, model_names)):
        sub_axis = figure.add_subplot(inner_grid[0, i])

        sub_axis.imshow(cm, cmap="Reds", vmin=0)

        sub_axis.set_title(name, fontsize=10)
        sub_axis.set_xticks(range(len(class_names)))
        sub_axis.set_yticks(range(len(class_names)))

        sub_axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        sub_axis.set_yticklabels(class_names, fontsize=7)

        sub_axis.set_xlabel("Predicted", fontsize=8)

        if i == 0:
            sub_axis.set_ylabel("Actual", fontsize=8)

        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                sub_axis.text(
                    col, row, str(cm[row, col]),
                    ha="center", va="center",
                    fontsize=7,
                    color="black"
                )

# main function
def visualize(data_file_path: Path):
    data = load_json(data_file_path)
    
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    figure.suptitle("Models Evaluation Board", fontsize = 16)

    #axis 
    top_left_ax = axes[0, 0]
    top_right_ax = axes[0, 1]
    bottom_left_ax = axes[1, 0]
    bottom_right_ax = axes[1, 1]

    draw_metric(data, top_left_ax)
    draw_loss_history(data, top_right_ax)
    draw_time(data, bottom_left_ax)
    draw_confusion_matrix(data, bottom_right_ax, figure)

    plt.tight_layout()
    plt.show()

