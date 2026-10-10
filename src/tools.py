from pathlib import Path
import json
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_ROOT = PROJECT_ROOT / "processed_data"

CATEGORY = [
    "Developer",
    "Business_Analyst",
    "Quality_Assurance",
    "UIUX_Designer"
]

def count_info_per_category():
    """
        in số lượng cv đã truy xuất thông tin quan trọng theo từng ngành nghề
        file_name là tên file chứa các json truy xuất thông tin
    """
    files = ["train", "evaluation", "test"]

    for file_name in files:
        file_name = file_name + ".json"
        count = {}
        for job in CATEGORY:
            count[job] = 0

        file_path = PROCESSED_DATA_ROOT / file_name
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            print(f"[ERR] Cannot find {file_name} in processed_data folder")

        # count information cv per category
        for i in range(len(data)):
            label = data[i]["label"]

            if label in CATEGORY:
                count[label] += 1

        print("#" * 60)
        print(f"Số lượng cv theo danh mục các ngành nghề {file_name}:")
        for job in CATEGORY:
            print(f"{job} có {count[job]} cv.")

if __name__ == "__main__":
    count_info_per_category()


