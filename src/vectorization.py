from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import json
import pandas as pd
import joblib

DATA_SET_NAME = {
    "train.json",
    "evaluation.json",
    "test.json"
}

ROOT_PROJECT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = ROOT_PROJECT / "processed_data"

def read_json(file_path: Path):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    else:
        print(f"[ERR] Cannot find {file_path.name} in the directory")
        return None

################################ Tính kinh nghiệm theo tháng ################################
def date_from_string(string_date):
    if string_date == "hien_tai":
        return datetime.now().year*12 + datetime.now().month

    try:
        date = datetime.strptime(string_date, "%Y-%m").date()
        year  = date.year
        month = date.month
        time = year * 12 + month
        return time
    except:
        pass

    try:
        date = datetime.strptime(string_date, "%Y-%m-%d").date()
        year  = date.year
        month = date.month
        time = year * 12 + month
        return time
    except:
        pass

    try:
        date = datetime.strptime(string_date, "%Y").date()
        year  = date.year
        time = year * 12
        return time
    except:
        pass

    
    print(f"[ERR] Cannot get date from string {string_date}")
    return None

def merge_overlap_month(time_list: list):
    if time_list == []:
        return []

    sorted_time_list = sorted(time_list, key=lambda x: x[0])

    merged = [sorted_time_list[0]]

    for start, end in sorted_time_list[1:]:
        current_end = merged[-1][1]

        if start <= current_end:
            merged[-1][1] = max(current_end, end)
        else:
            merged.append([start, end])

    return merged

def cal_experience_by_month(cv):
    time_list = []
    sum = 0
    cv_exp = cv["information"]["kinh_nghiem"]

    if cv_exp == []:
        return 0.0
    
    for exp in cv_exp:
        if exp["thoi_gian_bat_dau"] == "" or exp["thoi_gian_ket_thuc"] == "":
            continue

        start = date_from_string(exp["thoi_gian_bat_dau"])
        end = date_from_string(exp["thoi_gian_ket_thuc"])

        if start is not None and end is not None and end >= start:
            time_list.append([start, end])

    if time_list == []:
        return 0

    merged_time_list = merge_overlap_month(time_list)

    for start, end in merged_time_list:
        sum += end - start

    return round(sum / 12, 1)

################################################################################################

########################### Đếm số lượng skill trùng trong tập train ###########################
def get_cv_skills(cv_information: dict)->set:
    if not cv_information:
        return set()

    cv_info = cv_information["information"]   

    cv_skills = []

    cv_skills.extend(cv_info["ngon_ngu_lap_trinh"])
    cv_skills.extend(cv_info["ky_nang"])
    for project in cv_info["du_an"]:
        cv_skills.extend(project["cong_nghe_su_dung"])

    cv_skills = set(cv_skills)

    return cv_skills

def skill_frequency(data_set, rare_ratio: int = 2):
    skill_freq = {}

    for cv in data_set:
        cv_skills = get_cv_skills(cv)

        for skill in cv_skills:
            skill_freq[skill] = skill_freq.get(skill, 0) + 1

    res = {}
    for freq in skill_freq:
        if skill_freq[freq] >= rare_ratio:
            res[freq] = skill_freq[freq]

    res = sorted(res, key=lambda x: res[x])

    return res

################################################################################################

################################## Xây dựng cấu trúc đặc trưng #################################
def build_feature_structure(cv_information: dict, skill_dict: list):
    cv_skills = get_cv_skills(cv_information)
    info = cv_information["information"]
    feature = {}

    # Numeric
    feature["so_nam_kinh_nghiem"] = cal_experience_by_month(cv_information)
    feature["so_luong_skill"] = len(cv_skills)
    feature["so_du_an"] = len(info["du_an"])
    feature["so_cong_ty_da_lam"] = len(info["kinh_nghiem"])

    # Boolean
    feature["co_ngon_ngu_lap_trinh"] = 1 if len(info["ngon_ngu_lap_trinh"]) != 0 else 0
    feature["co_chung_chi"] = 1 if len(info["chung_chi"]) != 0 else 0
    feature["co_du_an"] = 1 if len(info["du_an"]) != 0 else 0
    feature["co_dang_di_lam"] = 1 if any(di_lam["thoi_gian_ket_thuc"] == "hien_tai" \
                                      for di_lam in info["kinh_nghiem"]) else 0

    # Multi-hot
    for skill_name in skill_dict:
        column_name = f"skill_{skill_name}"
        feature[column_name] = 1 if skill_name in cv_skills else 0

    return feature

def build_feature_table(data_set, skill_dict: list):
    if not data_set:
        return None
     
    feature_list = []

    for cv in data_set:
        feature_list.append(build_feature_structure(cv, skill_dict))

    feature_table = pd.DataFrame(feature_list)

    return feature_table

################################################################################################

########################## Xây dựng tập đặc trưng cho các thuật toán ###########################
def create_feature_for_LR_SVM(train_table, eval_table, test_table):
    scalar = StandardScaler()

    numeric_column = ["so_nam_kinh_nghiem", "so_luong_skill", "so_du_an", "so_cong_ty_da_lam"]
    boolean_column = [col for col in train_table.columns if col not in numeric_column]

    scalar.fit(train_table[numeric_column])

    numeric_train = pd.DataFrame(
        scalar.transform(train_table[numeric_column]),
        columns=numeric_column,
        index=train_table.index
    )

    numeric_eval = pd.DataFrame(
        scalar.transform(eval_table[numeric_column]),
        columns=numeric_column,
        index=eval_table.index
    )

    numeric_test = pd.DataFrame(
        scalar.transform(test_table[numeric_column]),
        columns=numeric_column,
        index=test_table.index 
    )

    feature_train = pd.concat([numeric_train, train_table[boolean_column]], axis=1)
    feature_eval = pd.concat([numeric_eval, eval_table[boolean_column]], axis=1)
    feature_test = pd.concat([numeric_test, test_table[boolean_column]], axis=1)

    return feature_train, feature_eval, feature_test, scalar

def load_dataset(filename: str):
    path = PROCESSED_DATA / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    else:
        print(f"[ERR] Cannot find {filename} in the directory")
        return None

def save_features():
    train_dataset = load_dataset("train.json")
    eval_dataset = load_dataset("evaluation.json")
    test_dataset = load_dataset("train.json")

    skill_dict = skill_frequency(train_dataset)

    train_table = build_feature_table(train_dataset, skill_dict)
    eval_table = build_feature_table(eval_dataset, skill_dict)
    test_table = build_feature_table(test_dataset, skill_dict)

    X_train_svm, X_eval_svm, X_test_svm, scalar = create_feature_for_LR_SVM(train_table, eval_table, test_table)
    X_train_nb, X_eval_nb, X_test_nb = train_table, eval_table, test_table

    Y_train = [cv["label"] for cv in train_dataset]
    Y_eval = [cv["label"] for cv in eval_dataset]
    Y_test = [cv["label"] for cv in test_dataset]

    package = {
        "X_train_LR_SVM": X_train_svm,
        "X_eval_LR_SVM": X_eval_svm, 
        "X_test_LR_SVM": X_test_svm, 
        "X_train_NB": X_train_nb, 
        "X_eval_NB": X_eval_nb, 
        "X_test_NB": X_test_nb,
        "Y_train": Y_train,
        "Y_eval": Y_eval,
        "Y_test": Y_test,
        "skill_dict": skill_dict,
        "scalar": scalar
    }

    output_path = PROCESSED_DATA / "features.pkl"
    joblib.dump(package, output_path)

################################################################################################

if __name__ == "__main__":
    save_features()