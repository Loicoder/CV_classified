from pathlib import Path
from datetime import datetime
import json

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
def date_from_string(string_date: str):
    if string_date == "hien_tai" or string_date == "":
        return datetime.now().year*12 + datetime.now().month

    try:
        date = datetime.strptime(string_date, "%Y-%m").date()
        year  = date.year
        month = date.month
        time = year * 12 + month
        return time
    except:
        print("[ERR] Cannot parse string into format date ""yyyy-mm""")
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

    for exp in cv["information"]["kinh_nghiem"]:
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
def skill_frequency(data_set, rare_ratio: int = 2):
    skill_freq = {}

    for cv in data_set:
        cv_info = cv["information"]

        cv_skills = []

        cv_skills.extend(cv_info["ngon_ngu_lap_trinh"])
        cv_skills.extend(cv_info["ky_nang"])
        for project in cv_info["du_an"]:
            cv_skills.extend(project["cong_nghe_su_dung"])

        cv_skills = set(cv_skills)

        for skill in cv_skills:
            skill_freq[skill] = skill_freq.get(skill, 0) + 1

    res = {}
    for freq in skill_freq:
        if skill_freq[freq] >= rare_ratio:
            res[freq] = skill_freq[freq]

    res = sorted(res, key=lambda x: res[x])
    return res.keys()

################################################################################################






