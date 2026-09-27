"""
file này dùng để gán nhãn cv lại lần nữa bằng api gpt
file này chỉnh sửa trực tiếp các cv trong các tập info
"""

import os
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from pathlib import Path
import json

load_dotenv()

API_KEY = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=API_KEY)

CATEGORY = [
    "Developer",
    "Business_Analyst",
    "Quality_Assurance",
    "UIUX_Designer"
]

def make_promt():
    return f"""
        Đọc đoạn văn bản thông tin trong CV được cung cấp. Xác định CV này phù hợp
        nhất với lĩnh vực nào trong danh sách sau:
        {CATEGORY}

        Dựa vào chức danh, kỹ năng, và mô tả công việc trong TOÀN BỘ
        văn bản để quyết định, không chỉ dựa vào 1 từ khóa đơn lẻ.

        Trả về DUY NHẤT 1 TEXT chính là ngành nghề phù hợp của CV đó, ví dụ "Developer".
        TUYỆT ĐỐI KHÔNG bịa thêm ngành nghề không có trong danh sách trên
        KHÔNG thêm giải thích.
    """

def relabel_one_file(info_text, prompt):
    retry_attemps = 2
    for attemp in range(retry_attemps):
        try:
            response = client.chat.completions.create(
                model = "gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": info_text
                    }
                ],
                temperature=0
            )

            result = response.choices[0].message.content
            return result, None
        
        except OpenAIError as e:
            return None, f"Lỗi API: {e}"

def relabel_set(file_path: Path):
    save_number = 10

    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("[ERR] Cannot find information file")
        return

    file_len = len(data)

    prompt = make_promt()

    for i in range(file_len):
        percent = round((i/file_len)*100, 2)
        print(f"Percent: {percent}%", end="\r")
        info_text = str(data[i]["info"])

        new_label, err = relabel_one_file(info_text, prompt)

        if err == None and new_label in CATEGORY:
            data[i]["label"] = new_label

        if (i+1)%save_number == 0:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii= False, indent=4)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii= False, indent=4)

if __name__ == "__main__":
    extract_sets = ["train", "evaluation", "test"]
    
    for set in extract_sets:
        print(f"Relabel {set} folder...")

        file_path = Path(f"./processed_data/{set}_info.json")
        relabel_set(file_path)
    