"""
file này dùng model gpt-4o-mini, gọi prompt để trích xuất các thông tin của từng cv
theo đúng format json đã định dạng sẵn trong EXTRACTED_INFO_STRUCTURE
sau đó lưu lại vào các file JSON info tương ứng với các tập
"""

from dotenv import load_dotenv
import os
from openai import OpenAIError, OpenAI
import time
import json
from pathlib import Path

load_dotenv()

API_KEY = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=API_KEY)

EXTRACTED_INFO_STRUCTURE = {
    "thong_tin_ca_nhan": {
        "ho_ten": "",
        "email": "",
        "so_dien_thoai": ""
    },
    "hoc_van": [
        {
            "truong": "",
            "chuyen_nganh": "",
            "bang_cap": "",
            "nam_tot_nghiep": None
        }   
    ],
    "ngon_ngu_lap_trinh": [],
    "ky_nang": [],
    "ky_nang_khac": [],
    "chung_chi": [
        {
            "ten_chung_chi": "",
            "to_chuc_cap": "",
            "nam_cap": None
        }
    ],
    "kinh_nghiem": [
        {
            "cong_ty": "",
            "chuc_danh": "",
            "thoi_gian_bat_dau": "",
            "thoi_gian_ket_thuc": "",
            "mo_ta_cong_viec": ""
        }
    ],
    "du_an": [
        {
            "ten_du_an": "",
            "vai_tro": "",
            "cong_nghe_su_dung": [],
            "mo_ta": ""
        }
    ]
}

def make_prompt(json_structure):
    json_structure_str = json.dumps(json_structure, ensure_ascii=False, indent=2)
    return f"""
    Bạn là công cụ trích xuất thông tin từ CV.
    Đọc đoạn văn bản CV được cung cấp, trả về DUY NHẤT 1 file JSON
    object đúng theo cấu trúc sau, KHÔNG thêm giải thích,
    KHÔNG bọc trong dấu ```json```:
    {json_structure_str}
    QUY TẮC BẮT BUỘC:
    - Nếu không tìm thấy thông tin:
        + Với trường mảng, thì để mảng rỗng [].
        + Với trường có giá trị mặc định là None, thì để là None.
        + Với trường chuỗi "", để chuỗi rỗng "".
    - Ngày tháng luôn ở định dạng "YYYY-MM".
    - Nếu đang làm việc hiện tại (chưa nghỉ), thoi_gian_ket_thuc = "hien_tai".
    - Trường "ngon_ngu_lap_trinh": CHỈ liệt kê các ngôn ngữ lập trình
    thực sự (ví dụ: Python, Java, JavaScript, C#, PHP, Kotlin, Swift,
    Go, C++). KHÔNG bao gồm framework (React, Spring Boot...), công
    cụ (Figma, Git, Selenium...), hay ngôn ngữ truy vấn (SQL) - những
    thứ này đưa vào "ky_nang". Một kỹ năng CHỈ được xuất hiện ở ĐÚNG
    1 trong 2 trường "ngon_ngu_lap_trinh" hoặc "ky_nang", không lặp
    lại ở cả 2.
    - TUYỆT ĐỐI không bịa thông tin không có trong text gốc.
    """

def extract_one_file(raw_text: str, prompt: str):
    retry_attempts = 2
    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                    "content": prompt},
                    {"role": "user",
                    "content": raw_text
                    }
                ],
                response_format={
                    "type": "json_object"
                },
                temperature = 0
            )

            response_text = response.choices[0].message.content

            try:
                result = json.loads(response_text)
                return result, None
            except json.JSONDecodeError as e:
                return None, f"Lỗi parse JSON {e}"
        except OpenAIError as e:
            if attempt < retry_attempts - 1:
                wait_time = 2 ** attempt
                print(f"Over speed processing {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return None, f"Lỗi gọi API {e}"

def read_json_file(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    else:
        print(f"[Err] Cannot find {path} file")

def extract_data(input_path: Path, output_path: Path, err_path: Path):
    raw_data = []
    save_number = 10

    raw_data = read_json_file(input_path)
    raw_data_len = len(raw_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and err_path.exists():
        result_data = read_json_file(output_path)
        err_data = read_json_file(err_path)
        checkpoint = len(result_data)
        count = len(result_data) + len(err_data)
    else:
        result_data = []
        err_data = []
        checkpoint = 0
        count = 0


    prompt = make_prompt(EXTRACTED_INFO_STRUCTURE)

    if raw_data is not None:
        for i in range(checkpoint, raw_data_len):
            progress = round((float(i)/raw_data_len)*100, 2)
            print(f"Percent: {progress}%", end="\r")

            raw_text = raw_data[i]["text"]
            label = raw_data[i]["label"]
            source = raw_data[i]["source"]


            response, err = extract_one_file(raw_text, prompt)

            if err == None:
                result_data.append({
                    "info": response,
                    "label": label,
                    "source": source
                })
                count += 1
            else:
                err_data.append({
                    "info": None,
                    "raw_text": raw_text,
                    "label": label,
                    "source": source,
                    "err_status": err,
                    "index": count
                })

            if (i+1) % save_number == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=4)
                    
                with open(err_path, "w", encoding="utf-8") as f:
                    json.dump(err_data, f, ensure_ascii=False, indent=4)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
        
    with open(err_path, "w", encoding="utf-8") as f:
        json.dump(err_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    extract_sets = ["train", "evaluation", "test"]

    for set in extract_sets:
        print(f"Processing {set} folder...")

        input_path = Path(f"./processed_data/{set}_raw.json")
        output_path = Path(f"./processed_data/{set}_info.json")
        error_path = Path(f"./processed_data/{set}_err.json")
        extract_data(input_path, output_path, error_path)

