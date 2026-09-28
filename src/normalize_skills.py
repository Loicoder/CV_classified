from pathlib import Path
from openai import OpenAI, OpenAIError
import json
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=API_KEY)

def collect_raw_skills():
    sets = ["train", "evaluation", "test"]
        
    skills = []
    for set_name in sets:
        folder_path = Path(f"./processed_data/{set_name}_info.json")
            
        if folder_path.exists():
            with open(folder_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            print("[ERR] Cannot find the folder in the directory")
            return 
        
            
        for i in range(len(data)):
            if data[i]["info"]:
                data_info = data[i]["info"]
                for skill in data_info["ngon_ngu_lap_trinh"]:
                    skills.append(skill.strip())
                for skill in data_info["ky_nang"]:
                    skills.append(skill.strip())
                for project in data_info["du_an"]:
                    for skill in project["cong_nghe_su_dung"]:
                        skills.append(skill.strip())
    skills = list(set(skills))
    return skills


def make_prompt(skill_list):
    return f"""
        Dưới đây là danh sách các kỹ năng/công nghệ được trích xuất
        thô từ nhiều CV, có thể chứa nhiều cách viết khác nhau cho
        cùng 1 công nghệ (ví dụ "ReactJS", "React.js", "React" là
        cùng 1 thứ).

        Danh sách: {skill_list}

        Hãy nhóm các skill có cùng ý nghĩa lại với nhau, chọn 1 tên
        chuẩn, viết đúng chính tả phổ biến nhất cho mỗi nhóm.

        Trả về DUY NHẤT 1 JSON object dạng ánh xạ:
        {{"ten_skill_goc_1": "ten_chuan", ...}}
        Bao gồm TẤT CẢ skill trong danh sách đầu vào, kể cả những
        skill không cần đổi tên (ánh xạ về chính nó).
    """

def process_one_batch(batch_arr, prompt):
    retry_attemps = 2
    json.dumps(batch_arr, ensure_ascii=False)
    for attemp in range(retry_attemps):
        try:
            response = client.chat.completions.create(
                model = "gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                ],
                response_format={"type":"json_object"},
                temperature=0
            )

            result = response.choices[0].message.content
            return json.loads(result)
        
        except (OpenAIError, json.JSONDecodeError) as e:
            print(f"[ERR] Lỗi api: {e}")
            return None

def normalize_skill():
    batch = 10
    normalized_skill_count = 0
    batch_count = 0

    skills = collect_raw_skills()
    normalized_skills = {}
    skills_len = len(skills)
    
    # while normalized_skill_count < skills_len:
    while normalized_skill_count < 10:
        batch_count += 1
        print(f"Đang khởi tạo batch {batch_count}/{skills_len/batch}")
        batch_arr = []
        print("Đang chia dữ liệu thành từng batch")
        for i in range(normalized_skill_count, normalized_skill_count + batch):
            if skills[i]:
                normalized_skill_count += 1
                batch_arr.append(skills[i])
        print("Hoàn tất việc chia batch")
                
        prompt = make_prompt(batch_arr)
        print("Đang chờ phản hồi từ api")
        batch_result = process_one_batch(batch_arr, prompt)

        if batch_result is not None:
            normalized_skills.update(batch_result)

    destination_file_path = "./processed_data/normalized_skills.json"

    print(f"Đang lưu file json vào thư mục {destination_file_path}")
    with open(destination_file_path, "w", encoding="utf-8") as f:
        json.dump(normalized_skills, f, ensure_ascii=False, indent=4)
    print("Hoàn tất việc lưu file")

    return normalized_skills

def reflex(skill_mapping, info_file_path: Path):
    if info_file_path.exists():
        with open(info_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("[ERR] Cannot find the file in the directory")

    data_len = len(data)
    for i in range(data_len):
        if data[i]["info"]:
            new_skills = []
            data_info = data[i]["info"]

            for current_skill in data_info["ky_nang"]:
                pass

if __name__ == "__main__":
    skill_mapping = normalize_skill()
    reflex(skill_mapping, Path("./processed_data/train_info.json"))
        