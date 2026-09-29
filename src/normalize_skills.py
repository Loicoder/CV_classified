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
                    raw_skill = skill.strip()
                    if raw_skill != "":
                        skills.append(raw_skill.strip())
                for skill in data_info["ky_nang"]:
                    raw_skill = skill.strip()
                    if raw_skill != "":
                        skills.append(raw_skill.strip())
                for project in data_info["du_an"]:
                    for skill in project["cong_nghe_su_dung"]:
                        raw_skill = skill.strip()
                        if raw_skill != "":
                            skills.append(raw_skill.strip())
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
            continue
    return None

def normalize_skill():
    batch = 100
    normalized_skill_count = 0
    batch_count = 0

    skills = collect_raw_skills()
    normalized_skills = {}
    skills_len = len(skills)
    
    for i in range(0, skills_len, batch):
        print("#"*60)
        batch_count += 1
        print(f"Đang khởi tạo batch {batch_count}/{-(-skills_len//batch)}")
        print("Đang chia dữ liệu thành từng batch")
        batch_arr = skills[i:i+batch]
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
    print(f"Bắt đầu quá trình ánh xạ {info_file_path.name}")
    if info_file_path.exists():
        with open(info_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("[ERR] Cannot find the file in the directory")
        return 
    
    data_len = len(data)
    for i in range(data_len):
        if data[i]["info"]:
            data_info = data[i]["info"]

            # ánh xạ ngôn ngữ lập trình
            new_languages = []
            for current_language in data_info["ngon_ngu_lap_trinh"]:
                normalize_name = skill_mapping.get(current_language.strip(), current_language)
                new_languages.append(normalize_name)
            data_info["ngon_ngu_lap_trinh"] = new_languages

            # ánh xạ kỹ năng
            new_skills = []
            for current_skill in data_info["ky_nang"]:
                normalize_skill_name = skill_mapping.get(current_skill.strip(), current_skill)
                new_skills.append(normalize_skill_name)
            data_info["ky_nang"] = new_skills

            # ánh xạ công nghệ dự án
            for project in data_info["du_an"]:
                new_techs = []
                for current_tech in project["cong_nghe_su_dung"]:
                    normalize_tech = skill_mapping.get(current_tech.strip(), current_tech)
                    new_techs.append(normalize_tech)
                project["cong_nghe_su_dung"] = new_techs  

    with open(info_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)              
    print("Hoàn tất ánh xạ")

if __name__ == "__main__":
    skill_mapping = normalize_skill()
    
    sets = ["train", "evaluation", "test"]
    for set_name in sets:
        file_path = Path(f"./processed_data/{set_name}_info.json")
        reflex(skill_mapping, file_path)
        