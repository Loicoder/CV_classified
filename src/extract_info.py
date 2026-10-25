"""
hàm đọc dữ liệu từ cv, sau đó gửi vào prompt cho model để trsich xuất thông tin và phân loại, sau đó chuyển
file cv vào thư mục tương ứng, đồng thời gom các thông tin vào file json
"""
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from pathlib import Path
from cv_reader import CVReader
import json
import os
import random
import shutil

load_dotenv()

API_KEY = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=API_KEY)

CATEGORY = (
    "Business_Analyst",
    "Developer",
    "Quality_Assurance",
    "UIUX_Designer",
    "Other"
)

EXTRACTED_INFO_STRUCTURE = {
    "information": {
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
    },
    "label": ""
}

SUPPORTED_EXTENSIONS = CVReader.SUPPORTED_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = PROJECT_ROOT / "processed_data"
RAW_DATA = PROJECT_ROOT / "raw_data"

# hàm đọc dữ liệu trong file cv
def read_text(reader, file_path: Path):
    try:
        text = reader.read(file_path)
        return text
    except Exception as e:
        print(f"[ERR] Cannot read text in cv {file_path.name}. Err: {e}")
        return None

# hàm gọi api lấy dữ liệu trả về là json
def call_api_for_one_file(raw_text, prompt):
    retries = 3
    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": raw_text
                    }
                ],
                response_format={"type":"json_object"},
                temperature=0
            )
            
            raw_json = response.choices[0].message.content
            usage = response.usage
            try:
                parsed_json = json.loads(raw_json)
                return parsed_json, usage
            except json.JSONDecodeError as e:
                print(f"[ERR] Cannot parse the api's response into JSON")
                return None
        except OpenAIError as e:
            print(f"[ERR] API error. Err: {e}")
    return None, None
# hàm tạo prompt để gửi lên api
def generate_prompt():
    schema_example = json.dumps(EXTRACTED_INFO_STRUCTURE, ensure_ascii=False, indent=2)
    return f""" 
    Bạn là công cụ đọc CV, dùng để vừa PHÂN LOẠI ngành nghề, vừa TRÍCH XUẤT
    thông tin hữu ích có cấu trúc từ văn bản CV được cung cấp.

    Trả về DUY NHẤT 1 JSON object đúng theo cấu trúc sau, KHÔNG thêm giải
    thích, KHÔNG bọc trong dấu markdown:

    {schema_example}

    QUY TẮC CHO "label" (BẮT BUỘC làm trước, dựa trên TOÀN BỘ văn bản):
    - Chọn ĐÚNG 1 giá trị trong danh sách sau, viết CHÍNH XÁC như liệt kê
    (kể cả dấu gạch dưới, chữ hoa/thường): {CATEGORY}
    - "Developer": lập trình phần mềm, phát triển ứng dụng/web/mobile.
    - "Business_Analyst": phân tích nghiệp vụ, thu thập yêu cầu, BRD/BPMN.
    - "Quality_Assurance": kiểm thử phần mềm, test case, automation test.
    - "UIUX_Designer": thiết kế giao diện, trải nghiệm người dùng, wireframe.
    - "Other": CV không thuộc 4 nhóm trên (ví dụ kế toán, giáo viên, sales...).
    - Dựa vào chức danh, kỹ năng, và mô tả công việc trong TOÀN BỘ văn bản
    để quyết định, không chỉ dựa vào 1 từ khóa đơn lẻ xuất hiện ngẫu nhiên.
    - Nếu CV có dấu hiệu của nhiều hơn 1 nhóm (ví dụ vừa code vừa test),
    chọn nhóm chiếm phần lớn thời gian/trách nhiệm công việc gần nhất.

    QUY TẮC CHO CÁC TRƯỜNG TRÍCH XUẤT CÒN LẠI:
    - Trường không tìm thấy thông tin:
        + Với trường mảng, để mảng rỗng [].
        + Với trường số (nam_tot_nghiep, nam_cap), để null.
        + Với trường chuỗi, để chuỗi rỗng "".
    - Ngày tháng luôn ở định dạng "YYYY-MM". Nếu đang làm việc hiện tại
    (chưa nghỉ), thoi_gian_ket_thuc = "hien_tai".
    - "ngon_ngu_lap_trinh": CHỈ liệt kê ngôn ngữ lập trình thực sự (ví dụ:
    Python, Java, JavaScript, C#, PHP, Kotlin, Swift, Go, C++). KHÔNG bao
    gồm framework, công cụ, hay SQL - những thứ này đưa vào "ky_nang".
    Một kỹ năng chỉ xuất hiện ở ĐÚNG 1 trong 2 trường này, không lặp lại.
    - "ky_nang": công cụ, framework, phương pháp, công nghệ chuyên môn
    (ví dụ: React, Figma, Selenium, BPMN, SQL Server).
    - "ky_nang_khac": CHỈ chứa kỹ năng mềm và ngoại ngữ (ví dụ: giao tiếp,
    làm việc nhóm, tiếng Anh giao tiếp). KHÔNG chứa công nghệ/công cụ.
    - "kinh_nghiem.chuc_danh": vị trí đảm nhiệm TẠI CÔNG TY đó.
    - "du_an.vai_tro": vị trí đảm nhiệm TRONG DỰ ÁN đó, có thể khác
    chuc_danh chính. Nếu 1 đoạn CV vừa có tên công ty vừa có tên dự án cụ
    thể, điền vào CẢ HAI mục kinh_nghiem và du_an, không chỉ chọn 1.
    - TUYỆT ĐỐI không bịa thông tin không có trong văn bản gốc.
    - Giữ nguyên ngôn ngữ của nội dung trong CV. Không dịch, không diễn giải hoặc chuẩn hóa sang ngôn ngữ khác. Tên các trường JSON vẫn sử dụng tiếng Việt theo schema đã cung cấp.
    """

# hàm move file từ raw_data vào thư mục label tương ứng
def move_file_to_label_folder(file_path: Path, file_folder: Path, label: str):
    destination_folder = file_folder / label
    destination_folder.mkdir(parents=True, exist_ok=True) 

    destination_path = destination_folder / file_path.name
    shutil.move(str(file_path), str(destination_path))

def write_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# hàm lưu file json vào thư mục processed_data
def save_json(data, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            current_data = json.load(f)
    else:
        current_data = []

    if not isinstance(current_data, list):
        raise ValueError("File JSON hiện tại không phải danh sách")

    if isinstance(data, list):
        current_data.extend(data)
    else:
        current_data.append(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)

def main_process():
    raw_folder = RAW_DATA / "unclassified"
    output_path = PROCESSED_DATA / "information.json"
    BATCH_SIZE = 10
    auto_save_cache = []
    money_on_api_call = 0

    print("Khởi tạo reader cv...")
    reader = CVReader(ocr_lang="vie+eng", min_chars_for_valid_pdf_text=10)
    
    print("Đang gen prompt cho API...")
    prompt = generate_prompt()
    
    # main loop
    print("Đang xử lý file...")
    for ext in SUPPORTED_EXTENSIONS:
        files = list(raw_folder.glob(f"*{ext}"))
        if len(files) == 0: continue
        for index, file_path in enumerate(files, start=1):
            print(file_path.name)
            print(f"{ext}: {round(float(index)/len(files)*100)}%", end="\r")
            raw_text = read_text(reader, file_path)
            if raw_text is None: continue
            response, usage = call_api_for_one_file(raw_text, prompt)
        
            if response is None and auto_save_cache:
                save_json(auto_save_cache, output_path)
                auto_save_cache.clear()
                continue

            money_per_cv = (float(usage.prompt_tokens) * 0.2 + float(usage.completion_tokens) * 1.25 ) / 1000000
            money_on_api_call += money_per_cv
            

            cv_label = response["label"]

            if cv_label != "Other":
                if cv_label not in CATEGORY:
                    print(f"[ERR] Nhãn không hợp lệ: {cv_label}")
                    continue
                response["file_name"] = file_path.name
                auto_save_cache.append(response)

                if len(auto_save_cache) >= BATCH_SIZE:
                    save_json(auto_save_cache, output_path)
                    auto_save_cache.clear()

            print(f"Tổng số tiền đã dùng để gọi API: {money_on_api_call:.2f}")
            move_file_to_label_folder(file_path, RAW_DATA, cv_label)

        if auto_save_cache:
            save_json(auto_save_cache, output_path)
            auto_save_cache.clear()

        print()
    print("Hoàn tất việc đọc thông tin hữu ích từ CV")

def split_data(info_file_path: Path, output_folder: Path, train_ratio: float = 0.8, evaluation_ratio: float = 0.1):
    print("Bắt đầu quá trình chia và xáo trộn dữ liệu...")
    random_generator = random.Random()

    if info_file_path.exists():
        with open(info_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(f"[ERR] Cannot read the information file")
        return

    random_generator.shuffle(data)
    data_size = len(data)

    train = []
    evaluation = []
    test = []

    train_size = round(train_ratio * data_size)
    evaluation_size = round(evaluation_ratio * data_size)

    # slicing
    train.extend(data[:train_size])
    evaluation.extend(data[train_size:train_size+evaluation_size])
    test.extend(data[train_size+evaluation_size:])

    #saving
    write_json(train, output_folder / "train.json")
    write_json(evaluation, output_folder / "evaluation.json")
    write_json(test, output_folder / "test.json")

if __name__ == "__main__":
    # main_process()
    split_data(
        info_file_path=PROCESSED_DATA / "information.json",
        train_ratio=0.8,
        evaluation_ratio=0.1,
        output_folder=PROCESSED_DATA
    )