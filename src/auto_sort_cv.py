import re
import shutil
from pathlib import Path
from cv_reader import CVReader

# mảng từ khóa cho từng nhóm ngành
CATEGORY_KEYWORDS = {
    "Developer": [
        "developer"
    ],
    "Business_Analyst": [
        "business analyst"
    ],
    "UIUX_Designer": [
        "ui/ux designer"
    ],
    "Quality_Assurance": [
        "quality assurance"
    ],
}

source_folder = Path("raw_data/unclassified")
class_root = Path("raw_data")
reader = CVReader(ocr_lang="eng", min_chars_for_valid_pdf_text=10)

# biên dịch trước regrex nhằm tăng tốc độ tìm kiếm từ khóa
compiled_keywords = {} # là một dict có value là mảng các tuple (keyword, pattern)

def compile_keyword():
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            pattern = re.compile(r"(?<![\w])" +  # không có ký tự chữ số hoặc chữ cái trước từ khóa
                                re.escape(keyword) + # bảo vệ ký tự đặc biệt 
                                r"(?![\w])",  # không có ký tự chữ số hoặc chữ cái sau từ khóa
                                re.IGNORECASE) # không phân biệt hoa thường
            compiled_keywords.setdefault(category, []).append((keyword, pattern))

def sort():
    for ext in CVReader.SUPPORTED_EXTENSIONS:
        for file_path in source_folder.glob(f"*{ext}"):
            process_one_file(file_path)


def process_one_file(file_path: Path):
    try:
        content = reader.read(file_path)
    except Exception as e:
        print(f"[LỖI] Không đọc được {file_path.name}: {e}")
        return

    category = classify(content)

    if category:
        if len(category) > 1:
            for i in range(len(category)):
                destination_folder = class_root / category[i]
                transfer_file(file_path, destination_folder, i+1, len(category))
        else:
            destination_folder = class_root / category[0]
            transfer_file(file_path, destination_folder, 1, 1)

def classify(content: str):
    matched = []
    for category, keyword_patterns in compiled_keywords.items():
        for keyword, pattern in keyword_patterns:
            if pattern.search(content):
                matched.append(category)

    return matched

def transfer_file(file_path: Path, destination_folder: Path, current: int, total: int):
    destination_folder.mkdir(parents=True, exist_ok=True) # nếu thư mục chưa tồn tại thì tạo mới
    destination_path = destination_folder / file_path.name

    # Tránh ghi đè nếu trùng tên file bằng cách thêm biến đếm vào cuối tên file
    counter = 1
    while destination_path.exists():
        destination_path = destination_folder / f"{file_path.stem}_{counter}{file_path.suffix}"
        counter += 1

    if total > 1 and current < total: 
        shutil.copy(str(file_path), str(destination_path))  
    if total == 1 or current == total:
        shutil.move(str(file_path), str(destination_path))

if __name__ == "__main__":
    compile_keyword()
    sort()