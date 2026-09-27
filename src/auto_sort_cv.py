"""
đây là file dùng để tự động sắp xếp các file cv từ thư mục ban đầu sang các thư mục
nhóm ngành nghề tương ứng, sử dụng kĩ thuật word-maching, đầu tiên đọc text trong cv
sau đó so sánh với CATEGORY_KEYWORDS xem nó thuộc nghành nghề nào
"""

import re
import shutil
from pathlib import Path
from cv_reader import CVReader

# mảng từ khóa cho từng nhóm ngành
CATEGORY_KEYWORDS = {
    "Developer": [
        "developer",
        "software developer",
        "software engineer",
        "web developer",
        "frontend developer",
        "front-end developer",
        "backend developer",
        "back-end developer",
        "full stack developer",
        "full-stack developer",
        "mobile developer",
        "application developer",
        "programmer",
        "lập trình viên",
        "kỹ sư phần mềm",
        "phát triển phần mềm",
        "lập trình web",
        "lập trình backend",
        "lập trình frontend",
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "php",
        "kotlin",
        "swift",
        "go",
        "golang",
        "ruby",
        "dart",
        "html",
        "css",
        
    ],
    "Business_Analyst": [
        "business analyst",
        "business analysis",
        "system analyst",
        "systems analyst",
        "business systems analyst",
        "process analyst",
        "chuyên viên phân tích nghiệp vụ",
        "phân tích nghiệp vụ",
        "phân tích kinh doanh",
        "sql",
        "python",
        "r programming",
        "r language",
        "vba",
        "mdx",
    ],
    "UIUX_Designer": [
        "ui/ux designer",
        "ui ux designer",
        "ui designer",
        "ux designer",
        "user interface designer",
        "user experience designer",
        "interaction designer",
        "product designer",
        "visual designer",
        "thiết kế ui",
        "thiết kế ux",
        "thiết kế giao diện",
        "thiết kế trải nghiệm người dùng",
        "html",
        "css",
        "javascript",
        "typescript",
        "sass",
        "scss",
    ],
    "Quality_Assurance": [
        "quality assurance",
        "qa engineer",
        "qa tester",
        "software tester",
        "software test engineer",
        "test engineer",
        "manual tester",
        "automation tester",
        "quality control tester",
        "kiểm thử phần mềm",
        "kỹ sư kiểm thử",
        "nhân viên kiểm thử",
        "kiểm thử viên",
        "python",
        "java",
        "javascript",
        "typescript",
        "c#",
        "ruby",
        "kotlin",
        "swift",
        "sql",
    ],
}

source_folder = Path("raw_data/unclassified")
class_root = Path("raw_data")
reader = CVReader(ocr_lang="vie+eng", min_chars_for_valid_pdf_text=10)

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


# Hàm sắp xếp các file trong thư mục vào các thư mục ngành nghề
def sort():
    for ext in CVReader.SUPPORTED_EXTENSIONS:
        files = list(source_folder.glob(f"*{ext}"))
        for index, file_path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] Processing: {file_path.name}")
            process_one_file(file_path)

# Hàm xử lý từng file, đọc nội dung và phân loại vào thư mục con
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

# hàm phân loại nội dung dựa trên từ khóa đã biên dịch trước
def classify(content: str):
    matched = []
    for category, keyword_patterns in compiled_keywords.items():
        for keyword, pattern in keyword_patterns:
            if pattern.search(content):
                if category not in matched:
                    matched.append(category)
                break

    return matched

# hàm chuyển file vào thư mục tương ứng
def transfer_file(file_path: Path, destination_folder: Path, current: int, total: int):
    destination_folder.mkdir(parents=True, exist_ok=True) # nếu thư mục chưa tồn tại thì tạo mới
    destination_path = destination_folder / file_path.name

    # Tránh ghi đè nếu trùng tên file bằng cách thêm biến đếm vào cuối tên file
    counter = 1
    while destination_path.exists():
        destination_path = destination_folder / f"{file_path.stem}_{counter}{file_path.suffix}"
        counter += 1

    # nếu có nhiều nghề cùng khớp, thì copy vào các thư mục đó, nếu không thì move vào thư mục tương ứng
    if total > 1 and current < total: 
        shutil.copy(str(file_path), str(destination_path))  
    if total == 1 or current == total:
        shutil.move(str(file_path), str(destination_path))

# đếm số lượng file trong mỗi thư mục, nhằm kiểm tra tính cân bằng dữ liệu 
def count_cv(root_folder: Path = Path("raw_data")):
    if not root_folder.exists() or not root_folder.is_dir():
        print(f"[LỖI] Thư mục {root_folder} không tồn tại hoặc không phải là thư mục.")
        return

    for category in CATEGORY_KEYWORDS.keys():
        category_folder = root_folder / category
        if category_folder.exists() and category_folder.is_dir():
            count = len(list(category_folder.glob("*.*")))
            print(f"{category} có {count} CV")
        else:
            print(f"Không tồn tại thư mục {category}")

if __name__ == "__main__":
    compile_keyword()
    sort()
    count_cv()