import json
import random
from pathlib import Path

from cv_reader import CVReader


CATEGORIES = (
	"Business_Analyst",
	"Developer",
	"Quality_Assurance",
	"UIUX_Designer",
)
SUPPORTED_EXTENSIONS = CVReader.SUPPORTED_EXTENSIONS
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# hàm tạo file json, gán nhãn cho toàn bộ các CV trong thư mục raw_data
def collect_labeled_data(raw_data_dir: Path, reader: CVReader) -> dict[str, list[dict[str, str]]]:
    data_by_label = {}
    cache_path = PROJECT_ROOT / "processed_data" / ".cv_text_cache.json"
    try:
        text_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        text_cache = {}
    print(f"Đang tạo file JSON cho {raw_data_dir}...")

    for label in CATEGORIES:
        print(f"Đang xử lý nhãn: {label}")
        category_dir = raw_data_dir / label
        if not category_dir.is_dir():
            continue

        samples = []
        files = [
            file_path for file_path in category_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        for index, file_path in enumerate(files, start=1):
            print(f"Đang xử lý file [{index}|{len(files)}]")

            try:
                cache_key = str(file_path.resolve())
                file_stat = file_path.stat()
                cached_file = text_cache.get(cache_key)
                if (
                    cached_file
                    and cached_file["size"] == file_stat.st_size
                    and cached_file["mtime_ns"] == file_stat.st_mtime_ns
                ):
                    text = cached_file["text"]
                else:
                    text = reader.read(str(file_path)).strip()
                    text_cache[cache_key] = {
                        "size": file_stat.st_size,
                        "mtime_ns": file_stat.st_mtime_ns,
                        "text": text,
                    }
            except Exception as error:
                print(f"[LỖI] Không đọc được {file_path.name}: {error}")
                continue

            if text:
                samples.append({
					"text": text,
					"label": label,
					"source": file_path.name,
				})

        data_by_label[label] = samples

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(text_cache, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Hoàn tất việc tạo file JSON cho {raw_data_dir}.")
    return data_by_label

# hàm chia dữ liệu đã gán nhãn ngẫu nhiên thành tập train và test với tỉ lệ 4/1
def split_data(
	data_by_label: dict[str, list[dict[str, str]]],
	test_ratio: float = 0.2,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    print("Đang chia dữ liệu thành tập train và test...")
    random_generator = random.Random()
    train_data = []
    test_data = []

    for label, samples in data_by_label.items():
        shuffled_samples = samples.copy()
        random_generator.shuffle(shuffled_samples)

        if len(shuffled_samples) < 2:
            test_count = 0
        else:
            test_count = max(1, round(len(shuffled_samples) * test_ratio))

        test_data.extend(shuffled_samples[:test_count])
        train_data.extend(shuffled_samples[test_count:])

    print("Hoàn tất việc chia dữ liệu.")
    # tránh các ngành nghề giống nhau nằm kế nhau
    random_generator.shuffle(train_data)
    random_generator.shuffle(test_data)
    return train_data, test_data

# lưu 2 tập dữ liệu train và test vào 2 file json riêng biệt
def save_json(data: list[dict[str, str]], output_path: Path) -> None:
    print(f"Đang ghi dữ liệu vào {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
		json.dumps(data, ensure_ascii=False, indent=2),
		encoding="utf-8",
	) 
    print(f"Hoàn tất việc ghi dữ liệu.")

if __name__ == "__main__":
    raw_data_dir = PROJECT_ROOT / "raw_data"
    processed_data_dir = PROJECT_ROOT / "processed_data"
    reader = CVReader(ocr_lang="vie+eng", min_chars_for_valid_pdf_text=10)

    data_by_label = collect_labeled_data(raw_data_dir, reader)
    train_data, test_data = split_data(data_by_label)

    save_json(train_data, processed_data_dir / "train.json")
    save_json(test_data, processed_data_dir / "test.json")

    print(f"Đã ghi {len(train_data)} mẫu vào processed_data/train.json")
    print(f"Đã ghi {len(test_data)} mẫu vào processed_data/test.json")