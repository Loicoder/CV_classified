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

load_dotenv()

API_KEY = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=API_KEY)

CATEGORIES = (
    "Business_Analyst",
    "Developer",
    "Quality_Assurance",
    "UIUX_Designer",
)

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
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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

        try:
            parsed_json = json.load(raw_json)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"[ERR] Cannot parse the api's response into JSON")
            return None
    except OpenAIError as e:
        print(f"[ERR] API error. Err: {e}")

