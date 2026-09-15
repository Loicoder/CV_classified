"""
    Class CVReader đọc và trích xuất text từ các CV, hỗ trợ các định dạng: PDF, DOCX, PNG, JPG/JPEG.
    Các luồng xử lý:
        - PDF text: dùng thư viện pymupdf để trích xuất text
        - PDF ảnh: pymupdf -> BytesIO -> PIL -> pytesseract OCR
        - DOCX: dùng thư viện docx
        - Ảnh (PNG/JPG): PIL -> pytesseract OCR
"""

from pathlib import Path
from io import BytesIO
import pymupdf
import pytesseract
from PIL import Image
from docx import Document


class UnsupportedFileTypeError(Exception):
    """Raised khi định dạng file không được hỗ trợ."""
    pass


class CVReader:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}

    def __init__(self, ocr_lang: str = "eng", min_chars_for_valid_pdf_text: int = 20):
        self.ocr_lang = ocr_lang
        self.min_chars_for_valid_pdf_text = min_chars_for_valid_pdf_text # nếu file pdf ít hơn số ký tự này, coi là pdf ảnh

    """
        Hàm đọc đường dẫn file và gửi đến hàm đọc file tương ứng
        với định dạng file, nếu định dạng không hỗ trợ, raise error
    """
    def read(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        ext = path.suffix.lower() # đọc đuôi file

        if ext == ".pdf":
            return self._read_pdf(path)
        elif ext == ".docx":
            return self._read_docx(path)
        elif ext in (".png", ".jpg", ".jpeg"):
            return self._read_image(path)
        else:
            raise UnsupportedFileTypeError(
                f"Định dạng '{ext}' chưa được hỗ trợ. "
                f"Chỉ hỗ trợ: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )


    """
        Hàm đọc tất cả các file trong thư mục folder_path, nếu
        bật recursive, nó sẽ đọc cả những file nằm trong thư mục
        con của folder_path
        Kết quả trả về là một dict, với key là đường dẫn file còn
        value là text trích xuất được từ file đó
    """
    def read_folder(self, folder_path: str, recursive: bool = True) -> dict:
        folder = Path(folder_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Không phải thư mục hợp lệ: {folder_path}")

        pattern = "**/*" if recursive else "*"
        results = {}

        for file_path in folder.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    results[str(file_path)] = self.read(file_path)
                except Exception as e:
                    results[str(file_path)] = f"[LỖI] Không đọc được file: {e}"

        return results


    """
        Hàm đọc file PDF
        Đọc từng trang, nếu text trên trang >= min_chars_for_valid_pdf_text
        thì đọc text bình thường, còn ít hơn sẽ gọi hàm scan ocr
        Sau đó nối các trang lại với nhau
    """
    def _read_pdf(self, path: Path) -> str:
        text_parts = []

        with pymupdf.open(path) as pdf_file:
            for page in pdf_file:
                page_text = page.get_text().strip()

                if len(page_text) >= self.min_chars_for_valid_pdf_text:
                    text_parts.append(page_text)
                else:
                    ocr_text = self._ocr_pdf_page(page)
                    text_parts.append(ocr_text)

        return " ".join(text_parts) # nối các trang lại với nhau, cách nhau khoảng trắng


    """
        Hàm scan OCR cho từng trang PDF
    """
    def _ocr_pdf_page(self, page, zoom: float = 2.0) -> str:
        matrix = pymupdf.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix)
        image_bytes = pixmap.tobytes("png")
        image = Image.open(BytesIO(image_bytes))

        return pytesseract.image_to_string(image, lang=self.ocr_lang).strip()

    """
        Hàm đọc đoạn văn và bảng trong file word, sau đó nối các
        nội dung lại với nhau
    """
    def _read_docx(self, path: Path) -> str:
        document = Document(path)
        text_parts = []

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())

        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    text_parts.append(row_text)

        return " ".join(text_parts).strip() # nối các đoạn văn và bảng lại với nhau, cách nhau khoảng trắng


    """
        hàm đọc ảnh và trích xuất ra text bằng pytesseract
    """
    def _read_image(self, path: Path) -> str:
        image = Image.open(path)
        return pytesseract.image_to_string(image, lang=self.ocr_lang).strip()
