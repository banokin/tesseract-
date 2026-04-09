import io
from pathlib import Path

import pytesseract
from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image
from async_utils import safe_to_thread
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

router = APIRouter(tags=["ocr"])
MAX_FILE_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GB
OCR_UNSUPPORTED_DETAIL = (
    "Для OCR поддерживаются изображения (PNG, JPG, JPEG) и PDF (первая страница). "
    "Файлы DOCX и TXT Tesseract не сканирует как документы — "
    "конвертируйте страницу в изображение или извлеките текст другими средствами."
)

_BLOCKED_OCR_EXTENSIONS = frozenset({".mp3", ".docx", ".doc", ".txt"})
_BLOCKED_OCR_CONTENT_TYPES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/plain",
    }
)


def file_too_large_exception() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail="Файл слишком большой: 1 ГБ и больше не принимаются.",
    )


def unsupported_ocr_document_exception() -> HTTPException:
    return HTTPException(status_code=415, detail=OCR_UNSUPPORTED_DETAIL)


def pdf_not_supported_exception() -> HTTPException:
    return HTTPException(
        status_code=500,
        detail="Поддержка PDF не установлена на сервере (нужен PyMuPDF).",
    )


def convert_pdf_first_page_to_png(pdf_bytes: bytes) -> bytes:
    if fitz is None:
        raise pdf_not_supported_exception()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count < 1:
            raise HTTPException(status_code=400, detail="PDF-файл не содержит страниц")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        return pix.tobytes("png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось обработать PDF: {e!r}") from e


def img_ocr(contents: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(contents))
        return pytesseract.image_to_string(image, lang="rus+eng")
    except StopIteration as e:
        # Python 3.12 не позволяет прокидывать StopIteration в asyncio Future
        # из asyncio.to_thread(...), поэтому нормализуем в обычное исключение.
        raise RuntimeError("OCR engine failed while iterating image data") from e


def validate_upload(file: UploadFile, contents: bytes) -> None:
    if len(contents) >= MAX_FILE_SIZE_BYTES:
        raise file_too_large_exception()

    filename = (file.filename or "").lower()
    ext = Path(filename).suffix
    if ext in _BLOCKED_OCR_EXTENSIONS:
        raise unsupported_ocr_document_exception()

    ct = (file.content_type or "").split(";")[0].strip().lower()
    if ct in _BLOCKED_OCR_CONTENT_TYPES:
        raise unsupported_ocr_document_exception()


async def extract_text_from_upload(file: UploadFile) -> str:
    contents = await file.read()
    validate_upload(file, contents)
    filename = (file.filename or "").lower()
    ext = Path(filename).suffix
    ct = (file.content_type or "").split(";")[0].strip().lower()
    if ext == ".pdf" or ct == "application/pdf":
        contents = convert_pdf_first_page_to_png(contents)
    try:
        return await safe_to_thread(img_ocr, contents)
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail="Ошибка OCR-движка при обработке изображения",
        ) from e
