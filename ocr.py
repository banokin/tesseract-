import asyncio
import io
from pathlib import Path

import pytesseract
from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image

router = APIRouter(tags=["ocr"])
MAX_FILE_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GB
OCR_UNSUPPORTED_DETAIL = (
    "Для OCR поддерживаются только изображения (PNG, JPG, JPEG). "
    "Файлы PDF, DOCX и TXT Tesseract не сканирует как документы — "
    "конвертируйте страницу в изображение или извлеките текст другими средствами."
)

_BLOCKED_OCR_EXTENSIONS = frozenset({".mp3", ".pdf", ".docx", ".doc", ".txt"})
_BLOCKED_OCR_CONTENT_TYPES = frozenset(
    {
        "application/pdf",
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


def img_ocr(contents: bytes) -> str:
    image = Image.open(io.BytesIO(contents))
    return pytesseract.image_to_string(image, lang="rus+eng")


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
    return await asyncio.to_thread(img_ocr, contents)
