import asyncio
import io
from pathlib import Path

import pytesseract
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from exceptions import (
    empty_file_exception,
    file_too_large_exception,
    image_not_recognized_exception,
    ocr_exception,
    tesseract_exception,
    unsupported_mp3_exception,
    unsupported_ocr_document_exception,
)

router = APIRouter(tags=["ocr"])
MAX_FILE_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GB

_BLOCKED_OCR_EXTENSIONS = frozenset({".pdf", ".docx", ".doc", ".txt"})
_BLOCKED_OCR_CONTENT_TYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/plain",
    }
)

def img_ocr(contents: bytes) -> str:
    image = Image.open(io.BytesIO(contents))
    return pytesseract.image_to_string(image, lang="rus+eng")


def validate_upload(file: UploadFile, contents: bytes) -> None:
    if not contents:
        raise empty_file_exception()
    if len(contents) >= MAX_FILE_SIZE_BYTES:
        raise file_too_large_exception()

    filename = (file.filename or "").lower()
    ext = Path(filename).suffix
    if ext == ".mp3" or file.content_type in {"audio/mpeg", "audio/mp3"}:
        raise unsupported_mp3_exception()

    if ext in _BLOCKED_OCR_EXTENSIONS:
        raise unsupported_ocr_document_exception()

    ct = (file.content_type or "").split(";")[0].strip().lower()
    if ct in _BLOCKED_OCR_CONTENT_TYPES:
        raise unsupported_ocr_document_exception()


@router.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        validate_upload(file, contents)
        text_task = asyncio.create_task(asyncio.to_thread(img_ocr, contents))
        text = await text_task

        return {
            "filename": file.filename,
            "text": text,
        }
    except UnidentifiedImageError:
        raise image_not_recognized_exception()
    except pytesseract.pytesseract.TesseractError as e:
        raise tesseract_exception(e)
    except HTTPException:
        raise
    except Exception as e:
        raise ocr_exception(e)
