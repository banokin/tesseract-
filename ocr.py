"""
OCR через Tesseract: низкоуровневая функция и HTTP-роутер.
"""
import io
from pathlib import Path

import pytesseract
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from concurrency import run_blocking

router = APIRouter(tags=["ocr"])
MAX_FILE_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GB


def img_ocr(contents: bytes) -> str:
    image = Image.open(io.BytesIO(contents))
    return pytesseract.image_to_string(image, lang="rus+eng")


def validate_upload(file: UploadFile, contents: bytes) -> None:
    if not contents:
        raise HTTPException(status_code=400, detail="Файл пустой")
    if len(contents) >= MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Файл слишком большой: 1 ГБ и больше не принимаются.",
        )

    filename = (file.filename or "").lower()
    ext = Path(filename).suffix
    if ext == ".mp3" or file.content_type in {"audio/mpeg", "audio/mp3"}:
        raise HTTPException(status_code=415, detail="Файлы MP3 не поддерживаются.")


@router.post("/ocr")
async def ocr_image(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        validate_upload(file, contents)
        text = await run_blocking(request, img_ocr, contents)

        return {
            "filename": file.filename,
            "text": text,
        }
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=422,
            detail="Не удалось распознать изображение. Проверьте формат файла.",
        )
    except pytesseract.pytesseract.TesseractError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка Tesseract: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка OCR: {str(e)}")
