import asyncio
import io
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import pytesseract
from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

router = APIRouter(tags=["ocr"])
P = ParamSpec("P")
T = TypeVar("T")
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


def _run_sync_guarded(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    try:
        return func(*args, **kwargs)
    except StopIteration as e:
        raise RuntimeError("Background sync function raised StopIteration") from e


async def safe_to_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return await asyncio.to_thread(_run_sync_guarded, func, *args, **kwargs)


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


def upscale_jpeg_for_ocr(image_bytes: bytes, scale: float = 2.5) -> bytes:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception:
        return image_bytes

    image_format = (image.format or "").upper()
    if image_format not in {"JPEG", "JPG"}:
        return image_bytes

    width, height = image.size
    new_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    output = io.BytesIO()
    resized.save(output, format="JPEG", quality=95, optimize=True)
    return output.getvalue()


def img_ocr(contents: bytes, *, config: str | None = None) -> str:
    try:
        image = Image.open(io.BytesIO(contents))
        return pytesseract.image_to_string(image, lang="rus+eng", config=config or "")
    except StopIteration as e:
        # Python 3.12 не позволяет прокидывать StopIteration в asyncio Future
        # из asyncio.to_thread(...), поэтому нормализуем в обычное исключение.
        raise RuntimeError("OCR engine failed while iterating image data") from e


def _serialize_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_variants_for_ocr(image: Image.Image) -> list[Image.Image]:
    base = image.convert("RGB")
    gray = ImageOps.grayscale(base)
    boosted = ImageEnhance.Contrast(gray).enhance(1.9)
    sharpened = boosted.filter(ImageFilter.SHARPEN)
    binary = sharpened.point(lambda p: 255 if p > 165 else 0)
    return [base, gray, boosted, sharpened, binary]


def _crop_variants_for_ocr(image: Image.Image) -> list[Image.Image]:
    w, h = image.size
    if w < 400 or h < 400:
        return []
    crops: list[Image.Image] = []
    # Крупные горизонтальные фрагменты для таблиц ЕГРН.
    crops.append(image.crop((0, 0, w, int(h * 0.55))))
    crops.append(image.crop((0, int(h * 0.25), w, int(h * 0.85))))
    # Центральный блок с основными характеристиками.
    crops.append(image.crop((int(w * 0.05), int(h * 0.15), int(w * 0.95), int(h * 0.75))))
    # Правые вертикальные полосы (серия/номер паспорта часто печатаются вертикально).
    right_strip = image.crop((int(w * 0.78), 0, w, h))
    right_top = image.crop((int(w * 0.72), 0, w, int(h * 0.55)))
    right_bottom = image.crop((int(w * 0.72), int(h * 0.45), w, h))
    for strip in (right_strip, right_top, right_bottom):
        crops.append(strip)
        crops.append(strip.rotate(90, expand=True, fillcolor="white"))
        crops.append(strip.rotate(270, expand=True, fillcolor="white"))
    return [c for c in crops if c.size[0] > 100 and c.size[1] > 100]


def img_ocr_multi_pass(contents: bytes, *, include_crops: bool = True) -> str:
    """
    Несколько проходов OCR: разные предобработки + опционально фрагменты страницы.
    Возвращает объединённый текст для более устойчивого парсинга.
    """
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise RuntimeError("OCR engine failed to decode image") from e

    scans: list[str] = []
    variants = _image_variants_for_ocr(image)
    configs = (
        None,
        "--psm 6",
        "--psm 11",
    )
    for variant in variants:
        payload = _serialize_png(variant)
        for cfg in configs:
            scans.append(img_ocr(payload, config=cfg))

    if include_crops:
        for crop in _crop_variants_for_ocr(image.convert("RGB")):
            for crop_variant in _image_variants_for_ocr(crop):
                payload = _serialize_png(crop_variant)
                for cfg in configs:
                    scans.append(img_ocr(payload, config=cfg))

    # Склеиваем без дублей, сохраняя порядок.
    seen: set[str] = set()
    chunks: list[str] = []
    for s in scans:
        txt = (s or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        chunks.append(txt)
    return "\n\n==== OCR PASS ====\n\n".join(chunks)


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
    elif ext in {".jpg", ".jpeg"} or ct in {"image/jpeg", "image/jpg"}:
        contents = upscale_jpeg_for_ocr(contents, scale=2.5)
    try:
        return await safe_to_thread(img_ocr, contents)
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail="Ошибка OCR-движка при обработке изображения",
        ) from e
