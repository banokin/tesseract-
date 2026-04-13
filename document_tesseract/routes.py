"""FastAPI: Tesseract OCR + парсеры паспорта и выписки ЕГРН без LLM."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from .egrn_parser import parse_egrn_ocr_text
from .passport_parser import parse_passport_ocr_text
from .registration_parser import parse_registration_ocr_text
from huggin_face_scan.scan_passport_hf import (
    EgrnExtractData,
    PassportScanResponse,
    UnifiedDocumentsData,
    UnifiedDocumentsScanResponse,
    normalize_egrn_data,
    normalize_passport_data,
    normalize_registration_data,
    pdf_first_page_to_png,
    safe_to_thread,
    upscale_jpeg_for_ocr,
    validate_file_size,
    validate_image,
)
from tesseract_scan.ocr import img_ocr, img_ocr_multi_pass

router = APIRouter(tags=["document-tesseract"])


class EgrnScanTesseractResponse(BaseModel):
    ok: bool
    model: str
    data: EgrnExtractData
    raw_text: str


async def _bytes_to_ocr_image(contents: bytes, file_type: str) -> bytes:
    """Байты изображения для OCR (PDF → PNG, JPEG — upscale)."""
    ft = file_type.lower()
    if ft == "application/pdf":
        return await safe_to_thread(pdf_first_page_to_png, contents)
    await safe_to_thread(validate_image, contents)
    await safe_to_thread(validate_file_size, contents)
    if ft in ("image/jpeg", "image/jpg"):
        return await safe_to_thread(upscale_jpeg_for_ocr, contents, 2.5)
    return contents


@router.post(
    "/scan-passport-tesseract-structured",
    response_model=PassportScanResponse,
)
async def scan_passport_tesseract_structured(
    file: UploadFile = File(...),
) -> PassportScanResponse:
    """
    Те же поля, что у /scan-passport, но: только Tesseract + эвристики по тексту.
    """
    file_type = (file.content_type or "").lower()
    if not (file_type.startswith("image/") or file_type == "application/pdf"):
        raise HTTPException(
            status_code=400,
            detail="Загрузите изображение (JPEG/PNG/WebP) или PDF",
        )

    contents = await file.read()
    image_bytes = await _bytes_to_ocr_image(contents, file_type)

    raw = await _ocr_file_bytes_to_text_multi_pass(contents, file_type)

    parsed = parse_passport_ocr_text(raw)
    data = normalize_passport_data(parsed)

    return PassportScanResponse(
        ok=True,
        model="tesseract+rules",
        data=data,
        raw_text=raw,
    )


@router.post(
    "/scan-egrn-tesseract-structured",
    response_model=EgrnScanTesseractResponse,
)
async def scan_egrn_tesseract_structured(
    file: UploadFile = File(...),
) -> EgrnScanTesseractResponse:
    """
    Поля выписки ЕГРН, как в unified HF-потоке, но только Tesseract + эвристики.
    """
    file_type = (file.content_type or "").lower()
    if not (file_type.startswith("image/") or file_type == "application/pdf"):
        raise HTTPException(
            status_code=400,
            detail="Загрузите изображение (JPEG/PNG/WebP) или PDF",
        )

    contents = await file.read()
    image_bytes = await _bytes_to_ocr_image(contents, file_type)

    try:
        raw = await safe_to_thread(img_ocr, image_bytes)
    except RuntimeError:
        raise HTTPException(
            status_code=500,
            detail="Ошибка OCR-движка при обработке изображения",
        )

    parsed = parse_egrn_ocr_text(raw)
    data = normalize_egrn_data(parsed)

    return EgrnScanTesseractResponse(
        ok=True,
        model="tesseract+rules",
        data=data,
        raw_text=raw,
    )


async def _ocr_file_bytes_to_text(contents: bytes, file_type: str) -> str:
    image_bytes = await _bytes_to_ocr_image(contents, file_type)
    try:
        return await safe_to_thread(img_ocr, image_bytes)
    except RuntimeError:
        raise HTTPException(
            status_code=500,
            detail="Ошибка OCR-движка при обработке изображения",
        )


async def _ocr_file_bytes_to_text_multi_pass(contents: bytes, file_type: str) -> str:
    image_bytes = await _bytes_to_ocr_image(contents, file_type)
    try:
        return await safe_to_thread(img_ocr_multi_pass, image_bytes, include_crops=True)
    except RuntimeError:
        raise HTTPException(
            status_code=500,
            detail="Ошибка OCR-движка при обработке изображения",
        )


@router.post(
    "/scan-documents-unified-tesseract",
    response_model=UnifiedDocumentsScanResponse,
)
async def scan_documents_unified_tesseract(
    passport_main: UploadFile = File(...),
    passport_registration: UploadFile = File(...),
    egrn_extract: UploadFile = File(...),
) -> UnifiedDocumentsScanResponse:
    """
    Три документа: только Tesseract + эвристики. Формат ответа как у /scan-documents-unified (HF).
    """
    main_type = (passport_main.content_type or "").lower()
    reg_type = (passport_registration.content_type or "").lower()
    egrn_type = (egrn_extract.content_type or "").lower()
    if not (main_type.startswith("image/") or main_type == "application/pdf"):
        raise HTTPException(status_code=400, detail="passport_main: загрузите изображение или PDF")
    if not (reg_type.startswith("image/") or reg_type == "application/pdf"):
        raise HTTPException(
            status_code=400,
            detail="passport_registration: загрузите изображение или PDF",
        )
    if not (egrn_type.startswith("image/") or egrn_type == "application/pdf"):
        raise HTTPException(
            status_code=400,
            detail="egrn_extract: поддерживаются изображение или PDF",
        )

    main_bytes = await passport_main.read()
    registration_bytes = await passport_registration.read()
    egrn_bytes = await egrn_extract.read()

    raw_main, raw_reg, raw_egrn = await asyncio.gather(
        _ocr_file_bytes_to_text_multi_pass(main_bytes, main_type),
        _ocr_file_bytes_to_text(registration_bytes, reg_type),
        _ocr_file_bytes_to_text_multi_pass(egrn_bytes, egrn_type),
    )

    passport_data = normalize_passport_data(parse_passport_ocr_text(raw_main))
    registration_data = normalize_registration_data(parse_registration_ocr_text(raw_reg))
    egrn_data = normalize_egrn_data(parse_egrn_ocr_text(raw_egrn))

    return UnifiedDocumentsScanResponse(
        ok=True,
        model="tesseract+rules",
        data=UnifiedDocumentsData(
            passport_main=passport_data,
            passport_registration=registration_data,
            egrn_extract=egrn_data,
        ),
        raw_text={
            "passport_main": raw_main,
            "passport_registration": raw_reg,
            "egrn_extract": raw_egrn,
        },
    )
