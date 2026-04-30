from __future__ import annotations

import importlib
import logging
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from huggin_face_scan.scan_passport_hf import (
    PassportScanResponse,
    normalize_passport_data,
    pdf_first_page_to_png,
    safe_to_thread,
    validate_image,
)

router = APIRouter(tags=["passport-experiments-russian-docs-ocr"])
logger = logging.getLogger(__name__)


class RussianDocsOcrDebug(BaseModel):
    ok: bool
    model: str
    doctype: str
    quality: dict[str, Any]
    ocr: dict[str, Any] | None
    raw: dict[str, Any]


def _log(scan_id: str, stage: str, event: str, **fields: Any) -> None:
    suffix = " ".join(f"{key}={value!r}" for key, value in fields.items())
    logger.info("russian-docs-ocr %s: %s scan_id=%s %s", stage, event, scan_id, suffix)


async def _prepare_image_bytes(contents: bytes, content_type: str, scan_id: str) -> tuple[bytes, str]:
    _log(scan_id, "prepare", "start", content_type=content_type, input_bytes=len(contents))
    if content_type == "application/pdf":
        image_bytes = await safe_to_thread(pdf_first_page_to_png, contents)
        _log(scan_id, "prepare", "pdf_converted", output_bytes=len(image_bytes))
        return image_bytes, ".png"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите изображение или PDF")
    await safe_to_thread(validate_image, contents)
    suffix = ".jpg" if content_type in {"image/jpeg", "image/jpg"} else ".png"
    _log(scan_id, "prepare", "image_validated", output_bytes=len(contents), suffix=suffix)
    return contents, suffix


def _pipeline_result_to_dict(result: Any) -> dict[str, Any]:
    keys = ("ocr", "doctype", "quality", "text_fields", "words_patches")
    out: dict[str, Any] = {}
    for key in keys:
        try:
            out[key] = getattr(result, key)
        except Exception as exc:
            out[key] = f"<error reading {key}: {exc!r}>"
    return out


def _run_russian_docs_ocr(
    image_path: Path,
    *,
    model_format: str,
    device: str,
    check_quality: bool,
    img_size: int,
) -> dict[str, Any]:
    try:
        try:
            document_processing = importlib.import_module("russian_docs_ocr.document_processing")
        except ModuleNotFoundError:
            document_processing = importlib.import_module("document_processing")
        pipeline_cls = getattr(document_processing, "Pipeline")
    except (ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            "Не установлен RussianDocsOCR. Установите зависимость из GitHub: "
            "pip install 'russian_docs_ocr @ git+https://github.com/protei300/RussianDocsOCR.git' "
            "или добавьте пакет russian_docs_ocr/document_processing в PYTHONPATH."
        ) from exc

    pipeline = pipeline_cls(model_format=model_format, device=device)
    result = pipeline(
        img_path=str(image_path),
        check_quality=check_quality,
        low_quality=True,
        img_size=img_size,
    )
    return _pipeline_result_to_dict(result)


def _normalize_russian_docs_passport(ocr: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(ocr, dict):
        return {"confidence_note": "RussianDocsOCR не вернул ocr-словарь."}
    return {
        "issuing_authority": str(ocr.get("issuing_authority", "") or ocr.get("issued_by", "") or ""),
        "issue_date": str(ocr.get("issue_date", "") or ocr.get("date_of_issue", "") or ""),
        "department_code": str(ocr.get("department_code", "") or ocr.get("code", "") or ""),
        "passport_series": str(ocr.get("passport_series", "") or ocr.get("series", "") or ""),
        "passport_number": str(ocr.get("passport_number", "") or ocr.get("number", "") or ""),
        "surname": str(ocr.get("surname", "") or ocr.get("last_name", "") or ""),
        "name": str(ocr.get("name", "") or ocr.get("first_name", "") or ""),
        "patronymic": str(ocr.get("patronymic", "") or ocr.get("middle_name", "") or ""),
        "gender": str(ocr.get("gender", "") or ocr.get("sex", "") or ""),
        "birth_date": str(ocr.get("birth_date", "") or ocr.get("date_of_birth", "") or ""),
        "birth_place": str(ocr.get("birth_place", "") or ocr.get("place_of_birth", "") or ""),
        "confidence_note": "Экспериментальный результат RussianDocsOCR Pipeline.",
    }


@router.post("/scan-passport-russian-docs-ocr", response_model=PassportScanResponse)
async def scan_passport_russian_docs_ocr(
    file: UploadFile = File(...),
    model_format: str = Form("ONNX"),
    device: str = Form("cpu"),
    check_quality: bool = Form(False),
    img_size: int = Form(1500),
) -> PassportScanResponse:
    scan_id = uuid4().hex[:12]
    file_type = (file.content_type or "").lower()
    _log(
        scan_id,
        "request",
        "start",
        filename=file.filename,
        content_type=file_type,
        model_format=model_format,
        device=device,
        check_quality=check_quality,
        img_size=img_size,
    )
    contents = await file.read()
    image_bytes, suffix = await _prepare_image_bytes(contents, file_type, scan_id)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        image_path = Path(tmp.name)
        try:
            _log(scan_id, "pipeline", "start", image_path=str(image_path), image_bytes=len(image_bytes))
            raw = await safe_to_thread(
                _run_russian_docs_ocr,
                image_path,
                model_format=model_format,
                device=device,
                check_quality=check_quality,
                img_size=img_size,
            )
            _log(
                scan_id,
                "pipeline",
                "success",
                doctype=raw.get("doctype", ""),
                has_ocr=isinstance(raw.get("ocr"), dict),
            )
        except RuntimeError as exc:
            _log(scan_id, "pipeline", "failed", error=repr(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("RussianDocsOCR failed scan_id=%s", scan_id)
            raise HTTPException(status_code=500, detail=f"Ошибка RussianDocsOCR: {exc!r}") from exc

    parsed = _normalize_russian_docs_passport(raw.get("ocr"))
    data = normalize_passport_data(parsed)
    _log(scan_id, "request", "finish")
    return PassportScanResponse(
        ok=True,
        model=f"RussianDocsOCR:{model_format}:{device}",
        data=data,
        raw_text=str(raw),
    )


@router.post("/debug-russian-docs-ocr", response_model=RussianDocsOcrDebug)
async def debug_russian_docs_ocr(
    file: UploadFile = File(...),
    model_format: str = Form("ONNX"),
    device: str = Form("cpu"),
    check_quality: bool = Form(False),
    img_size: int = Form(1500),
) -> RussianDocsOcrDebug:
    scan_id = uuid4().hex[:12]
    file_type = (file.content_type or "").lower()
    contents = await file.read()
    image_bytes, suffix = await _prepare_image_bytes(contents, file_type, scan_id)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        raw = await safe_to_thread(
            _run_russian_docs_ocr,
            Path(tmp.name),
            model_format=model_format,
            device=device,
            check_quality=check_quality,
            img_size=img_size,
        )
    return RussianDocsOcrDebug(
        ok=True,
        model=f"RussianDocsOCR:{model_format}:{device}",
        doctype=str(raw.get("doctype", "") or ""),
        quality=raw.get("quality") if isinstance(raw.get("quality"), dict) else {},
        ocr=raw.get("ocr") if isinstance(raw.get("ocr"), dict) else None,
        raw=raw,
    )
