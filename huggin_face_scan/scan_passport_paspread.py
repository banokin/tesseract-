from __future__ import annotations

import importlib
import logging
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from huggin_face_scan.scan_passport_hf import (
    PassportScanResponse,
    normalize_passport_data,
    pdf_first_page_to_png,
    safe_to_thread,
    validate_image,
)

router = APIRouter(tags=["passport-experiments-paspread"])
logger = logging.getLogger(__name__)


def _log(scan_id: str, stage: str, event: str, **fields: Any) -> None:
    suffix = " ".join(f"{key}={value!r}" for key, value in fields.items())
    logger.info("paspread %s: %s scan_id=%s %s", stage, event, scan_id, suffix)


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


def _run_paspread(image_path: Path) -> dict[str, Any]:
    try:
        paspread = importlib.import_module("rupasportread")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Не установлен rupasportread. Установите зависимость: pip install rupasportread"
        ) from exc

    result = paspread.catching(str(image_path))
    if not isinstance(result, dict):
        raise RuntimeError(f"rupasportread вернул неожиданный тип: {type(result).__name__}")
    return result


def _normalize_paspread_payload(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "surname": str(payload.get("Surname", "") or ""),
        "name": str(payload.get("Name", "") or ""),
        "patronymic": str(payload.get("Mid", "") or ""),
        "birth_date": str(payload.get("Date", "") or ""),
        "passport_series": str(payload.get("Series", "") or ""),
        "passport_number": str(payload.get("Number", "") or ""),
        "confidence_note": "Экспериментальный результат rupasportread/paspread: MRZ + Tesseract.",
    }


@router.post("/scan-passport-paspread", response_model=PassportScanResponse)
async def scan_passport_paspread(file: UploadFile = File(...)) -> PassportScanResponse:
    scan_id = uuid4().hex[:12]
    file_type = (file.content_type or "").lower()
    _log(scan_id, "request", "start", filename=file.filename, content_type=file_type)

    contents = await file.read()
    image_bytes, suffix = await _prepare_image_bytes(contents, file_type, scan_id)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        image_path = Path(tmp.name)
        try:
            _log(scan_id, "paspread", "start", image_path=str(image_path), image_bytes=len(image_bytes))
            raw_payload = await safe_to_thread(_run_paspread, image_path)
            _log(scan_id, "paspread", "success", keys=",".join(sorted(raw_payload.keys())))
        except RuntimeError as exc:
            _log(scan_id, "paspread", "failed", error=repr(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("paspread failed scan_id=%s", scan_id)
            raise HTTPException(status_code=500, detail=f"Ошибка paspread/rupasportread: {exc!r}") from exc

    parsed = _normalize_paspread_payload(raw_payload)
    data = normalize_passport_data(parsed)
    _log(scan_id, "request", "finish")
    return PassportScanResponse(
        ok=True,
        model="rupasportread+paspread",
        data=data,
        raw_text=str(raw_payload),
    )
