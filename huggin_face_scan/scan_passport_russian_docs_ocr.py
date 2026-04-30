from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from document_tesseract.egrn_parser import parse_egrn_ocr_text
from document_tesseract.registration_parser import parse_registration_ocr_text
from huggin_face_scan.scan_passport_hf import (
    PassportScanResponse,
    UnifiedDocumentsData,
    UnifiedDocumentsScanResponse,
    normalize_egrn_data,
    normalize_passport_data,
    normalize_registration_data,
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


def _file_debug_payload(file: UploadFile, contents: bytes) -> dict[str, Any]:
    return {
        "filename": file.filename or "",
        "content_type": file.content_type or "",
        "bytes": len(contents),
        "sha256_12": hashlib.sha256(contents).hexdigest()[:12],
    }


def _duplicate_file_warnings(files: dict[str, dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    items = list(files.items())
    for index, (left_key, left) in enumerate(items):
        for right_key, right in items[index + 1 :]:
            if left.get("sha256_12") and left.get("sha256_12") == right.get("sha256_12"):
                warnings.append(f"{left_key} и {right_key}: одинаковый sha256_12")
    return warnings


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


def _pick_ocr_value(ocr: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = ocr.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return " ".join(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, dict):
            inner = value.get("ocr")
            if isinstance(inner, list):
                return " ".join(str(item).strip() for item in inner if str(item).strip())
            if isinstance(inner, str):
                return inner.strip()
    return ""


def _split_russian_docs_licence_number(value: str) -> tuple[str, str]:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) < 10:
        return "", ""
    return digits[:4], digits[4:10]


def _validate_fio_value(label: str, value: str, *, patronymic: bool = False) -> list[str]:
    text = re.sub(r"\s+", " ", str(value or "").strip().upper())
    if not text:
        return [f"{label} не распознано"]
    warnings: list[str] = []
    if not re.fullmatch(r"[А-ЯЁ][А-ЯЁ -]{1,64}", text):
        warnings.append(f"{label}: недопустимые символы или длина")
    if re.search(r"[A-Z0-9]", text):
        warnings.append(f"{label}: есть латиница или цифры")
    if re.search(r"(.)\1\1", text):
        warnings.append(f"{label}: есть три одинаковые буквы подряд")
    if re.search(r"(ДЙ|ТЙ|НЙ|ЛЙ|РЙ|СЙ|ЗЙ|ВЙ|БЙ|ПЙ|ФЙ|ГЙ|КЙ|ХЙ|ЖЙ|ШЙ|ЩЙ|ЧЙ|ЦЙ|ЙМ|ЙН|ЙР|ЙЛ)", text):
        warnings.append(f"{label}: подозрительное сочетание букв, проверьте OCR")
    if patronymic and not re.search(r"(ОВИЧ|ЕВИЧ|ИЧ|ОВНА|ЕВНА|ИНИЧНА|ЫЧ|КЫЗЫ|ОГЛЫ)$", text):
        warnings.append(f"{label}: нет типичного окончания отчества")
    return warnings


def _fio_validation_note(payload: dict[str, str]) -> str:
    warnings = [
        *_validate_fio_value("Фамилия", payload.get("surname", "")),
        *_validate_fio_value("Имя", payload.get("name", "")),
        *_validate_fio_value("Отчество", payload.get("patronymic", ""), patronymic=True),
    ]
    return "; ".join(warnings)


def _compact_raw_payload(raw: dict[str, Any]) -> str:
    """Keep debug OCR values, but avoid serializing image arrays returned by RussianDocsOCR."""
    payload = {
        "ocr": raw.get("ocr") if isinstance(raw.get("ocr"), dict) else None,
        "doctype": raw.get("doctype"),
        "quality": raw.get("quality") if isinstance(raw.get("quality"), dict) else {},
    }
    text_fields = raw.get("text_fields")
    if isinstance(text_fields, tuple) and text_fields:
        payload["text_fields_count"] = len(text_fields[0]) if isinstance(text_fields[0], list) else 0
    elif isinstance(text_fields, list):
        payload["text_fields_count"] = len(text_fields)
    words_patches = raw.get("words_patches")
    if isinstance(words_patches, dict):
        payload["words_patch_fields"] = sorted(words_patches.keys())
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _russian_docs_ocr_to_text(raw: dict[str, Any]) -> str:
    ocr = raw.get("ocr")
    if not isinstance(ocr, dict):
        return ""
    lines: list[str] = []
    for key in ocr:
        text = _pick_ocr_value(ocr, key)
        if text:
            lines.append(f"{key}: {text}")
    return "\n".join(lines)


def _looks_like_passport_main_page(raw: dict[str, Any], text: str) -> bool:
    ocr = raw.get("ocr")
    if not isinstance(ocr, dict):
        return False
    passport_keys = {
        "Issue_organization_ru",
        "Issue_organisation_ru",
        "Issue_date",
        "Issue_organisation_code",
        "Issue_organization_code",
        "Last_name_ru",
        "First_name_ru",
        "Middle_name_ru",
        "Licence_number",
        "Birth_date",
        "Birth_place_ru",
    }
    matched_keys = passport_keys.intersection(ocr.keys())
    has_registration_words = bool(re.search(r"(регистрац|место\s+жительства|зарегистрирован)", text, re.I))
    return len(matched_keys) >= 4 and not has_registration_words


def _looks_like_issuing_authority(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip().upper())
    if not text:
        return False
    has_authority_words = bool(re.search(r"(ОТДЕЛОМ|ОТДЕЛ\s+|УФМС|МВД|ГУВМ|ОВД|РОССИИ|ВЫДАН|КОД\s+ПОДРАЗД)", text))
    has_address_words = bool(
        re.search(r"(УЛ\.|УЛИЦА|Д\.|ДОМ|КВ\.|КВАРТИРА|ПР-КТ|ПРОСПЕКТ|ПЕР\.|ПЕРЕУЛОК|Ш\.|ШОССЕ)", text)
    )
    return has_authority_words and not has_address_words


def _clear_invalid_registration_address(registration_data: Any) -> None:
    suspicious_values = [
        registration_data.address,
        registration_data.region,
        registration_data.city,
        registration_data.settlement,
        registration_data.street,
    ]
    if not any(_looks_like_issuing_authority(value) for value in suspicious_values):
        return
    registration_data.address = ""
    registration_data.region = ""
    registration_data.city = ""
    registration_data.settlement = ""
    registration_data.street = ""
    registration_data.house = ""
    registration_data.building = ""
    registration_data.apartment = ""
    note = "OCR вернул орган выдачи вместо адреса регистрации; заполните адрес вручную."
    registration_data.confidence_note = f"{registration_data.confidence_note}; {note}".strip("; ")


def _clear_invalid_egrn_data(egrn_data: Any, passport_data: Any, *, source_is_passport: bool) -> None:
    has_core_egrn_fields = bool(egrn_data.cadastral_number or egrn_data.address or egrn_data.right_holders)
    passport_dates = {passport_data.birth_date, passport_data.issue_date} - {""}
    date_looks_copied_from_passport = bool(egrn_data.extract_date and egrn_data.extract_date in passport_dates)
    if not source_is_passport and (has_core_egrn_fields or not date_looks_copied_from_passport):
        return
    egrn_data.cadastral_number = ""
    egrn_data.object_type = ""
    egrn_data.address = ""
    egrn_data.area_sq_m = ""
    egrn_data.ownership_type = ""
    egrn_data.right_holders = []
    egrn_data.extract_date = ""
    note = "RussianDocsOCR распознал файл ЕГРН как паспортный разворот; заполните данные ЕГРН вручную."
    egrn_data.confidence_note = f"{egrn_data.confidence_note}; {note}".strip("; ")


async def _scan_file_with_russian_docs(
    contents: bytes,
    content_type: str,
    scan_id: str,
    *,
    model_format: str,
    device: str,
    check_quality: bool,
    img_size: int,
) -> tuple[dict[str, Any], str, str]:
    image_bytes, suffix = await _prepare_image_bytes(contents, content_type, scan_id)
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

    return raw, _compact_raw_payload(raw), _russian_docs_ocr_to_text(raw)


async def _scan_passport_main_with_russian_docs(
    contents: bytes,
    content_type: str,
    scan_id: str,
    *,
    model_format: str,
    device: str,
    check_quality: bool,
    img_size: int,
) -> tuple[Any, str]:
    raw, raw_debug, _ = await _scan_file_with_russian_docs(
        contents,
        content_type,
        scan_id,
        model_format=model_format,
        device=device,
        check_quality=check_quality,
        img_size=img_size,
    )
    parsed = _normalize_russian_docs_passport(raw.get("ocr"))
    data = normalize_passport_data(parsed)
    return data, raw_debug


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
    licence_number = _pick_ocr_value(ocr, "Licence_number", "License_number", "licence_number", "license_number")
    passport_series, passport_number = _split_russian_docs_licence_number(licence_number)
    payload = {
        "issuing_authority": _pick_ocr_value(
            ocr,
            "Issue_organization_ru",
            "Issue_organisation_ru",
            "issuing_authority",
            "issued_by",
        ),
        "issue_date": _pick_ocr_value(ocr, "Issue_date", "issue_date", "date_of_issue"),
        "department_code": _pick_ocr_value(
            ocr,
            "Issue_organisation_code",
            "Issue_organization_code",
            "department_code",
            "code",
        ),
        "passport_series": passport_series or _pick_ocr_value(ocr, "passport_series", "series"),
        "passport_number": passport_number or _pick_ocr_value(ocr, "passport_number", "number"),
        "surname": _pick_ocr_value(ocr, "Last_name_ru", "surname", "last_name"),
        "name": _pick_ocr_value(ocr, "First_name_ru", "name", "first_name"),
        "patronymic": _pick_ocr_value(ocr, "Middle_name_ru", "patronymic", "middle_name"),
        "gender": _pick_ocr_value(ocr, "Sex_ru", "gender", "sex"),
        "birth_date": _pick_ocr_value(ocr, "Birth_date", "birth_date", "date_of_birth"),
        "birth_place": _pick_ocr_value(ocr, "Birth_place_ru", "birth_place", "place_of_birth"),
        "confidence_note": "Экспериментальный результат RussianDocsOCR Pipeline.",
    }
    fio_note = _fio_validation_note(payload)
    if fio_note:
        payload["confidence_note"] = f"{payload['confidence_note']} ФИО требует проверки: {fio_note}"
    return payload


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
    data, raw_text = await _scan_passport_main_with_russian_docs(
        contents,
        file_type,
        scan_id,
        model_format=model_format,
        device=device,
        check_quality=check_quality,
        img_size=img_size,
    )
    _log(
        scan_id,
        "request",
        "finish",
        has_passport_number=bool(data.passport_series and data.passport_number),
        has_fio=bool(data.surname and data.name),
    )
    return PassportScanResponse(
        ok=True,
        model=f"RussianDocsOCR:{model_format}:{device}",
        data=data,
        raw_text=raw_text,
    )


@router.post("/scan-documents-russian-docs-ocr", response_model=UnifiedDocumentsScanResponse)
async def scan_documents_russian_docs_ocr(
    passport_main: UploadFile = File(...),
    passport_registration: UploadFile = File(...),
    egrn_extract: UploadFile = File(...),
) -> UnifiedDocumentsScanResponse:
    scan_id = uuid4().hex[:12]
    main_type = (passport_main.content_type or "").lower()
    reg_type = (passport_registration.content_type or "").lower()
    egrn_type = (egrn_extract.content_type or "").lower()
    _log(
        scan_id,
        "unified_request",
        "start",
        passport_main=main_type,
        passport_registration=reg_type,
        egrn_extract=egrn_type,
    )

    main_bytes = await passport_main.read()
    registration_bytes = await passport_registration.read()
    egrn_bytes = await egrn_extract.read()
    files_debug = {
        "passport_main": _file_debug_payload(passport_main, main_bytes),
        "passport_registration": _file_debug_payload(passport_registration, registration_bytes),
        "egrn_extract": _file_debug_payload(egrn_extract, egrn_bytes),
    }
    file_warnings = _duplicate_file_warnings(files_debug)
    _log(
        scan_id,
        "unified_request",
        "files_read",
        passport_main=files_debug["passport_main"],
        passport_registration=files_debug["passport_registration"],
        egrn_extract=files_debug["egrn_extract"],
        warnings=file_warnings,
    )

    main_raw, passport_raw, main_text = await _scan_file_with_russian_docs(
        main_bytes,
        main_type,
        scan_id,
        model_format="ONNX",
        device="cpu",
        check_quality=False,
        img_size=1500,
    )
    reg_raw, reg_debug, reg_text = await _scan_file_with_russian_docs(
        registration_bytes,
        reg_type,
        scan_id,
        model_format="ONNX",
        device="cpu",
        check_quality=False,
        img_size=1500,
    )
    egrn_raw, egrn_debug, egrn_text = await _scan_file_with_russian_docs(
        egrn_bytes,
        egrn_type,
        scan_id,
        model_format="ONNX",
        device="cpu",
        check_quality=False,
        img_size=1500,
    )
    passport_data = normalize_passport_data(_normalize_russian_docs_passport(main_raw.get("ocr")))
    if _looks_like_passport_main_page(reg_raw, reg_text):
        registration_data = normalize_registration_data(
            {
                "confidence_note": (
                    "RussianDocsOCR распознал файл прописки как основной разворот паспорта; "
                    "заполните адрес регистрации вручную."
                )
            }
        )
    else:
        registration_data = normalize_registration_data(parse_registration_ocr_text(reg_text))
    _clear_invalid_registration_address(registration_data)
    egrn_source_is_passport = _looks_like_passport_main_page(egrn_raw, egrn_text)
    if egrn_source_is_passport:
        egrn_data = normalize_egrn_data(
            {
                "confidence_note": (
                    "RussianDocsOCR распознал файл ЕГРН как основной разворот паспорта; "
                    "заполните данные ЕГРН вручную."
                )
            }
        )
    else:
        egrn_data = normalize_egrn_data(parse_egrn_ocr_text(egrn_text))
    _clear_invalid_egrn_data(egrn_data, passport_data, source_is_passport=egrn_source_is_passport)

    # If the registration page is recognized as a passport page, keep its OCR visible for manual review.
    if not (registration_data.address or registration_data.region or registration_data.city):
        registration_data.confidence_note = (
            "RussianDocsOCR не извлек структурированный адрес прописки; проверьте raw_text.passport_registration."
        )
    if not (egrn_data.cadastral_number or egrn_data.address or egrn_data.right_holders):
        egrn_data.confidence_note = (
            "RussianDocsOCR не извлек структурированные поля ЕГРН; проверьте raw_text.egrn_extract."
        )

    # Preserve main page OCR text next to compact debug payload.
    passport_raw = f"{passport_raw}\n\n--- russian_docs_ocr_text ---\n{main_text}"
    _log(
        scan_id,
        "unified_request",
        "finish",
        has_passport_number=bool(passport_data.passport_series and passport_data.passport_number),
        registration_chars=len(reg_text or ""),
        egrn_chars=len(egrn_text or ""),
    )

    return UnifiedDocumentsScanResponse(
        ok=True,
        model="RussianDocsOCR:ONNX:cpu+rules",
        data=UnifiedDocumentsData(
            passport_main=passport_data,
            passport_registration=registration_data,
            egrn_extract=egrn_data,
        ),
        raw_text={
            "passport_main": passport_raw,
            "passport_registration": f"{reg_debug}\n\n--- russian_docs_ocr_text ---\n{reg_text}",
            "egrn_extract": f"{egrn_debug}\n\n--- russian_docs_ocr_text ---\n{egrn_text}",
            "_files": json.dumps(files_debug, ensure_ascii=False, indent=2),
            "_warnings": json.dumps(file_warnings, ensure_ascii=False, indent=2),
        },
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
