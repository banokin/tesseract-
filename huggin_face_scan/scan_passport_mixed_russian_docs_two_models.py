from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from document_tesseract.egrn_parser import parse_egrn_ocr_text
from document_tesseract.registration_parser import parse_registration_ocr_text
from huggin_face_scan.model_config import LLAMA_4_SCOUT_MODEL, QWEN_30_MODEL, TWO_MODELS_MAP
from huggin_face_scan.scan_passport_hf import (
    UnifiedDocumentsData,
    UnifiedDocumentsScanResponse,
    build_egrn_prompt,
    build_registration_prompt,
    enrich_egrn_fields,
    enrich_registration_fields,
    extract_generic_json_from_text,
    normalize_egrn_data,
    normalize_passport_data,
    normalize_registration_data,
    run_hf_document_extraction,
)
from huggin_face_scan.scan_passport_hf_two_models import (
    _extract_ocr_text as _extract_two_models_ocr_text,
    _merge_model_with_ocr,
    _prepare_ocr_bytes as _prepare_two_models_ocr_bytes,
    _prompt_with_ocr_context,
)
from huggin_face_scan.scan_passport_russian_docs_ocr import (
    _clear_invalid_egrn_data,
    _clear_invalid_registration_address,
    _compact_raw_payload,
    _duplicate_file_warnings,
    _file_debug_payload,
    _normalize_russian_docs_passport,
    _scan_file_with_russian_docs,
)

router = APIRouter(tags=["passport-mixed-russian-docs-two-models"])
logger = logging.getLogger(__name__)


def _log(scan_id: str, stage: str, event: str, **fields: Any) -> None:
    suffix = " ".join(f"{key}={value!r}" for key, value in fields.items())
    logger.info("mixed-russian-docs-two-models %s: %s scan_id=%s %s", stage, event, scan_id, suffix)


def _registration_score(data: Any) -> int:
    return sum(
        bool(str(value or "").strip())
        for value in (
            data.address,
            data.region,
            data.city,
            data.settlement,
            data.street,
            data.house,
            data.building,
            data.apartment,
            data.registration_date,
        )
    )


def _egrn_score(data: Any) -> int:
    return sum(
        (
            bool(str(data.cadastral_number or "").strip()),
            bool(str(data.object_type or "").strip()),
            bool(str(data.address or "").strip()),
            bool(str(data.area_sq_m or "").strip()),
            bool(str(data.ownership_type or "").strip()),
            bool(data.right_holders),
            bool(str(data.extract_date or "").strip()),
        )
    )


async def _scan_registration_egrn_with_model(
    registration_ocr_bytes: bytes,
    egrn_ocr_bytes: bytes,
    registration_ocr_text: str,
    egrn_ocr_text: str,
    model_name: str,
    scan_id: str,
) -> tuple[Any, Any, dict[str, str], str]:
    _log(scan_id, "two_models", "model_start", model=model_name)
    try:
        (registration_raw, model_used), (egrn_raw, _) = await asyncio.gather(
            run_hf_document_extraction(
                registration_ocr_bytes,
                _prompt_with_ocr_context(
                    build_registration_prompt(),
                    registration_ocr_text,
                    "страница регистрации паспорта РФ",
                ),
                max_tokens=600,
                model_name=model_name,
            ),
            run_hf_document_extraction(
                egrn_ocr_bytes,
                _prompt_with_ocr_context(build_egrn_prompt(), egrn_ocr_text, "выписка ЕГРН"),
                max_tokens=700,
                model_name=model_name,
            ),
        )
    except Exception as exc:
        logger.exception("mixed two-models failed scan_id=%s model=%s", scan_id, model_name)
        raise HTTPException(status_code=502, detail=f"Ошибка сканирования two-models ({model_name}): {exc!r}") from exc

    try:
        registration_payload = extract_generic_json_from_text(registration_raw)
        registration_ocr_payload = parse_registration_ocr_text(registration_ocr_text) if registration_ocr_text else {}
        registration_data = normalize_registration_data(
            _merge_model_with_ocr(registration_payload, registration_ocr_payload)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON прописки ({model_name}): {registration_raw[:1000]}",
        ) from exc

    try:
        egrn_payload = extract_generic_json_from_text(egrn_raw)
        egrn_ocr_payload = parse_egrn_ocr_text(egrn_ocr_text) if egrn_ocr_text else {}
        egrn_data = normalize_egrn_data(_merge_model_with_ocr(egrn_payload, egrn_ocr_payload))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON ЕГРН ({model_name}): {egrn_raw[:1000]}",
        ) from exc

    registration_data, egrn_data = await asyncio.gather(
        enrich_registration_fields(registration_ocr_bytes, registration_data, model_name=model_name),
        enrich_egrn_fields(egrn_ocr_bytes, egrn_data, model_name=model_name),
    )
    _log(
        scan_id,
        "two_models",
        "model_finish",
        model=model_name,
        registration_score=_registration_score(registration_data),
        egrn_score=_egrn_score(egrn_data),
    )
    return (
        registration_data,
        egrn_data,
        {
            "passport_registration": f"{registration_raw}\n\n--- tesseract_ocr_context ---\n{registration_ocr_text}",
            "egrn_extract": f"{egrn_raw}\n\n--- tesseract_ocr_context ---\n{egrn_ocr_text}",
        },
        model_used,
    )


def _pick_best_result(results: list[tuple[Any, Any, dict[str, str], str]]) -> tuple[Any, Any, dict[str, str], dict[str, str]]:
    best_registration = max(results, key=lambda item: _registration_score(item[0]))
    best_egrn = max(results, key=lambda item: _egrn_score(item[1]))
    raw_text = {
        "qwen30_passport_registration": results[0][2]["passport_registration"],
        "qwen30_egrn_extract": results[0][2]["egrn_extract"],
        "llama4scout_passport_registration": results[1][2]["passport_registration"],
        "llama4scout_egrn_extract": results[1][2]["egrn_extract"],
    }
    models_used = {
        "passport_registration": best_registration[3],
        "egrn_extract": best_egrn[3],
    }
    return best_registration[0], best_egrn[1], raw_text, models_used


@router.post("/scan-documents-russian-docs-two-models", response_model=UnifiedDocumentsScanResponse)
async def scan_documents_russian_docs_two_models(
    passport_main: UploadFile = File(...),
    passport_registration: UploadFile = File(...),
    egrn_extract: UploadFile = File(...),
) -> UnifiedDocumentsScanResponse:
    scan_id = uuid4().hex[:12]
    main_type = (passport_main.content_type or "").lower()
    reg_type = (passport_registration.content_type or "").lower()
    egrn_type = (egrn_extract.content_type or "").lower()
    _log(scan_id, "request", "start", passport_main=main_type, passport_registration=reg_type, egrn_extract=egrn_type)

    main_bytes = await passport_main.read()
    registration_bytes = await passport_registration.read()
    egrn_bytes = await egrn_extract.read()
    files_debug = {
        "passport_main": _file_debug_payload(passport_main, main_bytes),
        "passport_registration": _file_debug_payload(passport_registration, registration_bytes),
        "egrn_extract": _file_debug_payload(egrn_extract, egrn_bytes),
    }
    file_warnings = _duplicate_file_warnings(files_debug)
    _log(scan_id, "request", "files_read", files=files_debug, warnings=file_warnings)

    main_raw, passport_raw, main_text = await _scan_file_with_russian_docs(
        main_bytes,
        main_type,
        scan_id,
        model_format="ONNX",
        device="cpu",
        check_quality=False,
        img_size=1500,
    )
    passport_data = normalize_passport_data(_normalize_russian_docs_passport(main_raw.get("ocr")))

    registration_ocr_bytes, egrn_ocr_bytes = await asyncio.gather(
        _prepare_two_models_ocr_bytes(registration_bytes, reg_type),
        _prepare_two_models_ocr_bytes(egrn_bytes, egrn_type),
    )
    registration_ocr_text, egrn_ocr_text = await asyncio.gather(
        _extract_two_models_ocr_text(registration_ocr_bytes, include_crops=True),
        _extract_two_models_ocr_text(egrn_ocr_bytes, include_crops=True),
    )

    qwen_result, llama_result = await asyncio.gather(
        _scan_registration_egrn_with_model(
            registration_ocr_bytes,
            egrn_ocr_bytes,
            registration_ocr_text,
            egrn_ocr_text,
            QWEN_30_MODEL,
            scan_id,
        ),
        _scan_registration_egrn_with_model(
            registration_ocr_bytes,
            egrn_ocr_bytes,
            registration_ocr_text,
            egrn_ocr_text,
            LLAMA_4_SCOUT_MODEL,
            scan_id,
        ),
    )

    registration_data, egrn_data, two_models_raw, models_used = _pick_best_result([qwen_result, llama_result])
    _clear_invalid_registration_address(registration_data)
    _clear_invalid_egrn_data(egrn_data, passport_data, source_is_passport=False)
    _log(
        scan_id,
        "request",
        "finish",
        registration_score=_registration_score(registration_data),
        egrn_score=_egrn_score(egrn_data),
        models=models_used,
    )

    return UnifiedDocumentsScanResponse(
        ok=True,
        model="RussianDocsOCR:ONNX:cpu + two-models:registration/egrn",
        data=UnifiedDocumentsData(
            passport_main=passport_data,
            passport_registration=registration_data,
            egrn_extract=egrn_data,
        ),
        raw_text={
            "passport_main": (
                f"{_compact_raw_payload(main_raw)}\n\n--- russian_docs_ocr_text ---\n{main_text}\n\n"
                f"--- selected_model ---\nRussianDocsOCR:ONNX:cpu"
            ),
            "passport_registration": (
                f"--- selected_model ---\n{models_used['passport_registration']}\n\n"
                f"{two_models_raw['qwen30_passport_registration']}\n\n"
                f"--- llama4scout_candidate ---\n{two_models_raw['llama4scout_passport_registration']}"
            ),
            "egrn_extract": (
                f"--- selected_model ---\n{models_used['egrn_extract']}\n\n"
                f"{two_models_raw['qwen30_egrn_extract']}\n\n"
                f"--- llama4scout_candidate ---\n{two_models_raw['llama4scout_egrn_extract']}"
            ),
            "_files": json.dumps(files_debug, ensure_ascii=False, indent=2),
            "_warnings": json.dumps(
                {
                    "files": file_warnings,
                    "models": {
                        **TWO_MODELS_MAP,
                        "passport_main": "RussianDocsOCR:ONNX:cpu",
                        **models_used,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    )
