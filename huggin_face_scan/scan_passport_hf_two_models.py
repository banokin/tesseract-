from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from contextvars import ContextVar
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from document_tesseract.egrn_parser import parse_egrn_ocr_text
from document_tesseract.passport_parser import parse_passport_ocr_text
from document_tesseract.registration_parser import parse_registration_ocr_text
from huggin_face_scan.model_config import LLAMA_4_SCOUT_MODEL, QWEN_30_MODEL, TWO_MODELS_MAP
from huggin_face_scan.scan_passport_hf import (
    UnifiedDocumentsData,
    UnifiedDocumentsScanResponse,
    build_egrn_prompt,
    build_prompt as build_passport_prompt,
    build_registration_prompt,
    enrich_egrn_fields,
    enrich_passport_fields,
    enrich_registration_fields,
    extract_generic_json_from_text,
    extract_json_from_text,
    normalize_egrn_data,
    normalize_passport_data,
    normalize_registration_data,
    pdf_first_page_to_png,
    run_hf_document_extraction,
    safe_to_thread,
    validate_image,
)
from tesseract_scan.ocr import img_ocr_multi_pass

router = APIRouter(tags=["passport-two-models"])
logger = logging.getLogger(__name__)
_scan_id_ctx: ContextVar[str | None] = ContextVar("two_models_scan_id", default=None)


def _format_log_fields(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    return " " + " ".join(f"{key}={value!r}" for key, value in fields.items())


def _mask_digits(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def _log_stage(stage: str, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    scan_id = _scan_id_ctx.get()
    extra = {"stage": stage, **fields}
    if scan_id:
        extra["scan_id"] = scan_id
        fields = {"scan_id": scan_id, **fields}
    logger.log(level, "two-models %s: %s%s", stage, event, _format_log_fields(fields), extra=extra)


def _log_stage_exception(stage: str, event: str, **fields: Any) -> None:
    scan_id = _scan_id_ctx.get()
    extra = {"stage": stage, **fields}
    if scan_id:
        extra["scan_id"] = scan_id
        fields = {"scan_id": scan_id, **fields}
    logger.exception("two-models %s: %s%s", stage, event, _format_log_fields(fields), extra=extra)

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    cv2 = None
    np = None

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    pytesseract = None


class TwoModelsUnifiedResponse(BaseModel):
    ok: bool
    models: Dict[str, str]
    data: Dict[str, UnifiedDocumentsScanResponse]
    passport_number_consensus: str
    needs_review: bool
    extracted_numbers: Dict[str, Dict[str, str]]
    recommended_passport_number: Dict[str, str]
    passport_registration_validation: Dict[str, Any]
    extraction_debug: Dict[str, str]


async def _prepare_ocr_bytes(raw: bytes, content_type: str) -> bytes:
    _log_stage("prepare_ocr_bytes", "start", content_type=content_type, input_bytes=len(raw))
    if content_type == "application/pdf":
        prepared = await safe_to_thread(pdf_first_page_to_png, raw)
        _log_stage("prepare_ocr_bytes", "pdf_converted", output_bytes=len(prepared))
        return prepared
    await safe_to_thread(validate_image, raw)
    _log_stage("prepare_ocr_bytes", "image_validated", output_bytes=len(raw))
    return raw


def _truncate_ocr_context(text: str, max_chars: int = 5000) -> str:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[OCR truncated: {len(text) - max_chars} chars omitted]"


def _prompt_with_ocr_context(prompt: str, ocr_text: str, document_name: str) -> str:
    if not ocr_text.strip():
        return prompt
    return (
        f"{prompt}\n\n"
        f"Ниже шумный OCR-текст документа «{document_name}». "
        "Используй его как подсказку для букв, дат и номеров, но главным источником считай изображение. "
        "Не придумывай значения, если их нет ни на изображении, ни в OCR. Верни только JSON по исходной схеме.\n"
        "OCR_TEXT_BEGIN\n"
        f"{_truncate_ocr_context(ocr_text)}\n"
        "OCR_TEXT_END"
    )


async def _extract_ocr_text(contents: bytes, *, include_crops: bool = True) -> str:
    _log_stage(
        "extract_ocr_text",
        "start",
        input_bytes=len(contents),
        include_crops=include_crops,
    )
    try:
        text = await safe_to_thread(img_ocr_multi_pass, contents, include_crops=include_crops)
        _log_stage("extract_ocr_text", "success", chars=len(text or ""))
        return text
    except Exception as e:
        _log_stage_exception("extract_ocr_text", "failed", error_repr=repr(e))
        return ""


def _is_empty_ocr_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def _merge_model_with_ocr(model_payload: dict[str, Any], ocr_payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(model_payload)
    for key, ocr_value in ocr_payload.items():
        if _is_empty_ocr_value(ocr_value):
            continue
        if key not in merged or _is_empty_ocr_value(merged.get(key)):
            merged[key] = ocr_value
    return merged


def _order_quad_points(points: "np.ndarray") -> "np.ndarray":
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _preprocess_for_passport_number(contents: bytes) -> bytes:
    _log_stage("passport_number_preprocess", "start", input_bytes=len(contents))
    if cv2 is None or np is None:
        _log_stage("passport_number_preprocess", "opencv_unavailable")
        return contents

    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        _log_stage("passport_number_preprocess", "image_decode_failed", level=logging.WARNING)
        return contents

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        _log_stage("passport_number_preprocess", "decoded", width=w, height=h)

        # CLAHE improves local contrast on dark/uneven mobile photos.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Deskew: estimate angle from text-like foreground pixels.
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(th)
        if coords is not None and len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.3:
                h, w = gray.shape[:2]
                _log_stage("passport_number_preprocess", "deskew", angle=round(float(angle), 3))
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                gray = cv2.warpAffine(
                    gray,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )

        # Perspective correction from largest 4-point contour.
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype("float32")
                rect = _order_quad_points(quad)
                (tl, tr, br, bl) = rect
                width_a = np.linalg.norm(br - bl)
                width_b = np.linalg.norm(tr - tl)
                max_w = max(int(width_a), int(width_b))
                height_a = np.linalg.norm(tr - br)
                height_b = np.linalg.norm(tl - bl)
                max_h = max(int(height_a), int(height_b))
                if max_w > 0 and max_h > 0:
                    _log_stage(
                        "passport_number_preprocess",
                        "perspective_correction",
                        width=max_w,
                        height=max_h,
                    )
                    dst = np.array(
                        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                        dtype="float32",
                    )
                    matrix = cv2.getPerspectiveTransform(rect, dst)
                    gray = cv2.warpPerspective(gray, matrix, (max_w, max_h))
                break

        ok, encoded = cv2.imencode(".jpg", gray)
        result = encoded.tobytes() if ok else contents
        _log_stage(
            "passport_number_preprocess",
            "finish",
            encoded=ok,
            output_bytes=len(result),
        )
        return result
    except Exception as e:
        _log_stage_exception("passport_number_preprocess", "failed", error_repr=repr(e))
        return contents


def _make_rotation_variants(crop: "np.ndarray") -> dict[str, "np.ndarray"]:
    return {
        "original": crop,
        "rotate_90_clockwise": cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
        "rotate_90_counterclockwise": cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
        "rotate_180": cv2.rotate(crop, cv2.ROTATE_180),
    }


def _preprocess_number_crop(rotated_crop: "np.ndarray") -> bytes | None:
    if rotated_crop.size == 0:
        return None
    upscaled = cv2.resize(rotated_crop, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(upscaled)
    _, binarized = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ok, encoded = cv2.imencode(".jpg", binarized)
    return encoded.tobytes() if ok else None


def _build_passport_number_variants(contents: bytes) -> list[tuple[str, bytes]]:
    _log_stage("passport_number_variants", "start", input_bytes=len(contents))
    if cv2 is None or np is None:
        _log_stage("passport_number_variants", "opencv_unavailable", variants=1)
        return [("original", contents)]

    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        _log_stage("passport_number_variants", "image_decode_failed", level=logging.WARNING, variants=1)
        return [("original", contents)]

    h, w = image.shape[:2]
    _log_stage("passport_number_variants", "decoded", width=w, height=h)
    crops: dict[str, "np.ndarray"] = {
        "right_vertical": image[int(h * 0.05) : int(h * 0.95), int(w * 0.78) : int(w * 0.98)],
        "left_vertical": image[int(h * 0.05) : int(h * 0.95), int(w * 0.02) : int(w * 0.22)],
        "bottom_right": image[int(h * 0.45) : h, int(w * 0.45) : w],
        "top_right": image[int(h * 0.02) : int(h * 0.25), int(w * 0.55) : w],
        "full": image,
    }

    variants: list[tuple[str, bytes]] = []
    for crop_name, crop in crops.items():
        if crop.size == 0:
            _log_stage("passport_number_variants", "empty_crop_skipped", crop=crop_name)
            continue
        # Rotate the raw crop first; OCR/VLM reads horizontal digits more reliably after that.
        for rotation_name, rotated_crop in _make_rotation_variants(crop).items():
            encoded = _preprocess_number_crop(rotated_crop)
            if encoded is not None:
                variants.append((f"{crop_name}:{rotation_name}", encoded))

    if variants:
        _log_stage("passport_number_variants", "built", variants=len(variants))
        return variants
    _log_stage("passport_number_variants", "fallback_original", variants=1)
    return [("original", contents)]


def _sanitize_passport_digits(raw_value: str) -> str:
    mapped = (
        raw_value.upper()
        .replace("O", "0")
        .replace("О", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("З", "3")
        .replace("B", "8")
        .replace("В", "8")
    )
    return "".join(ch for ch in mapped if ch.isdigit())


def _extract_json_object(text: str) -> dict[str, str]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        return {}
    return {}


def _validate_passport_series_number(raw_text: str, model_name: str) -> tuple[str, str]:
    _log_stage("validate_passport_number", "start", model=model_name, raw_chars=len(raw_text or ""))
    payload = _extract_json_object(raw_text)
    series_raw = payload.get("series", "")
    number_raw = payload.get("number", "")
    direct = _sanitize_passport_digits(f"{series_raw}{number_raw}")

    if len(direct) != 10:
        # Fallback: try to extract 10 digits from plain model text.
        fallback = _sanitize_passport_digits(raw_text)
        if len(fallback) >= 10:
            _log_stage("validate_passport_number", "json_incomplete_plain_text_fallback", model=model_name)
            direct = fallback[:10]

    if len(direct) != 10:
        _log_stage(
            "validate_passport_number",
            "failed_length",
            level=logging.WARNING,
            model=model_name,
            digit_count=len(direct),
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"Не удалось надежно распознать серию/номер паспорта ({model_name}). "
                "Сделайте фото четче: хороший свет, без бликов, документ полностью в кадре."
            ),
        )

    series, number = direct[:4], direct[4:]
    full = f"{series}{number}"
    if full in {"0000000000", "1111111111", "1234567890", "0123456789", "9999999999"}:
        _log_stage(
            "validate_passport_number",
            "failed_suspicious_value",
            level=logging.WARNING,
            model=model_name,
            value_masked=_mask_digits(full),
        )
        raise HTTPException(
            status_code=422,
            detail=f"Распознан технически валидный, но подозрительный номер паспорта ({model_name}).",
        )
    _log_stage("validate_passport_number", "success", model=model_name, value_masked=_mask_digits(full))
    return series, number


def _validate_passport_series_number_no_raise(raw_text: str, model_name: str) -> tuple[str | None, str | None]:
    try:
        series, number = _validate_passport_series_number(raw_text, model_name)
        return series, number
    except HTTPException:
        return None, None


def _passport_full(series: str | None, number: str | None) -> str:
    return f"{series or ''}{number or ''}"


def _is_valid_passport_full(full: str) -> bool:
    return len(full) == 10 and full not in {
        "0000000000",
        "1111111111",
        "1234567890",
        "0123456789",
        "9999999999",
    }


def _extract_registration_passport_from_ocr(ocr_text: str) -> tuple[str | None, str | None, str]:
    _log_stage("registration_passport_ocr", "start", ocr_chars=len(ocr_text or ""))
    if not ocr_text.strip():
        _log_stage("registration_passport_ocr", "empty")
        return None, None, "ocr_empty"
    try:
        parsed = parse_passport_ocr_text(ocr_text)
    except Exception as e:
        _log_stage_exception("registration_passport_ocr", "parser_failed", error_repr=repr(e))
        parsed = {}
    series = _sanitize_passport_digits(str(parsed.get("passport_series", "") or ""))[:4]
    number = _sanitize_passport_digits(str(parsed.get("passport_number", "") or ""))[:6]
    if _is_valid_passport_full(_passport_full(series, number)):
        _log_stage(
            "registration_passport_ocr",
            "success",
            value_masked=_mask_digits(_passport_full(series, number)),
        )
        return series, number, "ocr_parser"
    _log_stage("registration_passport_ocr", "not_found", series_chars=len(series), number_chars=len(number))
    return None, None, "ocr_not_found"


async def _extract_registration_passport_number(
    registration_bytes: bytes,
    registration_ocr_text: str,
    model_name: str,
) -> tuple[str | None, str | None, str]:
    _log_stage(
        "registration_passport_number",
        "start",
        model=model_name,
        input_bytes=len(registration_bytes),
        ocr_chars=len(registration_ocr_text or ""),
    )
    ocr_series, ocr_number, ocr_source = _extract_registration_passport_from_ocr(registration_ocr_text)
    if ocr_series and ocr_number:
        _log_stage("registration_passport_number", "ocr_success", model=model_name, source=ocr_source)
        return ocr_series, ocr_number, ocr_source

    prompt = _prompt_with_ocr_context(
        (
            "На изображении страницы прописки паспорта РФ найди серию и номер паспорта, "
            "к которому относится эта страница. Нужно вернуть ровно 10 цифр: 4 цифры серии и 6 цифр номера. "
            "Верни строго JSON без пояснений:\n"
            '{"series":"1234","number":"567890"}\n'
            "Если серия/номер на странице не видны, верни:\n"
            '{"series":null,"number":null}'
        ),
        registration_ocr_text,
        "страница регистрации паспорта РФ",
    )
    try:
        _log_stage("registration_passport_number", "vlm_start", model=model_name, ocr_source=ocr_source)
        raw_text, _ = await run_hf_document_extraction(
            registration_bytes,
            prompt,
            max_tokens=120,
            model_name=model_name,
        )
        _log_stage(
            "registration_passport_number",
            "vlm_response",
            model=model_name,
            raw_chars=len(raw_text or ""),
        )
    except Exception as e:
        _log_stage_exception(
            "registration_passport_number",
            "vlm_failed",
            model=model_name,
            ocr_source=ocr_source,
            error_repr=repr(e),
        )
        return None, None, f"{ocr_source}+vlm_error"

    series, number = _validate_passport_series_number_no_raise(raw_text, model_name)
    if series and number:
        _log_stage("registration_passport_number", "vlm_success", model=model_name, ocr_source=ocr_source)
        return series, number, f"{ocr_source}+vlm"
    _log_stage("registration_passport_number", "not_found", model=model_name, ocr_source=ocr_source)
    return None, None, f"{ocr_source}+vlm_not_found"


def _build_registration_validation(
    main_full: str,
    registration_candidates: Dict[str, str],
) -> Dict[str, Any]:
    valid_candidates = {
        source: full
        for source, full in registration_candidates.items()
        if _is_valid_passport_full(full)
    }
    main = {
        "series": main_full[:4] if len(main_full) == 10 else "",
        "number": main_full[4:10] if len(main_full) == 10 else "",
        "full": main_full if len(main_full) == 10 else "",
    }
    if not _is_valid_passport_full(main_full):
        return {
            "status": "main_not_found",
            "main": main,
            "registration": {"series": "", "number": "", "full": ""},
            "sources": registration_candidates,
            "message": "Не удалось надежно определить серию/номер на основной странице.",
        }

    for source, full in valid_candidates.items():
        if full == main_full:
            return {
                "status": "match",
                "main": main,
                "registration": {"series": full[:4], "number": full[4:10], "full": full},
                "sources": valid_candidates,
                "source": source,
                "message": "Серия и номер на основной странице и странице прописки совпадают.",
            }

    if valid_candidates:
        full, _ = Counter(valid_candidates.values()).most_common(1)[0]
        return {
            "status": "mismatch",
            "main": main,
            "registration": {"series": full[:4], "number": full[4:10], "full": full},
            "sources": valid_candidates,
            "message": "Серия/номер на странице прописки не совпадают с основной страницей.",
        }

    return {
        "status": "not_found",
        "main": main,
        "registration": {"series": "", "number": "", "full": ""},
        "sources": registration_candidates,
        "message": "Не удалось найти серию/номер на странице прописки для проверки совпадения.",
    }


async def _extract_series_and_number(
    passport_main_bytes: bytes,
    model_name: str,
) -> tuple[str, str, str]:
    _log_stage(
        "strict_passport_number",
        "start",
        model=model_name,
        input_bytes=len(passport_main_bytes),
    )
    preprocessed = await safe_to_thread(_preprocess_for_passport_number, passport_main_bytes)
    _log_stage(
        "strict_passport_number",
        "preprocessed",
        model=model_name,
        output_bytes=len(preprocessed),
    )
    variants_primary = await safe_to_thread(_build_passport_number_variants, preprocessed)
    variants_original = await safe_to_thread(_build_passport_number_variants, passport_main_bytes)
    variants = variants_primary + variants_original[:6]
    _log_stage(
        "strict_passport_number",
        "variants_ready",
        model=model_name,
        primary_variants=len(variants_primary),
        original_variants=len(variants_original),
        total_variants=len(variants),
    )
    prompts = [
        (
            "Ты извлекаешь только серию и номер паспорта РФ. "
            "На изображении могут быть вертикальные цифры. "
            "Нужно найти ровно 10 цифр: 4 цифры серии и 6 цифр номера. "
            "Верни строго JSON без пояснений:\n"
            '{"series":"1234","number":"567890"}\n'
            "Если хотя бы одна цифра неразборчива, верни:\n"
            '{"series":null,"number":null}'
        ),
        (
            "Найди серию и номер паспорта РФ. "
            "Формат: серия 4 цифры и номер 6 цифр. "
            "Ответ строго JSON: {'series':'1234','number':'567890'}."
        ),
        (
            "Извлеки 10 цифр паспорта РФ (обычно XX YY NNNNNN), "
            "верни только JSON вида {'series':'1234','number':'567890'}."
        ),
    ]
    last_error: HTTPException | None = None
    for prompt_idx, prompt in enumerate(prompts):
        for variant_idx, (variant_name, variant) in enumerate(variants):
            _log_stage(
                "strict_passport_number",
                "vlm_attempt_start",
                model=model_name,
                prompt_idx=prompt_idx,
                variant_idx=variant_idx,
                variant=variant_name,
                variant_bytes=len(variant),
            )
            try:
                raw_text, _ = await run_hf_document_extraction(
                    variant,
                    prompt,
                    max_tokens=120,
                    model_name=model_name,
                )
            except Exception as e:
                _log_stage_exception(
                    "strict_passport_number",
                    "vlm_attempt_failed",
                    model=model_name,
                    prompt_idx=prompt_idx,
                    variant_idx=variant_idx,
                    variant=variant_name,
                    error_repr=repr(e),
                )
                raise
            _log_stage(
                "strict_passport_number",
                "vlm_attempt_response",
                model=model_name,
                prompt_idx=prompt_idx,
                variant_idx=variant_idx,
                variant=variant_name,
                raw_chars=len(raw_text or ""),
            )
            try:
                series, number = _validate_passport_series_number(raw_text, model_name)
                debug = f"success prompt={prompt_idx} variant={variant_idx} {variant_name}"
                _log_stage(
                    "strict_passport_number",
                    "success",
                    model=model_name,
                    prompt_idx=prompt_idx,
                    variant_idx=variant_idx,
                    variant=variant_name,
                    value_masked=_mask_digits(f"{series}{number}"),
                )
                return series, number, f"{debug}\n{raw_text}"
            except HTTPException as e:
                if e.status_code == 422:
                    _log_stage(
                        "strict_passport_number",
                        "attempt_rejected",
                        level=logging.WARNING,
                        model=model_name,
                        prompt_idx=prompt_idx,
                        variant_idx=variant_idx,
                        variant=variant_name,
                        status_code=e.status_code,
                    )
                    last_error = e
                    continue
                raise

    if last_error is not None:
        # Last chance: direct plain-text extraction without strict JSON format.
        _log_stage("strict_passport_number", "plain_text_fallback_start", model=model_name)
        try:
            raw_text, _ = await run_hf_document_extraction(
                passport_main_bytes,
                "Напиши только 10 цифр серии и номера паспорта РФ без пояснений.",
                max_tokens=80,
                model_name=model_name,
            )
        except Exception as e:
            _log_stage_exception(
                "strict_passport_number",
                "plain_text_fallback_error",
                model=model_name,
                error_repr=repr(e),
            )
            raise
        digits = _sanitize_passport_digits(raw_text)
        if len(digits) >= 10:
            series, number = digits[:4], digits[4:10]
            if f"{series}{number}" not in {"0000000000", "1111111111", "1234567890", "0123456789", "9999999999"}:
                _log_stage(
                    "strict_passport_number",
                    "plain_text_fallback_success",
                    model=model_name,
                    raw_chars=len(raw_text or ""),
                    value_masked=_mask_digits(f"{series}{number}"),
                )
                return series, number, f"fallback plain-text\n{raw_text}"
        _log_stage(
            "strict_passport_number",
            "plain_text_fallback_failed",
            level=logging.WARNING,
            model=model_name,
            raw_chars=len(raw_text or ""),
            digit_count=len(digits),
        )
        raise last_error
    _log_stage("strict_passport_number", "failed_no_attempts", level=logging.ERROR, model=model_name)
    raise HTTPException(
        status_code=422,
        detail=f"Не удалось надежно распознать серию/номер паспорта ({model_name}) после мульти-кропа.",
    )


async def _scan_with_model(
    main_ocr_bytes: bytes,
    registration_ocr_bytes: bytes,
    egrn_ocr_bytes: bytes,
    passport_ocr_text: str,
    registration_ocr_text: str,
    egrn_ocr_text: str,
    model_name: str,
) -> UnifiedDocumentsScanResponse:
    _log_stage(
        "scan_with_model",
        "start",
        model=model_name,
        main_bytes=len(main_ocr_bytes),
        registration_bytes=len(registration_ocr_bytes),
        egrn_bytes=len(egrn_ocr_bytes),
        passport_ocr_chars=len(passport_ocr_text or ""),
        registration_ocr_chars=len(registration_ocr_text or ""),
        egrn_ocr_chars=len(egrn_ocr_text or ""),
    )
    try:
        _log_stage("scan_with_model", "document_extraction_start", model=model_name)
        (
            (passport_raw, model_used),
            (registration_raw, _),
            (egrn_raw, _),
        ) = await asyncio.gather(
            run_hf_document_extraction(
                main_ocr_bytes,
                _prompt_with_ocr_context(build_passport_prompt(), passport_ocr_text, "паспорт РФ"),
                max_tokens=700,
                model_name=model_name,
            ),
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
        _log_stage(
            "scan_with_model",
            "document_extraction_success",
            model=model_name,
            model_used=model_used,
            passport_raw_chars=len(passport_raw or ""),
            registration_raw_chars=len(registration_raw or ""),
            egrn_raw_chars=len(egrn_raw or ""),
        )
    except HTTPException:
        _log_stage("scan_with_model", "document_extraction_http_error", level=logging.WARNING, model=model_name)
        raise
    except Exception as e:
        _log_stage_exception("scan_with_model", "document_extraction_failed", model=model_name, error_repr=repr(e))
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка сканирования документов ({model_name}): {e!r}",
        ) from e

    # Strict second pass for passport series+number with validation.
    _log_stage("scan_with_model", "strict_number_start", model=model_name)
    strict_series, strict_number, strict_passport_raw = await _extract_series_and_number(
        main_ocr_bytes,
        model_name,
    )
    _log_stage(
        "scan_with_model",
        "strict_number_success",
        model=model_name,
        value_masked=_mask_digits(f"{strict_series}{strict_number}"),
    )

    try:
        _log_stage("scan_with_model", "passport_json_parse_start", model=model_name)
        passport_payload = extract_json_from_text(passport_raw)
        passport_ocr_payload = parse_passport_ocr_text(passport_ocr_text) if passport_ocr_text else {}
        passport_data = normalize_passport_data(
            _merge_model_with_ocr(passport_payload, passport_ocr_payload)
        )
    except Exception as e:
        _log_stage_exception("scan_with_model", "passport_json_parse_failed", model=model_name, error_repr=repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON паспорта ({model_name}): {passport_raw[:1000]}",
        ) from e
    _log_stage(
        "scan_with_model",
        "passport_json_parse_success",
        model=model_name,
        model_fields=len(passport_payload),
        ocr_fields=len(passport_ocr_payload),
    )

    try:
        _log_stage("scan_with_model", "registration_json_parse_start", model=model_name)
        registration_payload = extract_generic_json_from_text(registration_raw)
        registration_ocr_payload = (
            parse_registration_ocr_text(registration_ocr_text) if registration_ocr_text else {}
        )
        registration_data = normalize_registration_data(
            _merge_model_with_ocr(registration_payload, registration_ocr_payload)
        )
    except Exception as e:
        _log_stage_exception("scan_with_model", "registration_json_parse_failed", model=model_name, error_repr=repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON страницы регистрации ({model_name}): {registration_raw[:1000]}",
        ) from e
    _log_stage(
        "scan_with_model",
        "registration_json_parse_success",
        model=model_name,
        model_fields=len(registration_payload),
        ocr_fields=len(registration_ocr_payload),
    )

    try:
        _log_stage("scan_with_model", "egrn_json_parse_start", model=model_name)
        egrn_payload = extract_generic_json_from_text(egrn_raw)
        egrn_ocr_payload = parse_egrn_ocr_text(egrn_ocr_text) if egrn_ocr_text else {}
        egrn_data = normalize_egrn_data(_merge_model_with_ocr(egrn_payload, egrn_ocr_payload))
    except Exception as e:
        _log_stage_exception("scan_with_model", "egrn_json_parse_failed", model=model_name, error_repr=repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON выписки ЕГРН ({model_name}): {egrn_raw[:1000]}",
        ) from e
    _log_stage(
        "scan_with_model",
        "egrn_json_parse_success",
        model=model_name,
        model_fields=len(egrn_payload),
        ocr_fields=len(egrn_ocr_payload),
    )

    _log_stage("scan_with_model", "enrich_start", model=model_name)
    passport_data, registration_data, egrn_data = await asyncio.gather(
        enrich_passport_fields(main_ocr_bytes, passport_data, model_name=model_name),
        enrich_registration_fields(
            registration_ocr_bytes,
            registration_data,
            model_name=model_name,
        ),
        enrich_egrn_fields(egrn_ocr_bytes, egrn_data, model_name=model_name),
    )
    _log_stage("scan_with_model", "enrich_success", model=model_name)
    passport_data.passport_series = strict_series
    passport_data.passport_number = strict_number
    note = (passport_data.confidence_note or "").strip()
    strict_note = "Серия/номер верифицированы строгим пайплайном (preprocess + JSON validation)."
    passport_data.confidence_note = f"{note} {strict_note}".strip() if note else strict_note

    _log_stage("scan_with_model", "finish", model=model_name, model_used=model_used)
    return UnifiedDocumentsScanResponse(
        ok=True,
        model=model_used,
        data=UnifiedDocumentsData(
            passport_main=passport_data,
            passport_registration=registration_data,
            egrn_extract=egrn_data,
        ),
        raw_text={
            "passport_main": (
                f"{passport_raw}\n\n--- strict_series_number_pass ---\n{strict_passport_raw}"
                f"\n\n--- tesseract_ocr_context ---\n{passport_ocr_text}"
            ),
            "passport_registration": (
                f"{registration_raw}\n\n--- tesseract_ocr_context ---\n{registration_ocr_text}"
            ),
            "egrn_extract": f"{egrn_raw}\n\n--- tesseract_ocr_context ---\n{egrn_ocr_text}",
        },
    )


def _build_consensus_payload(
    qwen_result: UnifiedDocumentsScanResponse,
    llama_result: UnifiedDocumentsScanResponse,
    tesseract_full: str | None = None,
) -> tuple[str, bool, Dict[str, Dict[str, str]]]:
    _log_stage(
        "consensus",
        "start",
        has_tesseract=bool(tesseract_full),
    )
    qwen_series = (qwen_result.data.passport_main.passport_series or "").strip()
    qwen_number = (qwen_result.data.passport_main.passport_number or "").strip()
    llama_series = (llama_result.data.passport_main.passport_series or "").strip()
    llama_number = (llama_result.data.passport_main.passport_number or "").strip()
    qwen_full = f"{qwen_series}{qwen_number}"
    llama_full = f"{llama_series}{llama_number}"

    votes = [v for v in [qwen_full, llama_full, tesseract_full] if v]
    top_value = ""
    top_count = 0
    if votes:
        counter = Counter(votes)
        top_value, top_count = counter.most_common(1)[0]

    high_confidence = top_count >= 2 and bool(top_value)
    consensus = "high" if high_confidence else "needs_review"
    needs_review = not high_confidence
    extracted = {
        "qwen30": {
            "series": qwen_series,
            "number": qwen_number,
            "full": qwen_full,
        },
        "llama4scout": {
            "series": llama_series,
            "number": llama_number,
            "full": llama_full,
        },
    }
    if tesseract_full:
        extracted["tesseract"] = {
            "series": tesseract_full[:4],
            "number": tesseract_full[4:10],
            "full": tesseract_full,
        }
    _log_stage(
        "consensus",
        "finish",
        consensus=consensus,
        needs_review=needs_review,
        vote_count=len(votes),
        top_count=top_count,
        top_value_masked=_mask_digits(top_value) if top_value else "",
    )
    return consensus, needs_review, extracted


def _extract_with_tesseract(contents: bytes) -> str | None:
    _log_stage("tesseract_number", "start", input_bytes=len(contents))
    if pytesseract is None or cv2 is None or np is None:
        _log_stage("tesseract_number", "dependencies_unavailable")
        return None
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        _log_stage("tesseract_number", "image_decode_failed", level=logging.WARNING)
        return None
    try:
        prepared = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(
            prepared,
            config="--psm 6 -c tessedit_char_whitelist=0123456789OIOLBВЗО",
        )
        digits = _sanitize_passport_digits(text)
        if len(digits) >= 10:
            full = digits[:10]
            if full not in {"0000000000", "1111111111", "1234567890", "0123456789", "9999999999"}:
                _log_stage(
                    "tesseract_number",
                    "success",
                    raw_chars=len(text or ""),
                    digit_count=len(digits),
                    value_masked=_mask_digits(full),
                )
                return full
        _log_stage("tesseract_number", "not_found", raw_chars=len(text or ""), digit_count=len(digits))
    except Exception as e:
        _log_stage_exception("tesseract_number", "failed", error_repr=repr(e))
        return None
    _log_stage("tesseract_number", "finish_empty")
    return None


@router.post("/scan-documents-unified-two-models", response_model=TwoModelsUnifiedResponse)
async def scan_documents_unified_two_models(
    passport_main: UploadFile = File(...),
    passport_registration: UploadFile = File(...),
    egrn_extract: UploadFile = File(...),
) -> TwoModelsUnifiedResponse:
    scan_id = uuid4().hex[:12]
    _scan_id_ctx.set(scan_id)
    _log_stage("request", "start")
    main_type = (passport_main.content_type or "").lower()
    reg_type = (passport_registration.content_type or "").lower()
    egrn_type = (egrn_extract.content_type or "").lower()
    _log_stage(
        "request",
        "content_types",
        passport_main=main_type,
        passport_registration=reg_type,
        egrn_extract=egrn_type,
    )
    if not (main_type.startswith("image/") or main_type == "application/pdf"):
        _log_stage("request", "invalid_content_type", level=logging.WARNING, field="passport_main", content_type=main_type)
        raise HTTPException(status_code=400, detail="passport_main: загрузите изображение или PDF")
    if not (reg_type.startswith("image/") or reg_type == "application/pdf"):
        _log_stage(
            "request",
            "invalid_content_type",
            level=logging.WARNING,
            field="passport_registration",
            content_type=reg_type,
        )
        raise HTTPException(status_code=400, detail="passport_registration: загрузите изображение или PDF")
    if not (egrn_type.startswith("image/") or egrn_type == "application/pdf"):
        _log_stage("request", "invalid_content_type", level=logging.WARNING, field="egrn_extract", content_type=egrn_type)
        raise HTTPException(status_code=400, detail="egrn_extract: поддерживаются изображение или PDF")

    _log_stage("request", "read_files_start")
    main_bytes = await passport_main.read()
    registration_bytes = await passport_registration.read()
    egrn_bytes = await egrn_extract.read()
    _log_stage(
        "request",
        "read_files_success",
        passport_main_bytes=len(main_bytes),
        passport_registration_bytes=len(registration_bytes),
        egrn_extract_bytes=len(egrn_bytes),
    )

    _log_stage("request", "prepare_ocr_bytes_start")
    main_ocr_bytes, registration_ocr_bytes, egrn_ocr_bytes = await asyncio.gather(
        _prepare_ocr_bytes(main_bytes, main_type),
        _prepare_ocr_bytes(registration_bytes, reg_type),
        _prepare_ocr_bytes(egrn_bytes, egrn_type),
    )
    _log_stage(
        "request",
        "prepare_ocr_bytes_success",
        passport_main_bytes=len(main_ocr_bytes),
        passport_registration_bytes=len(registration_ocr_bytes),
        egrn_extract_bytes=len(egrn_ocr_bytes),
    )

    _log_stage("request", "ocr_context_start")
    passport_ocr_text, registration_ocr_text, egrn_ocr_text = await asyncio.gather(
        _extract_ocr_text(main_ocr_bytes, include_crops=True),
        _extract_ocr_text(registration_ocr_bytes, include_crops=True),
        _extract_ocr_text(egrn_ocr_bytes, include_crops=True),
    )
    _log_stage(
        "request",
        "ocr_context_success",
        passport_ocr_chars=len(passport_ocr_text or ""),
        registration_ocr_chars=len(registration_ocr_text or ""),
        egrn_ocr_chars=len(egrn_ocr_text or ""),
    )

    _log_stage("request", "two_model_scan_start", qwen_model=QWEN_30_MODEL, llama_model=LLAMA_4_SCOUT_MODEL)
    try:
        qwen_result, llama_result = await asyncio.gather(
            _scan_with_model(
                main_ocr_bytes,
                registration_ocr_bytes,
                egrn_ocr_bytes,
                passport_ocr_text,
                registration_ocr_text,
                egrn_ocr_text,
                QWEN_30_MODEL,
            ),
            _scan_with_model(
                main_ocr_bytes,
                registration_ocr_bytes,
                egrn_ocr_bytes,
                passport_ocr_text,
                registration_ocr_text,
                egrn_ocr_text,
                LLAMA_4_SCOUT_MODEL,
            ),
        )
    except Exception as e:
        _log_stage_exception("request", "two_model_scan_failed", error_repr=repr(e))
        raise
    _log_stage("request", "two_model_scan_success")
    _log_stage("request", "tesseract_passport_number_start")
    tesseract_full = await safe_to_thread(_extract_with_tesseract, main_ocr_bytes)
    _log_stage(
        "request",
        "tesseract_passport_number_finish",
        found=bool(tesseract_full),
        value_masked=_mask_digits(tesseract_full) if tesseract_full else "",
    )
    _log_stage("request", "consensus_start")
    passport_number_consensus, needs_review, extracted_numbers = _build_consensus_payload(
        qwen_result,
        llama_result,
        tesseract_full=tesseract_full,
    )
    _log_stage("request", "consensus_success", consensus=passport_number_consensus, needs_review=needs_review)
    recommended = {"series": "", "number": ""}
    winner_full = ""
    if not needs_review:
        full_values = [value.get("full", "") for value in extracted_numbers.values() if value.get("full", "")]
        winner_full = Counter(full_values).most_common(1)[0][0] if full_values else ""
        if len(winner_full) == 10:
            recommended = {"series": winner_full[:4], "number": winner_full[4:10]}
    _log_stage(
        "request",
        "recommendation_ready",
        has_recommendation=bool(recommended.get("series") and recommended.get("number")),
        value_masked=_mask_digits(winner_full) if winner_full else "",
    )

    _log_stage("request", "registration_passport_validation_extract_start")
    (reg_qwen_series, reg_qwen_number, reg_qwen_source), (
        reg_llama_series,
        reg_llama_number,
        reg_llama_source,
    ) = await asyncio.gather(
        _extract_registration_passport_number(registration_ocr_bytes, registration_ocr_text, QWEN_30_MODEL),
        _extract_registration_passport_number(
            registration_ocr_bytes,
            registration_ocr_text,
            LLAMA_4_SCOUT_MODEL,
        ),
    )
    _log_stage(
        "request",
        "registration_passport_validation_extract_success",
        qwen_source=reg_qwen_source,
        llama_source=reg_llama_source,
        qwen_found=bool(reg_qwen_series and reg_qwen_number),
        llama_found=bool(reg_llama_series and reg_llama_number),
    )
    registration_candidates: Dict[str, str] = {}
    reg_qwen_full = _passport_full(reg_qwen_series, reg_qwen_number)
    reg_llama_full = _passport_full(reg_llama_series, reg_llama_number)
    if _is_valid_passport_full(reg_qwen_full):
        registration_candidates[f"qwen30_registration:{reg_qwen_source}"] = reg_qwen_full
    if _is_valid_passport_full(reg_llama_full):
        registration_candidates[f"llama4scout_registration:{reg_llama_source}"] = reg_llama_full

    main_full_for_validation = _passport_full(recommended.get("series"), recommended.get("number"))
    if not _is_valid_passport_full(main_full_for_validation):
        main_full_for_validation = winner_full
    if not _is_valid_passport_full(main_full_for_validation):
        full_values = [value.get("full", "") for value in extracted_numbers.values() if value.get("full", "")]
        main_full_for_validation = Counter(full_values).most_common(1)[0][0] if full_values else ""

    _log_stage("request", "registration_validation_start", candidates=len(registration_candidates))
    passport_registration_validation = _build_registration_validation(
        main_full_for_validation,
        registration_candidates,
    )
    _log_stage(
        "request",
        "registration_validation_success",
        status=passport_registration_validation.get("status", ""),
    )
    if passport_registration_validation.get("status") != "match":
        needs_review = True
    reg_validation_full = str(passport_registration_validation.get("registration", {}).get("full", ""))
    if _is_valid_passport_full(reg_validation_full):
        extracted_numbers["passport_registration"] = {
            "series": reg_validation_full[:4],
            "number": reg_validation_full[4:10],
            "full": reg_validation_full,
        }

    debug = {
        "tesseract_used": "yes" if any([passport_ocr_text, registration_ocr_text, egrn_ocr_text]) else "no",
        "tesseract_passport_number_used": "yes" if tesseract_full else "no",
        "passport_ocr_chars": str(len(passport_ocr_text)),
        "registration_ocr_chars": str(len(registration_ocr_text)),
        "egrn_ocr_chars": str(len(egrn_ocr_text)),
        "passport_registration_validation": str(passport_registration_validation.get("status", "")),
    }

    _log_stage(
        "request",
        "finish",
        consensus=passport_number_consensus,
        needs_review=needs_review,
        extracted_sources=",".join(sorted(extracted_numbers.keys())),
    )
    return TwoModelsUnifiedResponse(
        ok=True,
        models=TWO_MODELS_MAP,
        data={
            "qwen30": qwen_result,
            "llama4scout": llama_result,
        },
        passport_number_consensus=passport_number_consensus,
        needs_review=needs_review,
        extracted_numbers=extracted_numbers,
        recommended_passport_number=recommended,
        passport_registration_validation=passport_registration_validation,
        extraction_debug=debug,
    )
