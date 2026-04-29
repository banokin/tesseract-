from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from huggin_face_scan.model_config import LLAMA_4_SCOUT_MODEL, QWEN_30_MODEL, TWO_MODELS_MAP
from huggin_face_scan.scan_passport_hf import (
    UnifiedDocumentsData,
    UnifiedDocumentsScanResponse,
    build_egrn_prompt,
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
    run_hf_passport_extraction,
    safe_to_thread,
    validate_image,
)

router = APIRouter(tags=["passport-two-models"])

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
    extraction_debug: Dict[str, str]


async def _prepare_ocr_bytes(raw: bytes, content_type: str) -> bytes:
    if content_type == "application/pdf":
        return await safe_to_thread(pdf_first_page_to_png, raw)
    await safe_to_thread(validate_image, raw)
    return raw


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
    if cv2 is None or np is None:
        return contents


def _build_passport_number_variants(contents: bytes) -> list[bytes]:
    if cv2 is None or np is None:
        return [contents]

    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return [contents]

    h, w = image.shape[:2]
    crops: dict[str, "np.ndarray"] = {
        "right_vertical": image[int(h * 0.05) : int(h * 0.95), int(w * 0.78) : int(w * 0.98)],
        "bottom_right": image[int(h * 0.45) : h, int(w * 0.45) : w],
        "top_right": image[int(h * 0.02) : int(h * 0.25), int(w * 0.55) : w],
        "full": image,
    }

    variants: list[bytes] = []
    for crop in crops.values():
        if crop.size == 0:
            continue
        rotated_variants = [
            crop,
            cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(crop, cv2.ROTATE_180),
        ]
        for variant in rotated_variants:
            upscaled = cv2.resize(variant, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            _, binarized = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ok, encoded = cv2.imencode(".jpg", binarized)
            if ok:
                variants.append(encoded.tobytes())

    return variants or [contents]

    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return contents

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
                    dst = np.array(
                        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                        dtype="float32",
                    )
                    matrix = cv2.getPerspectiveTransform(rect, dst)
                    gray = cv2.warpPerspective(gray, matrix, (max_w, max_h))
                break

        # Crop bottom-right region where passport number is typically located.
        h, w = gray.shape[:2]
        y1 = int(h * 0.45)
        x1 = int(w * 0.45)
        roi = gray[y1:h, x1:w]
        if roi.size == 0:
            roi = gray

        upscaled = cv2.resize(roi, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        _, final = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ok, encoded = cv2.imencode(".jpg", final)
        return encoded.tobytes() if ok else contents
    except Exception:
        return contents


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
    payload = _extract_json_object(raw_text)
    series_raw = payload.get("series", "")
    number_raw = payload.get("number", "")
    direct = _sanitize_passport_digits(f"{series_raw}{number_raw}")

    if len(direct) != 10:
        # Fallback: try to extract 10 digits from plain model text.
        fallback = _sanitize_passport_digits(raw_text)
        if len(fallback) >= 10:
            direct = fallback[:10]

    if len(direct) != 10:
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
        raise HTTPException(
            status_code=422,
            detail=f"Распознан технически валидный, но подозрительный номер паспорта ({model_name}).",
        )
    return series, number


def _validate_passport_series_number_no_raise(raw_text: str, model_name: str) -> tuple[str | None, str | None]:
    try:
        series, number = _validate_passport_series_number(raw_text, model_name)
        return series, number
    except HTTPException:
        return None, None


async def _extract_series_and_number(
    passport_main_bytes: bytes,
    model_name: str,
) -> tuple[str, str, str]:
    preprocessed = await safe_to_thread(_preprocess_for_passport_number, passport_main_bytes)
    variants_primary = await safe_to_thread(_build_passport_number_variants, preprocessed)
    variants_original = await safe_to_thread(_build_passport_number_variants, passport_main_bytes)
    variants = variants_primary + variants_original[:6]
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
        for variant_idx, variant in enumerate(variants):
            raw_text, _ = await run_hf_document_extraction(
                variant,
                prompt,
                max_tokens=120,
                model_name=model_name,
            )
            try:
                series, number = _validate_passport_series_number(raw_text, model_name)
                debug = f"success prompt={prompt_idx} variant={variant_idx}"
                return series, number, f"{debug}\n{raw_text}"
            except HTTPException as e:
                if e.status_code == 422:
                    last_error = e
                    continue
                raise

    if last_error is not None:
        # Last chance: direct plain-text extraction without strict JSON format.
        raw_text, _ = await run_hf_document_extraction(
            passport_main_bytes,
            "Напиши только 10 цифр серии и номера паспорта РФ без пояснений.",
            max_tokens=80,
            model_name=model_name,
        )
        digits = _sanitize_passport_digits(raw_text)
        if len(digits) >= 10:
            series, number = digits[:4], digits[4:10]
            if f"{series}{number}" not in {"0000000000", "1111111111", "1234567890", "0123456789", "9999999999"}:
                return series, number, f"fallback plain-text\n{raw_text}"
        raise last_error
    raise HTTPException(
        status_code=422,
        detail=f"Не удалось надежно распознать серию/номер паспорта ({model_name}) после мульти-кропа.",
    )


async def _scan_with_model(
    main_ocr_bytes: bytes,
    registration_ocr_bytes: bytes,
    egrn_ocr_bytes: bytes,
    model_name: str,
) -> UnifiedDocumentsScanResponse:
    try:
        (
            (passport_raw, model_used),
            (registration_raw, _),
            (egrn_raw, _),
        ) = await asyncio.gather(
            run_hf_passport_extraction(main_ocr_bytes, model_name=model_name),
            run_hf_document_extraction(
                registration_ocr_bytes,
                build_registration_prompt(),
                max_tokens=600,
                model_name=model_name,
            ),
            run_hf_document_extraction(
                egrn_ocr_bytes,
                build_egrn_prompt(),
                max_tokens=700,
                model_name=model_name,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка сканирования документов ({model_name}): {e!r}",
        ) from e

    # Strict second pass for passport series+number with validation.
    strict_series, strict_number, strict_passport_raw = await _extract_series_and_number(
        main_ocr_bytes,
        model_name,
    )

    try:
        passport_data = normalize_passport_data(extract_json_from_text(passport_raw))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON паспорта ({model_name}): {passport_raw[:1000]}",
        ) from e

    try:
        registration_data = normalize_registration_data(extract_generic_json_from_text(registration_raw))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON страницы регистрации ({model_name}): {registration_raw[:1000]}",
        ) from e

    try:
        egrn_data = normalize_egrn_data(extract_generic_json_from_text(egrn_raw))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON выписки ЕГРН ({model_name}): {egrn_raw[:1000]}",
        ) from e

    passport_data, registration_data, egrn_data = await asyncio.gather(
        enrich_passport_fields(main_ocr_bytes, passport_data, model_name=model_name),
        enrich_registration_fields(
            registration_ocr_bytes,
            registration_data,
            model_name=model_name,
        ),
        enrich_egrn_fields(egrn_ocr_bytes, egrn_data, model_name=model_name),
    )
    passport_data.passport_series = strict_series
    passport_data.passport_number = strict_number
    note = (passport_data.confidence_note or "").strip()
    strict_note = "Серия/номер верифицированы строгим пайплайном (preprocess + JSON validation)."
    passport_data.confidence_note = f"{note} {strict_note}".strip() if note else strict_note

    return UnifiedDocumentsScanResponse(
        ok=True,
        model=model_used,
        data=UnifiedDocumentsData(
            passport_main=passport_data,
            passport_registration=registration_data,
            egrn_extract=egrn_data,
        ),
        raw_text={
            "passport_main": f"{passport_raw}\n\n--- strict_series_number_pass ---\n{strict_passport_raw}",
            "passport_registration": registration_raw,
            "egrn_extract": egrn_raw,
        },
    )


def _build_consensus_payload(
    qwen_result: UnifiedDocumentsScanResponse,
    llama_result: UnifiedDocumentsScanResponse,
    tesseract_full: str | None = None,
) -> tuple[str, bool, Dict[str, Dict[str, str]]]:
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
    return consensus, needs_review, extracted


def _extract_with_tesseract(contents: bytes) -> str | None:
    if pytesseract is None or cv2 is None or np is None:
        return None
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
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
                return full
    except Exception:
        return None
    return None


@router.post("/scan-documents-unified-two-models", response_model=TwoModelsUnifiedResponse)
async def scan_documents_unified_two_models(
    passport_main: UploadFile = File(...),
    passport_registration: UploadFile = File(...),
    egrn_extract: UploadFile = File(...),
) -> TwoModelsUnifiedResponse:
    main_type = (passport_main.content_type or "").lower()
    reg_type = (passport_registration.content_type or "").lower()
    egrn_type = (egrn_extract.content_type or "").lower()
    if not (main_type.startswith("image/") or main_type == "application/pdf"):
        raise HTTPException(status_code=400, detail="passport_main: загрузите изображение или PDF")
    if not (reg_type.startswith("image/") or reg_type == "application/pdf"):
        raise HTTPException(status_code=400, detail="passport_registration: загрузите изображение или PDF")
    if not (egrn_type.startswith("image/") or egrn_type == "application/pdf"):
        raise HTTPException(status_code=400, detail="egrn_extract: поддерживаются изображение или PDF")

    main_bytes = await passport_main.read()
    registration_bytes = await passport_registration.read()
    egrn_bytes = await egrn_extract.read()

    main_ocr_bytes, registration_ocr_bytes, egrn_ocr_bytes = await asyncio.gather(
        _prepare_ocr_bytes(main_bytes, main_type),
        _prepare_ocr_bytes(registration_bytes, reg_type),
        _prepare_ocr_bytes(egrn_bytes, egrn_type),
    )

    qwen_result, llama_result = await asyncio.gather(
        _scan_with_model(main_ocr_bytes, registration_ocr_bytes, egrn_ocr_bytes, QWEN_30_MODEL),
        _scan_with_model(main_ocr_bytes, registration_ocr_bytes, egrn_ocr_bytes, LLAMA_4_SCOUT_MODEL),
    )
    tesseract_full = await safe_to_thread(_extract_with_tesseract, main_ocr_bytes)
    passport_number_consensus, needs_review, extracted_numbers = _build_consensus_payload(
        qwen_result,
        llama_result,
        tesseract_full=tesseract_full,
    )
    recommended = {"series": "", "number": ""}
    if not needs_review:
        full_values = [value.get("full", "") for value in extracted_numbers.values() if value.get("full", "")]
        winner_full = Counter(full_values).most_common(1)[0][0] if full_values else ""
        if len(winner_full) == 10:
            recommended = {"series": winner_full[:4], "number": winner_full[4:10]}
    debug = {"tesseract_used": "yes" if tesseract_full else "no"}

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
        extraction_debug=debug,
    )
