from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from huggin_face_scan.scan_passport_hf import (
    PassportScanResponse,
    _inference_client_for_provider,
    _message_content_to_str,
    _parse_router_model_id,
    extract_json_from_text,
    normalize_passport_data,
    pdf_first_page_to_png,
    run_hf_document_extraction,
    safe_to_thread,
    settings,
    validate_image,
)

router = APIRouter(tags=["passport-experiments-deepseek-qwen"])
logger = logging.getLogger(__name__)

DEEPSEEK_OCR_MODEL = os.getenv("HF_DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")
QWEN_STRUCTURER_MODEL = os.getenv("HF_QWEN_STRUCTURER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def _log(scan_id: str, stage: str, event: str, **fields: Any) -> None:
    suffix = " ".join(f"{key}={value!r}" for key, value in fields.items())
    logger.info("deepseek-qwen %s: %s scan_id=%s %s", stage, event, scan_id, suffix)


async def _prepare_image_bytes(contents: bytes, content_type: str, scan_id: str) -> tuple[bytes, str]:
    _log(scan_id, "prepare", "start", content_type=content_type, input_bytes=len(contents))
    if content_type == "application/pdf":
        image_bytes = await safe_to_thread(pdf_first_page_to_png, contents)
        _log(scan_id, "prepare", "pdf_converted", output_bytes=len(image_bytes))
        return image_bytes, "image/png"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите изображение или PDF")
    await safe_to_thread(validate_image, contents)
    _log(scan_id, "prepare", "image_validated", output_bytes=len(contents))
    return contents, content_type or "image/png"


def _run_hf_qwen_structuring(ocr_text: str) -> str:
    prompt = f"""
Извлеки данные паспорта РФ из OCR-текста и верни только валидный JSON.
Поля:
- issuing_authority
- issue_date
- department_code
- passport_series
- passport_number
- surname
- name
- patronymic
- gender
- birth_date
- birth_place
- confidence_note

Правила:
1. Не придумывай значения.
2. Если поле не найдено, верни пустую строку.
3. Серия паспорта: 4 цифры, номер: 6 цифр.
4. Ответ строго JSON без markdown.

OCR:
\"\"\"
{ocr_text[:12000]}
\"\"\"
""".strip()
    repo_id, provider = _parse_router_model_id(QWEN_STRUCTURER_MODEL)
    client = _inference_client_for_provider(provider)
    completion = client.chat.completions.create(
        model=repo_id,
        messages=[
            {
                "role": "system",
                "content": "Ты эксперт по структуризации данных из документов. Отвечай только JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    return _message_content_to_str(completion.choices[0].message.content)


@router.post("/scan-passport-deepseek-qwen", response_model=PassportScanResponse)
async def scan_passport_deepseek_qwen(
    file: UploadFile = File(...),
    ocr_prompt: str = Form("Free OCR."),
    timeout_sec: int = Form(180),
) -> PassportScanResponse:
    scan_id = uuid4().hex[:12]
    file_type = (file.content_type or "").lower()
    _log(
        scan_id,
        "request",
        "start",
        filename=file.filename,
        content_type=file_type,
        deepseek_model=DEEPSEEK_OCR_MODEL,
        qwen_model=QWEN_STRUCTURER_MODEL,
    )
    contents = await file.read()
    image_bytes, mime_type = await _prepare_image_bytes(contents, file_type, scan_id)

    try:
        _log(
            scan_id,
            "deepseek_ocr",
            "start",
            image_bytes=len(image_bytes),
            mime_type=mime_type,
            prompt=ocr_prompt,
            model=DEEPSEEK_OCR_MODEL,
        )
        ocr_text, ocr_model_used = await run_hf_document_extraction(
            image_bytes,
            ocr_prompt,
            max_tokens=3500,
            model_name=DEEPSEEK_OCR_MODEL,
        )
        _log(scan_id, "deepseek_ocr", "success", chars=len(ocr_text or ""), model_used=ocr_model_used)
    except HTTPException:
        _log(scan_id, "deepseek_ocr", "http_exception", model=DEEPSEEK_OCR_MODEL)
        raise
    except Exception as exc:
        logger.exception("DeepSeek-OCR failed scan_id=%s", scan_id)
        raise HTTPException(
            status_code=502,
            detail=(
                "Ошибка Hugging Face DeepSeek-OCR. "
                f"Проверьте HF_TOKEN, доступность модели {DEEPSEEK_OCR_MODEL} "
                f"и HF_REQUEST_TIMEOUT_SEC={settings.hf_request_timeout_sec}. "
                f"Техническая ошибка: {exc!r}"
            ),
        ) from exc

    try:
        _log(
            scan_id,
            "qwen_structuring",
            "start",
            ocr_chars=len(ocr_text or ""),
            model=QWEN_STRUCTURER_MODEL,
        )
        structured_text = await asyncio.wait_for(
            safe_to_thread(_run_hf_qwen_structuring, ocr_text),
            timeout=float(timeout_sec),
        )
        _log(scan_id, "qwen_structuring", "success", chars=len(structured_text or ""))
        payload = extract_json_from_text(structured_text)
    except TimeoutError as exc:
        logger.exception("Qwen structuring timeout scan_id=%s", scan_id)
        raise HTTPException(
            status_code=504,
            detail=f"Превышено время ожидания Hugging Face Qwen ({timeout_sec} с).",
        ) from exc
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.exception("Qwen JSON parse failed scan_id=%s", scan_id)
        raise HTTPException(
            status_code=500,
            detail=f"Qwen не вернул валидный JSON: {structured_text[:1000]}",
        ) from exc
    except Exception as exc:
        logger.exception("Qwen structuring failed scan_id=%s", scan_id)
        raise HTTPException(
            status_code=502,
            detail=(
                "Ошибка Hugging Face Qwen при структуризации. "
                f"Проверьте HF_TOKEN, доступность модели {QWEN_STRUCTURER_MODEL} "
                f"и HF_REQUEST_TIMEOUT_SEC={settings.hf_request_timeout_sec}. "
                f"Техническая ошибка: {exc!r}"
            ),
        ) from exc

    payload["confidence_note"] = str(payload.get("confidence_note") or "DeepSeek-OCR + Qwen через Hugging Face API.")
    data = normalize_passport_data(payload)
    _log(scan_id, "request", "finish")
    return PassportScanResponse(
        ok=True,
        model=f"{DEEPSEEK_OCR_MODEL}+{QWEN_STRUCTURER_MODEL}",
        data=data,
        raw_text=f"--- deepseek_ocr ---\n{ocr_text}\n\n--- qwen_json ---\n{structured_text}",
    )
