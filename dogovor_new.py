from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter, File, HTTPException, UploadFile

from async_utils import safe_to_thread
from dogovor import (
    ContractData,
    OUTPUT_DIR,
    create_doc,
    passport_scan_to_contract_data,
)
from scan_passport_hf import (
    extract_json_from_text,
    run_hf_passport_extraction,
    settings as hf_settings,
    validate_image,
)

router = APIRouter(tags=["dogovor-new"])


def passport_json_to_contract_data(passport_json: Mapping[str, Any]) -> ContractData:
    """
    Преобразует JSON сканирования паспорта в ``ContractData``.

    Поддерживается полный ответ API
    ``{"ok": true, "model": "...", "data": {...}}`` или только объект полей
    (как в ``data``: issuing_authority, issue_date, passport_series, …).
    """
    return passport_scan_to_contract_data(passport_json)


def create_contract_docx_from_passport_json(
    passport_json: Mapping[str, Any],
    output_path: Path | None = None,
) -> Path:
    """
    Заполняет шаблон ``шаблон договора.docx`` данными из JSON паспорта
    (через ``create_doc`` в ``dogovor.py``), сохраняет файл в ``generated_docs``.

    :param passport_json: ответ API или только блок полей паспорта
    :param output_path: путь к .docx; если ``None``, имя вида ``dogovor_<uuid>.docx``
    :returns: путь к созданному файлу
    """
    contract_data = passport_json_to_contract_data(passport_json)
    if output_path is None:
        output_path = OUTPUT_DIR / f"dogovor_{uuid.uuid4().hex}.docx"
    output_path = Path(output_path)
    create_doc(contract_data, output_path)
    return output_path


@router.post("/scan-passport-to-contract-hf")
async def scan_passport_to_contract_hf(file: UploadFile = File(...)):
    """Скан паспорта через HF → JSON → заполнение шаблона договора."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите изображение паспорта")

    contents = await file.read()
    validate_image(contents)

    raw_text, model_used = await run_hf_passport_extraction(contents)
    try:
        parsed = extract_json_from_text(raw_text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Модель вернула невалидный JSON. Ответ модели: {raw_text[:1000]}",
        ) from e

    contract_data = passport_scan_to_contract_data({"data": parsed})
    output_name = f"dogovor_{uuid.uuid4().hex}.docx"
    output_path = OUTPUT_DIR / output_name
    await safe_to_thread(create_doc, contract_data, output_path)

    return {
        "message": "Договор создан по данным паспорта (Hugging Face)",
        "filename": file.filename,
        "generated_filename": output_name,
        "download_url": f"/download/{output_name}",
        "json_data": {
            "source": "huggingface_passport",
            "model": model_used or hf_settings.hf_model,
            "passport_data": parsed,
            "contract_data": contract_data.model_dump(),
            "raw_text": raw_text,
        },
    }
