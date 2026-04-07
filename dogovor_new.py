from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from async_utils import safe_to_thread
from dogovor import (
    ContractData,
    OUTPUT_DIR,
    build_short_fio,
    create_doc,
    normalize_person_fio,
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


def _build_registration_address(reg: Mapping[str, Any]) -> str:
    ordered_parts = [
        str(reg.get("region", "") or "").strip(),
        str(reg.get("city", "") or "").strip(),
        str(reg.get("settlement", "") or "").strip(),
        str(reg.get("street", "") or "").strip(),
    ]
    house = str(reg.get("house", "") or "").strip()
    building = str(reg.get("building", "") or "").strip()
    apartment = str(reg.get("apartment", "") or "").strip()
    if house:
        ordered_parts.append(f"д. {house}")
    if building:
        ordered_parts.append(f"корп. {building}")
    if apartment:
        ordered_parts.append(f"кв. {apartment}")
    return ", ".join(part for part in ordered_parts if part)


def unified_json_to_contract_data(payload: Mapping[str, Any]) -> ContractData:
    raw_data = payload.get("data")
    if isinstance(raw_data, Mapping):
        data = raw_data
    else:
        data = payload

    passport_main = data.get("passport_main")
    passport_registration = data.get("passport_registration")
    egrn_extract = data.get("egrn_extract")
    if not isinstance(passport_main, Mapping) or not isinstance(passport_registration, Mapping) or not isinstance(
        egrn_extract, Mapping
    ):
        raise HTTPException(status_code=400, detail="Ожидается unified JSON с блоками passport_main, passport_registration, egrn_extract")

    surname = str(passport_main.get("surname", "") or "").strip()
    name = str(passport_main.get("name", "") or "").strip()
    patronymic = str(passport_main.get("patronymic", "") or "").strip()
    customer_fio_raw = " ".join(part for part in (surname, name, patronymic) if part)
    customer_fio = normalize_person_fio(customer_fio_raw)
    customer_fio_short = build_short_fio(customer_fio)

    override_address = str(payload.get("customer_registration_address_override", "") or "").strip()
    registration_address = override_address or _build_registration_address(passport_registration)
    ownership_basis = str(payload.get("ownership_basis_document_override", "") or "").strip()
    customer_email = str(payload.get("customer_email_override", "") or "").strip()
    customer_phone = str(payload.get("customer_phone_override", "") or "").strip()

    return ContractData(
        contract_city="",
        customer_fio=customer_fio,
        customer_fio_short=customer_fio_short,
        customer_registration_address=registration_address,
        customer_email=customer_email,
        customer_phone=customer_phone,
        passport_series=str(passport_main.get("passport_series", "") or "").strip(),
        passport_number=str(passport_main.get("passport_number", "") or "").strip(),
        passport_issued_by=str(passport_main.get("issuing_authority", "") or "").strip(),
        passport_issue_date=str(passport_main.get("issue_date", "") or "").strip(),
        passport_code=str(passport_main.get("department_code", "") or "").strip(),
        birth_place=str(passport_main.get("birth_place", "") or "").strip(),
        birth_date=str(passport_main.get("birth_date", "") or "").strip(),
        property_name=str(egrn_extract.get("object_type", "") or "").strip(),
        property_purpose=str(egrn_extract.get("ownership_type", "") or "").strip(),
        property_area=str(egrn_extract.get("area_sq_m", "") or "").strip(),
        property_address=str(egrn_extract.get("address", "") or "").strip(),
        property_cadastral_number=str(egrn_extract.get("cadastral_number", "") or "").strip(),
        ownership_basis_document=ownership_basis,
    )


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


@router.post("/unified-json-to-contract")
async def unified_json_to_contract(payload: dict[str, Any] = Body(...)):
    """
    Генерация договора на основе объединённого JSON (3 документа).
    """
    contract_data = await safe_to_thread(unified_json_to_contract_data, payload)
    output_name = f"dogovor_{uuid.uuid4().hex}.docx"
    output_path = OUTPUT_DIR / output_name
    await safe_to_thread(create_doc, contract_data, output_path)

    return {
        "message": "Договор создан по объединенному JSON",
        "generated_filename": output_name,
        "download_url": f"/download/{output_name}",
        "json_data": {
            "source": "unified_json",
            "contract_data": contract_data.model_dump(),
        },
    }
