import asyncio
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import ValidationError

from dogovor import ContractData, create_doc, OUTPUT_DIR
from scan_passport_hf import (
    extract_json_from_text,
    normalize_passport_data,
    run_hf_passport_extraction,
    settings as hf_settings,
    validate_image,
)

router = APIRouter(tags=["dogovor-new"])


def _passport_to_contract_data(passport_data) -> ContractData:
    fio_parts = [passport_data.surname, passport_data.name, passport_data.patronymic]
    fio = " ".join(part for part in fio_parts if part).strip()
    return ContractData(
        customer_fio=fio,
        passport_series=passport_data.passport_series,
        passport_number=passport_data.passport_number,
        passport_issued_by=passport_data.issuing_authority,
        passport_issue_date=passport_data.issue_date,
        passport_code=passport_data.department_code,
        birth_date=passport_data.birth_date,
        birth_place=passport_data.birth_place,
    )


@router.post("/scan-passport-to-contract-hf")
async def scan_passport_to_contract_hf(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите изображение паспорта")

    contents = await file.read()
    validate_image(contents)

    raw_text = await run_hf_passport_extraction(contents)
    try:
        parsed = extract_json_from_text(raw_text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Модель вернула невалидный JSON. Ответ модели: {raw_text[:1000]}",
        ) from e

    try:
        passport_data = normalize_passport_data(parsed)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Невалидные данные паспорта: {e.errors()}") from e
    contract_data = _passport_to_contract_data(passport_data)

    output_name = f"dogovor_{uuid.uuid4().hex}.docx"
    output_path = OUTPUT_DIR / output_name
    await asyncio.to_thread(create_doc, contract_data, output_path)

    return {
        "message": "Договор создан по данным паспорта (Hugging Face)",
        "filename": file.filename,
        "generated_filename": output_name,
        "download_url": f"/download/{output_name}",
        "json_data": {
            "source": "huggingface_passport",
            "model": hf_settings.hf_model,
            "passport_data": passport_data.model_dump(),
            "contract_data": contract_data.model_dump(),
            "raw_text": raw_text,
        },
    }
