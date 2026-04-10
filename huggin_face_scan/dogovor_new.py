from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, ParamSpec, TypeVar

from docxtpl import DocxTemplate
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from tesseract_scan.ocr import extract_text_from_upload
from huggin_face_scan.scan_passport_hf import (
    extract_json_from_text,
    pdf_first_page_to_png,
    run_hf_passport_extraction,
    settings as hf_settings,
    validate_image,
)

router = APIRouter(tags=["dogovor-new"])

P = ParamSpec("P")
T = TypeVar("T")


def _run_sync_guarded(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    try:
        return func(*args, **kwargs)
    except StopIteration as e:
        raise RuntimeError("Background sync function raised StopIteration") from e


async def safe_to_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return await asyncio.to_thread(_run_sync_guarded, func, *args, **kwargs)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_CANDIDATES = (
    BASE_DIR / "шаблон договора (1) (1) новый.docx",
)
OUTPUT_DIR = BASE_DIR / "generated_docs"
OUTPUT_DIR.mkdir(exist_ok=True)
_DOWNLOAD_NAME = re.compile(r"^dogovor_[0-9a-f]{32}\.docx$")


class ContractData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    contract_city: str = ""
    contract_number: str = ""
    contract_date: str = ""
    executor_name: str = ""
    executor_address: str = ""
    executor_email: str = ""
    executor_phone: str = ""
    executor_inn: str = ""
    executor_bik: str = ""
    executor_ogrn: str = ""
    executor_bank: str = ""
    executor_rs: str = ""
    executor_ks: str = ""
    customer_fio: str = ""
    customer_fio_short: str = ""
    customer_registration_address: str = ""
    customer_email: str = ""
    customer_phone: str = ""
    passport_series: str = ""
    passport_number: str = ""
    passport_issued_by: str = ""
    passport_issue_date: str = ""
    passport_code: str = ""
    birth_place: str = ""
    birth_date: str = ""
    property_name: str = ""
    property_purpose: str = ""
    property_area: str = ""
    property_address: str = ""
    property_cadastral_number: str = ""
    ownership_basis_document: str = ""


class ContractPayload(BaseModel):
    source: str = Field(default="tesseract")
    ocr_text: str
    contract_data: ContractData


def _find(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else default


def normalize_person_fio(raw_fio: str) -> str:
    value = str(raw_fio or "").strip()
    if not value:
        return ""
    parts = [p for p in re.split(r"\s+", value) if p]
    normalized: list[str] = []
    for part in parts:
        low = part.lower()
        normalized.append(low[:1].upper() + low[1:])
    return " ".join(normalized)


def build_short_fio(full_fio: str) -> str:
    parts = [p for p in re.split(r"\s+", str(full_fio or "").strip()) if p]
    if not parts:
        return ""
    surname = parts[0]
    initials = "".join(f"{p[0].upper()}." for p in parts[1:] if p)
    return f"{surname} {initials}".strip()


def _section_text(text: str, header: str) -> str:
    headers = ("исполнитель", "заказчик", "паспорт", "объект")
    start_match = re.search(rf"(?im)^\s*{header}(?:\s+\w+)?\s*:?\s*$", text)
    if not start_match:
        return ""

    start = start_match.end()
    end = len(text)
    for candidate in headers:
        if candidate == header:
            continue
        m = re.search(rf"(?im)^\s*{candidate}(?:\s+\w+)?\s*:?\s*$", text[start:])
        if m:
            end = min(end, start + m.start())
    return text[start:end]


def extract_fields(text: str) -> dict[str, str]:
    fields = {
        "contract_city": _find(
            r"г\.\s*([А-ЯЁA-Zа-яёa-z\-]+(?:[ \t]+[А-ЯЁA-Zа-яёa-z\-]+)*)", text
        ),
        "contract_number": _find(r"договор\s*№\s*([A-Za-zА-Яа-я0-9\-\/]+)", text),
        "contract_date": _find(r"дата договора[:\-]?\s*(\d{2}\.\d{2}\.\d{4})", text),
        "executor_name": _find(r"фио исполнителя[:\-]?\s*(.+)", text),
        "executor_address": _find(r"адрес\s+регистрации\s+исполнителя[:\-]?\s*(.+)", text),
        "executor_phone": _find(r"телефон исполнителя[:\-]?\s*([\+\d\(\)\-\s]+)", text),
        "executor_email": _find(r"email исполнителя[:\-]?\s*([^\s]+)", text),
        "executor_inn": _find(r"инн исполнителя[:\-]?\s*(\d+)", text),
        "executor_ogrn": _find(r"огрн исполнителя[:\-]?\s*(\d+)", text),
        "executor_bik": _find(r"бик\s+исполнителя[:\-]?\s*(\d{8,9})", text),
        "executor_bank": _find(r"(?:банк|наименование\s+банка)\s+исполнителя[:\-]?\s*(.+)", text),
        "executor_rs": _find(
            r"(?:р[/\\]?с|расч[её]тный\s+сч[её]т)\s+исполнителя[:\-]?\s*([\d\s]+)",
            text,
        ),
        "executor_ks": _find(
            r"(?:корреспондентск\w*\s+сч[её]т|к[/\\]?с|корр\.?\s*сч[её]т)\s+исполнителя[:\-]?\s*([\d\s]+)",
            text,
        ),
        "customer_fio": _find(r"фио заказчика[:\-]?\s*(.+)", text),
        "customer_registration_address": _find(
            r"адрес\s+регистрации\s+заказчика[:\-]?\s*(.+)", text
        ),
        "customer_phone": _find(r"телефон заказчика[:\-]?\s*([\+\d\(\)\-\s]+)", text),
        "customer_email": _find(r"email заказчика[:\-]?\s*([^\s]+)", text),
        "birth_place": _find(r"место\s+рождения(?:\s+заказчика)?[:\-]?\s*(.+)", text),
        "birth_date": _find(r"дата\s+рождения(?:\s+заказчика)?[:\-]?\s*(\d{2}\.\d{2}\.\d{4})", text),
        "property_name": _find(r"название объекта[:\-]?\s*(.+)", text),
        "property_purpose": _find(r"назначение объекта[:\-]?\s*(.+)", text),
        "property_area": _find(r"площадь объекта[:\-]?\s*([\d\.]+)", text),
        "property_address": _find(r"адрес объекта[:\-]?\s*(.+)", text),
        "property_cadastral_number": _find(r"кадастровый номер[:\-]?\s*([0-9:\-]+)", text),
        "ownership_basis_document": _find(r"основание собственности[:\-]?\s*(.+)", text),
    }
    return fields


def _extract_passport_data_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    inner = payload.get("data")
    if isinstance(inner, Mapping):
        return dict(inner)
    if any(k in payload for k in ("issuing_authority", "passport_series", "surname", "passport_number")):
        return dict(payload)
    return {}


def passport_scan_to_contract_data(passport_payload: Mapping[str, Any]) -> ContractData:
    data = _extract_passport_data_dict(passport_payload)

    surname = str(data.get("surname", "") or "").strip()
    name = str(data.get("name", "") or "").strip()
    patronymic = str(data.get("patronymic", "") or "").strip()
    customer_fio_raw = " ".join(p for p in (surname, name, patronymic) if p)
    customer_fio = normalize_person_fio(customer_fio_raw)
    customer_fio_short = build_short_fio(customer_fio)

    return ContractData(
        customer_fio=customer_fio,
        customer_fio_short=customer_fio_short,
        passport_series=str(data.get("passport_series", "") or ""),
        passport_number=str(data.get("passport_number", "") or ""),
        passport_issued_by=str(data.get("issuing_authority", "") or ""),
        passport_issue_date=str(data.get("issue_date", "") or ""),
        passport_code=str(data.get("department_code", "") or ""),
        birth_place=str(data.get("birth_place", "") or ""),
        birth_date=str(data.get("birth_date", "") or ""),
    )


def parse_ocr_to_contract_data(text: str) -> ContractData:
    return ContractData.model_validate(extract_fields(text))


def build_payload_from_ocr_text(ocr_text: str) -> ContractPayload:
    return ContractPayload(
        source="tesseract",
        ocr_text=ocr_text,
        contract_data=parse_ocr_to_contract_data(ocr_text),
    )


def create_doc(contract_data: ContractData, output_path: Path) -> None:
    template_path = next((p for p in TEMPLATE_CANDIDATES if p.exists()), None)
    if template_path is None:
        raise HTTPException(status_code=500, detail="Файл шаблона договора не найден")

    doc = DocxTemplate(str(template_path))
    context = contract_data.model_dump()
    # Гарантируем сокращенное ФИО для блока подписи (например: Иванов И.И.).
    customer_fio = normalize_person_fio(str(context.get("customer_fio", "") or ""))
    customer_fio_short = str(context.get("customer_fio_short", "") or "").strip()
    if not customer_fio_short and customer_fio:
        customer_fio_short = build_short_fio(customer_fio)
    context["customer_fio_short"] = customer_fio_short
    # Совместимость с новым шаблоном: customer_fio_abr ожидает сокращенное ФИО.
    context["customer_fio_abr"] = customer_fio_short
    doc.render(context)
    doc.save(str(output_path))


def _resolve_download_path(filename: str) -> Path:
    safe_name = Path(filename).name
    if not _DOWNLOAD_NAME.fullmatch(safe_name):
        raise HTTPException(status_code=404, detail="Файл не найден")
    return OUTPUT_DIR / safe_name


@router.post("/ocr-to-contract")
async def ocr_to_contract(file: UploadFile = File(...)):
    text = await extract_text_from_upload(file)
    payload = await safe_to_thread(build_payload_from_ocr_text, text)

    output_name = f"dogovor_{uuid.uuid4().hex}.docx"
    output_path = OUTPUT_DIR / output_name
    await safe_to_thread(create_doc, payload.contract_data, output_path)

    return {
        "message": "Договор успешно заполнен",
        "filename": file.filename,
        "generated_filename": output_name,
        "json_data": payload.model_dump(),
        "download_url": f"/download/{output_name}",
    }


@router.get("/download/{filename}")
async def download_file(filename: str):
    file_path = _resolve_download_path(filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


def passport_json_to_contract_data(passport_json: Mapping[str, Any]) -> ContractData:
    return passport_scan_to_contract_data(passport_json)


def create_contract_docx_from_passport_json(
    passport_json: Mapping[str, Any],
    output_path: Path | None = None,
) -> Path:
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
    file_type = (file.content_type or "").lower()
    if not (file_type.startswith("image/") or file_type == "application/pdf"):
        raise HTTPException(status_code=400, detail="Загрузите изображение или PDF паспорта")

    contents = await file.read()
    if file_type == "application/pdf":
        contents = await safe_to_thread(pdf_first_page_to_png, contents)
    else:
        await safe_to_thread(validate_image, contents)

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
