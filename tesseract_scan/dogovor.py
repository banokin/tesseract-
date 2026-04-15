import asyncio
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, ParamSpec, TypeVar

from docxtpl import DocxTemplate
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from tesseract_scan.ocr import extract_text_from_upload

router = APIRouter(tags=["dogovor"])

P = ParamSpec("P")
T = TypeVar("T")


def _run_sync_guarded(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    try:
        return func(*args, **kwargs)
    except StopIteration as e:
        raise RuntimeError("Background sync function raised StopIteration") from e


async def safe_to_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return await asyncio.to_thread(_run_sync_guarded, func, *args, **kwargs)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_CANDIDATES = (
    BASE_DIR / "шаблон договора (1) (1).docx",
    BASE_DIR / "шаблон договора.docx",
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


def _uppercase_for_contract(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().upper()


def _normalize_inline_address(value: str) -> str:
    compact = re.sub(r"\s*,\s*", ", ", re.sub(r"\s+", " ", str(value or "")).strip())
    if not compact:
        return ""

    def _is_city_like(part: str) -> bool:
        return bool(re.match(r"^(г\.|город|пгт|пос\.|с\.|дер\.)\s+", part, re.I))

    def _is_street_like(part: str) -> bool:
        return bool(re.match(r"^(ул\.|улица|просп\.|проспект|пер\.|переулок|бул\.|бульвар|ш\.|шоссе)\s+", part, re.I))

    def _is_building_like(part: str) -> bool:
        return bool(re.match(r"^(д\.|дом|кв\.|корп\.|стр\.)\s*", part, re.I))

    parts = [p.strip() for p in compact.split(",") if p.strip()]
    normalized_parts: list[str] = []
    for idx, part in enumerate(parts):
        p = part
        p = re.sub(r"\bобласть\b", "обл.", p, flags=re.I)
        p = re.sub(r"^\s*город\s+", "г. ", p, flags=re.I)
        p = re.sub(r"^\s*улица\s+", "ул. ", p, flags=re.I)
        if idx == 1 and not _is_city_like(p) and not _is_street_like(p) and not _is_building_like(p):
            p = f"г. {p}"
        normalized_parts.append(p)
    return ", ".join(normalized_parts)


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


def extract_fields(text: str) -> dict:
    exec_text = _section_text(text, "исполнитель")
    cust_text = _section_text(text, "заказчик")

    fields = {
        # город договора 
        "contract_city": _find( r"г\.\s*([А-ЯЁA-Zа-яёa-z\-]+(?:[ \t]+[А-ЯЁA-Zа-яёa-z\-]+)*)", text),

        # договор
        "contract_number": _find(r"договор\s*№\s*([A-Za-zА-Яа-я0-9\-\/]+)", text),
        
        "contract_date": _find(r"дата договора[:\-]?\s*(\d{2}\.\d{2}\.\d{4})", text),

        # исполнитель (часто в форме: «Адрес регистрации:», «р/с», «к/с» без слова «исполнителя»)
        "executor_name": _find(r"фио исполнителя[:\-]?\s*(.+)", text),
        
        "executor_address": _find(r"адрес\s+регистрации\s+исполнителя[:\-]?\s*(.+)", text),
        
        "executor_phone": _find(r"телефон исполнителя[:\-]?\s*([\+\d\(\)\-\s]+)", text),

        
        "executor_email": _find(r"email исполнителя[:\-]?\s*([^\s]+)", text),
        
        "executor_inn": _find(r"инн исполнителя[:\-]?\s*(\d+)", text),

        
        "executor_ogrn": _find(r"огрн исполнителя[:\-]?\s*(\d+)", text),
   
        
        "executor_bik": _find(r"бик\s+исполнителя[:\-]?\s*(\d{8,9})", text),
      
        
        "executor_bank": _find( r"(?:банк|наименование\s+банка)\s+исполнителя[:\-]?\s*(.+)", text),

        
        "executor_rs": _find( r"(?:р[/\\]?с|расч[её]тный\s+сч[её]т)\s+исполнителя[:\-]?\s*([\d\s]+)", text),

        "executor_ks": _find( r"(?:корреспондентск\w*\s+сч[её]т|к[/\\]?с|корр\.?\s*сч[её]т)\s+исполнителя[:\-]?\s*([\d\s]+)", text),

        # заказчик
        "customer_fio": _find(r"фио заказчика[:\-]?\s*(.+)", text),
        
        "customer_registration_address": _find( r"адрес\s+регистрации\s+заказчика[:\-]?\s*(.+)", text),

        "customer_phone": _find(r"телефон заказчика[:\-]?\s*([\+\d\(\)\-\s]+)", text),
       
        "customer_email": _find(r"email заказчика[:\-]?\s*([^\s]+)", text),
        
        "birth_place": _find( r"место\s+рождения(?:\s+заказчика)?[:\-]?\s*(.+)", text),
        
        "birth_date": _find( r"дата\s+рождения(?:\s+заказчика)?[:\-]?\s*(\d{2}\.\d{2}\.\d{4})", text),

        # объект
        "property_name": _find(r"название объекта[:\-]?\s*(.+)", text),
        
        "property_purpose": _find(r"назначение объекта[:\-]?\s*(.+)", text),
        
        "property_area": _find(r"площадь объекта[:\-]?\s*([\d\.]+)", text),
        
        "property_address": _find(r"адрес объекта[:\-]?\s*(.+)", text),
        
        "property_cadastral_number": _find(r"кадастровый номер[:\-]?\s*([0-9:\-]+)", text),

        # основание
        "ownership_basis_document": _find(r"основание собственности[:\-]?\s*(.+)", text),
    }

    return fields


def _extract_passport_data_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Достаёт объект полей паспорта из полного ответа API или из плоского dict."""
    if not isinstance(payload, Mapping):
        return {}
    inner = payload.get("data")
    if isinstance(inner, Mapping):
        return dict(inner)
    if any(
        k in payload
        for k in ("issuing_authority", "passport_series", "surname", "passport_number")
    ):
        return dict(payload)
    return {}


def passport_scan_to_contract_data(passport_payload: Mapping[str, Any]) -> ContractData:
    """
    Заполняет ContractData данными из JSON сканирования паспорта (HF и т.п.).

    Ожидается либо полный ответ API: ``{"ok": true, "data": {...}}``,
    либо только объект ``data`` с полями как у PassportData.

    Поля договора/исполнителя/объекта, которых нет в паспорте, остаются пустыми
    (как незаполненные строки в ``extract_fields``).
    """
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
        passport_issued_by=_uppercase_for_contract(str(data.get("issuing_authority", "") or "")),
        passport_issue_date=str(data.get("issue_date", "") or ""),
        passport_code=str(data.get("department_code", "") or ""),
        birth_place=_uppercase_for_contract(str(data.get("birth_place", "") or "")),
        birth_date=str(data.get("birth_date", "") or ""),
    )


def passport_scan_to_contract_fields_dict(passport_payload: Mapping[str, Any]) -> dict[str, str]:
    """
    Тот же набор ключей, что возвращает ``extract_fields`` (см. dogovor.py ~90–151),
    но значения берутся из скана паспорта; остальное — пустые строки.
    Удобно для ``st.json`` или слияния с полями из OCR текста договора.
    """
    return passport_scan_to_contract_data(passport_payload).model_dump()


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
    context["passport_issued_by"] = _uppercase_for_contract(str(context.get("passport_issued_by", "") or ""))
    context["birth_place"] = _uppercase_for_contract(str(context.get("birth_place", "") or ""))
    context["customer_registration_address"] = _normalize_inline_address(
        str(context.get("customer_registration_address", "") or "")
    )
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