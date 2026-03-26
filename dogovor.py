import asyncio
import re
import uuid
from pathlib import Path

from docxtpl import DocxTemplate
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ocr import extract_text_from_upload

router = APIRouter(tags=["dogovor"])

BASE_DIR = Path(__file__).resolve().parent
# Шаблон эксклюзивного договора (docxtpl): «шаблон договора.docx» в корне проекта
TEMPLATE_PATH = BASE_DIR / "шаблон договора.docx"
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
        # город договора (без захвата следующей строки «ДОГОВОР …»)
        "contract_city": _find(
            r"г\.\s*([А-ЯЁA-Zа-яёa-z\-]+(?:[ \t]+[А-ЯЁA-Zа-яёa-z\-]+)*)",
            text,
        ),

        # договор
        "contract_number": _find(r"договор\s*№\s*([A-Za-zА-Яа-я0-9\-\/]+)", text),
        "contract_date": _find(r"дата договора[:\-]?\s*(\d{2}\.\d{2}\.\d{4})", text),

        # исполнитель (часто в форме: «Адрес регистрации:», «р/с», «к/с» без слова «исполнителя»)
        "executor_name": _find(r"фио исполнителя[:\-]?\s*(.+)", text),
        "executor_address": _find(
            r"адрес\s+регистрации\s+исполнителя[:\-]?\s*(.+)", text
        )
        or _find(r"адрес\s+регистрации[:\-]?\s*(.+)", exec_text)
        or _find(r"адрес\s+исполнителя[:\-]?\s*(.+)", text),
        "executor_phone": _find(r"телефон исполнителя[:\-]?\s*([\+\d\(\)\-\s]+)", text)
        or _find(r"(?:тел\.?|телефон)[:\-]?\s*([\+\d\(\)\-\s]+)", exec_text),
        "executor_email": _find(r"email исполнителя[:\-]?\s*([^\s]+)", text)
        or _find(r"(?:email|e-?mail|почта)[:\-]?\s*([^\s]+)", exec_text),
        "executor_inn": _find(r"инн исполнителя[:\-]?\s*(\d+)", text)
        or _find(r"инн[:\-]?\s*(\d{10,12})", exec_text),
        "executor_ogrn": _find(r"огрн исполнителя[:\-]?\s*(\d+)", text)
        or _find(r"огрн[:\-]?\s*(\d+)", exec_text),
        "executor_bik": _find(r"бик\s+исполнителя[:\-]?\s*(\d{8,9})", text)
        or _find(r"бик[:\-]?\s*(\d{8,9})", exec_text),
        "executor_bank": _find(
            r"(?:банк|наименование\s+банка)\s+исполнителя[:\-]?\s*(.+)", text
        )
        or _find(r"банк[:\-]?\s*(.+)", exec_text),
        "executor_rs": _find(
            r"(?:р[/\\]?с|расч[её]тный\s+сч[её]т)\s+исполнителя[:\-]?\s*([\d\s]+)",
            text,
        )
        or _find(r"(?:р[/\\]?с|р\s*\.\s*с)[:\-]?\s*([\d\s]+)", exec_text),
        "executor_ks": _find(
            r"(?:корреспондентск\w*\s+сч[её]т|к[/\\]?с|корр\.?\s*сч[её]т)\s+исполнителя[:\-]?\s*([\d\s]+)",
            text,
        )
        or _find(
            r"корреспондентск\w*\s+сч[её]т[:\-]?\s*([\d\s]+)",
            exec_text,
        )
        or _find(r"(?:к[/\\]?с|к\s*\.\s*с)[:\-]?\s*([\d\s]+)", exec_text),

        # заказчик
        "customer_fio": _find(r"фио заказчика[:\-]?\s*(.+)", text),
        "customer_registration_address": _find(
            r"адрес\s+регистрации\s+заказчика[:\-]?\s*(.+)", text
        )
        or _find(
            r"адрес\s+регистрац(?:ии)?\s+заказчика[:\-]?\s*(.+)", text
        )
        or _find(r"адрес\s+регистрац(?:ии)?[:\-]?\s*(.+)", cust_text),
        "customer_phone": _find(r"телефон заказчика[:\-]?\s*([\+\d\(\)\-\s]+)", text),
        "customer_email": _find(r"email заказчика[:\-]?\s*([^\s]+)", text),

        # паспорт
        "passport_series": _find(r"серия[:\-]?\s*(\d{4})", text),
        "passport_number": _find(r"номер[:\-]?\s*(\d{6})", text),
        "passport_issued_by": _find(r"выдан[:\-]?\s*(.+)", text),
        "passport_issue_date": _find(r"дата выдачи[:\-]?\s*(\d{2}\.\d{2}\.\d{4})", text),
        "passport_code": _find(r"код подразделения[:\-]?\s*([\d\-]+)", text),
        "birth_place": _find(
            r"место\s+рождения(?:\s+заказчика)?[:\-]?\s*(.+)", text
        ),
        "birth_date": _find(
            r"дата\s+рождения(?:\s+заказчика)?[:\-]?\s*(\d{2}\.\d{2}\.\d{4})", text
        ),

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


def parse_ocr_to_contract_data(text: str) -> ContractData:
    """Собирает ContractData из результата extract_fields (единый набор regex)."""
    return ContractData.model_validate(extract_fields(text))


def build_payload_from_ocr_text(ocr_text: str) -> ContractPayload:
    return ContractPayload(
        source="tesseract",
        ocr_text=ocr_text,
        contract_data=parse_ocr_to_contract_data(ocr_text),
    )


def create_doc(contract_data: ContractData, output_path: Path) -> None:
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=500, detail="Файл шаблона договора не найден")

    doc = DocxTemplate(str(TEMPLATE_PATH))
    doc.render(contract_data.model_dump())
    doc.save(str(output_path))


def _resolve_download_path(filename: str) -> Path:
    safe_name = Path(filename).name
    if not _DOWNLOAD_NAME.fullmatch(safe_name):
        raise HTTPException(status_code=404, detail="Файл не найден")
    return OUTPUT_DIR / safe_name


@router.post("/ocr-to-contract")
async def ocr_to_contract(file: UploadFile = File(...)):
    text = await extract_text_from_upload(file)
    payload = await asyncio.to_thread(build_payload_from_ocr_text, text)

    output_name = f"dogovor_{uuid.uuid4().hex}.docx"
    output_path = OUTPUT_DIR / output_name
    await asyncio.to_thread(create_doc, payload.contract_data, output_path)

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