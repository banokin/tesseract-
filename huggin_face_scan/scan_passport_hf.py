from __future__ import annotations

import asyncio
import base64
import ast
import imghdr
import json
import logging
import re
from io import BytesIO
from typing import Any, Callable, Dict, List, ParamSpec, TypeVar

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from huggingface_hub import InferenceClient
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tesseract_scan.ocr import extract_text_from_upload, preprocess_image_for_ocr

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_token: str = Field(validation_alias="HF_TOKEN")
    # Для скана документов нужна мультимодальная модель (image+text). Чисто текстовые модели дают пустой content на image_url.
    hf_model: str = Field(
        default="meta-llama/Llama-4-Scout-17B-16E-Instruct:novita",
        validation_alias="HF_MODEL",
    )
    # Используется при необходимости; InferenceClient ходит в HF Inference API.
    hf_base_url: str = Field(default="https://router.huggingface.co/v1", validation_alias="HF_BASE_URL")
    max_file_size_mb: int = Field(default=10, validation_alias="MAX_FILE_SIZE_MB")
    hf_request_timeout_sec: int = Field(default=90, validation_alias="HF_REQUEST_TIMEOUT_SEC")

    @field_validator("hf_model", mode="before")
    @classmethod
    def normalize_hf_model_id(cls, v: object) -> object:
        """Убираем пробелы и случайную точку в конце строки из .env (иначе HFValidationError на repo id)."""
        if not isinstance(v, str):
            return v
        s = v.strip()
        while s.endswith("."):
            s = s[:-1]
        return s


settings = Settings()


def _parse_router_model_id(spec: str) -> tuple[str, str | None]:
    """Разделяет Hub repo id и суффикс провайдера Router (например :together, :novita).

    InferenceClient ожидает в `model` только валидный repo id; провайдер задаётся отдельно.
    """
    spec = spec.strip()
    if not spec or ":" not in spec:
        return spec, None
    left, right = spec.rsplit(":", 1)
    if not left or not right or "/" not in left or "/" in right:
        return spec, None
    return left, right


def _inference_client_for_provider(provider: str | None) -> InferenceClient:
    kwargs: dict[str, Any] = {
        "token": settings.hf_token,
        "timeout": float(settings.hf_request_timeout_sec),
    }
    if provider:
        kwargs["provider"] = provider
    return InferenceClient(**kwargs)


logger = logging.getLogger(__name__)

router = APIRouter(tags=["passport"])

P = ParamSpec("P")
T = TypeVar("T")


def _run_sync_guarded(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    try:
        return func(*args, **kwargs)
    except StopIteration as e:
        raise RuntimeError("Background sync function raised StopIteration") from e


async def safe_to_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return await asyncio.to_thread(_run_sync_guarded, func, *args, **kwargs)


class PassportData(BaseModel):
    issuing_authority: str = ""
    issue_date: str = ""
    department_code: str = ""
    passport_series: str = ""
    passport_number: str = ""
    surname: str = ""
    name: str = ""
    patronymic: str = ""
    gender: str = ""
    birth_date: str = ""
    birth_place: str = ""
    confidence_note: str = ""


class PassportScanResponse(BaseModel):
    ok: bool
    model: str
    data: PassportData
    raw_text: str


class PassportScanTesseractResponse(BaseModel):
    ok: bool
    engine: str
    text: str


class PassportRegistrationData(BaseModel):
    address: str = ""
    region: str = ""
    city: str = ""
    settlement: str = ""
    street: str = ""
    house: str = ""
    building: str = ""
    apartment: str = ""
    registration_date: str = ""
    confidence_note: str = ""


class EgrnExtractData(BaseModel):
    cadastral_number: str = ""
    object_type: str = ""
    address: str = ""
    area_sq_m: str = ""
    ownership_type: str = ""
    right_holders: List[str] = Field(default_factory=list)
    extract_date: str = ""
    confidence_note: str = ""


class UnifiedDocumentsData(BaseModel):
    passport_main: PassportData
    passport_registration: PassportRegistrationData
    egrn_extract: EgrnExtractData


class UnifiedDocumentsScanResponse(BaseModel):
    ok: bool
    model: str
    data: UnifiedDocumentsData
    raw_text: Dict[str, str]


def validate_file_size(contents: bytes) -> None:
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой. Максимум: {settings.max_file_size_mb} MB",
        )


def validate_image(contents: bytes) -> None:
    validate_file_size(contents)
    try:
        Image.open(BytesIO(contents)).verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Файл не является валидным изображением",
        )


def detect_mime(contents: bytes) -> str:
    kind = imghdr.what(None, h=contents)
    if kind == "jpeg":
        return "image/jpeg"
    if kind == "png":
        return "image/png"
    if kind == "webp":
        return "image/webp"
    raise HTTPException(
        status_code=400,
        detail="Поддерживаются только JPEG, PNG, WEBP",
    )


def image_to_data_url(contents: bytes, mime: str) -> str:
    encoded = base64.b64encode(contents).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def upscale_jpeg_for_ocr(image_bytes: bytes, scale: float = 2.5) -> bytes:
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Exception:
        return image_bytes

    image_format = (image.format or "").upper()
    if image_format not in {"JPEG", "JPG"}:
        return image_bytes

    width, height = image.size
    new_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    resized.save(buffer, format="JPEG", quality=95, optimize=True)
    return buffer.getvalue()


def pdf_first_page_to_png(pdf_bytes: bytes) -> bytes:
    validate_file_size(pdf_bytes)
    if fitz is None:
        raise HTTPException(
            status_code=500,
            detail="Поддержка PDF не установлена на сервере (нужен PyMuPDF).",
        )
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count < 1:
            raise HTTPException(status_code=400, detail="PDF-файл не содержит страниц")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        png_bytes = pix.tobytes("png")
        validate_image(png_bytes)
        return png_bytes
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось обработать PDF: {e!r}") from e


def _first_balanced_json_object(text: str) -> str | None:
    """
    Находит первый объект `{...}` с корректной балансировкой скобок,
    учитывая строки в двойных кавычках (экранирование \\").
    Не обрабатывает одинарные кавычки внутри JSON — для плоского паспорта достаточно.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_passport_keys_loose(text: str) -> Dict[str, Any]:
    """Последний шанс: значения строковых полей до запятой или } (без вложенных кавычек в значении)."""
    keys = [
        "issuing_authority",
        "issue_date",
        "department_code",
        "passport_series",
        "passport_number",
        "surname",
        "name",
        "patronymic",
        "gender",
        "birth_date",
        "birth_place",
        "confidence_note",
    ]
    extracted: Dict[str, Any] = {}
    for key in keys:
        m = re.search(
            rf'["\']{re.escape(key)}["\']\s*:\s*'
            r'(?:"([^"]*)"|\'([^\']*)\'|(\d+))',
            text,
            flags=re.DOTALL,
        )
        if m:
            val = (m.group(1) or m.group(2) or m.group(3) or "").strip()
            extracted[key] = val
    return extracted


def _passport_loose_usable(d: Dict[str, Any]) -> bool:
    if len(d) < 2:
        return False
    if len(d) >= 4:
        return True
    return any(
        d.get(k)
        for k in ("surname", "name", "passport_number", "issuing_authority", "passport_series")
    )


def _normalize_jsonish_text(text: str) -> str:
    """
    Нормализует "почти JSON" ответы модели:
    - снимает внешние обертки вида (...) или "...";
    - разворачивает экранирование \\n и \\" для строковых payload-ов;
    - при необходимости добавляет фигурные скобки вокруг key:value списка.
    """
    s = text.strip()
    if not s:
        return s

    # Иногда upstream возвращает repr-кортежа: ('{...}', 'model-id')
    if s.startswith("(") and s.endswith(")"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, tuple) and parsed:
                first = parsed[0]
                if isinstance(first, str) and first.strip():
                    s = first.strip()
        except Exception:
            pass
        # Fallback для строк вида ("{\\n ... }", "model"), которые не парсятся literal_eval.
        if s.startswith("(") and s.endswith(")"):
            m = re.search(r'^\(\s*"((?:\\.|[^"\\])*)"\s*,', s, flags=re.DOTALL)
            if m:
                inner = m.group(1)
                s = inner.replace('\\"', '"').replace("\\n", "\n").strip()

    # Иногда модель возвращает JSON в скобках: ( {...} ) или ("k":"v", ...)
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        s = s[1:-1].strip()

    # Если весь payload пришел как одна экранированная строка, снимем внешние кавычки
    # и разэкранируем минимально необходимое для JSON-парсинга.
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
        s = s.replace('\\"', '"').replace("\\n", "\n")

    # Если пришел список полей без {}, завернем в объект.
    if "{" not in s and "}" not in s and re.search(r'["\']\w+["\']\s*:', s):
        s = "{%s}" % s

    return s.strip()


def _repair_passport_jsonish(text: str) -> Dict[str, Any]:
    """
    Восстанавливает паспортные поля из "битого JSON"-текста:
    - ищет ключи из фиксированного белого списка;
    - корректно режет значение до следующего ключа;
    - убирает кавычки/мусорные разделители и хвостовые запятые.
    """
    keys = [
        "issuing_authority",
        "issue_date",
        "department_code",
        "passport_series",
        "passport_number",
        "surname",
        "name",
        "patronymic",
        "gender",
        "birth_date",
        "birth_place",
        "confidence_note",
    ]
    key_group = "|".join(re.escape(k) for k in keys)
    # Ищем "key": или 'key':
    key_re = re.compile(rf'["\']({key_group})["\']\s*:\s*', flags=re.DOTALL)
    matches = list(key_re.finditer(text))
    if not matches:
        return {}

    result: Dict[str, Any] = {}
    for idx, m in enumerate(matches):
        key = m.group(1)
        value_start = m.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        raw_val = text[value_start:value_end].strip()

        # Срезаем типичные хвосты между полями
        raw_val = raw_val.lstrip(",")
        raw_val = raw_val.rstrip(",")
        raw_val = raw_val.strip()

        # Убираем обёртки, если значение пришло в кавычках
        if len(raw_val) >= 2 and (
            (raw_val[0] == '"' and raw_val[-1] == '"')
            or (raw_val[0] == "'" and raw_val[-1] == "'")
        ):
            raw_val = raw_val[1:-1]

        # Нормализуем экранирование
        raw_val = raw_val.replace('\\"', '"').replace("\\n", "\n").strip()

        # Если в конце случайно остался символ закрытия объекта/скобки
        raw_val = re.sub(r"[}\])]+$", "", raw_val).strip()
        result[key] = raw_val

    return result


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = _normalize_jsonish_text(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        rest = text[fence.end() :]
        closing = rest.find("```")
        block = rest[:closing] if closing >= 0 else rest
        fb = _first_balanced_json_object(block)
        if fb:
            try:
                return json.loads(fb.strip())
            except Exception:
                text = fb.strip()
        else:
            text = block.strip()

    balanced = _first_balanced_json_object(text)
    if balanced:
        candidate = balanced.strip()
        candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        loose = _extract_passport_keys_loose(candidate)
        if _passport_loose_usable(loose):
            return loose

    loose = _extract_passport_keys_loose(text)
    if not _passport_loose_usable(loose):
        # Второй проход для ответов с экранированными кавычками/переносами.
        unescaped = text.replace('\\"', '"').replace("\\n", "\n")
        loose = _extract_passport_keys_loose(unescaped)
    if not _passport_loose_usable(loose):
        repaired = _repair_passport_jsonish(text)
        if _passport_loose_usable(repaired):
            return repaired
    if _passport_loose_usable(loose):
        return loose

    raise ValueError("Не удалось извлечь JSON из ответа модели")


def _looks_like_russian_patronymic(s: str) -> bool:
    """Грубая проверка: отчество обычно оканчивается на -ович/-евич/-ич/-овна/-евна и т.д."""
    t = re.sub(r"\s+", "", str(s or "")).upper()
    if len(t) < 4:
        return False
    return bool(
        re.search(
            r"(ОВИЧ|ЕВИЧ|ЬЕВИЧ|ОВНА|ЕВНА|ИНИЧНА|ИЧНА|ИЧ)$",
            t,
        )
    )


def _maybe_swap_name_and_patronymic(name: str, patronymic: str) -> tuple[str, str]:
    """Модели часто путают имя и отчество: исправляем, если одно явно похоже на отчество, другое — нет."""
    n, p = str(name or "").strip(), str(patronymic or "").strip()
    if not n or not p:
        return n, p
    n_pat, p_pat = _looks_like_russian_patronymic(n), _looks_like_russian_patronymic(p)
    if n_pat and not p_pat:
        return p, n
    return n, p


def _normalize_passport_series_number(series_raw: str, number_raw: str) -> tuple[str, str]:
    """Паспорт РФ: ровно 10 цифр — первые 4 серия, последние 6 номер (часто печатают как 4518 478497)."""

    def _d(value: str) -> str:
        return re.sub(r"\D+", "", str(value or ""))

    s, n = _d(series_raw), _d(number_raw)

    if len(s) == 10 and not n:
        return s[:4], s[4:10]
    if len(n) == 10 and not s:
        return n[:4], n[4:10]

    combined = s + n
    if len(combined) == 10:
        return combined[:4], combined[4:10]
    if len(combined) > 10:
        combined = combined[:10]
        return combined[:4], combined[4:10]

    if len(s) == 4 and len(n) == 6:
        return s, n

    return s[:4], n[:6]


def normalize_passport_data(payload: Dict[str, Any]) -> PassportData:
    def _normalize_date(value: str) -> str:
        s = str(value or "").strip()
        if not s:
            return ""
        m = re.search(r"\b(\d{2})[.\-/](\d{2})[.\-/](\d{4})\b", s)
        if m:
            return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
        m = re.search(r"\b(\d{4})[.\-/](\d{2})[.\-/](\d{2})\b", s)
        if m:
            return f"{m.group(3)}.{m.group(2)}.{m.group(1)}"
        return s

    def _digits_only(value: str) -> str:
        return re.sub(r"\D+", "", str(value or ""))

    dept_raw = str(payload.get("department_code", "") or "")
    dept_digits = _digits_only(dept_raw)
    department_code = (
        f"{dept_digits[:3]}-{dept_digits[3:6]}" if len(dept_digits) >= 6 else dept_raw.strip()
    )

    raw_name = str(payload.get("name", "") or "")
    raw_patronymic = str(payload.get("patronymic", "") or "")
    name, patronymic = _maybe_swap_name_and_patronymic(raw_name, raw_patronymic)

    series_num = _normalize_passport_series_number(
        str(payload.get("passport_series", "") or ""),
        str(payload.get("passport_number", "") or ""),
    )

    return PassportData(
        issuing_authority=str(payload.get("issuing_authority", "") or ""),
        issue_date=_normalize_date(str(payload.get("issue_date", "") or "")),
        department_code=department_code,
        passport_series=series_num[0],
        passport_number=series_num[1],
        surname=str(payload.get("surname", "") or ""),
        name=name,
        patronymic=patronymic,
        gender=str(payload.get("gender", "") or ""),
        birth_date=_normalize_date(str(payload.get("birth_date", "") or "")),
        birth_place=str(payload.get("birth_place", "") or ""),
        confidence_note=str(payload.get("confidence_note", "") or ""),
    )


def normalize_registration_data(payload: Dict[str, Any]) -> PassportRegistrationData:
    def _normalize_date(value: str) -> str:
        s = str(value or "").strip()
        if not s:
            return ""
        m = re.search(r"\b(\d{2})[.\-/](\d{2})[.\-/](\d{4})\b", s)
        if m:
            return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
        m = re.search(r"\b(\d{4})[.\-/](\d{2})[.\-/](\d{2})\b", s)
        if m:
            return f"{m.group(3)}.{m.group(2)}.{m.group(1)}"
        return s

    return PassportRegistrationData(
        address=str(payload.get("address", "") or ""),
        region=str(payload.get("region", "") or ""),
        city=str(payload.get("city", "") or ""),
        settlement=str(payload.get("settlement", "") or ""),
        street=str(payload.get("street", "") or ""),
        house=str(payload.get("house", "") or ""),
        building=str(payload.get("building", "") or ""),
        apartment=str(payload.get("apartment", "") or ""),
        registration_date=_normalize_date(str(payload.get("registration_date", "") or "")),
        confidence_note=str(payload.get("confidence_note", "") or ""),
    )


def normalize_egrn_data(payload: Dict[str, Any]) -> EgrnExtractData:
    def _first_non_empty(*keys: str) -> str:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            s = str(value).strip()
            if s:
                return s
        return ""

    def _normalize_area(value: str) -> str:
        if not value:
            return ""
        text = value.replace(",", ".")
        text = re.sub(r"[^\d.]", "", text)
        if not text:
            return ""
        m = re.search(r"\d+(?:\.\d+)?", text)
        if not m:
            return ""
        number = m.group(0)
        return number.rstrip("0").rstrip(".") if "." in number else number

    def _normalize_date(value: str) -> str:
        if not value:
            return ""
        s = str(value).strip()
        m = re.search(r"\b(\d{2})[.\-/](\d{2})[.\-/](\d{4})\b", s)
        if m:
            return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
        m = re.search(r"\b(\d{4})[.\-/](\d{2})[.\-/](\d{2})\b", s)
        if m:
            return f"{m.group(3)}.{m.group(2)}.{m.group(1)}"
        return s

    raw_holders = payload.get("right_holders", [])
    right_holders: List[str]
    if isinstance(raw_holders, list):
        right_holders = [str(item).strip() for item in raw_holders if str(item).strip()]
    elif isinstance(raw_holders, str):
        right_holders = [item.strip() for item in raw_holders.split(",") if item.strip()]
    else:
        right_holders = []

    return EgrnExtractData(
        cadastral_number=_first_non_empty("cadastral_number", "cad_number", "cadastralNo"),
        object_type=_first_non_empty("object_type", "property_type", "object_name", "kind"),
        address=_first_non_empty("address", "object_address", "property_address"),
        area_sq_m=_normalize_area(_first_non_empty("area_sq_m", "area", "total_area", "square")),
        ownership_type=_first_non_empty("ownership_type", "right_type", "purpose"),
        right_holders=right_holders,
        extract_date=_normalize_date(_first_non_empty("extract_date", "statement_date", "egrn_date", "date")),
        confidence_note=_first_non_empty("confidence_note", "note"),
    )


def _passport_missing_or_invalid_fields(data: PassportData) -> List[str]:
    fields: List[str] = []
    if not data.surname.strip():
        fields.append("surname")
    if not data.name.strip():
        fields.append("name")
    if not re.fullmatch(r"\d{4}", data.passport_series.strip() or ""):
        fields.append("passport_series")
    if not re.fullmatch(r"\d{6}", data.passport_number.strip() or ""):
        fields.append("passport_number")
    if data.department_code.strip() and not re.fullmatch(r"\d{3}-\d{3}", data.department_code.strip()):
        fields.append("department_code")
    if data.issue_date.strip() and not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", data.issue_date.strip()):
        fields.append("issue_date")
    if data.birth_date.strip() and not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", data.birth_date.strip()):
        fields.append("birth_date")
    return list(dict.fromkeys(fields))


def _registration_missing_or_invalid_fields(data: PassportRegistrationData) -> List[str]:
    fields: List[str] = []
    if not any([data.city.strip(), data.settlement.strip(), data.street.strip()]):
        fields.extend(["city", "settlement", "street"])
    if not data.house.strip():
        fields.append("house")
    if data.registration_date.strip() and not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", data.registration_date.strip()):
        fields.append("registration_date")
    return list(dict.fromkeys(fields))


def _egrn_missing_or_invalid_fields(data: EgrnExtractData) -> List[str]:
    fields: List[str] = []
    if not re.fullmatch(r"\d{2}:\d{2}:\d{7,}:\d+", data.cadastral_number.strip() or ""):
        fields.append("cadastral_number")
    if not data.object_type.strip():
        fields.append("object_type")
    if not data.address.strip():
        fields.append("address")
    if not re.fullmatch(r"\d+(?:\.\d+)?", data.area_sq_m.strip() or ""):
        fields.append("area_sq_m")
    if data.extract_date.strip() and not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", data.extract_date.strip()):
        fields.append("extract_date")
    if not data.extract_date.strip():
        fields.append("extract_date")
    return list(dict.fromkeys(fields))


def extract_generic_json_from_text(text: str) -> Dict[str, Any]:
    normalized = _normalize_jsonish_text(text)

    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    balanced = _first_balanced_json_object(normalized)
    if balanced:
        candidate = balanced.strip()
        candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    raise ValueError("Не удалось извлечь JSON-объект из ответа модели")


def _message_content_to_str(content: Any) -> str:
    """HF chat completion может вернуть content строкой или списком блоков (VLM)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, tuple):
        # В некоторых интеграциях может прийти (text, model/provider meta)
        if len(content) >= 1 and isinstance(content[0], str):
            return content[0].strip()
        parts = [str(item) for item in content if item is not None]
        return "\n".join(parts).strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if t is not None:
                    parts.append(str(t))
                elif item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"]).strip()
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def build_prompt() -> str:
    return """
Ты OCR/Document AI модуль.
Извлеки данные ТОЛЬКО из изображения паспорта.

Верни ответ СТРОГО в JSON без markdown и без пояснений.
Не добавляй текст до JSON и после JSON.

Формат ответа:
{
  "issuing_authority": "",
  "issue_date": "",
  "department_code": "",
  "passport_series": "",
  "passport_number": "",
  "surname": "",
  "name": "",
  "patronymic": "",
  "gender": "",
  "birth_date": "",
  "birth_place": "",
  "confidence_note": ""
}

Правила:
- Если поля нет или не удалось уверенно прочитать — верни пустую строку.
- Не выдумывай значения.
- Даты приводи к формату DD.MM.YYYY, если это возможно.
- Серия и номер паспорта РФ: всего 10 цифр подряд — первые 4 цифры это passport_series, последние 6 — passport_number.
  На бланке часто две группы «4518» и «478497» или «45 18» и «478497»; всё равно разбей строго: series = 4 цифры, number = 6 цифр, без пробелов внутри значения.
- ФИО (критично): на развороте РФ обычно три отдельных поля/строки — фамилия, имя, отчество сверху вниз или слева направо.
  - surname — только фамилия.
  - name — только личное имя (например Вячеслав, Мария), не отчество.
  - patronymic — только отчество; в русском языке чаще оканчивается на -ович/-евич/-ич (муж.) или -овна/-евна/-инична (жен.).
  Не путай: отчество никогда не клади в name, а имя — в patronymic.
- confidence_note: коротко укажи, какие поля могли быть распознаны неуверенно.
""".strip()


def build_registration_prompt() -> str:
    return """
Ты OCR/Document AI модуль.
Извлеки данные ТОЛЬКО со страницы паспорта с регистрацией (пропиской).

Верни ответ СТРОГО в JSON без markdown и без пояснений.
Не добавляй текст до JSON и после JSON.

Формат ответа:
{
  "region": "",
  "city": "",
  "settlement": "",
  "street": "",
  "house": "",
  "building": "",
  "apartment": "",
  "registration_date": "",
  "confidence_note": ""
}

Правила:
- Если поля нет или не удалось уверенно прочитать — верни пустую строку.
- Не выдумывай значения.
- Дату регистрации приводи к формату DD.MM.YYYY, если это возможно.
- confidence_note: коротко укажи, какие поля могли быть распознаны неуверенно.
""".strip()


def build_egrn_prompt() -> str:
    return """
Ты OCR/Document AI модуль.
Извлеки данные ТОЛЬКО из выписки ЕГРН.

Верни ответ СТРОГО в JSON без markdown и без пояснений.
Не добавляй текст до JSON и после JSON.

Формат ответа:
{
  "cadastral_number": "",
  "object_type": "",
  "address": "",
  "area_sq_m": "",
  "ownership_type": "",
  "right_holders": [],
  "extract_date": "",
  "confidence_note": ""
}

Правила:
- Если поля нет или не удалось уверенно прочитать — верни пустую строку.
- Не выдумывай значения.
- right_holders верни массивом строк.
- extract_date приводи к формату DD.MM.YYYY, если это возможно.
- confidence_note: коротко укажи, какие поля могли быть распознаны неуверенно.
""".strip()


def _build_focus_prompt(title: str, missing_fields: List[str], extra_rules: List[str]) -> str:
    fields_json = ",\n  ".join(f'"{field}": ""' for field in missing_fields)
    rules = "\n".join(f"- {rule}" for rule in extra_rules)
    return f"""
Ты OCR/Document AI модуль.
{title}

Верни ответ СТРОГО в JSON без markdown и без пояснений.
Не добавляй текст до JSON и после JSON.

Формат ответа:
{{
  {fields_json},
  "confidence_note": ""
}}

Правила:
- Верни только перечисленные поля + confidence_note.
- Если поле не найдено уверенно, верни пустую строку.
- Не выдумывай значения.
{rules}
""".strip()


def build_passport_focus_prompt(fields: List[str]) -> str:
    return _build_focus_prompt(
        "На изображении паспорта извлеки ТОЛЬКО недостающие/невалидные поля.",
        fields,
        [
            "Для дат используй формат DD.MM.YYYY.",
            "passport_series — ровно 4 цифры, passport_number — ровно 6 цифр.",
            "department_code — формат XXX-XXX.",
        ],
    )


def build_registration_focus_prompt(fields: List[str]) -> str:
    return _build_focus_prompt(
        "На изображении страницы прописки извлеки ТОЛЬКО недостающие/невалидные поля.",
        fields,
        [
            "Для registration_date используй формат DD.MM.YYYY.",
            "Если город отсутствует, используй settlement.",
            "house укажи как номер дома без префикса.",
        ],
    )


def build_egrn_focus_prompt(fields: List[str]) -> str:
    return _build_focus_prompt(
        "На изображении/странице ЕГРН извлеки ТОЛЬКО недостающие/невалидные поля.",
        fields,
        [
            "Для даты используй формат DD.MM.YYYY.",
            "Для площади верни только число в м2 (например 47.3).",
            "cadastral_number верни в формате 77:04:0002001:9976 (цифры и двоеточия).",
        ],
    )


async def enrich_passport_fields(
    contents: bytes, current: PassportData, model_name: str | None = None
) -> PassportData:
    fields = _passport_missing_or_invalid_fields(current)
    if not fields:
        return current

    try:
        focused_raw, _ = await run_hf_document_extraction(
            contents,
            build_passport_focus_prompt(fields),
            max_tokens=350,
            model_name=model_name,
        )
        focused_payload = extract_json_from_text(focused_raw)
        focused = normalize_passport_data(focused_payload)
    except Exception:
        return current

    merged = PassportData(
        issuing_authority=current.issuing_authority or focused.issuing_authority,
        issue_date=current.issue_date or focused.issue_date,
        department_code=current.department_code or focused.department_code,
        passport_series=current.passport_series or focused.passport_series,
        passport_number=current.passport_number or focused.passport_number,
        surname=current.surname or focused.surname,
        name=current.name or focused.name,
        patronymic=current.patronymic or focused.patronymic,
        gender=current.gender or focused.gender,
        birth_date=current.birth_date or focused.birth_date,
        birth_place=current.birth_place or focused.birth_place,
        confidence_note=(current.confidence_note or focused.confidence_note).strip(),
    )
    return merged


async def enrich_registration_fields(
    contents: bytes, current: PassportRegistrationData, model_name: str | None = None
) -> PassportRegistrationData:
    fields = _registration_missing_or_invalid_fields(current)
    if not fields:
        return current
    try:
        focused_raw, _ = await run_hf_document_extraction(
            contents,
            build_registration_focus_prompt(fields),
            max_tokens=300,
            model_name=model_name,
        )
        focused_payload = extract_generic_json_from_text(focused_raw)
        focused = normalize_registration_data(focused_payload)
    except Exception:
        return current

    merged = PassportRegistrationData(
        region=current.region or focused.region,
        city=current.city or focused.city,
        settlement=current.settlement or focused.settlement,
        street=current.street or focused.street,
        house=current.house or focused.house,
        building=current.building or focused.building,
        apartment=current.apartment or focused.apartment,
        registration_date=current.registration_date or focused.registration_date,
        confidence_note=(current.confidence_note or focused.confidence_note).strip(),
    )
    return merged


async def enrich_egrn_fields(
    egrn_bytes: bytes, current: EgrnExtractData, model_name: str | None = None
) -> EgrnExtractData:
    fields = _egrn_missing_or_invalid_fields(current)
    if not fields:
        return current
    try:
        focused_raw, _ = await run_hf_document_extraction(
            egrn_bytes,
            build_egrn_focus_prompt(fields),
            max_tokens=350,
            model_name=model_name,
        )
        focused_payload = extract_generic_json_from_text(focused_raw)
        focused = normalize_egrn_data(focused_payload)
    except Exception:
        return current

    merged = EgrnExtractData(
        cadastral_number=current.cadastral_number or focused.cadastral_number,
        object_type=current.object_type or focused.object_type,
        address=current.address or focused.address,
        area_sq_m=current.area_sq_m or focused.area_sq_m,
        ownership_type=current.ownership_type or focused.ownership_type,
        right_holders=current.right_holders or focused.right_holders,
        extract_date=current.extract_date or focused.extract_date,
        confidence_note=(current.confidence_note or focused.confidence_note).strip(),
    )
    return merged


async def run_hf_document_extraction(
    contents: bytes,
    prompt: str,
    max_tokens: int = 700,
    model_name: str | None = None,
) -> tuple[str, str]:
    contents = await safe_to_thread(preprocess_image_for_ocr, contents)
    contents = await safe_to_thread(upscale_jpeg_for_ocr, contents, 2.5)
    mime = await safe_to_thread(detect_mime, contents)

    # Жёсткий предел: если HTTP-клиент HF не вернёт управление, поток не должен висеть вечно.
    hard_timeout_sec = float(settings.hf_request_timeout_sec) + 45.0

    image_url = image_to_data_url(contents, mime)

    def _chat_completion_for_model(model_name: str):
        repo_id, provider = _parse_router_model_id(model_name)
        client = _inference_client_for_provider(provider)
        return client.chat.completions.create(
            model=repo_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
        )

    def _text_from_completion(completion: Any) -> str:
        raw = completion.choices[0].message.content
        return _message_content_to_str(raw)

    def _sync_chat_completion() -> tuple[Any, str]:
        primary = (model_name or settings.hf_model).strip()
        completion, used = _chat_completion_for_model(primary), primary

        if not _text_from_completion(completion):
            raise ValueError("empty content")

        return completion, used

    last_error: Exception | None = None
    last_error_cause: BaseException | None = None
    for attempt in range(3):
        try:
            completion, used_model = await asyncio.wait_for(
                safe_to_thread(_sync_chat_completion),
                timeout=hard_timeout_sec,
            )
            break
        except TimeoutError as e:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Превышено время ожидания ответа от Hugging Face (~{int(hard_timeout_sec)} с). "
                    "Для скана паспорта нужна мультимодальная модель (например meta-llama/Llama-4-Scout-17B-16E-Instruct:novita в HF_MODEL). "
                    "Проверьте .env и HF_REQUEST_TIMEOUT_SEC."
                ),
            ) from e
        except Exception as e:
            last_error = e
            last_error_cause = e.__cause__
            logger.exception(
                "HF passport extraction attempt failed",
                extra={
                    "attempt": attempt + 1,
                    "model": model_name or settings.hf_model,
                    "error_repr": repr(e),
                    "cause_repr": repr(e.__cause__) if e.__cause__ else None,
                },
            )
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
    else:
        tech_error = repr(last_error) if last_error else "unknown"
        tech_cause = repr(last_error_cause) if last_error_cause else "unknown"
        detail = (
            "Hugging Face временно недоступен. "
            "Попробуйте повторить запрос через 10-30 секунд "
            "или используйте режим Tesseract. "
            f"Техническая ошибка: {tech_error} "
            f"Причина: {tech_cause}; "
            f"модель: {model_name or settings.hf_model}; "
            f"таймаут HTTP: {settings.hf_request_timeout_sec} с."
        )
        logger.error("HF: все попытки исчерпаны: %s", detail)
        raise HTTPException(status_code=502, detail=detail)

    try:
        raw = completion.choices[0].message.content
        text = _message_content_to_str(raw)
        if not text:
            raise ValueError("empty content")
        return text, used_model
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Пустой или неожиданный ответ от Hugging Face",
        )


async def run_hf_passport_extraction(
    contents: bytes, model_name: str | None = None
) -> tuple[str, str]:
    return await run_hf_document_extraction(
        contents,
        build_prompt(),
        max_tokens=700,
        model_name=model_name,
    )


@router.post("/scan-passport", response_model=PassportScanResponse)
async def scan_passport(file: UploadFile = File(...)) -> PassportScanResponse:
    file_type = (file.content_type or "").lower()
    if not (file_type.startswith("image/") or file_type == "application/pdf"):
        raise HTTPException(status_code=400, detail="Загрузите изображение или PDF")

    contents = await file.read()
    if file_type == "application/pdf":
        contents = await safe_to_thread(pdf_first_page_to_png, contents)
    else:
        await safe_to_thread(validate_image, contents)

    raw_text, model_used = await run_hf_passport_extraction(contents)

    try:
        parsed = extract_json_from_text(raw_text)
        passport_data = normalize_passport_data(parsed)
        passport_data = await enrich_passport_fields(contents, passport_data)
    except Exception as e:
        # Last-resort: не падаем 500, если удалось достать поля эвристикой.
        normalized = _normalize_jsonish_text(raw_text)
        repaired = _repair_passport_jsonish(normalized)
        if not _passport_loose_usable(repaired):
            repaired = _extract_passport_keys_loose(normalized)
        if _passport_loose_usable(repaired):
            passport_data = normalize_passport_data(repaired)
            note = (passport_data.confidence_note or "").strip()
            fallback_note = "JSON восстановлен эвристически из ответа модели."
            passport_data.confidence_note = f"{note} {fallback_note}".strip() if note else fallback_note
        else:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Модель вернула невалидный JSON и данные не удалось восстановить. "
                    f"Фрагмент ответа: {raw_text[:1000]}"
                ),
            ) from e

    return PassportScanResponse(
        ok=True,
        model=model_used,
        data=passport_data,
        raw_text=raw_text,
    )


@router.post("/scan-documents-unified", response_model=UnifiedDocumentsScanResponse)
async def scan_documents_unified(
    passport_main: UploadFile = File(...),
    passport_registration: UploadFile = File(...),
    egrn_extract: UploadFile = File(...),
    hf_model: str | None = Form(default=None),
) -> UnifiedDocumentsScanResponse:
    main_type = (passport_main.content_type or "").lower()
    reg_type = (passport_registration.content_type or "").lower()
    if not (main_type.startswith("image/") or main_type == "application/pdf"):
        raise HTTPException(status_code=400, detail="passport_main: загрузите изображение или PDF")
    if not (reg_type.startswith("image/") or reg_type == "application/pdf"):
        raise HTTPException(status_code=400, detail="passport_registration: загрузите изображение или PDF")
    egrn_type = (egrn_extract.content_type or "").lower()
    if not (egrn_type.startswith("image/") or egrn_type == "application/pdf"):
        raise HTTPException(
            status_code=400,
            detail="egrn_extract: поддерживаются изображение или PDF",
        )

    main_bytes = await passport_main.read()
    registration_bytes = await passport_registration.read()
    egrn_bytes = await egrn_extract.read()

    async def _prepare_ocr_bytes(raw: bytes, content_type: str) -> bytes:
        if content_type == "application/pdf":
            return await safe_to_thread(pdf_first_page_to_png, raw)
        await safe_to_thread(validate_image, raw)
        return raw

    main_ocr_bytes, registration_ocr_bytes, egrn_ocr_bytes = await asyncio.gather(
        _prepare_ocr_bytes(main_bytes, main_type),
        _prepare_ocr_bytes(registration_bytes, reg_type),
        _prepare_ocr_bytes(egrn_bytes, egrn_type),
    )

    try:
        (
            (passport_raw, model_used),
            (registration_raw, _),
            (egrn_raw, _),
        ) = await asyncio.gather(
            run_hf_passport_extraction(main_ocr_bytes, model_name=hf_model),
            run_hf_document_extraction(
                registration_ocr_bytes,
                build_registration_prompt(),
                max_tokens=600,
                model_name=hf_model,
            ),
            run_hf_document_extraction(
                egrn_ocr_bytes,
                build_egrn_prompt(),
                max_tokens=700,
                model_name=hf_model,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("scan-documents-unified: неожиданная ошибка до разбора JSON")
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка сканирования документов: {e!r}",
        ) from e

    try:
        passport_data = normalize_passport_data(extract_json_from_text(passport_raw))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON паспорта: {passport_raw[:1000]}",
        ) from e

    try:
        registration_data = normalize_registration_data(
            extract_generic_json_from_text(registration_raw)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON страницы регистрации: {registration_raw[:1000]}",
        ) from e

    try:
        egrn_data = normalize_egrn_data(extract_generic_json_from_text(egrn_raw))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось разобрать JSON выписки ЕГРН: {egrn_raw[:1000]}",
        ) from e

    passport_data, registration_data, egrn_data = await asyncio.gather(
        enrich_passport_fields(main_ocr_bytes, passport_data, model_name=hf_model),
        enrich_registration_fields(registration_ocr_bytes, registration_data, model_name=hf_model),
        enrich_egrn_fields(egrn_ocr_bytes, egrn_data, model_name=hf_model),
    )

    return UnifiedDocumentsScanResponse(
        ok=True,
        model=model_used,
        data=UnifiedDocumentsData(
            passport_main=passport_data,
            passport_registration=registration_data,
            egrn_extract=egrn_data,
        ),
        raw_text={
            "passport_main": passport_raw,
            "passport_registration": registration_raw,
            "egrn_extract": egrn_raw,
        },
    )


@router.post("/scan-passport-tesseract", response_model=PassportScanTesseractResponse)
async def scan_passport_tesseract(file: UploadFile = File(...)) -> PassportScanTesseractResponse:
    text = await extract_text_from_upload(file)
    return PassportScanTesseractResponse(
        ok=True,
        engine="tesseract",
        text=text,
    )
