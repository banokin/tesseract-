from __future__ import annotations

import asyncio
import base64
import ast
import imghdr
import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import OpenAI, APITimeoutError
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from async_utils import safe_to_thread
from ocr import extract_text_from_upload

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_token: str = Field(validation_alias="HF_TOKEN")
    hf_model: str = Field(default="Qwen/Qwen3-VL-8B-Instruct:novita", validation_alias="HF_MODEL")
    hf_fallback_model: str = Field(default="", validation_alias="MODEL_2")
    hf_base_url: str = Field(default="https://router.huggingface.co/v1", validation_alias="HF_BASE_URL")
    max_file_size_mb: int = Field(default=10, validation_alias="MAX_FILE_SIZE_MB")
    hf_request_timeout_sec: int = Field(default=90, validation_alias="HF_REQUEST_TIMEOUT_SEC")


settings = Settings()
logger = logging.getLogger(__name__)

router = APIRouter(tags=["passport"])


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


client = OpenAI(
    base_url=settings.hf_base_url,
    api_key=settings.hf_token,
    timeout=float(settings.hf_request_timeout_sec),
)


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
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
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


def normalize_passport_data(payload: Dict[str, Any]) -> PassportData:
    return PassportData(
        issuing_authority=str(payload.get("issuing_authority", "") or ""),
        issue_date=str(payload.get("issue_date", "") or ""),
        department_code=str(payload.get("department_code", "") or ""),
        passport_series=str(payload.get("passport_series", "") or ""),
        passport_number=str(payload.get("passport_number", "") or ""),
        surname=str(payload.get("surname", "") or ""),
        name=str(payload.get("name", "") or ""),
        patronymic=str(payload.get("patronymic", "") or ""),
        gender=str(payload.get("gender", "") or ""),
        birth_date=str(payload.get("birth_date", "") or ""),
        birth_place=str(payload.get("birth_place", "") or ""),
        confidence_note=str(payload.get("confidence_note", "") or ""),
    )


def normalize_registration_data(payload: Dict[str, Any]) -> PassportRegistrationData:
    return PassportRegistrationData(
        region=str(payload.get("region", "") or ""),
        city=str(payload.get("city", "") or ""),
        settlement=str(payload.get("settlement", "") or ""),
        street=str(payload.get("street", "") or ""),
        house=str(payload.get("house", "") or ""),
        building=str(payload.get("building", "") or ""),
        apartment=str(payload.get("apartment", "") or ""),
        registration_date=str(payload.get("registration_date", "") or ""),
        confidence_note=str(payload.get("confidence_note", "") or ""),
    )


def normalize_egrn_data(payload: Dict[str, Any]) -> EgrnExtractData:
    raw_holders = payload.get("right_holders", [])
    right_holders: List[str]
    if isinstance(raw_holders, list):
        right_holders = [str(item).strip() for item in raw_holders if str(item).strip()]
    elif isinstance(raw_holders, str):
        right_holders = [item.strip() for item in raw_holders.split(",") if item.strip()]
    else:
        right_holders = []

    return EgrnExtractData(
        cadastral_number=str(payload.get("cadastral_number", "") or ""),
        object_type=str(payload.get("object_type", "") or ""),
        address=str(payload.get("address", "") or ""),
        area_sq_m=str(payload.get("area_sq_m", "") or ""),
        ownership_type=str(payload.get("ownership_type", "") or ""),
        right_holders=right_holders,
        extract_date=str(payload.get("extract_date", "") or ""),
        confidence_note=str(payload.get("confidence_note", "") or ""),
    )


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
- passport_series и passport_number верни отдельно, если это возможно.
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


async def run_hf_document_extraction(contents: bytes, prompt: str, max_tokens: int = 700) -> tuple[str, str]:
    mime = detect_mime(contents)
    image_url = image_to_data_url(contents, mime)

    # Жёсткий предел: если HTTP-клиент HF не вернёт управление, поток не должен висеть вечно.
    hard_timeout_sec = float(settings.hf_request_timeout_sec) + 45.0

    def _chat_completion_for_model(model_name: str):
        return client.chat.completions.create(
            model=model_name,
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

    def _sync_chat_completion() -> tuple[Any, str]:
        primary = settings.hf_model
        fallback = settings.hf_fallback_model.strip()
        try:
            return _chat_completion_for_model(primary), primary
        except Exception as e:
            # На router-маршруте иногда модель/провайдер может быть недоступен.
            if fallback and fallback != primary:
                logger.warning(
                    "HF router primary model failed, switching to fallback",
                    extra={"primary_model": primary, "fallback_model": fallback},
                )
                return _chat_completion_for_model(fallback), fallback
            raise e

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
                    "Для скана паспорта нужна vision-модель (например Qwen/Qwen2.5-VL-7B-Instruct в HF_MODEL). "
                    "Проверьте .env и HF_REQUEST_TIMEOUT_SEC."
                ),
            ) from e
        except APITimeoutError as e:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Превышено время ожидания ответа от Hugging Face Router ({settings.hf_request_timeout_sec} с). "
                    "Повторите позже, смените модель в .env (HF_MODEL) или увеличьте HF_REQUEST_TIMEOUT_SEC."
                ),
            ) from e
        except Exception as e:
            last_error = e
            last_error_cause = e.__cause__
            logger.exception(
                "HF passport extraction attempt failed",
                extra={
                    "attempt": attempt + 1,
                    "model": settings.hf_model,
                    "error_repr": repr(e),
                    "cause_repr": repr(e.__cause__) if e.__cause__ else None,
                },
            )
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
    else:
        tech_error = repr(last_error) if last_error else "unknown"
        tech_cause = repr(last_error_cause) if last_error_cause else "unknown"
        raise HTTPException(
            status_code=502,
            detail=(
                "Hugging Face временно недоступен. "
                "Попробуйте повторить запрос через 10-30 секунд "
                "или используйте режим Tesseract. "
                f"Техническая ошибка: {tech_error}. "
                f"Причина: {tech_cause}. "
                f"Модель: {settings.hf_model}. "
                f"Fallback: {settings.hf_fallback_model or 'не задан'}. "
                f"Таймаут HTTP: {settings.hf_request_timeout_sec} сек."
            ),
        )

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


async def run_hf_passport_extraction(contents: bytes) -> tuple[str, str]:
    return await run_hf_document_extraction(contents, build_prompt(), max_tokens=700)


@router.post("/scan-passport", response_model=PassportScanResponse)
async def scan_passport(file: UploadFile = File(...)) -> PassportScanResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите изображение")

    contents = await file.read()
    validate_image(contents)

    raw_text, model_used = await run_hf_passport_extraction(contents)

    try:
        parsed = extract_json_from_text(raw_text)
        passport_data = normalize_passport_data(parsed)
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
) -> UnifiedDocumentsScanResponse:
    if not passport_main.content_type or not passport_main.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="passport_main: загрузите изображение")
    if not passport_registration.content_type or not passport_registration.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="passport_registration: загрузите изображение")
    egrn_type = (egrn_extract.content_type or "").lower()
    if not (egrn_type.startswith("image/") or egrn_type == "application/pdf"):
        raise HTTPException(
            status_code=400,
            detail="egrn_extract: поддерживаются изображение или PDF",
        )

    main_bytes = await passport_main.read()
    registration_bytes = await passport_registration.read()
    egrn_bytes = await egrn_extract.read()

    validate_image(main_bytes)
    validate_image(registration_bytes)
    if egrn_type == "application/pdf":
        egrn_ocr_bytes = pdf_first_page_to_png(egrn_bytes)
    else:
        validate_image(egrn_bytes)
        egrn_ocr_bytes = egrn_bytes

    try:
        passport_raw, model_used = await run_hf_passport_extraction(main_bytes)
        registration_raw, _ = await run_hf_document_extraction(
            registration_bytes,
            build_registration_prompt(),
            max_tokens=600,
        )
        egrn_raw, _ = await run_hf_document_extraction(
            egrn_ocr_bytes,
            build_egrn_prompt(),
            max_tokens=700,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка сканирования документов: {e!r}") from e

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
