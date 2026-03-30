from __future__ import annotations

import asyncio
import base64
import ast
import imghdr
import json
import re
from io import BytesIO
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile
from huggingface_hub import InferenceClient
from huggingface_hub.errors import InferenceTimeoutError
from requests.exceptions import Timeout as RequestsTimeout
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from ocr import extract_text_from_upload


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_token: str
    hf_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_file_size_mb: int = 10
    hf_request_timeout_sec: int = 90


settings = Settings()

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


client = InferenceClient(
    api_key=settings.hf_token,
    timeout=float(settings.hf_request_timeout_sec),
)


def validate_image(contents: bytes) -> None:
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой. Максимум: {settings.max_file_size_mb} MB",
        )

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


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()

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


def _message_content_to_str(content: Any) -> str:
    """HF chat completion может вернуть content строкой или списком блоков (VLM)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
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


async def run_hf_passport_extraction(contents: bytes) -> str:
    mime = detect_mime(contents)
    image_url = image_to_data_url(contents, mime)
    prompt = build_prompt()

    # Жёсткий предел: если HTTP-клиент HF не вернёт управление, поток не должен висеть вечно.
    hard_timeout_sec = float(settings.hf_request_timeout_sec) + 45.0

    def _sync_chat_completion():
        return client.chat_completion(
            model=settings.hf_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=700,
            temperature=0,
        )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            completion = await asyncio.wait_for(
                asyncio.to_thread(_sync_chat_completion),
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
        except (InferenceTimeoutError, RequestsTimeout) as e:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Превышено время ожидания ответа от Hugging Face ({settings.hf_request_timeout_sec} с). "
                    "Повторите позже, смените модель в .env (HF_MODEL) или увеличьте HF_REQUEST_TIMEOUT_SEC."
                ),
            ) from e
        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
    else:
        raise HTTPException(
            status_code=502,
            detail=(
                "Hugging Face временно недоступен. "
                "Попробуйте повторить запрос через 10-30 секунд "
                "или используйте режим Tesseract. "
                f"Техническая ошибка: {last_error}. "
                f"Таймаут HTTP: {settings.hf_request_timeout_sec} сек."
            ),
        )

    try:
        raw = completion.choices[0].message.content
        text = _message_content_to_str(raw)
        if not text:
            raise ValueError("empty content")
        return text
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Пустой или неожиданный ответ от Hugging Face",
        )


@router.post("/scan-passport", response_model=PassportScanResponse)
async def scan_passport(file: UploadFile = File(...)) -> PassportScanResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите изображение")

    contents = await file.read()
    validate_image(contents)

    raw_text = await run_hf_passport_extraction(contents)

    try:
        parsed = extract_json_from_text(raw_text)
        passport_data = normalize_passport_data(parsed)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Модель вернула ответ, из которого не удалось извлечь JSON. "
                f"Фрагмент ответа: {raw_text[:1000]}"
            ),
        ) from e

    return PassportScanResponse(
        ok=True,
        model=settings.hf_model,
        data=passport_data,
        raw_text=raw_text,
    )


@router.post("/scan-passport-tesseract", response_model=PassportScanTesseractResponse)
async def scan_passport_tesseract(file: UploadFile = File(...)) -> PassportScanTesseractResponse:
    text = await extract_text_from_upload(file)
    return PassportScanTesseractResponse(
        ok=True,
        engine="tesseract",
        text=text,
    )
