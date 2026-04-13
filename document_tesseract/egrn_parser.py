"""
Эвристический разбор текста OCR выписки ЕГРН (Росреестр) в поля для normalize_egrn_data.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_DATE = re.compile(r"\b(\d{2})[.\-/](\d{2})[.\-/](\d{4})\b")
# Кадастровый номер: группы через двоеточие (допускаем пробелы у OCR)
_CADASTRAL = re.compile(
    r"\b(\d{1,2})\s*:\s*(\d{1,2})\s*:\s*(\d{6,12})\s*:\s*(\d{1,10})\b"
)
# ФИО: три слова кириллицей
_FIO_LINE = re.compile(
    r"^[А-ЯЁ][а-яё\-]+\s+[А-ЯЁ][а-яё\-]+\s+[А-ЯЁ][а-яё\-]+$",
)


def _norm_space(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r", "\n")).strip()


def _first_date_after(text: str, *markers: str) -> str:
    low = text.lower()
    for mk in markers:
        idx = low.find(mk.lower())
        if idx < 0:
            continue
        window = text[idx : idx + 200]
        m = _DATE.search(window)
        if m:
            return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    return ""


def _extract_cadastral(full: str) -> str:
    m = _CADASTRAL.search(full)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)}:{m.group(4)}"
    return ""


def _extract_after_label(full: str, label_pattern: str, max_len: int = 500) -> str:
    m = re.search(
        rf"{label_pattern}\s*[:\s]+\s*(.+?)(?:\n\n|\n(?=[А-ЯA-Z])|$)",
        full,
        re.I | re.DOTALL,
    )
    if m:
        line = re.sub(r"\s+", " ", m.group(1)).strip()
        return line[:max_len]
    return ""


def _extract_object_type(full: str) -> str:
    for pat in (
        r"вид\s+объекта\s+недвижимости\s*[:\s]+([^\n]+)",
        r"назначение\s+помещен\w*\s*[:\s]+([^\n]+)",
        r"тип\s+объекта\s*[:\s]+([^\n]+)",
        r"вид\s+объекта\s*[:\s]+([^\n]+)",
    ):
        m = re.search(pat, full, re.I)
        if m:
            return m.group(1).strip()[:300]
    m = re.search(
        r"\b(жилое\s+помещение|квартира|земельн\w+\s+участок|здание|"
        r"нежилое\s+помещение|машино-место|сооружение)\b",
        full,
        re.I,
    )
    if m:
        return m.group(1).strip()
    return ""


def _extract_address(full: str) -> str:
    for pat in (
        r"(?:адрес|местоположен\w*|место\s+нахожден\w*)\s*(?:объекта)?\s*[:\s]+\s*([^\n]+)",
        r"местоположение\s+установлено\s*[:\s]*\s*([^\n]+)",
    ):
        m = re.search(pat, full, re.I)
        if m:
            return m.group(1).strip()[:500]
    for ln in full.split("\n"):
        line = ln.strip()
        if len(line) < 20 or len(line) > 240:
            continue
        if re.search(r"\b(кадастров|стоимость|номер|дата|лист|раздел)\b", line, re.I):
            continue
        if re.search(
            r"\b(край|область|республика|г\.|город|ул\.|улица|проспект|линия|участок|дом|д\.)\b",
            line,
            re.I,
        ):
            return line[:500]
    return ""


def _extract_area(full: str) -> str:
    m = re.search(
        r"(?:площадь|общая\s+площадь)\s*[:\s]*\s*"
        r"(\d+[.,]?\d*)\s*(?:кв\.?\s*м|м\s*[²2]|кв\.м\.?)",
        full,
        re.I,
    )
    if m:
        return m.group(1).replace(",", ".").strip()
    m = re.search(r"(\d+[.,]\d+)\s*(?:кв|м)", full, re.I)
    if m:
        return m.group(1).replace(",", ".").strip()
    return ""


def _extract_ownership(full: str) -> str:
    for pat in (
        r"(?:вид\s+зарегистрированн\w+\s+права|вид\s+права)\s*[:\s]+\s*([^\n]+)",
        r"(?:собственность|долевая\s+собственность|аренда|сервитут|"
        r"оперативное\s+управление|хозяйственное\s+ведение)\b[^\n]*",
    ):
        m = re.search(pat, full, re.I)
        if m:
            if m.lastindex:
                return m.group(1).strip()[:300]
            return m.group(0).strip()[:300]
    return ""


def _extract_right_holders(full: str) -> List[str]:
    holders: List[str] = []
    sec = re.search(
        r"(?:правообладател\w*|сведения\s+о\s+правообладател\w*)"
        r"[^\n]*\n((?:.|\n){1,1200}?)(?=\n\s*(?:вид\s+права|ограничен|обремен|кадастр|$))",
        full,
        re.I,
    )
    block = sec.group(1) if sec else full
    for ln in block.split("\n"):
        ln = ln.strip()
        if not ln or len(ln) > 120:
            continue
        if _FIO_LINE.match(ln):
            holders.append(ln)
    seen: set[str] = set()
    out: List[str] = []
    for h in holders:
        if h not in seen:
            seen.add(h)
            out.append(h)
    if out:
        return out[:20]
    m = re.search(
        r"правообладател\w*\s*[:\s]+\s*([А-ЯЁ][^\n]{5,100})",
        full,
        re.I,
    )
    if m:
        part = m.group(1).strip()
        if re.search(r"[А-ЯЁа-яё]{3,}", part):
            return [part[:200]]
    return []


def _extract_extract_date(full: str) -> str:
    d = _first_date_after(
        full,
        "дата формирования",
        "дата выдачи выписки",
        "выписка сформирована",
        "дата получения сведений",
        "дата создания выписки",
    )
    if d:
        return d
    dates: List[str] = []
    for m in _DATE.finditer(full):
        dates.append(f"{m.group(1)}.{m.group(2)}.{m.group(3)}")
    return dates[-1] if dates else ""


def parse_egrn_ocr_text(ocr_text: str) -> Dict[str, Any]:
    """
    Словарь полей для :func:`normalize_egrn_data` из `scan_passport_hf`.
    """
    text = _norm_space(ocr_text)
    full = text

    cadastral_number = _extract_cadastral(full)
    object_type = _extract_object_type(full) or _extract_after_label(
        full, r"объект\s+недвижимости"
    )
    address = _extract_address(full)
    area_sq_m = _extract_area(full)
    ownership_type = _extract_ownership(full)
    right_holders = _extract_right_holders(full)
    extract_date = _extract_extract_date(full)

    notes: List[str] = []
    if not cadastral_number:
        notes.append("кадастровый номер не распознан по шаблону")
    if not address:
        notes.append("адрес не найден по подписям")
    if not right_holders:
        notes.append("правообладатели не выделены (проверьте ФИО вручную)")

    return {
        "cadastral_number": cadastral_number,
        "object_type": object_type,
        "address": address,
        "area_sq_m": area_sq_m,
        "ownership_type": ownership_type,
        "right_holders": right_holders,
        "extract_date": extract_date,
        "confidence_note": "; ".join(notes) if notes else "OCR Tesseract + правила (ЕГРН)",
    }
