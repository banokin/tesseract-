"""
Эвристический разбор OCR страницы паспорта с регистрацией (пропиской).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_DATE = re.compile(r"\b(\d{2})[.\-/](\d{2})[.\-/](\d{4})\b")


def _norm_space(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r", "\n")).strip()


def _pick_date_near(text: str, patterns: tuple[str, ...]) -> str:
    low = text.lower()
    for p in patterns:
        idx = low.find(p)
        if idx < 0:
            continue
        window = text[idx : idx + 160]
        m = _DATE.search(window)
        if m:
            return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    return ""


def _extract_address_blob(full: str) -> str:
    for pat in (
        r"(?:место\s+жительства|адрес\s+регистрац\w*|зарегистрирован\w*\s+по\s+адресу)"
        r"\s*[:\s]*\s*(.+?)(?=\n\s*(?:дата|свед|подпис|$))",
        r"(?:по\s+месту\s+жительства)\s*[:\s]*\s*(.+?)(?=\n\n|$)",
    ):
        m = re.search(pat, full, re.I | re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()[:800]
    # одна длинная строка с «обл.», «г.», «ул.»
    m = re.search(
        r"([А-ЯЁа-яё0-9\s,\.\-]+(?:обл\.|респ\.|край|г\.|ул\.|д\.|кв\.)[А-ЯЁа-яё0-9\s,\.\-]+)",
        full,
        re.I,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:800]
    lines = [ln.strip(" ,.;:") for ln in full.split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        if not re.search(r"(адрес|место\s+жительства|зарегистрирован)", ln, re.I):
            continue
        chunk: list[str] = []
        for j in range(i, min(i + 4, len(lines))):
            cur = lines[j].strip(" ,.;:")
            if not cur:
                continue
            if j > i and re.search(r"(дата|выдан|подпись|код\s+подраздел)", cur, re.I):
                break
            if re.search(r"(обл|респ|край|г\.|ул\.|д\.|кв\.|дом|корп|стр\.)", cur, re.I):
                chunk.append(cur)
        if chunk:
            return re.sub(r"\s+", " ", ", ".join(chunk)).strip()[:800]
    return ""


def _split_address_parts(blob: str) -> Dict[str, str]:
    out = {
        "address": "",
        "region": "",
        "city": "",
        "settlement": "",
        "street": "",
        "house": "",
        "building": "",
        "apartment": "",
    }
    if not blob:
        return out
    out["address"] = blob.strip()
    hm = re.search(r"(?:^|[,\s])д\.?\s*([0-9]+[а-яА-Яа-яёЁ]?)", blob, re.I)
    if hm:
        out["house"] = hm.group(1).strip()
    bm = re.search(r"(?:корп\.|стр\.)\s*([0-9]+[а-яА-Я]?)", blob, re.I)
    if bm:
        out["building"] = bm.group(1).strip()
    am = re.search(r"кв\.?\s*([0-9]+[а-яА-Я]?)", blob, re.I)
    if am:
        out["apartment"] = am.group(1).strip()

    parts = [p.strip() for p in re.split(r",\s*", blob) if p.strip()]
    for p in parts:
        pl = p.lower()
        if re.search(r"\b(обл\.|респ\.|край|ао\b|автономный округ)", pl):
            out["region"] = p
        elif re.search(r"^\s*г\.\s*", pl, re.I):
            out["city"] = re.sub(r"^\s*г\.\s*", "", p, flags=re.I).strip()
        elif re.search(r"^\s*(пгт|пос\.|с\.|дер\.)\s*", pl, re.I):
            out["settlement"] = p
        elif re.search(r"^\s*(ул\.|просп\.|пер\.|бул\.|ш\.)\s*", pl, re.I):
            out["street"] = p
    return out


def parse_registration_ocr_text(ocr_text: str) -> Dict[str, Any]:
    text = _norm_space(ocr_text)
    full = text
    blob = _extract_address_blob(full)
    parts = _split_address_parts(blob)

    registration_date = _pick_date_near(
        full,
        ("дата регистрац", "зарегистрирован", "регистрац"),
    )

    notes: List[str] = []
    if not blob and not any(parts.values()):
        notes.append("адрес регистрации не выделен; проверьте OCR")

    return {
        "address": blob,
        "region": parts["region"],
        "city": parts["city"],
        "settlement": parts["settlement"],
        "street": parts["street"],
        "house": parts["house"],
        "building": parts["building"],
        "apartment": parts["apartment"],
        "registration_date": registration_date,
        "confidence_note": "; ".join(notes) if notes else "OCR Tesseract + правила (прописка)",
    }
