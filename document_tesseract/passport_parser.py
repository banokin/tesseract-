"""
Эвристический разбор текста OCR паспорта РФ в плоский dict полей.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_DATE = re.compile(r"\b(\d{2})\s*[.\-/]\s*(\d{2})\s*[.\-/]\s*(\d{4})\b")
_DEPT = re.compile(r"\b(\d{3})[-\s]?(\d{3})\b")
_CYR_TOKEN = re.compile(r"^[А-ЯЁа-яё\-]+$")
_PASSPORT_OCR_DIGIT_MAP = str.maketrans(
    {
        "О": "0",
        "о": "0",
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "Б": "6",
        "В": "8",
        "B": "8",
    }
)
_FIO_NOISE = re.compile(
    r"\b(уфмс|мвд|россии|рф|выдан|выдачи|области|городе|район|код|подразделен|паспорт|дата)\b",
    re.I,
)
_FIO_STOPWORDS = {
    "имя",
    "фамилия",
    "отчество",
    "личный",
    "подпись",
    "россия",
    "россии",
    "мвд",
    "уфмс",
    "выдан",
    "дата",
    "рождения",
    "рожд",
    "город",
    "область",
    "области",
    "код",
    "серия",
    "номер",
}


def _norm_space(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r", "\n")).strip()


def _all_dates(text: str) -> List[str]:
    out: List[str] = []
    for m in _DATE.finditer(text):
        out.append(f"{m.group(1)}.{m.group(2)}.{m.group(3)}")
    return out


def _date_to_key(date_str: str) -> tuple[int, int, int]:
    d, m, y = (int(x) for x in date_str.split("."))
    return y, m, d


def _extract_birth_date_from_lines(lines: List[str], issue_date: str = "") -> str:
    issue_digits = re.sub(r"\D", "", issue_date or "")
    for i, ln in enumerate(lines):
        if not re.search(r"(дата\s*рожд|рожд[её]н|рождени|рожден)", ln, re.I):
            continue
        candidates = [ln]
        if i + 1 < len(lines):
            candidates.append(lines[i + 1])
        if i + 2 < len(lines):
            candidates.append(lines[i + 2])
        for cand in candidates:
            m = _DATE.search(cand)
            if not m:
                continue
            d = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
            if issue_digits and re.sub(r"\D", "", d) == issue_digits:
                continue
            return d
    return ""


def _pick_date_near(text: str, patterns: tuple[str, ...], default: str = "") -> str:
    low = text.lower()
    for p in patterns:
        idx = low.find(p)
        if idx < 0:
            continue
        window = text[idx : idx + 120]
        m = _DATE.search(window)
        if m:
            return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    return default


def _value_after_label(lines: List[str], label_pattern: str) -> str:
    def _cleanup_label_tail(value: str) -> str:
        cleaned = re.sub(r"\b(фамилия|имя|отчество)\b.*$", "", value, flags=re.I).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.strip(".,:;!?()[]{}\"'")
        return cleaned

    def _extract_token(value: str) -> str:
        candidate = _cleanup_label_tail(value)
        if not candidate:
            return ""
        # Берём последний "словоподобный" токен (чаще это целевое ФИО значение).
        tokens = re.findall(r"[А-ЯЁа-яё\-]{2,40}", candidate)
        if not tokens:
            return ""
        return tokens[-1].upper()

    def _candidate_score(token: str) -> int:
        if not token:
            return -100
        if re.search(r"\d", token):
            return -100
        low = token.lower()
        if low in _FIO_STOPWORDS:
            return -100
        if _FIO_NOISE.search(token):
            return -100
        if not re.match(r"^[А-ЯЁ\-]+$", token):
            return -50
        score = 0
        if 3 <= len(token) <= 20:
            score += 6
        elif len(token) == 2:
            score -= 4
        else:
            score -= 2
        if re.match(r"^[А-ЯЁ]{3,20}$", token):
            score += 3
        return score

    block = "\n".join(lines)
    candidates: list[str] = []

    for cm in re.finditer(
        rf"{label_pattern}\s*[:\s]+\s*([А-ЯЁа-яё\-\s]+?)(?:\n|$)",
        block,
        re.I,
    ):
        token = _extract_token(cm.group(1).strip().split("\n")[0].strip())
        if token:
            candidates.append(token)

    for i, ln in enumerate(lines):
        if re.search(label_pattern, ln, re.I):
            # Формат "Фамилия ХАЛАБУДИНА" (значение после метки в той же строке).
            parts = re.split(label_pattern, ln, maxsplit=1, flags=re.I)
            if len(parts) > 1:
                same_line_tail = parts[1].strip(" :;-")
                token = _extract_token(same_line_tail)
                if token:
                    candidates.append(token)
            # OCR часто даёт формат: "ХАЛАБУДИНА Фамилия" (значение перед меткой).
            before = re.split(label_pattern, ln, maxsplit=1, flags=re.I)[0].strip(" :;-")
            if before:
                before_tokens = [t for t in before.split() if t]
                if before_tokens:
                    cand = _extract_token(before_tokens[-1])
                    if cand:
                        candidates.append(cand)
            if ":" in ln:
                tail = ln.split(":", 1)[1].strip()
                if tail and len(tail) < 80:
                    cand = _extract_token(tail)
                    if cand:
                        candidates.append(cand)
            if i + 1 < len(lines):
                cand = lines[i + 1].strip()
                if _CYR_TOKEN.match(cand) or re.match(r"^[А-ЯЁ][а-яё\-]+$", cand):
                    token = _extract_token(cand)
                    if token:
                        candidates.append(token)
            if i > 0:
                prev = lines[i - 1].strip()
                if _CYR_TOKEN.match(prev) or re.match(r"^[А-ЯЁ][а-яё\-]+$", prev):
                    token = _extract_token(prev)
                    if token:
                        candidates.append(token)
    if not candidates:
        return ""
    # Считаем общий скор (качество + частота появления).
    best = ""
    best_score = -10**9
    uniq = set(candidates)
    for cand in uniq:
        first_idx = candidates.index(cand)
        score = _candidate_score(cand) + candidates.count(cand) * 2 + max(0, 20 - first_idx)
        if score > best_score:
            best_score = score
            best = cand
    return best


def _looks_like_person_name(value: str, *, allow_two_words: bool = False) -> bool:
    txt = re.sub(r"\s+", " ", value).strip()
    if not txt:
        return False
    if _FIO_NOISE.search(txt):
        return False
    if re.search(r"\d", txt):
        return False
    words = txt.split()
    normalized_words = [w.strip("-").lower() for w in words if w.strip("-")]
    if any(w in _FIO_STOPWORDS for w in normalized_words):
        return False
    token_re = r"(?:[А-ЯЁ]{3,30}|[А-ЯЁ][а-яё\-]{2,30})"
    if len(words) == 1:
        return bool(re.match(rf"^{token_re}$", words[0]))
    if allow_two_words and len(words) == 2:
        return all(re.match(rf"^{token_re}$", w) for w in words)
    return False


def _extract_fio_labeled(lines: List[str]) -> tuple[str, str, str]:
    surname = _value_after_label(lines, r"фамили") or _value_after_label(lines, r"famil")
    name = _value_after_label(lines, r"(?<![а-яё])имя(?![а-яё])") or _value_after_label(
        lines, r"given\s*name"
    )
    patronymic = _value_after_label(lines, r"отчеств") or _value_after_label(lines, r"patronym")
    surname = surname if _looks_like_person_name(surname, allow_two_words=True) else ""
    name = name if _looks_like_person_name(name) else ""
    patronymic = patronymic if _looks_like_person_name(patronymic) else ""

    if not (surname and name and patronymic):
        cyr_lines = [
            ln
            for ln in lines
            if re.search(r"[А-ЯЁа-яё]", ln)
            and not _DATE.search(ln)
            and not _DEPT.search(ln)
            and not _FIO_NOISE.search(ln)
            and len(ln) < 60
        ]
        short = [
            ln
            for ln in cyr_lines
            if not re.search(r"\d", ln)
            and _looks_like_person_name(ln, allow_two_words=True)
        ]
        if len(short) >= 3:
            # Часто ФИО находится ближе к низу OCR-блока страницы.
            fio_tail = short[-3:]
            if not surname:
                surname = fio_tail[0]
            if not name:
                name = fio_tail[1]
            if not patronymic:
                patronymic = fio_tail[2]
            if surname and name and surname == name:
                surname = fio_tail[0]
                name = fio_tail[1]

    return surname, name, patronymic


def _extract_gender(full: str) -> str:
    m = re.search(r"(?:пол|sex)\s*[:\s]*([МЖ])\b", full, re.I)
    if m:
        g = m.group(1).upper()
        return "М" if g in "MМ" else "Ж"
    m = re.search(r"\b(МУЖ|ЖЕН)\w*", full, re.I)
    if m:
        return "М" if m.group(1).upper().startswith("М") else "Ж"
    return ""


def _extract_department(full: str) -> str:
    km = re.search(r"код\s*подразделен\w*[^\d]*(\d{3})[-\s]?(\d{3})", full, re.I)
    if km:
        return f"{km.group(1)}-{km.group(2)}"
    for m in _DEPT.finditer(full):
        a, b = m.group(1), m.group(2)
        if a == "000" and b == "000":
            continue
        return f"{a}-{b}"
    return ""


def _extract_series_number(full: str, issue_date: str = "", department_code: str = "") -> tuple[str, str]:
    def _ocr_digits_only(text: str) -> str:
        normalized = text.translate(_PASSPORT_OCR_DIGIT_MAP)
        return re.sub(r"\D", "", normalized)

    def _collect_pass_candidates(text: str) -> list[tuple[str, str, bool]]:
        normalized_text = text.translate(_PASSPORT_OCR_DIGIT_MAP)
        local: list[tuple[str, str, bool]] = []
        for m in re.finditer(
            r"(?<!\d)(\d{2})[\s\-_.]*(\d{2})[\s\-_.]*(\d{6})(?!\d)",
            normalized_text,
        ):
            pos = m.start()
            window = normalized_text[max(0, pos - 80) : pos + 80]
            near_keywords = bool(re.search(r"(серия|номер|паспорт)", window, re.I))
            local.append((f"{m.group(1)}{m.group(2)}", m.group(3), near_keywords))
        for m in re.finditer(r"(?<!\d)(\d{4})[\s\-_.]*(\d{6})(?!\d)", normalized_text):
            pos = m.start()
            window = normalized_text[max(0, pos - 80) : pos + 80]
            near_keywords = bool(re.search(r"(серия|номер|паспорт)", window, re.I))
            local.append((m.group(1), m.group(2), near_keywords))
        return local

    lines = [ln.strip() for ln in full.split("\n") if ln.strip()]
    issue_digits = _ocr_digits_only(issue_date)
    dept_digits = _ocr_digits_only(department_code)

    series = ""
    number = ""

    # Если текст собран из многих OCR-проходов, выбираем пару серия/номер по "голосованию".
    # Это снижает влияние единичных шумных проходов.
    pass_chunks = [chunk.strip() for chunk in re.split(r"====\s*OCR PASS\s*====", full) if chunk.strip()]
    if pass_chunks:
        stats: dict[tuple[str, str], dict[str, int]] = {}
        for chunk in pass_chunks:
            seen_in_chunk: set[tuple[str, str]] = set()
            for cand_series, cand_number, near_keywords in _collect_pass_candidates(chunk):
                key = (cand_series, cand_number)
                if key in seen_in_chunk:
                    continue
                seen_in_chunk.add(key)
                bucket = stats.setdefault(key, {"count": 0, "keyword_hits": 0, "penalty": 0})
                bucket["count"] += 1
                if near_keywords:
                    bucket["keyword_hits"] += 1
                combo = f"{cand_series}{cand_number}"
                if issue_digits and issue_digits in combo:
                    bucket["penalty"] += 2
                if dept_digits and dept_digits in combo:
                    bucket["penalty"] += 2
                if cand_series == "0000" or cand_number == "000000":
                    bucket["penalty"] += 2
        if stats:
            best_key: tuple[str, str] | None = None
            best_score = -10**9
            for key, meta in stats.items():
                cand_series, cand_number = key
                score = meta["count"] * 8 + meta["keyword_hits"] * 3 - meta["penalty"] * 4
                if 1 <= int(cand_series[:2]) <= 99:
                    score += 1
                if 0 <= int(cand_series[2:]) <= 99:
                    score += 1
                if score > best_score:
                    best_score = score
                    best_key = key
            if best_key is not None and best_score >= 8:
                return best_key

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "серия" in low and not series:
            tail = ln.split(":", 1)[1] if ":" in ln else ln[low.find("серия") + len("серия") :]
            cand = _ocr_digits_only(tail)
            if len(cand) >= 4:
                series = cand[:4]
            elif i + 1 < len(lines):
                nxt = _ocr_digits_only(lines[i + 1])
                if len(nxt) >= 4:
                    series = nxt[:4]
        if "номер" in low and not number:
            tail = ln.split(":", 1)[1] if ":" in ln else ln[low.find("номер") + len("номер") :]
            cand = _ocr_digits_only(tail)
            if len(cand) >= 6:
                number = cand[-6:] if len(cand) >= 10 else cand[:6]
            elif i + 1 < len(lines):
                nxt = _ocr_digits_only(lines[i + 1])
                if len(nxt) >= 6:
                    number = nxt[-6:] if len(nxt) >= 10 else nxt[:6]
        if series and number:
            combo = f"{series}{number}"
            if issue_digits and issue_digits in combo:
                series = ""
                number = ""
            elif dept_digits and dept_digits in combo:
                series = ""
                number = ""
            else:
                return series, number

    # Частый OCR-вид: "45 03 123456" или "4503-123456".
    normalized = full.translate(_PASSPORT_OCR_DIGIT_MAP)
    candidates: list[tuple[str, str, int]] = []
    for m in re.finditer(
        r"(?<!\d)(\d{2})[\s\-_.]*(\d{2})[\s\-_.]*(\d{6})(?!\d)",
        normalized,
    ):
        candidates.append((f"{m.group(1)}{m.group(2)}", m.group(3), m.start()))
    for m in re.finditer(r"(?<!\d)(\d{4})[\s\-_.]*(\d{6})(?!\d)", normalized):
        candidates.append((m.group(1), m.group(2), m.start()))
    if candidates:
        best: tuple[str, str, int] | None = None
        best_score = -10**9
        for cand_series, cand_number, pos in candidates:
            combo = f"{cand_series}{cand_number}"
            score = 0
            window = normalized[max(0, pos - 80) : pos + 80].lower()
            if re.search(r"(серия|номер|паспорт)", window, re.I):
                score += 4
            if issue_digits and issue_digits in combo:
                score -= 6
            if issue_digits and cand_number.endswith(issue_digits[-4:]):
                score -= 4
            if dept_digits and (dept_digits in combo or cand_number.startswith(dept_digits)):
                score -= 5
            if cand_series == "0000" or cand_number == "000000":
                score -= 5
            if 1 <= int(cand_series[:2]) <= 99:
                score += 1
            if 0 <= int(cand_series[2:]) <= 99:
                score += 1
            if score > best_score:
                best_score = score
                best = (cand_series, cand_number, pos)
        if best is not None and best_score >= 0:
            return best[0], best[1]

    # Слитно 10 цифр.
    compact = re.sub(r"\s+", "", normalized)
    m = re.search(r"(?<!\d)(\d{4})(\d{6})(?!\d)", compact)
    if m:
        series_guess, number_guess = m.group(1), m.group(2)
        combo = f"{series_guess}{number_guess}"
        if (issue_digits and issue_digits in combo) or (dept_digits and dept_digits in combo):
            return "", ""
        return series_guess, number_guess
    m = re.search(r"(?<!\d)(\d{10})(?!\d)", compact)
    if m:
        d = m.group(1)
        return d[:4], d[4:10]
    return "", ""


def _extract_issuing_authority(full: str) -> str:
    lines = [ln.strip() for ln in full.split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        if not re.search(r"(выдан|орган\s*выдач)", ln, re.I):
            continue
        chunk: list[str] = []
        for j in range(i, min(i + 6, len(lines))):
            cur = lines[j]
            if j > i and re.search(
                r"(код|серия|номер|дата\s*выдач|фамили|имя|отчеств|пол|рожд)",
                cur,
                re.I,
            ):
                break
            chunk.append(cur)
            if len(chunk) >= 3:
                break
        s = re.sub(r"\s+", " ", " ".join(chunk)).strip()
        if s:
            return s[:500]
    for ln in lines:
        if re.search(r"ОУФМС|УФМС|МВД|ГУ\s*МВД|отдел\w* УФМС", ln, re.I):
            return ln.strip()[:500]
    return ""


def _extract_birth_place(full: str) -> str:
    def _clean_birth_place(value: str) -> str:
        s = re.sub(r"\s+", " ", value).strip()
        # Убираем короткий OCR-мусор перед маркером населенного пункта: "ви г. ...", "xx город ...".
        s = re.sub(
            r"^[A-Za-zА-ЯЁа-яё]{1,3}\s+(?=(г\.?|город|пос\.?|пгт|с\.|дер\.?)\b)",
            "",
            s,
            flags=re.I,
        )
        # Частый OCR-шум: мусорные символы/буквы перед "г.", "город", "пос." и т.д.
        marker = re.search(r"\b(г\.?|город|пос\.?|пгт|с\.|дер\.?)\b", s, re.I)
        if marker and marker.start() > 0:
            s = s[marker.start() :].strip()
        # Удаляем ведущий неалфавитный мусор.
        s = re.sub(r"^[^А-ЯЁа-яё0-9]+", "", s)
        return s[:500]

    m = re.search(
        r"(?:место\s*рожд)[^\n:]*[:\s]*\n?\s*(.+?)(?=\n|граждан|пол|серия|$)",
        full,
        re.I | re.DOTALL,
    )
    if m:
        return _clean_birth_place(m.group(1))
    lines = [ln.strip() for ln in full.split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        if not re.search(r"\b(г\.?|гор(?:од)?|пос\.?|с\.|дер\.?)\s*[А-ЯЁ]", ln, re.I):
            continue
        # Пропускаем строки про "кем выдан".
        if _FIO_NOISE.search(ln):
            continue
        neighborhood = " ".join(lines[max(0, i - 3) : min(len(lines), i + 2)])
        if re.search(r"(рожд|дата\s*рожд|пол|отчеств|имя|фамили)", neighborhood, re.I) or _DATE.search(
            neighborhood
        ):
            return _clean_birth_place(ln)
    return ""


def parse_passport_ocr_text(ocr_text: str) -> Dict[str, Any]:
    """
    Преобразует плоский текст OCR в словарь полей (как у JSON из LLM).
    """
    text = _norm_space(ocr_text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    full = "\n".join(lines)
    dates = _all_dates(full)

    birth_date = _extract_birth_date_from_lines(lines)
    if not birth_date:
        birth_date = _pick_date_near(
        full,
        ("дата рожд", "д.р.", "родил", "рождения"),
        )
    issue_date = _pick_date_near(
        full,
        ("дата выдач", "выдан", "выдачи"),
    )
    if not birth_date and dates:
        birth_date = dates[0]
    if not issue_date and len(dates) >= 2:
        issue_date = dates[-1] if dates[-1] != birth_date else dates[1]
    elif not issue_date and len(dates) == 1 and not birth_date:
        issue_date = dates[0]
    # Страхуемся от дублирования даты выдачи и рождения, если OCR дал обе даты.
    if birth_date and issue_date and birth_date == issue_date and len(dates) >= 2:
        alternatives = [d for d in dates if d != issue_date]
        if alternatives:
            birth_date = min(alternatives, key=_date_to_key)

    surname, name, patronymic = _extract_fio_labeled(lines)
    gender = _extract_gender(full)
    department_code = _extract_department(full)
    series, number = _extract_series_number(
        full,
        issue_date=issue_date,
        department_code=department_code,
    )
    issuing_authority = _extract_issuing_authority(full)
    birth_place = _extract_birth_place(full)

    notes: List[str] = []
    if not department_code:
        notes.append("код подразделения не найден по шаблону")
    if not (series and number):
        notes.append("серия/номер не извлечены полностью")
    if not (surname and name):
        notes.append("ФИО частично или по эвристике строк")

    return {
        "issuing_authority": issuing_authority,
        "issue_date": issue_date,
        "department_code": department_code,
        "passport_series": series,
        "passport_number": number,
        "surname": surname,
        "name": name,
        "patronymic": patronymic,
        "gender": gender,
        "birth_date": birth_date,
        "birth_place": birth_place,
        "confidence_note": "; ".join(notes) if notes else "OCR Tesseract + правила",
    }
