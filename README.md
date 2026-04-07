# Tesseract API (OCR + договоры)

Бэкенд на `FastAPI` для:
- OCR документов (Tesseract и Qwen через Hugging Face Router),
- сканирования паспорта и выписки ЕГРН,
- формирования `.docx` договора из распознанного JSON.

## Основные возможности

- `POST /scan-passport` — скан основной страницы паспорта через Qwen.
- `POST /scan-documents-unified` — единый скан 3 файлов:
  - паспорт (основная страница),
  - страница паспорта с пропиской,
  - выписка ЕГРН.
- `POST /unified-json-to-contract` — генерация договора из объединенного JSON с ручными override-полями.
- `GET /download/{filename}` — скачивание готового `.docx`.

## Быстрый старт

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API по умолчанию: `http://127.0.0.1:8000`

## Переменные окружения

Файл: `.env`

Ключевые настройки:
- `HF_TOKEN` — токен Hugging Face Router.
- `HF_MODEL` — vision-модель для OCR (например, Qwen VL).
- `HF_REQUEST_TIMEOUT_SEC` — таймаут запросов к HF.
- `MAX_FILE_SIZE_MB` — лимит размера изображения.

## Стек

- `fastapi`, `uvicorn`
- `pydantic`
- `openai` (HF Router compatible API)
- `docxtpl` (заполнение шаблона договора)

