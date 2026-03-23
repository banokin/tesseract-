from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from docx import Document
import pytesseract
import io
import asyncio
import re
import uuid
from pathlib import Path


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "Новый договор.docx"
OUTPUT_DIR = BASE_DIR / "generated_docs"
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "OCR API работает"}


def img_ocr(contents: bytes) -> str:
    image = Image.open(io.BytesIO(contents))
    return pytesseract.image_to_string(image, lang="rus+eng")


def extract_fields(text: str) -> dict:
    """
    Пример извлечения полей из OCR текста.
    Регулярки потом подстроишь под реальный формат сканов.
    """
    def find(pattern: str, default: str = "") -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else default

    fields = {
        "Укажите номер договора": find(r"договор[а-я\s№]*[:\-]?\s*([A-Za-zА-Яа-я0-9\-\/]+)"),
        "Укажите дату договора": find(r"дата[а-я\s]*[:\-]?\s*(\d{2}\.\d{2}\.\d{4})"),
        "Укажите ФИО": find(r"фио[:\-]?\s*([А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z\s\-]+)"),
        "Укажите адрес регистрации": find(r"адрес регистрации[:\-]?\s*(.+)"),
        "Укажите номер телефона": find(r"(?:телефон|тел\.?)[:\-]?\s*([\+\d\-\(\)\s]{6,})"),
        "Укажите адрес почты": find(r"(?:e-?mail|почта)[:\-]?\s*([^\s]+@[^\s]+)"),
        "укажите серию паспорта": find(r"серия[:\-]?\s*(\d{4})"),
        "укажите номер паспорта": find(r"номер[:\-]?\s*(\d{6})"),
        "Укажите орган, выдавший паспорт": find(r"выдан[:\-]?\s*(.+)"),
        "укажите дату выдачи паспорта": find(r"дата выдачи[:\-]?\s*(\d{2}\.\d{2}\.\d{4})"),
        "укажите код подразделения": find(r"код подразделения[:\-]?\s*([\d\-]+)"),
        "укажите место рождения": find(r"место рождения[:\-]?\s*(.+)"),
        "укажите дату рождения": find(r"дата рождения[:\-]?\s*(\d{2}\.\d{2}\.\d{4})"),
        "Выберите название объекта": find(r"объект[:\-]?\s*(.+)"),
        "Укажите назначение объекта": find(r"назначение[:\-]?\s*(.+)"),
        "Укажите площадь объекта м2": find(r"площадь[:\-]?\s*([\d\.,]+)"),
        "Укажите адрес объекта": find(r"адрес объекта[:\-]?\s*(.+)"),
        "Укажите кадастровый номер объекта": find(r"кадастровый номер[:\-]?\s*([0-9:\-]+)"),
        "Укажите документ-основание права собственности": find(r"основание права собственности[:\-]?\s*(.+)"),
    }

    return fields


def replace_text_in_paragraph(paragraph, replacements: dict):
    for old_text, new_text in replacements.items():
        if old_text in paragraph.text:
            for run in paragraph.runs:
                run.text = run.text.replace(old_text, new_text)


def replace_text_in_table(table, replacements: dict):
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                replace_text_in_paragraph(paragraph, replacements)
            for nested_table in cell.tables:
                replace_text_in_table(nested_table, replacements)


def fill_contract(template_path: str, output_path: str, fields: dict):
    doc = Document(template_path)

    replacements = {}
    for key, value in fields.items():
        if value:
            replacements[key] = value

    for paragraph in doc.paragraphs:
        replace_text_in_paragraph(paragraph, replacements)

    for table in doc.tables:
        replace_text_in_table(table, replacements)

    doc.save(output_path)


def _validate_upload(contents: bytes):
    if not contents:
        raise HTTPException(status_code=400, detail="Файл пустой")


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        _validate_upload(contents)
        text = await asyncio.to_thread(img_ocr, contents)

        return {
            "filename": file.filename,
            "text": text
        }
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=422,
            detail="Не удалось распознать изображение. Проверьте формат файла.",
        )
    except pytesseract.pytesseract.TesseractError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка Tesseract: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка OCR: {str(e)}")


@app.post("/ocr-to-contract")
async def ocr_to_contract(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        _validate_upload(contents)

        # 1. OCR
        text = await asyncio.to_thread(img_ocr, contents)

        # 2. Извлечение полей
        fields = await asyncio.to_thread(extract_fields, text)

        # 3. Проверка шаблона
        template_file = TEMPLATE_PATH
        if not template_file.exists():
            raise HTTPException(status_code=500, detail="Файл шаблона договора не найден")

        # 4. Генерация нового файла
        output_name = f"filled_contract_{uuid.uuid4().hex}.docx"
        output_path = OUTPUT_DIR / output_name

        await asyncio.to_thread(
            fill_contract,
            str(template_file),
            str(output_path),
            fields
        )

        return {
            "message": "Договор успешно заполнен",
            "filename": file.filename,
            "ocr_text": text,
            "fields": fields,
            "download_url": f"/download/{output_name}"
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=422,
            detail="Не удалось распознать изображение. Проверьте формат файла.",
        )
    except pytesseract.pytesseract.TesseractError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка Tesseract: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки документа: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )