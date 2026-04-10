from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="OCR и договор", page_icon="📄")

API_BASE_URL = "http://127.0.0.1:8000"
OCR_TO_CONTRACT_API_URL = f"{API_BASE_URL}/ocr-to-contract"
MAX_FILE_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GB

_BLOCKED_EXTENSIONS = {".mp3", ".pdf", ".docx", ".doc", ".txt"}
_BLOCKED_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
}


def blocked_upload_error_message(ext: str, upload_type: str) -> str:
    if ext == ".mp3" or upload_type in {"audio/mpeg", "audio/mp3"}:
        return "Файлы MP3 не поддерживаются. Загрузите изображение PNG/JPG/JPEG."
    return "Файлы PDF, DOC/DOCX и TXT для OCR не поддерживаются — загрузите фото PNG, JPG или JPEG."

st.title("Договор по фотографии")
st.write(
    "Загрузите фото — договор заполнится автоматически (OCR + шаблон). "
    "Ниже появится кнопка скачивания готового файла .docx."
)
st.caption(
    "Ограничения: файлы от 1 ГБ и больше не принимаются. "
    "MP3, PDF, DOC/DOCX, TXT для OCR не подходят — Tesseract обрабатывает только изображения (PNG, JPG, JPEG)."
)


def show_api_error(response):
    st.error(f"Ошибка API: {response.status_code}")
    try:
        error_data = response.json()
        st.write(error_data.get("detail", response.text))
    except Exception:
        st.write(response.text)


uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.stop()

uploaded_raw = uploaded_file.getvalue()
filename_lower = (uploaded_file.name or "").lower()
if len(uploaded_raw) >= MAX_FILE_SIZE_BYTES:
    st.error("Файл слишком большой: 1 ГБ и больше не принимаются.")
    st.stop()

ext = Path(filename_lower).suffix

upload_type = (uploaded_file.type or "").split(";")[0].strip().lower()

if ext in _BLOCKED_EXTENSIONS or upload_type in _BLOCKED_TYPES:
    st.error(blocked_upload_error_message(ext, upload_type))
    st.stop()

st.image(uploaded_file, caption="Загруженное изображение")

err = None
docx = None
filename = None
ocr_text = ""
json_data = None

with st.spinner("Распознаю текст и заполняю договор..."):
    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }
        response = requests.post(
            OCR_TO_CONTRACT_API_URL,
            files=files,
            timeout=120,
        )
        if response.status_code != 200:
            err = response
        else:
            data = response.json()
            download_url = data.get("download_url")
            if not download_url:
                err = "API не вернул ссылку на файл договора."
            else:
                file_response = requests.get(
                    f"{API_BASE_URL}{download_url}",
                    timeout=120,
                )
                if file_response.status_code == 200:
                    docx = file_response.content
                    filename = data.get("generated_filename") or download_url.rsplit("/", 1)[-1]
                    ocr_text = data.get("ocr_text", "")
                    json_data = data.get("json_data")
                else:
                    err = file_response
    except requests.exceptions.ConnectionError:
        err = "connection"
    except requests.exceptions.Timeout:
        err = "timeout"
    except Exception as e:
        err = str(e)

if err == "connection":
    st.error("Не удалось подключиться к FastAPI. Запустите сервер: `uvicorn main:app --reload`")
elif err == "timeout":
    st.error("Сервер отвечает слишком долго.")
elif isinstance(err, str) and err:
    st.error(err)
elif hasattr(err, "status_code"):
    show_api_error(err)
elif docx:
    st.success("Договор сформирован по данным с фото.")
    st.download_button(
        label="Скачать заполненный договор (.docx)",
        data=docx,
        file_name=filename,
    )
    if json_data and st.button("Показать JSON после сканирования"):
        st.subheader("JSON для заполнения договора")
        st.json(json_data)
    if ocr_text:
        with st.expander("Распознанный текст (OCR)"):
            st.text_area("Текст", ocr_text, height=240, disabled=True)

