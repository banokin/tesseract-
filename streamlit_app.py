import hashlib
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="OCR и договор", page_icon="📄")

API_BASE_URL = "http://127.0.0.1:8000"
OCR_API_URL = f"{API_BASE_URL}/ocr"
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


def file_cache_key(uploaded_file) -> str:
    """Стабильный ключ по имени и содержимому (размер + хэш начала файла)."""
    raw = uploaded_file.getvalue()
    h = hashlib.sha256(raw[:65536]).hexdigest()[:16]
    return f"{uploaded_file.name}|{len(raw)}|{h}"


uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    if "contract_cache" in st.session_state:
        st.session_state.contract_cache = {}
    st.stop()

uploaded_raw = uploaded_file.getvalue()
filename_lower = (uploaded_file.name or "").lower()
if len(uploaded_raw) >= MAX_FILE_SIZE_BYTES:
    st.error("Файл слишком большой: 1 ГБ и больше не принимаются.")
    st.stop()
ext = Path(filename_lower).suffix
upload_type = (uploaded_file.type or "").split(";")[0].strip().lower()
if ext in _BLOCKED_EXTENSIONS or upload_type in _BLOCKED_TYPES:
    if ext == ".mp3" or upload_type in {"audio/mpeg", "audio/mp3"}:
        st.error("Файлы MP3 не поддерживаются. Загрузите изображение PNG/JPG/JPEG.")
    else:
        st.error(
            "Файлы PDF, DOC/DOCX и TXT для OCR не поддерживаются — загрузите фото PNG, JPG или JPEG."
        )
    st.stop()

st.image(uploaded_file, caption="Загруженное изображение")

cache = st.session_state.setdefault("contract_cache", {})
key = file_cache_key(uploaded_file)

if cache.get("key") != key:
    cache["key"] = key
    cache.pop("docx", None)
    cache.pop("filename", None)
    cache.pop("err", None)
    cache.pop("ocr_text", None)
    cache.pop("json_data", None)

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
                cache["err"] = response
            else:
                data = response.json()
                download_url = data.get("download_url")
                if not download_url:
                    cache["err"] = "API не вернул ссылку на файл договора."
                else:
                    file_response = requests.get(
                        f"{API_BASE_URL}{download_url}",
                        timeout=120,
                    )
                    if file_response.status_code != 200:
                        cache["err"] = file_response
                    else:
                        cache["docx"] = file_response.content
                        cache["filename"] = (
                            data.get("generated_filename")
                            or download_url.rsplit("/", 1)[-1]
                            or "filled_contract.docx"
                        )
                        cache["ocr_text"] = data.get("ocr_text", "")
                        cache["json_data"] = data.get("json_data")
        except requests.exceptions.ConnectionError:
            cache["err"] = "connection"
        except requests.exceptions.Timeout:
            cache["err"] = "timeout"
        except Exception as e:
            cache["err"] = str(e)

if cache.get("err") == "connection":
    st.error("Не удалось подключиться к FastAPI. Запустите сервер: `uvicorn dogovor:app --reload`")
elif cache.get("err") == "timeout":
    st.error("Сервер отвечает слишком долго.")
elif isinstance(cache.get("err"), str) and cache["err"]:
    st.error(cache["err"])
elif hasattr(cache.get("err"), "status_code"):
    show_api_error(cache["err"])
elif cache.get("docx"):
    st.success("Договор сформирован по данным с фото.")
    st.download_button(
        label="Скачать заполненный договор (.docx)",
        data=cache["docx"],
        file_name=cache["filename"],
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    if cache.get("json_data") and st.button("Показать JSON после сканирования"):
        st.subheader("JSON для заполнения договора")
        st.json(cache["json_data"])
    if cache.get("ocr_text"):
        with st.expander("Распознанный текст (OCR)"):
            st.text_area("Текст", cache["ocr_text"], height=240, disabled=True)

if st.button("Только распознать текст (без договора)"):
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }
    try:
        with st.spinner("Распознаю текст..."):
            response = requests.post(OCR_API_URL, files=files, timeout=60)
        if response.status_code == 200:
            data = response.json()
            st.subheader("Результат OCR")
            st.text_area("Текст", data.get("text", ""), height=300)
        else:
            show_api_error(response)
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к FastAPI.")
    except requests.exceptions.Timeout:
        st.error("Сервер отвечает слишком долго.")
    except Exception as e:
        st.error(f"Ошибка: {e}")
