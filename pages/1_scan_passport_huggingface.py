import os

import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"
PASSPORT_HF_API_URL = f"{API_BASE_URL}/scan-passport"
PASSPORT_TO_CONTRACT_HF_API_URL = f"{API_BASE_URL}/scan-passport-to-contract-hf"

# Согласовано с HF_REQUEST_TIMEOUT_SEC в .env на бэкенде (запас на сеть и парсинг)
_HF_SEC = int(os.environ.get("HF_REQUEST_TIMEOUT_SEC", "90"))
HTTP_TIMEOUT = (10, _HF_SEC + 45)


def show_api_error(response):
    st.error(f"Ошибка API: {response.status_code}")
    try:
        error_data = response.json()
        st.write(error_data.get("detail", response.text))
    except Exception:
        st.write(response.text)


st.title("Скан паспорта — Hugging Face")
st.caption(
    "Извлечение структурированных полей паспорта через модель Hugging Face. "
    f"Ожидание ответа HF — до ~{_HF_SEC} с (настройка HF_REQUEST_TIMEOUT_SEC)."
)

st.session_state.setdefault("hf_passport_scan_json", None)
st.session_state.setdefault("hf_passport_upload_snapshot", None)
st.session_state.setdefault("hf_contract_docx_bytes", None)
st.session_state.setdefault("hf_contract_filename", None)

passport_file = st.file_uploader(
    "Загрузите фото паспорта",
    type=["png", "jpg", "jpeg", "webp"],
    key="passport_hf_uploader",
)

if passport_file is not None and st.button("Сканировать (Hugging Face)"):
    files = {
        "file": (
            passport_file.name,
            passport_file.getvalue(),
            passport_file.type,
        )
    }
    try:
        with st.spinner("Сканирую паспорт через Hugging Face..."):
            response = requests.post(PASSPORT_HF_API_URL, files=files, timeout=HTTP_TIMEOUT)
        if response.status_code == 200:
            st.session_state["hf_passport_scan_json"] = response.json()
            st.session_state["hf_passport_upload_snapshot"] = (
                passport_file.name,
                passport_file.getvalue(),
                passport_file.type,
            )
            st.session_state["hf_contract_docx_bytes"] = None
            st.session_state["hf_contract_filename"] = None
            st.success("Сканирование Hugging Face завершено.")
        else:
            st.session_state["hf_passport_scan_json"] = None
            show_api_error(response)
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к FastAPI.")
    except requests.exceptions.Timeout:
        st.error(
            f"Превышено время ожидания (бэкенд HF ~{_HF_SEC} с). "
            "Проверьте, что uvicorn запущен; при перегрузке HF увеличьте HF_REQUEST_TIMEOUT_SEC в .env."
        )
    except Exception as e:
        st.error(f"Ошибка: {e}")

data = st.session_state.get("hf_passport_scan_json")
if data:
    st.subheader("Данные паспорта (JSON)")
    st.json(data.get("data", {}))
    if st.button("Показать полный JSON ответа API", key="show_hf_full_json"):
        st.json(data)
    raw_text = data.get("raw_text", "")
    if raw_text:
        with st.expander("Сырой текст ответа модели"):
            st.text_area("Текст", raw_text, height=220, disabled=True)

    snapshot = st.session_state.get("hf_passport_upload_snapshot")
    can_build = snapshot is not None

    st.subheader("Договор")
    if st.button(
        "Сформировать договор (.docx) по данным паспорта",
        key="hf_build_contract",
        disabled=not can_build,
        help="Нужен успешный скан и тот же файл в сессии (пересканируйте при сбое).",
    ):
        name, body, ctype = snapshot
        files = {"file": (name, body, ctype)}
        try:
            with st.spinner("Создаю договор на сервере..."):
                response = requests.post(
                    PASSPORT_TO_CONTRACT_HF_API_URL, files=files, timeout=HTTP_TIMEOUT
                )
            if response.status_code == 200:
                payload = response.json()
                download_url = payload.get("download_url")
                if not download_url:
                    st.error("API не вернул ссылку для скачивания договора.")
                else:
                    file_response = requests.get(
                        f"{API_BASE_URL}{download_url}", timeout=120
                    )
                    if file_response.status_code == 200:
                        st.session_state["hf_contract_docx_bytes"] = file_response.content
                        st.session_state["hf_contract_filename"] = payload.get(
                            "generated_filename", "dogovor.docx"
                        )
                        st.success("Договор сформирован — скачайте файл ниже.")
                    else:
                        show_api_error(file_response)
            else:
                show_api_error(response)
        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к FastAPI.")
        except requests.exceptions.Timeout:
            st.error(
                f"Превышено время ожидания (HF ~{_HF_SEC} с). Повторите или увеличьте HF_REQUEST_TIMEOUT_SEC."
            )
        except Exception as e:
            st.error(f"Ошибка: {e}")

    docx_bytes = st.session_state.get("hf_contract_docx_bytes")
    if docx_bytes:
        st.download_button(
            label="Скачать договор (.docx)",
            data=docx_bytes,
            file_name=st.session_state.get("hf_contract_filename", "dogovor.docx"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="hf_download_contract_docx",
        )
