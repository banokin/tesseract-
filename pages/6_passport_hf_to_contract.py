import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"
PASSPORT_TO_CONTRACT_HF_API_URL = f"{API_BASE_URL}/scan-passport-to-contract-hf"
PASSPORT_HF_API_URL = f"{API_BASE_URL}/scan-passport"


def show_api_error(response):
    st.error(f"Ошибка API: {response.status_code}")
    try:
        error_data = response.json()
        st.write(error_data.get("detail", response.text))
    except Exception:
        st.write(response.text)


st.title("Паспорт HF -> Договор")
st.caption("Сканирование паспорта через Hugging Face и автоматическое создание договора.")

passport_file = st.file_uploader(
    "Загрузите фото паспорта",
    type=["png", "jpg", "jpeg", "webp"],
    key="passport_to_contract_hf_uploader",
)

if passport_file is None:
    st.stop()

files = {
    "file": (
        passport_file.name,
        passport_file.getvalue(),
        passport_file.type,
    )
}

st.session_state.setdefault("passport_scan_json", None)
st.session_state.setdefault("contract_response_json", None)
st.session_state.setdefault("contract_docx_bytes", None)
st.session_state.setdefault("contract_filename", None)

# Этап 1: отдельное сканирование паспорта
if st.button("1) Сканировать паспорт", key="scan_passport_step"):
    try:
        with st.spinner("Сканирую паспорт через Hugging Face..."):
            response = requests.post(PASSPORT_HF_API_URL, files=files, timeout=180)
        if response.status_code == 200:
            st.session_state["passport_scan_json"] = response.json()
            st.success("Паспорт успешно отсканирован.")
        else:
            show_api_error(response)
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к FastAPI.")
    except requests.exceptions.Timeout:
        st.error("Сервер отвечает слишком долго.")
    except Exception as e:
        st.error(f"Ошибка: {e}")

if st.session_state.get("passport_scan_json"):
    st.subheader("Результат сканирования паспорта (полный JSON ответа API)")
    st.json(st.session_state["passport_scan_json"])

# Этап 2: создание договора после этапа 1
if st.button("2) Создать договор", key="create_contract_step", disabled=st.session_state.get("passport_scan_json") is None):
    try:
        with st.spinner("Создаю договор по данным паспорта..."):
            response = requests.post(PASSPORT_TO_CONTRACT_HF_API_URL, files=files, timeout=180)
        if response.status_code == 200:
            data = response.json()
            st.session_state["contract_response_json"] = data

            download_url = data.get("download_url")
            if download_url:
                file_response = requests.get(f"{API_BASE_URL}{download_url}", timeout=120)
                if file_response.status_code == 200:
                    st.session_state["contract_docx_bytes"] = file_response.content
                    st.session_state["contract_filename"] = data.get("generated_filename", "filled_contract.docx")
                    st.success("Договор успешно создан.")
                else:
                    show_api_error(file_response)
            else:
                st.error("API не вернул ссылку для скачивания договора.")
        else:
            show_api_error(response)
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к FastAPI.")
    except requests.exceptions.Timeout:
        st.error("Сервер отвечает слишком долго.")
    except Exception as e:
        st.error(f"Ошибка: {e}")

if st.session_state.get("contract_docx_bytes"):
    st.download_button(
        label="Скачать заполненный договор (.docx)",
        data=st.session_state["contract_docx_bytes"],
        file_name=st.session_state.get("contract_filename", "filled_contract.docx"),
    )
    if st.button("Показать JSON", key="show_passport_to_contract_json"):
        st.subheader("JSON ответа")
        st.json(st.session_state.get("contract_response_json", {}).get("json_data", {}))
