import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"
PASSPORT_HF_API_URL = f"{API_BASE_URL}/scan-passport"


def show_api_error(response):
    st.error(f"Ошибка API: {response.status_code}")
    try:
        error_data = response.json()
        st.write(error_data.get("detail", response.text))
    except Exception:
        st.write(response.text)


st.title("Скан паспорта — Hugging Face")
st.caption("Извлечение структурированных полей паспорта через модель Hugging Face.")

st.session_state.setdefault("hf_passport_scan_json", None)

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
            response = requests.post(PASSPORT_HF_API_URL, files=files, timeout=180)
        if response.status_code == 200:
            st.session_state["hf_passport_scan_json"] = response.json()
            st.success("Сканирование Hugging Face завершено.")
        else:
            st.session_state["hf_passport_scan_json"] = None
            show_api_error(response)
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к FastAPI.")
    except requests.exceptions.Timeout:
        st.error("Сервер отвечает слишком долго.")
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
