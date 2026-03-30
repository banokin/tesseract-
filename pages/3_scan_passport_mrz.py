import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"
PASSPORT_MRZ_API_URL = f"{API_BASE_URL}/scan-passport-mrz"


def show_api_error(response):
    st.error(f"Ошибка API: {response.status_code}")
    try:
        error_data = response.json()
        st.write(error_data.get("detail", response.text))
    except Exception:
        st.write(response.text)


st.title("Скан паспорта — MRZ")
st.caption("Чтение машиночитаемой зоны паспорта (MRZ) через PassportEye.")

passport_file = st.file_uploader(
    "Загрузите фото паспорта",
    type=["png", "jpg", "jpeg", "webp"],
    key="passport_mrz_uploader",
)

if passport_file is not None and st.button("Сканировать MRZ"):
    files = {
        "file": (
            passport_file.name,
            passport_file.getvalue(),
            passport_file.type,
        )
    }
    try:
        with st.spinner("Сканирую MRZ..."):
            response = requests.post(PASSPORT_MRZ_API_URL, files=files, timeout=180)
        if response.status_code == 200:
            data = response.json()
            st.success("Сканирование MRZ завершено.")
            st.json(data.get("data", {}))
        else:
            show_api_error(response)
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к FastAPI.")
    except requests.exceptions.Timeout:
        st.error("Сервер отвечает слишком долго.")
    except Exception as e:
        st.error(f"Ошибка: {e}")
