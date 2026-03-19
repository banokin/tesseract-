import streamlit as st
import requests

st.set_page_config(page_title="OCR через Tesseract", page_icon="📄")

API_URL = "http://127.0.0.1:8000/ocr"

st.title("Распознавание текста с изображения")
st.write("Загрузи фото, и Tesseract вытащит из него текст.")

uploaded_file = st.file_uploader(
    "Выбери изображение",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Загруженное изображение")

    if st.button("Распознать текст"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        try:
            with st.spinner("Распознаю текст..."):
                response = requests.post(API_URL, files=files, timeout=60)

            if response.status_code == 200:
                data = response.json()
                st.success("Текст успешно распознан")
                st.subheader("Результат:")
                st.text_area("Текст", data.get("text", ""), height=300)
            else:
                st.error(f"Ошибка API: {response.status_code}")
                try:
                    error_data = response.json()
                    st.write(error_data.get("detail", response.text))
                except:
                    st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к FastAPI. Проверь, запущен ли сервер.")
        except requests.exceptions.Timeout:
            st.error("Сервер отвечает слишком долго.")
        except Exception as e:
            st.error(f"Ошибка: {e}")