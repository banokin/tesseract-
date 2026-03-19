import streamlit as st
import requests

st.set_page_config(page_title="OCR через Tesseract", page_icon="📄")

st.title("Распознавание текста с изображения")
st.write("Загрузи фото, и Tesseract вытащит из него текст.")

uploaded_file = st.file_uploader(
    "Выбери изображение",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # По умолчанию `width=None` означает: использовать естественную ширину
    # изображения, но не превышать ширину колонки (и при этом не использовать
    # deprecated `use_column_width`).
    st.image(uploaded_file, caption="Загруженное изображение")

    if st.button("Распознать текст"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        try:
            response = requests.post("http://127.0.0.1:8000/ocr", files=files)

            if response.status_code == 200:
                data = response.json()
                st.success("Текст успешно распознан")
                st.subheader("Результат:")
                st.text_area("Текст", data["text"], height=300)
            else:
                st.error(f"Ошибка API: {response.status_code}")
                st.write(response.text)

        except Exception as e:
            st.error(f"Не удалось подключиться к FastAPI: {e}")