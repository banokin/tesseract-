import os
import subprocess
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Разрешаем запросы от Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "OCR API работает"}

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    # читаем загруженный файл в память
    contents = await file.read()

    # Чтобы избежать зависимости от `pytesseract` (и его тяжёлых пакетов),
    # вызываем бинарник `tesseract` напрямую.
    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
        ext = ".png"

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input{ext}")
        output_base = os.path.join(tmpdir, "result")

        with open(input_path, "wb") as f:
            f.write(contents)

        try:
            proc = subprocess.run(
                ["tesseract", input_path, output_base, "-l", "rus+eng"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail="Бинарник `tesseract` не найден") from e

        # Текст записывается в файл `<output_base>.txt`
        txt_path = f"{output_base}.txt"
        if not os.path.exists(txt_path) or proc.returncode != 0:
            # Сохраняем stderr для отладки, но не раздуваем ответ.
            detail = proc.stderr.strip() or f"tesseract exit code: {proc.returncode}"
            raise HTTPException(status_code=500, detail=detail[:1000])

        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    return {
        "filename": file.filename,
        "text": text
    }