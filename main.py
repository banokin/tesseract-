from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import io

app = FastAPI()

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
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(image, lang="rus+eng")

        return {
            "filename": file.filename,
            "text": text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка OCR: {str(e)}")