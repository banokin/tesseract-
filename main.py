from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dogovor import router as dogovor_router
from ocr import router as ocr_router

app = FastAPI(
    title="Сканер для договоров",
)

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


app.include_router(ocr_router)
app.include_router(dogovor_router)
