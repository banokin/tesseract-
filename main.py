from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from huggin_face_scan.dogovor_new import router as dogovor_new_router
from document_tesseract import router as document_tesseract_router
from huggin_face_scan.scan_passport_hf import router as passport_router
from huggin_face_scan.scan_passport_hf_two_models import router as passport_two_models_router
from huggin_face_scan.scan_passport_paspread import router as passport_paspread_router
from huggin_face_scan.scan_passport_russian_docs_ocr import router as passport_russian_docs_ocr_router
from huggin_face_scan.scan_passport_deepseek_qwen import router as passport_deepseek_qwen_router

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


app.include_router(dogovor_new_router)
app.include_router(passport_router)
app.include_router(passport_two_models_router)
app.include_router(passport_paspread_router)
app.include_router(passport_russian_docs_ocr_router)
app.include_router(passport_deepseek_qwen_router)
app.include_router(document_tesseract_router)
