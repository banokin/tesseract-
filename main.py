from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from huggin_face_scan.dogovor_new import router as dogovor_new_router
from huggin_face_scan.scan_passport_hf import router as passport_router

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
