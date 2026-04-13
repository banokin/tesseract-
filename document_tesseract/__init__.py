"""
Документы (паспорт РФ, выписка ЕГРН): Tesseract OCR + правила разбора без LLM.

- ``router`` — эндпоинты
  ``POST /scan-passport-tesseract-structured``,
  ``POST /scan-egrn-tesseract-structured``;
- ``parse_passport_ocr_text``, ``parse_egrn_ocr_text`` — разбор готового текста OCR;
- ``EgrnScanTesseractResponse`` — модель ответа по ЕГРН.
"""

from .egrn_parser import parse_egrn_ocr_text
from .passport_parser import parse_passport_ocr_text
from .routes import EgrnScanTesseractResponse, router

__all__ = [
    "router",
    "parse_passport_ocr_text",
    "parse_egrn_ocr_text",
    "EgrnScanTesseractResponse",
]
