from fastapi import HTTPException

OCR_UNSUPPORTED_DETAIL = (
    "Для OCR поддерживаются только изображения (PNG, JPG, JPEG). "
    "Файлы PDF, DOCX и TXT Tesseract не сканирует как документы — "
    "конвертируйте страницу в изображение или извлеките текст другими средствами."
)


def empty_file_exception() -> HTTPException:
    return HTTPException(status_code=400, detail="Файл пустой")


def file_too_large_exception() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail="Файл слишком большой: 1 ГБ и больше не принимаются.",
    )


def unsupported_mp3_exception() -> HTTPException:
    return HTTPException(status_code=415, detail="Файлы MP3 не поддерживаются.")


def unsupported_ocr_document_exception() -> HTTPException:
    return HTTPException(status_code=415, detail=OCR_UNSUPPORTED_DETAIL)


def image_not_recognized_exception() -> HTTPException:
    return HTTPException(
        status_code=422,
        detail="Не удалось распознать изображение. Проверьте формат файла.",
    )


def tesseract_exception(error: Exception) -> HTTPException:
    return HTTPException(status_code=500, detail=f"Ошибка Tesseract: {str(error)}")


def ocr_exception(error: Exception) -> HTTPException:
    return HTTPException(status_code=500, detail=f"Ошибка OCR: {str(error)}")


def invalid_filename_exception() -> HTTPException:
    return HTTPException(status_code=400, detail="Некорректное имя файла.")


def invalid_filepath_exception() -> HTTPException:
    return HTTPException(status_code=400, detail="Некорректный путь к файлу.")


def template_not_found_exception() -> HTTPException:
    return HTTPException(status_code=500, detail="Файл шаблона договора не найден")


def document_processing_exception(error: Exception) -> HTTPException:
    return HTTPException(status_code=500, detail=f"Ошибка обработки документа: {str(error)}")


def file_not_found_exception() -> HTTPException:
    return HTTPException(status_code=404, detail="Файл не найден")
