from fastapi import HTTPException

import exceptions as module  # <-- замени при необходимости


def assert_http_exception(exc: HTTPException, status_code: int, detail: str):
    assert isinstance(exc, HTTPException)
    assert exc.status_code == status_code
    assert exc.detail == detail


def test_empty_file_exception():
    exc = module.empty_file_exception()

    assert_http_exception(
        exc,
        400,
        "Файл пустой",
    )


def test_file_too_large_exception():
    exc = module.file_too_large_exception()

    assert_http_exception(
        exc,
        413,
        "Файл слишком большой: 1 ГБ и больше не принимаются.",
    )


def test_unsupported_mp3_exception():
    exc = module.unsupported_mp3_exception()

    assert_http_exception(
        exc,
        415,
        "Файлы MP3 не поддерживаются.",
    )


def test_unsupported_ocr_document_exception():
    exc = module.unsupported_ocr_document_exception()

    assert isinstance(exc, HTTPException)
    assert exc.status_code == 415
    assert exc.detail == module.OCR_UNSUPPORTED_DETAIL


def test_image_not_recognized_exception():
    exc = module.image_not_recognized_exception()

    assert_http_exception(
        exc,
        422,
        "Не удалось распознать изображение. Проверьте формат файла.",
    )


def test_tesseract_exception():
    error = Exception("tesseract failed")
    exc = module.tesseract_exception(error)

    assert_http_exception(
        exc,
        500,
        "Ошибка Tesseract: tesseract failed",
    )


def test_ocr_exception():
    error = Exception("unexpected ocr error")
    exc = module.ocr_exception(error)

    assert_http_exception(
        exc,
        500,
        "Ошибка OCR: unexpected ocr error",
    )


def test_invalid_filename_exception():
    exc = module.invalid_filename_exception()

    assert_http_exception(
        exc,
        400,
        "Некорректное имя файла.",
    )


def test_invalid_filepath_exception():
    exc = module.invalid_filepath_exception()

    assert_http_exception(
        exc,
        400,
        "Некорректный путь к файлу.",
    )


def test_template_not_found_exception():
    exc = module.template_not_found_exception()

    assert_http_exception(
        exc,
        500,
        "Файл шаблона договора не найден",
    )


def test_document_processing_exception():
    error = Exception("docx render failed")
    exc = module.document_processing_exception(error)

    assert_http_exception(
        exc,
        500,
        "Ошибка обработки документа: docx render failed",
    )


def test_file_not_found_exception():
    exc = module.file_not_found_exception()

    assert_http_exception(
        exc,
        404,
        "Файл не найден",
    )


def test_ocr_unsupported_detail_constant():
    assert isinstance(module.OCR_UNSUPPORTED_DETAIL, str)
    assert "PNG" in module.OCR_UNSUPPORTED_DETAIL
    assert "JPG" in module.OCR_UNSUPPORTED_DETAIL
    assert "JPEG" in module.OCR_UNSUPPORTED_DETAIL
    assert "PDF" in module.OCR_UNSUPPORTED_DETAIL
    assert "DOCX" in module.OCR_UNSUPPORTED_DETAIL
    assert "TXT" in module.OCR_UNSUPPORTED_DETAIL