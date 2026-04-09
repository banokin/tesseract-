import io
import asyncio
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import ocr as module


def make_upload_file(
    filename: str,
    content: bytes,
    content_type: str = "image/png",
) -> UploadFile:
    file = UploadFile(filename=filename, file=io.BytesIO(content))
    file.headers = {"content-type": content_type}
    return file


def test_validate_upload_ok():
    file = make_upload_file("image.png", b"12345", "image/png")

    module.validate_upload(file, b"12345")


def test_validate_upload_pdf_ok():
    file = make_upload_file("file.pdf", b"12345", "application/pdf")

    module.validate_upload(file, b"12345")


def test_validate_upload_blocked_document():
    file = make_upload_file(
        "file.docx",
        b"12345",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    with pytest.raises(HTTPException) as exc:
        module.validate_upload(file, b"12345")
    assert exc.value.status_code == 415


def test_extract_text_from_upload_pdf_converts_first_page(monkeypatch):
    converted_png = b"png-bytes"

    def fake_convert(pdf_bytes: bytes) -> bytes:
        assert pdf_bytes == b"pdf-bytes"
        return converted_png

    def fake_img_ocr(contents: bytes) -> str:
        assert contents == converted_png
        return "ok"

    monkeypatch.setattr(module, "convert_pdf_first_page_to_png", fake_convert)
    monkeypatch.setattr(module, "img_ocr", fake_img_ocr)

    file = make_upload_file("extract.pdf", b"pdf-bytes", "application/pdf")
    result = asyncio.run(module.extract_text_from_upload(file))
    assert result == "ok"
