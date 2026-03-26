import io
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


def test_validate_upload_blocked_document():
    file = make_upload_file("file.pdf", b"12345", "application/pdf")
    with pytest.raises(HTTPException) as exc:
        module.validate_upload(file, b"12345")
    assert exc.value.status_code == 415
