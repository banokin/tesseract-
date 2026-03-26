import io

import pytest
import pytesseract
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

import ocr as module


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(module.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


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


def test_validate_upload_empty_file():
    file = make_upload_file("image.png", b"", "image/png")
    try:
        module.validate_upload(file, b"")
        assert False
    except HTTPException as exc:
        assert exc.status_code == 400


def test_validate_upload_blocked_document():
    file = make_upload_file("file.pdf", b"12345", "application/pdf")
    try:
        module.validate_upload(file, b"12345")
        assert False
    except HTTPException as exc:
        assert exc.status_code == 415


def test_img_ocr_calls_tesseract(monkeypatch):
    calls = {}

    class FakeImage:
        pass

    def fake_open(buffer):
        calls["image_open_called"] = True
        assert isinstance(buffer, io.BytesIO)
        return FakeImage()

    def fake_image_to_string(image, lang):
        calls["ocr_called"] = True
        calls["lang"] = lang
        assert isinstance(image, FakeImage)
        return "распознанный текст"

    monkeypatch.setattr(module.Image, "open", fake_open)
    monkeypatch.setattr(module.pytesseract, "image_to_string", fake_image_to_string)

    result = module.img_ocr(b"fake-image-bytes")

    assert result == "распознанный текст"
    assert calls["image_open_called"] is True
    assert calls["ocr_called"] is True
    assert calls["lang"] == "rus+eng"


def test_ocr_image_success(client, monkeypatch):
    monkeypatch.setattr(module, "validate_upload", lambda file, contents: None)
    monkeypatch.setattr(module, "img_ocr", lambda contents: "hello from ocr")

    response = client.post(
        "/ocr",
        files={"file": ("test.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "test.png",
        "text": "hello from ocr",
    }


def test_ocr_image_tesseract_error(client, monkeypatch):
    monkeypatch.setattr(module, "validate_upload", lambda file, contents: None)

    def fake_img_ocr(contents):
        raise pytesseract.pytesseract.TesseractError(1, "tesseract failed")

    monkeypatch.setattr(module, "img_ocr", fake_img_ocr)

    response = client.post(
        "/ocr",
        files={"file": ("test.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 500
    assert "Ошибка Tesseract" in response.json()["detail"]