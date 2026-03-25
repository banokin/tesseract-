import io

import pytest
import pytesseract
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import UnidentifiedImageError
from starlette.datastructures import UploadFile

import ocr as module  # <-- замени на имя своего файла


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

    with pytest.raises(HTTPException):
        module.validate_upload(file, b"")


def test_validate_upload_too_large(monkeypatch):
    file = make_upload_file("image.png", b"123", "image/png")
    monkeypatch.setattr(module, "MAX_FILE_SIZE_BYTES", 3)

    with pytest.raises(HTTPException):
        module.validate_upload(file, b"123")


def test_validate_upload_mp3_by_extension():
    file = make_upload_file("audio.mp3", b"12345", "application/octet-stream")

    with pytest.raises(HTTPException):
        module.validate_upload(file, b"12345")


def test_validate_upload_mp3_by_content_type():
    file = make_upload_file("audio.bin", b"12345", "audio/mpeg")

    with pytest.raises(HTTPException):
        module.validate_upload(file, b"12345")


@pytest.mark.parametrize(
    "filename, content_type",
    [
        ("file.pdf", "image/png"),
        ("file.docx", "image/png"),
        ("file.doc", "image/png"),
        ("file.txt", "image/png"),
    ],
)
def test_validate_upload_blocked_by_extension(filename, content_type):
    file = make_upload_file(filename, b"12345", content_type)

    with pytest.raises(HTTPException):
        module.validate_upload(file, b"12345")


@pytest.mark.parametrize(
    "content_type",
    [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/plain",
        "application/pdf; charset=utf-8",
    ],
)
def test_validate_upload_blocked_by_content_type(content_type):
    file = make_upload_file("image.png", b"12345", content_type)

    with pytest.raises(HTTPException):
        module.validate_upload(file, b"12345")


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


def test_ocr_image_unidentified_image_error(client, monkeypatch):
    monkeypatch.setattr(module, "validate_upload", lambda file, contents: None)

    def fake_img_ocr(contents):
        raise UnidentifiedImageError("cannot identify image file")

    monkeypatch.setattr(module, "img_ocr", fake_img_ocr)

    response = client.post(
        "/ocr",
        files={"file": ("broken.png", b"not-an-image", "image/png")},
    )

    assert response.status_code >= 400


def test_ocr_image_tesseract_error(client, monkeypatch):
    monkeypatch.setattr(module, "validate_upload", lambda file, contents: None)

    def fake_img_ocr(contents):
        raise pytesseract.pytesseract.TesseractError(1, "tesseract failed")

    monkeypatch.setattr(module, "img_ocr", fake_img_ocr)

    response = client.post(
        "/ocr",
        files={"file": ("test.png", b"fake-image", "image/png")},
    )

    assert response.status_code >= 400


def test_ocr_image_validate_upload_http_exception(client, monkeypatch):
    def fake_validate_upload(file, contents):
        raise HTTPException(status_code=415, detail="unsupported file")

    monkeypatch.setattr(module, "validate_upload", fake_validate_upload)

    response = client.post(
        "/ocr",
        files={"file": ("test.pdf", b"fake-pdf", "application/pdf")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "unsupported file"


def test_ocr_image_unexpected_exception(client, monkeypatch):
    monkeypatch.setattr(module, "validate_upload", lambda file, contents: None)

    def fake_img_ocr(contents):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(module, "img_ocr", fake_img_ocr)

    response = client.post(
        "/ocr",
        files={"file": ("test.png", b"fake-image", "image/png")},
    )

    assert response.status_code >= 400