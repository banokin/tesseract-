import io
import pathlib
import sys
import types

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

if "pytesseract" not in sys.modules:
    fake_module = types.SimpleNamespace()
    fake_module.image_to_string = lambda *_args, **_kwargs: "stub text"
    fake_module.pytesseract = types.SimpleNamespace(
        TesseractError=type("TesseractError", (Exception,), {})
    )
    sys.modules["pytesseract"] = fake_module

import main


@pytest.fixture()
def client():
    return TestClient(main.app)


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (10, 10), color="white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_root_returns_status_message(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "OCR API работает"}


def test_ocr_success(client):
    response = client.post(
        "/ocr",
        files={"file": ("ok.png", _make_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "ok.png"
    assert "text" in body
