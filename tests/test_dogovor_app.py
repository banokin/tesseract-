from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import dogovor


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(dogovor.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def sample_text():
    return """
г. Санкт-Петербург
ДОГОВОР № AB-123/2024
Дата договора: 15.03.2024

Исполнитель:
ФИО исполнителя: Петров Петр Петрович
Адрес регистрации: г. Москва, ул. Ленина, д. 10
Телефон: +7 (495) 111-22-33
Email: info@petrov.ru
ИНН: 123456789012
ОГРН: 123456789012345
БИК: 044525225
Банк: ПАО Сбербанк
р/с: 40702810900000000001
к/с: 30101810400000000225

Заказчик:
ФИО заказчика: Иванов Иван Иванович
Адрес регистрации: г. Санкт-Петербург, Невский пр., д. 20
Телефон заказчика: +7 (999) 123-45-67
Email заказчика: ivanov@example.com

Паспорт заказчика:
Серия: 4501
Номер: 123456
Выдан: ОВД Тверского района г. Москвы
Дата выдачи: 20.05.2015
Код подразделения: 770-001
Место рождения: г. Москва
Дата рождения: 10.10.1990

Объект:
Название объекта: Квартира
Назначение объекта: Жилое помещение
Площадь объекта: 45.6
Адрес объекта: г. Санкт-Петербург, ул. Пушкина, д. 5
Кадастровый номер: 78:12:0001111:222
Основание собственности: Договор купли-продажи
"""


def test_extract_fields(sample_text):
    fields = dogovor.extract_fields(sample_text)

    assert fields["contract_city"] == "Санкт-Петербург"
    assert fields["contract_number"] == "AB-123/2024"
    assert fields["executor_name"] == "Петров Петр Петрович"
    assert fields["customer_fio"] == "Иванов Иван Иванович"
    assert fields["property_cadastral_number"] == "78:12:0001111:222"


def test_resolve_download_path_invalid():
    try:
        dogovor._resolve_download_path("../hack.docx")
        assert False
    except Exception:
        assert True


def test_create_doc_success(monkeypatch, tmp_path):
    output_path = tmp_path / "result.docx"
    template_path = tmp_path / "template.docx"
    template_path.write_text("fake template")

    monkeypatch.setattr(dogovor, "TEMPLATE_PATH", template_path)

    calls = {}

    class FakeDocxTemplate:
        def __init__(self, path):
            calls["path"] = path

        def render(self, context):
            calls["context"] = context

        def save(self, path):
            calls["save_path"] = path
            Path(path).write_text("generated")

    monkeypatch.setattr(dogovor, "DocxTemplate", FakeDocxTemplate)

    data = dogovor.ContractData(
        contract_city="Москва",
        executor_name="Петров Петр Петрович",
    )

    dogovor.create_doc(data, output_path)

    assert calls["path"] == str(template_path)
    assert calls["context"]["contract_city"] == "Москва"
    assert calls["context"]["executor_name"] == "Петров Петр Петрович"
    assert calls["save_path"] == str(output_path)
    assert output_path.exists()


def test_ocr_to_contract_endpoint_success(client, monkeypatch, tmp_path, sample_text):
    monkeypatch.setattr(dogovor, "OUTPUT_DIR", tmp_path)

    async def fake_extract_text_from_upload(file):
        return sample_text

    def fake_create_doc(contract_data, output_path):
        output_path.write_text("fake docx")

    monkeypatch.setattr(dogovor, "extract_text_from_upload", fake_extract_text_from_upload)
    monkeypatch.setattr(dogovor, "create_doc", fake_create_doc)

    response = client.post(
        "/ocr-to-contract",
        files={"file": ("test.png", b"fake image bytes", "image/png")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["message"] == "Договор успешно заполнен"
    assert data["filename"] == "test.png"
    assert data["generated_filename"].startswith("dogovor_")
    assert data["generated_filename"].endswith(".docx")
    assert data["download_url"].startswith("/download/dogovor_")
    assert data["json_data"]["source"] == "tesseract"
    assert data["json_data"]["contract_data"]["executor_name"] == "Петров Петр Петрович"


def test_download_file_success(client, monkeypatch, tmp_path):
    monkeypatch.setattr(dogovor, "OUTPUT_DIR", tmp_path)

    filename = "dogovor_1234567890abcdef1234567890abcdef.docx"
    file_path = tmp_path / filename
    file_path.write_text("test file")

    response = client.get(f"/download/{filename}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def test_download_file_not_found(client, monkeypatch, tmp_path):
    monkeypatch.setattr(dogovor, "OUTPUT_DIR", tmp_path)

    filename = "dogovor_1234567890abcdef1234567890abcdef.docx"
    response = client.get(f"/download/{filename}")

    assert response.status_code == 404