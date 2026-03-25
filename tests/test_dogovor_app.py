import io
from pathlib import Path

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

# ВАЖНО:
# замени `dogovor` на имя своего модуля/файла, где лежит этот код
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


def test_find_returns_match():
    text = "Дата договора: 15.03.2024"
    result = dogovor._find(r"дата договора[:\-]?\s*(\d{2}\.\d{2}\.\d{4})", text)
    assert result == "15.03.2024"


def test_find_returns_default_when_no_match():
    text = "Нет нужного поля"
    result = dogovor._find(r"ИНН[:\-]?\s*(\d+)", text, default="не найдено")
    assert result == "не найдено"


def test_executor_section(sample_text):
    result = dogovor._executor_section(sample_text)
    assert "ФИО исполнителя" in result
    assert "ФИО заказчика" not in result


def test_customer_section(sample_text):
    result = dogovor._customer_section(sample_text)
    assert "ФИО заказчика" in result
    assert "ФИО исполнителя" not in result


def test_extract_fields(sample_text):
    fields = dogovor.extract_fields(sample_text)

    assert fields["contract_city"] == "Санкт-Петербург"
    assert fields["contract_number"] == "AB-123/2024"
    assert fields["contract_date"] == "15.03.2024"

    assert fields["executor_name"] == "Петров Петр Петрович"
    assert fields["executor_address"] == "г. Москва, ул. Ленина, д. 10"
    assert fields["executor_phone"] == "+7 (495) 111-22-33"
    assert fields["executor_email"] == "info@petrov.ru"
    assert fields["executor_inn"] == "123456789012"
    assert fields["executor_ogrn"] == "123456789012345"
    assert fields["executor_bik"] == "044525225"
    assert fields["executor_bank"] == "ПАО Сбербанк"
    assert fields["executor_rs"] == "40702810900000000001"
    assert fields["executor_ks"] == "30101810400000000225"

    assert fields["customer_fio"] == "Иванов Иван Иванович"
    assert fields["customer_registration_address"] == "г. Санкт-Петербург, Невский пр., д. 20"
    assert fields["customer_phone"] == "+7 (999) 123-45-67"
    assert fields["customer_email"] == "ivanov@example.com"

    assert fields["passport_series"] == "4501"
    assert fields["passport_number"] == "123456"
    assert fields["passport_issued_by"] == "ОВД Тверского района г. Москвы"
    assert fields["passport_issue_date"] == "20.05.2015"
    assert fields["passport_code"] == "770-001"
    assert fields["birth_place"] == "г. Москва"
    assert fields["birth_date"] == "10.10.1990"

    assert fields["property_name"] == "Квартира"
    assert fields["property_purpose"] == "Жилое помещение"
    assert fields["property_area"] == "45.6"
    assert fields["property_address"] == "г. Санкт-Петербург, ул. Пушкина, д. 5"
    assert fields["property_cadastral_number"] == "78:12:0001111:222"
    assert fields["ownership_basis_document"] == "Договор купли-продажи"


def test_parse_ocr_to_contract_data(sample_text):
    data = dogovor.parse_ocr_to_contract_data(sample_text)

    assert isinstance(data, dogovor.ContractData)
    assert data.contract_city == "Санкт-Петербург"
    assert data.executor_name == "Петров Петр Петрович"
    assert data.customer_fio == "Иванов Иван Иванович"


def test_build_payload_from_ocr_text(sample_text):
    payload = dogovor.build_payload_from_ocr_text(sample_text)

    assert isinstance(payload, dogovor.ContractPayload)
    assert payload.source == "tesseract"
    assert payload.ocr_text == sample_text
    assert payload.contract_data.contract_number == "AB-123/2024"


def test_resolve_download_path_valid(monkeypatch, tmp_path):
    monkeypatch.setattr(dogovor, "OUTPUT_DIR", tmp_path)

    filename = "dogovor_1234567890abcdef1234567890abcdef.docx"
    result = dogovor._resolve_download_path(filename)

    assert result == tmp_path / filename


def test_resolve_download_path_invalid():
    with pytest.raises(Exception):
        dogovor._resolve_download_path("../hack.docx")


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


def test_create_doc_template_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(dogovor, "TEMPLATE_PATH", tmp_path / "missing_template.docx")

    with pytest.raises(Exception):
        dogovor.create_doc(dogovor.ContractData(), tmp_path / "out.docx")


def test_ocr_to_contract_endpoint_success(client, monkeypatch, tmp_path, sample_text):
    monkeypatch.setattr(dogovor, "OUTPUT_DIR", tmp_path)

    def fake_validate_upload(file, contents):
        return None

    def fake_img_ocr(contents):
        return sample_text

    def fake_create_doc(contract_data, output_path):
        output_path.write_text("fake docx")

    monkeypatch.setattr(dogovor, "validate_upload", fake_validate_upload)
    monkeypatch.setattr(dogovor, "img_ocr", fake_img_ocr)
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

    assert response.status_code in (404, 400, 422, 500)