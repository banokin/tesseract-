import pytest
import sys
import types

docxtpl_stub = types.ModuleType("docxtpl")


class _DummyDocxTemplate:
    def __init__(self, *_args, **_kwargs):
        pass

    def render(self, *_args, **_kwargs):
        return None

    def save(self, *_args, **_kwargs):
        return None


docxtpl_stub.DocxTemplate = _DummyDocxTemplate
sys.modules.setdefault("docxtpl", docxtpl_stub)

pytesseract_stub = types.ModuleType("pytesseract")
pytesseract_stub.image_to_string = lambda *_args, **_kwargs: ""
sys.modules.setdefault("pytesseract", pytesseract_stub)

import huggin_face_scan.dogovor_new as dogovor_new


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
    fields = dogovor_new.extract_fields(sample_text)

    assert fields["contract_city"] == "Санкт-Петербург"
    assert fields["contract_number"] == "AB-123/2024"
    assert fields["executor_name"] == "Петров Петр Петрович"
    assert fields["customer_fio"] == "Иванов Иван Иванович"
    assert fields["property_cadastral_number"] == "78:12:0001111:222"


def test_resolve_download_path_invalid():
    with pytest.raises(Exception):
        dogovor_new._resolve_download_path("../hack.docx")


def test_unified_json_formats_caps_and_short_address():
    payload = {
        "data": {
            "passport_main": {
                "surname": "иванов",
                "name": "иван",
                "patronymic": "иванович",
                "issuing_authority": "Отделом МВД России по району арбат г. москвы",
                "birth_place": "г. москва",
                "passport_series": "4501",
                "passport_number": "123456",
                "issue_date": "20.05.2015",
                "department_code": "770-001",
                "birth_date": "10.10.1990",
            },
            "passport_registration": {
                "region": "Московская область",
                "city": "Зарайск",
                "street": "Рязанская",
                "house": "13",
            },
            "egrn_extract": {},
        }
    }

    contract_data = dogovor_new.unified_json_to_contract_data(payload)

    assert contract_data.passport_issued_by == "ОТДЕЛОМ МВД РОССИИ ПО РАЙОНУ АРБАТ Г. МОСКВЫ"
    assert contract_data.birth_place == "Г. МОСКВА"
    assert contract_data.customer_registration_address == "Московская обл., г. Зарайск, ул. Рязанская, д. 13"


def test_unified_json_formats_city_prefix_in_override_address():
    payload = {
        "data": {
            "passport_main": {
                "surname": "Иванов",
                "name": "Иван",
                "patronymic": "Иванович",
            },
            "passport_registration": {},
            "egrn_extract": {},
        },
        "customer_registration_address_override": "Оренбургская область, Оренбург, ул. Пионерская, д. 15, кв. 94",
    }

    contract_data = dogovor_new.unified_json_to_contract_data(payload)
    assert contract_data.customer_registration_address == "Оренбургская обл., г. Оренбург, ул. Пионерская, д. 15, кв. 94"