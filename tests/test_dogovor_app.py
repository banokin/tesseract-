import pytest

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