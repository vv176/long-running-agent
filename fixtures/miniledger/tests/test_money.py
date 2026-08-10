from app.utils.money import Money


def test_add():
    assert Money(10) + Money(5) == Money(15)


def test_paise():
    assert Money("12.34").as_paise() == 1234
