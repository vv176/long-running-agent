"""Money arithmetic. No local imports — a true leaf."""
from decimal import Decimal


class Money:
    def __init__(self, amount, currency="INR"):
        self.amount = Decimal(str(amount))
        self.currency = currency

    def __add__(self, other):
        assert self.currency == other.currency
        return Money(self.amount + other.amount, self.currency)

    def __eq__(self, other):
        return (self.amount, self.currency) == (other.amount, other.currency)

    def as_paise(self):
        return int(self.amount * 100)
