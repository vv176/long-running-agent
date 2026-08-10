"""Invoice rules. Framework-agnostic: no flask import anywhere."""
from app.models.invoice import Invoice
from app.utils.money import Money

_STORE = {}


def create(invoice_id, customer_id, amounts):
    inv = Invoice(invoice_id, customer_id, [Money(a) for a in amounts])
    _STORE[invoice_id] = inv
    return inv


def get(invoice_id):
    return _STORE.get(invoice_id)


def total_paise(invoice_id):
    inv = get(invoice_id)
    return inv.total().as_paise() if inv else 0
