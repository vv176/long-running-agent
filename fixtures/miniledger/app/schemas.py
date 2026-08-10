"""Request validation. Touches flask, so it is framework-coupled."""
from flask import request

from app.utils.money import Money


def invoice_payload():
    data = request.get_json(silent=True) or {}
    if "id" not in data:
        raise ValueError("id is required")
    amounts = data.get("amounts", [])
    for a in amounts:               # validate each parses; the service builds the objects
        Money(a)
    return data["id"], data.get("customer_id", ""), amounts
