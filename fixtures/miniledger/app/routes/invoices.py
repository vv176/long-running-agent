"""Invoice endpoints. Uses `from pkg import module` on purpose."""
from flask import Blueprint, jsonify

from app.services import invoice as invoice_service
from app.schemas import invoice_payload

bp = Blueprint("invoices", __name__, url_prefix="/invoices")


@bp.route("", methods=["POST"])
def create_invoice():
    invoice_id, customer_id, amounts = invoice_payload()
    inv = invoice_service.create(invoice_id, customer_id, amounts)
    return jsonify({"id": inv.id, "total_paise": inv.total().as_paise()}), 201


@bp.route("/<invoice_id>")
def get_invoice(invoice_id):
    inv = invoice_service.get(invoice_id)
    if inv is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"id": inv.id, "total_paise": inv.total().as_paise()})
