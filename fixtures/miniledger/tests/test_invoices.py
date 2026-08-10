def test_create_and_get(client):
    r = client.post("/invoices", json={"id": "INV-1", "customer_id": "C1", "amounts": [10, 5]})
    assert r.status_code == 201
    assert r.get_json()["total_paise"] == 1500

    r = client.get("/invoices/INV-1")
    assert r.get_json()["total_paise"] == 1500


def test_missing(client):
    assert client.get("/invoices/NOPE").status_code == 404
