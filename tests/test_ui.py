"""Tests para UI teleprompter — sin levantar server real."""

from fastapi.testclient import TestClient

from traductor.ui.app import app


def test_root_sirve_html() -> None:
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "Teleprompter" in res.text
    assert "feed" in res.text


def test_post_transcripcion_broadcast() -> None:
    client = TestClient(app)
    # Conecta un WS para recibir broadcast
    with client.websocket_connect("/ws") as ws:
        # POST fallback debe hacer broadcast al WS
        res = client.post("/api/transcripcion", json={"en": "hello", "es": "hola"})
        assert res.status_code == 200
        assert res.json() == {"ok": "true"}
        data = ws.receive_text()
        assert '"en": "hello"' in data or '"en":"hello"' in data.replace(" ", "")


def test_post_transcripcion_sin_ws_no_falla() -> None:
    client = TestClient(app)
    res = client.post("/api/transcripcion", json={"en": "hi", "es": "hola"})
    assert res.status_code == 200


def test_post_transcripcion_invalida_422() -> None:
    client = TestClient(app)
    res = client.post("/api/transcripcion", json={"en": "hi"})
    assert res.status_code == 422
