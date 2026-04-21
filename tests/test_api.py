"""Smoke tests that exercise every endpoint with the OCR engine mocked out.

Model load is stubbed so the tests run on any machine (no GPU, no model download).
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import service as service_mod
from app.main import app
from app.schemas import PageResult


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    """Replace real model load + inference with deterministic stubs."""
    async def fake_startup():
        service_mod.engine._ready = True
        service_mod.engine._device = "cpu"
        service_mod.engine._dtype = "float32"
        service_mod.engine._load_error = None

    async def fake_shutdown():
        service_mod.engine._ready = False

    async def fake_infer_page(page, *, prompt, max_new_tokens, do_sample, temperature, repetition_penalty, no_repeat_ngram_size):
        return PageResult(
            source=page.source,
            text=f"[ocr:{prompt}] {page.source}",
            prompt_used=prompt,
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
        )

    # Force synchronous model load in tests so /health is ready before requests.
    monkeypatch.setattr("app.main.settings.background_model_load", False)
    monkeypatch.setattr(service_mod.engine, "startup", fake_startup)
    monkeypatch.setattr(service_mod.engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(service_mod.engine, "infer_page", fake_infer_page)
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _png_bytes(color=(200, 50, 50), size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["model"] == "zai-org/GLM-OCR"


def test_parse_upload_single_image(client):
    r = client.post(
        "/ocr/parse",
        files={"files": ("hello.png", _png_bytes(), "image/png")},
        data={"task": "text"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["pages"]) == 1
    assert body["pages"][0]["source"] == "hello.png"
    assert body["pages"][0]["prompt_used"] == "Text Recognition:"
    assert "hello.png" in body["text"]


def test_parse_upload_multi_image(client):
    r = client.post(
        "/ocr/parse",
        files=[
            ("files", ("a.png", _png_bytes((10, 10, 10)), "image/png")),
            ("files", ("b.png", _png_bytes((240, 240, 240)), "image/png")),
        ],
        data={"task": "table"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["pages"]) == 2
    assert all(p["prompt_used"] == "Table Recognition:" for p in body["pages"])


def test_parse_upload_custom_prompt(client):
    r = client.post(
        "/ocr/parse",
        files={"files": ("x.png", _png_bytes(), "image/png")},
        data={"task": "custom", "prompt": "Extract invoice number only:"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["pages"][0]["prompt_used"] == "Extract invoice number only:"


def test_parse_upload_custom_requires_prompt(client):
    r = client.post(
        "/ocr/parse",
        files={"files": ("x.png", _png_bytes(), "image/png")},
        data={"task": "custom"},
    )
    assert r.status_code == 400
    assert "prompt" in r.json()["detail"]


def test_parse_upload_rejects_empty_file(client):
    r = client.post(
        "/ocr/parse",
        files={"files": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 400


def test_parse_upload_rejects_undecodable_bytes(client):
    r = client.post(
        "/ocr/parse",
        files={"files": ("junk.png", b"not-an-image", "image/png")},
    )
    assert r.status_code == 400


def test_parse_urls(client):
    r = client.post(
        "/ocr/parse/url",
        json={
            "images": ["https://example.com/a.png", "https://example.com/b.png"],
            "task": "formula",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert [p["source"] for p in body["pages"]] == [
        "https://example.com/a.png",
        "https://example.com/b.png",
    ]
    assert all(p["prompt_used"] == "Formula Recognition:" for p in body["pages"])


def test_parse_urls_empty_rejected(client):
    r = client.post("/ocr/parse/url", json={"images": []})
    assert r.status_code == 422  # pydantic min_length


def test_parse_returns_503_when_not_ready(client, monkeypatch):
    monkeypatch.setattr(service_mod.engine, "_ready", False)
    monkeypatch.setattr(service_mod.engine, "_load_error", "disk full")
    r = client.post("/ocr/parse/url", json={"images": ["https://x/a.png"]})
    assert r.status_code == 503
    assert r.json()["detail"] == "disk full"


def test_health_reflects_loading_state(client, monkeypatch):
    monkeypatch.setattr(service_mod.engine, "_ready", False)
    monkeypatch.setattr(service_mod.engine, "_load_error", None)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "loading"
    assert r.json()["ready"] is False


def test_batch_size_limit(client, monkeypatch):
    monkeypatch.setattr("app.main.settings.max_batch_size", 2)
    r = client.post(
        "/ocr/parse/url",
        json={"images": [f"https://x/{i}.png" for i in range(5)], "task": "text"},
    )
    assert r.status_code == 400
    assert "max_batch_size" in r.json()["detail"]
