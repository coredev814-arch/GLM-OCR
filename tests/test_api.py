"""Smoke tests that exercise every endpoint with the OCR engine mocked out.

Model load is stubbed so the tests run on any machine (no GPU, no model download).
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import layout as layout_mod
from app import pipeline as pipeline_mod
from app import service as service_mod
from app.main import app
from app.schemas import PageResult, RegionResult
from app.service import RegionOutput


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    """Replace real model + layout load/inference with deterministic stubs."""
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
            raw_text=f"[ocr:{prompt}] {page.source}",
            prompt_used=prompt,
            input_tokens=10,
            num_tokens=5,
            latency_ms=1,
        )

    async def fake_infer_region(pil_image, *, prompt, **_):
        return RegionOutput(
            raw_text=f"[region:{prompt}]",
            num_tokens=3,
            input_tokens=5,
            latency_ms=1,
        )

    async def fake_infer_url(url, *, prompt, **_):
        return RegionOutput(
            raw_text=f"[url:{prompt}] {url}",
            num_tokens=3,
            input_tokens=5,
            latency_ms=1,
        )

    async def fake_layout_startup():
        layout_mod.layout_engine._ready = True
        layout_mod.layout_engine._load_error = None

    async def fake_layout_shutdown():
        layout_mod.layout_engine._ready = False

    async def fake_layout_detect(image):
        return []

    monkeypatch.setattr("app.main.settings.background_model_load", False)
    monkeypatch.setattr("app.main.settings.layout_enabled", False)
    monkeypatch.setattr(service_mod.engine, "startup", fake_startup)
    monkeypatch.setattr(service_mod.engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(service_mod.engine, "infer_page", fake_infer_page)
    monkeypatch.setattr(service_mod.engine, "infer_region", fake_infer_region)
    monkeypatch.setattr(service_mod.engine, "infer_url", fake_infer_url)
    monkeypatch.setattr(layout_mod.layout_engine, "startup", fake_layout_startup)
    monkeypatch.setattr(layout_mod.layout_engine, "shutdown", fake_layout_shutdown)
    monkeypatch.setattr(layout_mod.layout_engine, "detect", fake_layout_detect)
    # Pretend layout is ready so task=auto passes readiness checks.
    layout_mod.layout_engine._ready = True
    layout_mod.layout_engine._load_error = None
    yield
    layout_mod.layout_engine._ready = False


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
    assert "layout_ready" in body


def test_parse_upload_single_image(client):
    r = client.post(
        "/ocr/parse",
        files={"files": ("hello.png", _png_bytes(), "image/png")},
        data={"task": "text"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["pages"]) == 1
    page = body["pages"][0]
    assert page["source"] == "hello.png"
    assert page["prompt_used"] == "Text Recognition:"
    assert "hello.png" in body["text"]
    assert page["ocr_engine"] == "glm-ocr"
    assert page["preset"] == "fast"
    assert page["attempts"] == 1
    assert page["flag"] in {"green", "yellow", "red"}
    assert isinstance(page["flag_details"], list)
    assert "num_tokens" in page
    assert "raw_text" in page


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
        data={"task": "text"},
    )
    assert r.status_code == 400


def test_parse_upload_rejects_undecodable_bytes(client):
    r = client.post(
        "/ocr/parse",
        files={"files": ("junk.png", b"not-an-image", "image/png")},
        data={"task": "text"},
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
    assert r.status_code == 422


def test_parse_urls_auto_falls_back_to_text(client):
    # URL inputs can't do layout — server silently rewrites auto → text.
    r = client.post(
        "/ocr/parse/url",
        json={"images": ["https://example.com/a.png"]},
    )
    assert r.status_code == 200
    assert r.json()["pages"][0]["prompt_used"] == "Text Recognition:"


def test_parse_returns_503_when_not_ready(client, monkeypatch):
    monkeypatch.setattr(service_mod.engine, "_ready", False)
    monkeypatch.setattr(service_mod.engine, "_load_error", "disk full")
    r = client.post("/ocr/parse/url", json={"images": ["https://x/a.png"], "task": "text"})
    assert r.status_code == 503
    assert r.json()["detail"] == "disk full"


def test_auto_mode_returns_503_when_layout_not_ready(client, monkeypatch):
    monkeypatch.setattr(layout_mod.layout_engine, "_ready", False)
    monkeypatch.setattr(layout_mod.layout_engine, "_load_error", "layout down")
    r = client.post(
        "/ocr/parse",
        files={"files": ("x.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 503
    assert r.json()["detail"] == "layout down"


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


def test_auto_mode_stitches_regions(client, monkeypatch):
    """task=auto routes through the layout-aware pipeline and stitches regions."""
    from app.layout import Region

    async def fake_detect(image):
        return [
            Region(
                index=0,
                label="doc_title",
                task_type="text",
                score=0.99,
                bbox_2d=[0, 0, 1000, 80],
                polygon=[],
                crop=image.crop((0, 0, 16, 16)),
            ),
            Region(
                index=1,
                label="text",
                task_type="text",
                score=0.95,
                bbox_2d=[0, 80, 1000, 400],
                polygon=[],
                crop=image.crop((0, 0, 16, 16)),
            ),
        ]

    async def fake_infer_region(pil_image, *, prompt, **_):
        # Return distinct content per call so we can verify ordering.
        if not hasattr(fake_infer_region, "n"):
            fake_infer_region.n = 0
        fake_infer_region.n += 1
        return RegionOutput(
            raw_text=f"Region-{fake_infer_region.n}",
            num_tokens=4,
            input_tokens=10,
            latency_ms=1,
        )

    monkeypatch.setattr(layout_mod.layout_engine, "detect", fake_detect)
    monkeypatch.setattr(service_mod.engine, "infer_region", fake_infer_region)

    r = client.post(
        "/ocr/parse",
        files={"files": ("doc.png", _png_bytes(size=(256, 256)), "image/png")},
        data={"task": "auto"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    page = body["pages"][0]
    assert page["prompt_used"] == "document-parse"
    assert page["regions"] is not None
    assert len(page["regions"]) == 2
    assert page["regions"][0]["label"] == "doc_title"
    assert page["regions"][0]["task_type"] == "text"
    assert "Region-1" in page["text"]
    assert "Region-2" in page["text"]


def test_auto_mode_empty_layout_falls_back_to_single_shot(client, monkeypatch):
    """When layout returns no regions, pipeline runs single-shot Text Recognition."""
    r = client.post(
        "/ocr/parse",
        files={"files": ("blank.png", _png_bytes(), "image/png")},
        data={"task": "auto"},
    )
    assert r.status_code == 200, r.text
    page = r.json()["pages"][0]
    assert "fallback" in page["prompt_used"].lower()
    assert page["regions"] == []
