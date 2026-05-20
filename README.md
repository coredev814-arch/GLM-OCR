# GLM-OCR FastAPI

Self-hosted FastAPI backend for [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), with layout-aware document parsing (PP-DocLayoutV3) and quality scoring.

## Requirements

- NVIDIA GPU with CUDA 12.1+ drivers
- Python 3.10+
- ~20 GB free disk (model weights + torch wheels)

## Install

```bash
make install-gpu   # CUDA 12.1 torch + requirements.txt
# or for CPU-only smoke (model load will be slow / OOM-prone):
make install-cpu
```

## Run

```bash
make run                       # uvicorn on 0.0.0.0:8887
bash scripts/run_supervised.sh # respawn-on-exit supervisor (recommended)
```

The default port is **8887**. It used to be 8889; we moved off because Cloudflare's HTTP proxy times out at ~100s (524) and large OCR requests routinely exceed that. Route 8887 *outside* the Cloudflare proxy (grey-cloud DNS, direct IP, or a tunnel) — changing the port alone doesn't escape the 524.

Override with `PORT=...`:
```bash
make run PORT=9000
PORT=9000 bash scripts/run_supervised.sh
```

## Endpoints

- `GET /health` — returns `ready` (OCR model), `layout_ready` (layout detector), and any load errors.
- `POST /ocr/parse` — multipart upload (`files=@page.png` + `task=...`).
- `POST /ocr/parse/url` — JSON body `{"images": ["https://..."], "task": "..."}`.

### Task field

```
text     single-prompt text recognition
table    table recognition
formula  formula recognition
custom   pass your own `prompt` field
auto     layout-aware: detect regions, OCR each with the right task
```

`task=auto` requires the layout detector. If `/health` shows `layout_ready: false`, `auto` requests 503. Use an explicit task in that case.

## Smoke test

```bash
curl http://localhost:8887/health | python3 -m json.tool

curl -X POST http://localhost:8887/ocr/parse \
  -F "files=@page.png" \
  -F "task=auto"
```

Or use `scripts/client.py` / `scripts/examples.sh`.

## Configuration

All settings can be set in `.env` (see `.env.example`) or as env vars. The defaults live in `app/config.py`.

Notable knobs:
- `MAX_BATCH_SIZE` — max files per request (default 16).
- `MAX_UPLOAD_BYTES` — per-file cap (default 50 MB).
- `REGION_MAX_NEW_TOKENS` — per-region generation cap for `task=auto`. Bounds page latency to stay under proxy timeouts.
- `PDF_DPI` / `PDF_MAX_PAGES` — PDF rasterization controls.

## Docker

`docker compose up` is **not currently working end-to-end** — the Dockerfile predates the dependency floor bump (torch ≥2.5, transformers ≥5.4) and doesn't pull `glmocr`/`opencv`/`scipy`. Use `make install-gpu && make run` instead until the Dockerfile is updated.
