.PHONY: help install install-cpu install-gpu run dev test lint clean \
        docker-build docker-up docker-down docker-logs smoke

PY ?= python3
PORT ?= 8889
HOST ?= 0.0.0.0

help:
	@echo "Targets:"
	@echo "  install-cpu   pip install deps + CPU-only torch (for local dev)"
	@echo "  install-gpu   pip install deps + CUDA 12.1 torch"
	@echo "  run           uvicorn app.main:app (HOST=$(HOST) PORT=$(PORT))"
	@echo "  dev           uvicorn with --reload"
	@echo "  test          pytest"
	@echo "  docker-build  docker compose build"
	@echo "  docker-up     docker compose up -d"
	@echo "  docker-down   docker compose down"
	@echo "  docker-logs   docker compose logs -f glm-ocr"
	@echo "  smoke         curl /health against a running server"

install-cpu:
	$(PY) -m pip install -U pip
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

install-gpu:
	$(PY) -m pip install -U pip
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	$(PY) -m pip install -r requirements.txt

run:
	uvicorn app.main:app --host $(HOST) --port $(PORT)

dev:
	uvicorn app.main:app --host $(HOST) --port $(PORT) --reload

test:
	$(PY) -m pytest tests/ -v

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f glm-ocr

smoke:
	curl -sS http://localhost:$(PORT)/health | $(PY) -m json.tool

clean:
	rm -rf .pytest_cache __pycache__ app/__pycache__ tests/__pycache__
