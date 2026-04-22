from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .layout import layout_engine
from .pipeline import infer_document
from .schemas import (
    ErrorResponse,
    HealthResponse,
    OcrTask,
    PageResult,
    ParseByUrlRequest,
    ParseResponse,
)
from .service import ImageInput, bytes_to_pages, engine, resolve_prompt

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("glmocr")


@asynccontextmanager
async def lifespan(_: FastAPI):
    async def _load():
        try:
            await engine.startup()
            logger.info("Model loaded: %s on %s (%s)", settings.model_path, engine.device, engine.dtype)
        except Exception:
            if settings.fail_fast_on_load_error:
                raise
            logger.error("Model load failed; /health will report not-ready")

        if settings.layout_enabled:
            try:
                await layout_engine.startup()
                logger.info("Layout detector loaded: %s", settings.layout_model_dir)
            except Exception:
                logger.error("Layout detector load failed; task=auto will 503")

    if settings.background_model_load:
        load_task = asyncio.create_task(_load())
        try:
            yield
        finally:
            load_task.cancel()
            await layout_engine.shutdown()
            await engine.shutdown()
    else:
        await _load()
        try:
            yield
        finally:
            await layout_engine.shutdown()
            await engine.shutdown()


app = FastAPI(
    title="GLM-OCR FastAPI (self-hosted)",
    description="Self-hosted FastAPI server around zai-org/GLM-OCR via transformers.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[%s] %s %s failed", request_id, request.method, request.url.path)
        raise
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response.headers["x-request-id"] = request_id
    logger.info(
        "[%s] %s %s -> %d (%dms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if engine.ready else ("error" if engine.load_error else "loading"),
        model=settings.model_path,
        device=engine.device,
        dtype=engine.dtype,
        ready=engine.ready,
        load_error=engine.load_error,
        layout_ready=layout_engine.ready,
        layout_error=layout_engine.load_error,
    )


@app.post(
    "/ocr/parse",
    response_model=ParseResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def parse_uploads(
    files: List[UploadFile] = File(..., description="Image or PDF files."),
    task: OcrTask = Form(OcrTask.auto),
    prompt: Optional[str] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    do_sample: Optional[bool] = Form(None),
    temperature: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    no_repeat_ngram_size: Optional[int] = Form(None),
) -> ParseResponse:
    _require_ready(task)
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files provided")
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Batch exceeds max_batch_size={settings.max_batch_size}",
        )

    pages: List[ImageInput] = []
    for f in files:
        data = await f.read()
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{f.filename!r} is empty")
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"{f.filename!r} exceeds max_upload_bytes={settings.max_upload_bytes}",
            )
        try:
            pages.extend(bytes_to_pages(data, source_name=f.filename or "upload"))
        except Exception as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Could not decode {f.filename!r}: {exc}",
            ) from exc

    return await _run(
        pages, task, prompt, max_new_tokens, do_sample, temperature,
        repetition_penalty, no_repeat_ngram_size,
    )


@app.post(
    "/ocr/parse/url",
    response_model=ParseResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def parse_urls(req: ParseByUrlRequest) -> ParseResponse:
    # URL inputs skip layout detection — force single-prompt mode.
    task = req.task if req.task != OcrTask.auto else OcrTask.text
    _require_ready(task)
    if len(req.images) > settings.max_batch_size:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Batch exceeds max_batch_size={settings.max_batch_size}",
        )
    pages = [ImageInput(source=u, url=u) for u in req.images]
    return await _run(
        pages,
        task,
        req.prompt,
        req.max_new_tokens,
        req.do_sample,
        req.temperature,
        req.repetition_penalty,
        req.no_repeat_ngram_size,
    )


def _require_ready(task: OcrTask) -> None:
    if not engine.ready:
        detail = engine.load_error or "model is still loading"
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail)
    if task == OcrTask.auto and not layout_engine.ready:
        detail = layout_engine.load_error or "layout detector is still loading"
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail)


async def _run(
    pages: List[ImageInput],
    task: OcrTask,
    prompt: Optional[str],
    max_new_tokens: Optional[int],
    do_sample: Optional[bool],
    temperature: Optional[float],
    repetition_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
) -> ParseResponse:
    mnt = max_new_tokens if max_new_tokens is not None else settings.max_new_tokens
    ds = do_sample if do_sample is not None else settings.do_sample
    temp = temperature if temperature is not None else settings.temperature
    rp = repetition_penalty if repetition_penalty is not None else settings.repetition_penalty
    nrng = no_repeat_ngram_size if no_repeat_ngram_size is not None else settings.no_repeat_ngram_size

    if task == OcrTask.auto:
        results = await _run_document_parse(pages, mnt, ds, temp, rp, nrng)
    else:
        results = await _run_single_prompt(pages, task, prompt, mnt, ds, temp, rp, nrng)

    return ParseResponse(
        id=f"glmocr-{uuid.uuid4().hex}",
        model=settings.model_path,
        device=engine.device,
        pages=results,
        text="\n\n---\n\n".join(r.text for r in results),
    )


async def _run_document_parse(
    pages: List[ImageInput],
    mnt: int,
    ds: bool,
    temp: float,
    rp: float,
    nrng: int,
) -> List[PageResult]:
    results: List[PageResult] = []
    for page in pages:
        try:
            res = await infer_document(
                page,
                max_new_tokens=mnt,
                do_sample=ds,
                temperature=temp,
                repetition_penalty=rp,
                no_repeat_ngram_size=nrng,
            )
        except RuntimeError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except Exception as exc:
            logger.exception("Document-parse failed for %s", page.source)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Document-parse failed for {page.source}: {exc}",
            ) from exc
        results.append(res)
    return results


async def _run_single_prompt(
    pages: List[ImageInput],
    task: OcrTask,
    prompt: Optional[str],
    mnt: int,
    ds: bool,
    temp: float,
    rp: float,
    nrng: int,
) -> List[PageResult]:
    try:
        resolved_prompt = resolve_prompt(task, prompt)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    results: List[PageResult] = []
    for page in pages:
        try:
            res = await engine.infer_page(
                page,
                prompt=resolved_prompt,
                max_new_tokens=mnt,
                do_sample=ds,
                temperature=temp,
                repetition_penalty=rp,
                no_repeat_ngram_size=nrng,
            )
        except RuntimeError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        except Exception as exc:
            logger.exception("Inference failed for %s", page.source)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Inference failed for {page.source}: {exc}",
            ) from exc
        results.append(res)
    return results


@app.exception_handler(RuntimeError)
async def _runtime_error_handler(_, exc: RuntimeError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(error="service_unavailable", detail=str(exc)).model_dump(),
    )
