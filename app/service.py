from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

from .config import settings
from .quality import QualityResult, score_page
from .schemas import TASK_PROMPTS, OcrTask, PageResult, PageScore

logger = logging.getLogger(__name__)


@dataclass
class ImageInput:
    """One page to OCR. Exactly one of (pil_image, url) must be set."""
    source: str
    pil_image: Optional[Image.Image] = None
    url: Optional[str] = None


@dataclass
class RegionOutput:
    """Raw per-region inference result, prior to stitching or scoring."""
    raw_text: str
    num_tokens: int
    input_tokens: int
    latency_ms: int


class OcrEngine:
    """Loads GLM-OCR once and serializes GPU-bound inference calls."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device: str = "unknown"
        self._dtype: str = settings.torch_dtype
        self._ready = False
        self._load_error: Optional[str] = None
        self._load_lock = asyncio.Lock()
        self._infer_sem = asyncio.Semaphore(settings.max_concurrent_inferences)

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    async def startup(self) -> None:
        async with self._load_lock:
            if self._ready:
                return
            try:
                await asyncio.to_thread(self._load)
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Model load failed")
                raise
            self._ready = True
            self._load_error = None

    async def shutdown(self) -> None:
        self._model = None
        self._processor = None
        self._ready = False

    def _resolve_dtype(self):
        import torch

        return {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[settings.torch_dtype]

    def _load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading processor from %s", settings.model_path)
        self._processor = AutoProcessor.from_pretrained(
            settings.model_path, trust_remote_code=settings.trust_remote_code
        )

        logger.info(
            "Loading model from %s (device_map=%s, dtype=%s)",
            settings.model_path,
            settings.device_map,
            settings.torch_dtype,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=settings.model_path,
            torch_dtype=self._resolve_dtype(),
            device_map=settings.device_map,
            trust_remote_code=settings.trust_remote_code,
        )
        self._model.eval()

        first_param = next(self._model.parameters())
        self._device = str(first_param.device)
        self._dtype = str(first_param.dtype).replace("torch.", "")
        logger.info("Model ready on %s (%s)", self._device, self._dtype)

    async def infer_region(
        self,
        pil_image: Image.Image,
        *,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ) -> RegionOutput:
        if not self._ready or self._model is None or self._processor is None:
            raise RuntimeError("OCR engine is not ready")

        async with self._infer_sem:
            return await asyncio.to_thread(
                self._infer_sync,
                pil_image,
                None,
                prompt,
                max_new_tokens,
                do_sample,
                temperature,
                repetition_penalty,
                no_repeat_ngram_size,
            )

    async def infer_url(
        self,
        url: str,
        *,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ) -> RegionOutput:
        """Inference path for URL inputs — skips layout detection."""
        if not self._ready or self._model is None or self._processor is None:
            raise RuntimeError("OCR engine is not ready")

        async with self._infer_sem:
            return await asyncio.to_thread(
                self._infer_sync,
                None,
                url,
                prompt,
                max_new_tokens,
                do_sample,
                temperature,
                repetition_penalty,
                no_repeat_ngram_size,
            )

    async def infer_page(
        self,
        page: ImageInput,
        *,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ) -> PageResult:
        """Single-prompt page inference — used for task=text/table/formula/custom overrides."""
        if page.pil_image is not None:
            output = await self.infer_region(
                page.pil_image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            width, height = page.pil_image.size
        elif page.url is not None:
            output = await self.infer_url(
                page.url,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            width = height = 0
        else:
            raise ValueError("ImageInput requires either pil_image or url")

        quality = score_page(
            raw_text=output.raw_text,
            num_tokens=output.num_tokens,
            max_tokens=max_new_tokens,
            image_width=width,
            image_height=height,
        )
        return build_page_result(
            source=page.source,
            prompt=prompt,
            input_tokens=output.input_tokens,
            num_tokens=output.num_tokens,
            latency_ms=output.latency_ms,
            quality=quality,
        )

    def _infer_sync(
        self,
        pil_image: Optional[Image.Image],
        url: Optional[str],
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ) -> RegionOutput:
        import torch

        content = [
            {"type": "image", "image": pil_image} if pil_image is not None
            else {"type": "image", "url": url},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        inputs.pop("token_type_ids", None)

        input_len = int(inputs["input_ids"].shape[1])

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = settings.top_p
        if repetition_penalty and repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)

        new_tokens = generated_ids[0][input_len:]
        num_tokens = int(new_tokens.shape[0])
        raw_text = self._processor.decode(new_tokens, skip_special_tokens=False)

        return RegionOutput(
            raw_text=raw_text,
            num_tokens=num_tokens,
            input_tokens=input_len,
            latency_ms=latency_ms,
        )


engine = OcrEngine()


def build_page_result(
    *,
    source: str,
    prompt: str,
    input_tokens: int,
    num_tokens: int,
    latency_ms: int,
    quality: QualityResult,
    regions: Optional[list] = None,
) -> PageResult:
    return PageResult(
        source=source,
        text=quality.text,
        raw_text=quality.raw_text,
        prompt_used=prompt,
        num_tokens=num_tokens,
        input_tokens=input_tokens,
        latency_ms=latency_ms,
        score=PageScore(composite=quality.composite, variables=quality.variables),
        flag=quality.flag,
        flag_message=quality.flag_message,
        flag_details=quality.flag_details,
        attempts=1,
        preset="fast",
        ocr_engine="glm-ocr",
        needs_external_ocr=quality.needs_external_ocr,
        regions=regions,
    )


def resolve_prompt(task: OcrTask, custom: Optional[str]) -> str:
    if task == OcrTask.custom:
        if not custom or not custom.strip():
            raise ValueError("task=custom requires a non-empty 'prompt'")
        return custom.strip()
    if custom and custom.strip():
        return custom.strip()
    return TASK_PROMPTS[task]


def bytes_to_pages(
    data: bytes, *, source_name: str
) -> List[ImageInput]:
    """Turn an uploaded blob (image or PDF) into a list of ImageInput pages."""
    if _looks_like_pdf(data):
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(
            data, dpi=settings.pdf_dpi, fmt="png", last_page=settings.pdf_max_pages
        )
        return [
            ImageInput(source=f"{source_name}:page={i + 1}", pil_image=img.convert("RGB"))
            for i, img in enumerate(images)
        ]

    img = Image.open(io.BytesIO(data)).convert("RGB")
    return [ImageInput(source=source_name, pil_image=img)]


def _looks_like_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"
