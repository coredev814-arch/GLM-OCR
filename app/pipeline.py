"""Document-parse pipeline: layout → per-region OCR → markdown stitching.

Mirrors the official glmocr flow without the HTTP indirection:
- PP-DocLayoutV3 detects regions (headings, text, tables, formulas).
- Each region is cropped and OCR'd with the prompt matching its task_type.
- glmocr.postprocess.ResultFormatter stitches region outputs into reading-order
  markdown (heading prefixes, `$$…$$` formulas, HTML tables).
- Quality scoring runs on the stitched page text.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

from .column_split import split_multi_column_regions
from .config import settings
from .layout import Region, layout_engine
from .quality import score_page
from .schemas import PageResult, RegionResult
from .service import ImageInput, build_page_result, engine

logger = logging.getLogger(__name__)

_REGION_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


async def infer_document(
    page: ImageInput,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> PageResult:
    """Layout-aware OCR for a single page image."""
    if page.pil_image is None:
        raise ValueError(
            "Document-parse requires a PIL image — URL inputs must use "
            "task=text/table/formula/custom (single-prompt override)."
        )

    start = time.perf_counter()
    regions = await layout_engine.detect(page.pil_image)
    if settings.column_split_enabled and regions:
        regions = split_multi_column_regions(regions)

    if not regions:
        # Empty layout: fall back to single-shot text recognition on the whole page.
        logger.info("[%s] layout returned 0 regions; falling back to Text Recognition", page.source)
        output = await engine.infer_region(
            page.pil_image,
            prompt=_REGION_PROMPTS["text"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        width, height = page.pil_image.size
        quality = score_page(
            raw_text=output.raw_text,
            num_tokens=output.num_tokens,
            max_tokens=max_new_tokens,
            image_width=width,
            image_height=height,
        )
        return build_page_result(
            source=page.source,
            prompt="document-parse (fallback: Text Recognition:)",
            input_tokens=output.input_tokens,
            num_tokens=output.num_tokens,
            latency_ms=int((time.perf_counter() - start) * 1000),
            quality=quality,
            regions=[],
        )

    region_records, region_results = await _ocr_regions(
        regions,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    markdown = _stitch_markdown(region_records)
    raw_concat = "\n\n".join(r["content"] for r in region_records if r.get("content"))
    width, height = page.pil_image.size

    quality = score_page(
        raw_text=markdown if markdown else raw_concat,
        num_tokens=sum(r.num_tokens for r in region_results),
        max_tokens=max_new_tokens * max(1, len(regions)),
        image_width=width,
        image_height=height,
    )
    # quality.text is the cleaned markdown; swap raw_text for the per-region
    # concatenation so callers can debug the raw model stream.
    quality.raw_text = raw_concat

    return build_page_result(
        source=page.source,
        prompt="document-parse",
        input_tokens=sum(r.input_tokens for r in region_results),
        num_tokens=sum(r.num_tokens for r in region_results),
        latency_ms=int((time.perf_counter() - start) * 1000),
        quality=quality,
        regions=region_results,
    )


async def _ocr_regions(
    regions: List[Region],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> Tuple[List[dict], List[RegionResult]]:
    """OCR each region serially (single-GPU) and return (formatter_input, api_output)."""
    formatter_input: List[dict] = []
    api_output: List[RegionResult] = []

    for region in regions:
        prompt = _REGION_PROMPTS.get(region.task_type)
        if prompt is None:
            continue
        try:
            out = await engine.infer_region(
                region.crop,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        except Exception as exc:
            logger.exception("Region %d (%s) OCR failed", region.index, region.label)
            content = f"[ocr_error: {type(exc).__name__}]"
            num_tokens = input_tokens = 0
            latency_ms = 0
        else:
            content = out.raw_text
            num_tokens = out.num_tokens
            input_tokens = out.input_tokens
            latency_ms = out.latency_ms

        formatter_input.append({
            "index": region.index,
            "label": region.label,
            "content": content,
            "bbox_2d": region.bbox_2d,
            "polygon": region.polygon,
            "score": region.score,
            "task_type": region.task_type,
        })
        api_output.append(RegionResult(
            index=region.index,
            label=region.label,
            task_type=region.task_type,
            bbox_2d=region.bbox_2d,
            content=content,
            score=region.score,
            num_tokens=num_tokens,
            input_tokens=input_tokens,
            latency_ms=latency_ms,
        ))

    return formatter_input, api_output


def _stitch_markdown(formatter_input: List[dict]) -> str:
    """Run ResultFormatter.process on the per-region outputs and return markdown."""
    if not formatter_input:
        return ""
    try:
        from glmocr.config import ResultFormatterConfig
        from glmocr.postprocess import ResultFormatter

        formatter = ResultFormatter(ResultFormatterConfig())
        _, markdown, _ = formatter.process([formatter_input])
    except Exception:
        logger.exception("ResultFormatter failed; falling back to naive concatenation")
        return "\n\n".join(
            _format_naive(r["label"], r["content"]) for r in formatter_input if r.get("content")
        )
    return markdown or ""


def _format_naive(label: str, content: str) -> str:
    content = (content or "").strip()
    if not content:
        return ""
    if label == "doc_title":
        return f"# {content}"
    if label == "paragraph_title":
        return f"## {content}"
    return content
