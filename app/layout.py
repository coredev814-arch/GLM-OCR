"""Layout detection via PP-DocLayoutV3.

Wraps glmocr.layout.PPDocLayoutDetector for in-process region detection. The
detector is loaded once at startup, runs serially alongside the OCR model on
the same device, and hands back per-region crops ready for the VLM.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image

from .config import settings

logger = logging.getLogger(__name__)


# Mirrors glmocr/config.yaml — safer to hard-code than depend on yaml parsing.
_LABEL_TASK_MAPPING: Dict[str, List[str]] = {
    "text": [
        "abstract", "algorithm", "content", "doc_title", "figure_title",
        "paragraph_title", "reference_content", "text", "vertical_text",
        "vision_footnote", "seal", "formula_number",
    ],
    "table": ["table"],
    "formula": ["display_formula", "inline_formula"],
    "skip": ["chart", "image"],
    "abandon": [
        "header", "footer", "number", "footnote", "aside_text",
        "reference", "footer_image", "header_image",
    ],
}

_ID2LABEL: Dict[int, str] = {
    0: "abstract", 1: "algorithm", 2: "aside_text", 3: "chart", 4: "content",
    5: "display_formula", 6: "doc_title", 7: "figure_title", 8: "footer",
    9: "footer_image", 10: "footnote", 11: "formula_number", 12: "header",
    13: "header_image", 14: "image", 15: "inline_formula", 16: "number",
    17: "paragraph_title", 18: "reference", 19: "reference_content",
    20: "seal", 21: "table", 22: "text", 23: "vertical_text",
    24: "vision_footnote",
}

_LAYOUT_MERGE_BBOXES_MODE = {i: ("small" if i == 18 else "large") for i in range(25)}


@dataclass
class Region:
    index: int              # reading order within the page
    label: str              # native label ("doc_title", "text", ...)
    task_type: str          # "text" | "table" | "formula" | "skip"
    score: float
    bbox_2d: List[int]      # normalized 0..1000 [x1, y1, x2, y2]
    polygon: List[List[int]]
    crop: Image.Image       # cropped image ready for OCR


class LayoutEngine:
    """Loads PP-DocLayoutV3 once and serializes detection calls."""

    def __init__(self) -> None:
        self._detector = None
        self._ready = False
        self._load_error: Optional[str] = None
        self._load_lock = asyncio.Lock()
        self._detect_lock = asyncio.Lock()

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
                logger.exception("Layout detector load failed")
                raise
            self._ready = True
            self._load_error = None

    async def shutdown(self) -> None:
        if self._detector is not None:
            try:
                self._detector.stop()
            except Exception:
                logger.exception("Layout detector stop failed")
        self._detector = None
        self._ready = False

    def _load(self) -> None:
        from glmocr.config import LayoutConfig
        from glmocr.layout import PPDocLayoutDetector

        device = settings.layout_device or _auto_device()
        logger.info("Loading PP-DocLayoutV3 from %s on %s", settings.layout_model_dir, device)

        config = LayoutConfig(
            model_dir=settings.layout_model_dir,
            device=device,
            threshold=settings.layout_threshold,
            batch_size=1,
            layout_nms=True,
            layout_unclip_ratio=[1.0, 1.0],
            layout_merge_bboxes_mode=_LAYOUT_MERGE_BBOXES_MODE,
            label_task_mapping=_LABEL_TASK_MAPPING,
            id2label=_ID2LABEL,
            use_polygon=False,
        )
        self._detector = PPDocLayoutDetector(config)
        self._detector.start()
        logger.info("PP-DocLayoutV3 ready on %s", device)

    async def detect(self, image: Image.Image) -> List[Region]:
        if not self._ready or self._detector is None:
            raise RuntimeError("Layout engine is not ready")
        async with self._detect_lock:
            return await asyncio.to_thread(self._detect_sync, image)

    def _detect_sync(self, image: Image.Image) -> List[Region]:
        results, _ = self._detector.process([image], save_visualization=False)
        regions_raw = results[0] if results else []

        width, height = image.size
        regions: List[Region] = []
        for item in regions_raw:
            task_type = item.get("task_type")
            if task_type not in ("text", "table", "formula"):
                # "skip" and "abandon" are already filtered by the detector;
                # any unexpected type (e.g. "image") gets dropped here.
                continue

            x1_n, y1_n, x2_n, y2_n = item["bbox_2d"]
            x1 = max(0, int(x1_n / 1000.0 * width))
            y1 = max(0, int(y1_n / 1000.0 * height))
            x2 = min(width, int(x2_n / 1000.0 * width))
            y2 = min(height, int(y2_n / 1000.0 * height))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))

            regions.append(
                Region(
                    index=int(item["index"]),
                    label=str(item["label"]),
                    task_type=str(task_type),
                    score=float(item.get("score", 0.0)),
                    bbox_2d=[int(v) for v in item["bbox_2d"]],
                    polygon=[[int(p[0]), int(p[1])] for p in item.get("polygon", [])],
                    crop=crop,
                )
            )
        regions.sort(key=lambda r: r.index)
        return regions


def _auto_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


layout_engine = LayoutEngine()
