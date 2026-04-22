from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OcrTask(str, Enum):
    auto = "auto"
    text = "text"
    formula = "formula"
    table = "table"
    custom = "custom"


TASK_PROMPTS: dict[OcrTask, str] = {
    OcrTask.text: "Text Recognition:",
    OcrTask.formula: "Formula Recognition:",
    OcrTask.table: "Table Recognition:",
}


class ParseByUrlRequest(BaseModel):
    images: List[str] = Field(
        ...,
        min_length=1,
        description="List of image sources (http(s):// or file:// URLs, or data: URIs).",
    )
    task: OcrTask = OcrTask.auto
    prompt: Optional[str] = Field(
        default=None,
        description="Required when task=custom; overrides the default task prompt otherwise.",
    )
    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None


class RegionResult(BaseModel):
    index: int
    label: str
    task_type: str
    bbox_2d: List[int] = Field(..., description="Normalized bbox 0..1000 [x1, y1, x2, y2].")
    content: str
    score: float
    num_tokens: int
    input_tokens: int = 0
    latency_ms: int


class PageScore(BaseModel):
    composite: float = Field(..., description="Weighted quality score in [0.0, 1.0].")
    variables: Dict[str, float]


class PageResult(BaseModel):
    source: str = Field(..., description="Origin of the page: filename, URL, or pdf:page=N")
    text: str = Field(..., description="Cleaned model output (stitched markdown in auto mode).")
    raw_text: str = Field("", description="Raw model output (concatenated per region in auto mode).")
    prompt_used: str
    num_tokens: int = Field(0, description="Output tokens generated (summed across regions in auto mode).")
    input_tokens: int = 0
    latency_ms: int = 0
    score: Optional[PageScore] = None
    flag: str = Field("green", description="green | yellow | red")
    flag_message: str = ""
    flag_details: List[str] = Field(default_factory=list)
    attempts: int = 1
    preset: str = "fast"
    ocr_engine: str = "glm-ocr"
    needs_external_ocr: bool = False
    regions: Optional[List[RegionResult]] = Field(
        default=None,
        description="Per-region detections; populated only in auto (document-parse) mode.",
    )


class ParseResponse(BaseModel):
    id: str
    model: str
    device: str
    pages: List[PageResult]
    text: str = Field(..., description="All page markdown joined with `\\n\\n---\\n\\n`.")


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    dtype: str
    ready: bool
    load_error: Optional[str] = None
    layout_ready: bool = False
    layout_error: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
