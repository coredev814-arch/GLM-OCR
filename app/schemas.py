from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class OcrTask(str, Enum):
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
    task: OcrTask = OcrTask.text
    prompt: Optional[str] = Field(
        default=None,
        description="Required when task=custom; overrides the default task prompt otherwise.",
    )
    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None


class PageResult(BaseModel):
    source: str = Field(..., description="Origin of the page: filename, URL, or pdf:page=N")
    text: str
    prompt_used: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class ParseResponse(BaseModel):
    id: str
    model: str
    device: str
    pages: List[PageResult]
    text: str = Field(..., description="All pages joined with `\\n\\n---\\n\\n`.")


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    dtype: str
    ready: bool
    load_error: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
