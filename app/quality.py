"""Quality scoring for GLM-OCR output.

Mirrors the deepseek-ocr scoring system: clean the raw model output, score on
6 weighted metrics, and attach severity-tagged flags so callers can decide
whether to re-OCR, fall back, or accept the result.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

_WEIGHTS: Dict[str, float] = {
    "hallucination_ratio": 0.25,
    "self_consistency": 0.20,
    "token_efficiency": 0.20,
    "content_density": 0.15,
    "structural_integrity": 0.10,
    "repetition_density": 0.10,
}

_LAYOUT_TOKEN_RE = re.compile(
    r"<\|[^|]*?\|>"
    r"|<ref>.*?</ref>"
    r"|<points?>.*?</points?>"
    r"|<box>.*?</box>",
    re.DOTALL,
)

_EOS_TAIL_RE = re.compile(
    r"<\|(?:user|assistant|endoftext|end_of_text)\|>"
)

_STRUCTURE_PATTERNS = [
    re.compile(r"^#{1,6}\s", re.MULTILINE),
    re.compile(r"\|.*\|"),
    re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b"),
    re.compile(r"[\$€£]?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?|\b\d+\.\d{2}\b"),
]


@dataclass
class QualityResult:
    text: str
    raw_text: str
    variables: Dict[str, float]
    composite: float
    flag: str
    flag_message: str
    flag_details: List[str]
    needs_external_ocr: bool


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    txt = _EOS_TAIL_RE.sub("", raw)
    txt = _LAYOUT_TOKEN_RE.sub("", txt)
    txt = _trim_repetition_tail(txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _trim_repetition_tail(txt: str) -> str:
    """Collapse a repeating suffix to a single copy.

    Catches the common GLM-OCR failure where the tail is the same short phrase
    repeated many times (e.g. "Extenuating Circumstances Code: $119" ×N).
    """
    n = len(txt)
    if n < 100:
        return txt
    max_block = min(500, n // 3)
    for block_size in range(10, max_block):
        block = txt[n - block_size:]
        repeats = 1
        pos = n - block_size
        while pos >= block_size and txt[pos - block_size:pos] == block:
            repeats += 1
            pos -= block_size
        if repeats >= 3:
            return txt[:pos + block_size]
    return txt


def _score_hallucination_ratio(clean: str, raw: str) -> float:
    if not raw:
        return 1.0
    return round(min(1.0, len(clean) / len(raw)), 3)


def _score_token_efficiency(num_tokens: int, max_tokens: int, clean: str) -> float:
    if max_tokens <= 0 or num_tokens < max_tokens - 10:
        return 1.0
    chars_per_token = len(clean) / max(num_tokens, 1)
    if chars_per_token >= 2.0:
        return 0.7
    return 0.2


def _score_content_density(clean: str, image_area: int) -> float:
    if image_area <= 0:
        return 1.0
    density = len(clean) / image_area
    return round(min(1.0, density / 1e-4), 3)


def _score_structural_integrity(clean: str) -> float:
    if not clean:
        return 0.0
    hits = sum(1 for p in _STRUCTURE_PATTERNS if p.search(clean))
    return round(hits / len(_STRUCTURE_PATTERNS), 3)


def _score_repetition_density(clean: str, n: int = 5) -> float:
    tokens = clean.split()
    if len(tokens) < n + 1:
        return 1.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return round(len(set(grams)) / len(grams), 3)


def _collect_flags(
    *,
    clean: str,
    num_tokens: int,
    max_tokens: int,
    variables: Dict[str, float],
) -> Tuple[List[str], bool]:
    flags: List[str] = []
    critical = False

    clean_len = len(clean.strip())
    if clean_len == 0:
        flags.append("no_content")
        critical = True
    elif clean_len <= 10:
        flags.append("blank_page")
        critical = True
    elif clean_len < 30:
        flags.append("low_content")

    if variables["hallucination_ratio"] < 0.25:
        flags.append("possible_hallucination")

    if max_tokens > 0 and num_tokens >= max_tokens - 10:
        flags.append("max_tokens_hit")

    if variables["repetition_density"] < 0.5:
        flags.append("repetitive_content")

    if variables["content_density"] < 0.3:
        flags.append("sparse_content")

    return flags, critical


def _flag_color(composite: float, critical: bool) -> str:
    if critical:
        return "red"
    if composite >= 0.80:
        return "green"
    if composite >= 0.50:
        return "yellow"
    return "red"


def _flag_message(color: str, composite: float) -> str:
    score = f"{composite:.2f}"
    return {
        "green": f"Good quality ({score}).",
        "yellow": f"Quality issues detected ({score}).",
        "red": f"Poor quality ({score}).",
    }[color]


def score_page(
    *,
    raw_text: str,
    num_tokens: int,
    max_tokens: int,
    image_width: int,
    image_height: int,
) -> QualityResult:
    clean = clean_text(raw_text)
    area = max(0, image_width) * max(0, image_height)

    variables = {
        "hallucination_ratio": _score_hallucination_ratio(clean, raw_text),
        "self_consistency": 1.0,
        "token_efficiency": _score_token_efficiency(num_tokens, max_tokens, clean),
        "content_density": _score_content_density(clean, area),
        "structural_integrity": _score_structural_integrity(clean),
        "repetition_density": _score_repetition_density(clean),
    }

    composite = sum(variables[k] * _WEIGHTS[k] for k in _WEIGHTS)

    clean_len = len(clean.strip())
    if clean_len <= 10:
        composite = min(composite, 0.10)
    elif clean_len <= 30:
        composite = min(composite, 0.30)

    composite = round(composite, 2)

    flags, critical = _collect_flags(
        clean=clean,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        variables=variables,
    )

    color = _flag_color(composite, critical)

    return QualityResult(
        text=clean,
        raw_text=raw_text,
        variables=variables,
        composite=composite,
        flag=color,
        flag_message=_flag_message(color, composite),
        flag_details=flags,
        needs_external_ocr=color == "red",
    )
