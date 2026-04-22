"""Unit tests for the quality scoring module."""
from __future__ import annotations

from app.quality import clean_text, score_page


def test_clean_text_strips_layout_tokens():
    raw = "<|begin_of_box|>Hello world<|end_of_box|><ref>42</ref>"
    assert clean_text(raw) == "Hello world"


def test_clean_text_strips_eos_tokens():
    # Tokens are stripped wherever they appear — safe for multi-region stitching
    # where each per-region output ends with <|user|>.
    raw = "Region A<|user|>\nRegion B<|user|>"
    assert clean_text(raw) == "Region A\nRegion B"


def test_clean_text_collapses_repetition_tail():
    # Common GLM-OCR stuck-loop failure mode
    head = "Header information for the form.\n" * 4
    block = "Extenuating Circumstances Code: $119\n"
    raw = head + block * 30
    cleaned = clean_text(raw)
    # One copy of the repeating block should remain
    assert cleaned.count("Extenuating Circumstances Code") <= 2
    assert "Header information" in cleaned


def test_clean_text_preserves_short_text():
    raw = "short note"
    assert clean_text(raw) == "short note"


def test_score_page_good_output():
    # A realistic-looking page: dense, has structural elements, no repetition
    text = (
        "# Invoice 2024-01-15\n"
        "Issued to Alice Johnson at 42 Maple Street, Oakland.\n"
        "| Item | Qty | Price |\n"
        "| Widget | 3 | $12.50 |\n"
        "| Gadget | 1 | $5.00 |\n"
        "| Sprocket | 7 | $2.25 |\n"
        "Subtotal: $57.25\n"
        "Tax (7%): $4.01\n"
        "Total due: $61.26\n"
        "Payment terms: net 30 days from 01/15/2024.\n"
        "Thank you for your business. Questions? support@example.com.\n"
    )
    result = score_page(
        raw_text=text,
        num_tokens=200,
        max_tokens=8192,
        image_width=2480,
        image_height=3508,
    )
    assert result.flag == "green"
    assert result.composite >= 0.80
    assert result.needs_external_ocr is False
    assert "no_content" not in result.flag_details


def test_score_page_no_content():
    result = score_page(
        raw_text="",
        num_tokens=0,
        max_tokens=8192,
        image_width=2480,
        image_height=3508,
    )
    assert "no_content" in result.flag_details
    assert result.flag == "red"
    assert result.needs_external_ocr is True
    assert result.composite <= 0.10


def test_score_page_blank_page_cap():
    result = score_page(
        raw_text="abc",
        num_tokens=1,
        max_tokens=8192,
        image_width=2480,
        image_height=3508,
    )
    assert "blank_page" in result.flag_details
    assert result.composite <= 0.10
    assert result.flag == "red"


def test_score_page_low_content():
    result = score_page(
        raw_text="Invoice #42 dated 2024-01-15",
        num_tokens=8,
        max_tokens=8192,
        image_width=2480,
        image_height=3508,
    )
    assert "low_content" in result.flag_details
    assert result.composite <= 0.30


def test_score_page_max_tokens_hit():
    # Hit cap with barely-any output → low token_efficiency, max_tokens_hit flag
    raw = "a" * 50
    result = score_page(
        raw_text=raw,
        num_tokens=8192,
        max_tokens=8192,
        image_width=2480,
        image_height=3508,
    )
    assert "max_tokens_hit" in result.flag_details
    assert result.variables["token_efficiency"] == 0.2


def test_score_page_repetitive_content_flag():
    # Internal repetition the tail-trimmer can't collapse (ends with unique text).
    raw = (
        "The quick brown fox jumps over the lazy dog. " * 40
        + "Now a distinct ending paragraph with different words, punctuation, dates: 2024-01-15, numbers like $42.00."
    )
    result = score_page(
        raw_text=raw,
        num_tokens=600,
        max_tokens=8192,
        image_width=2480,
        image_height=3508,
    )
    assert "repetitive_content" in result.flag_details


def test_score_page_possible_hallucination():
    # Big raw with massive repetition → cleanup removes >75% of it
    raw = "Start of document.\n" + ("Repeat this line over and over.\n" * 500)
    result = score_page(
        raw_text=raw,
        num_tokens=2000,
        max_tokens=8192,
        image_width=2480,
        image_height=3508,
    )
    assert "possible_hallucination" in result.flag_details


def test_score_page_unknown_image_dims_skip_density():
    # When we don't know image size (URL input), density scorer returns 1.0
    result = score_page(
        raw_text="Some clean text here that is long enough not to trip blank caps.",
        num_tokens=12,
        max_tokens=8192,
        image_width=0,
        image_height=0,
    )
    assert result.variables["content_density"] == 1.0


def test_score_page_variables_keys():
    result = score_page(
        raw_text="hello world",
        num_tokens=2,
        max_tokens=8192,
        image_width=100,
        image_height=100,
    )
    expected = {
        "hallucination_ratio",
        "self_consistency",
        "token_efficiency",
        "content_density",
        "structural_integrity",
        "repetition_density",
    }
    assert set(result.variables.keys()) == expected


def test_score_page_self_consistency_always_one():
    result = score_page(
        raw_text="hello",
        num_tokens=1,
        max_tokens=8192,
        image_width=100,
        image_height=100,
    )
    assert result.variables["self_consistency"] == 1.0
