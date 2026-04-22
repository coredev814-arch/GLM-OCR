"""Unit tests for the column-splitter.

We simulate columnar ink density using filled rectangles rather than rendered
text, so results don't depend on font availability or font metrics.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from app.column_split import (
    detect_column_gutters,
    detect_row_gutters,
    split_multi_column_regions,
    split_region_2d,
    split_region_by_columns,
    split_region_by_rows,
)
from app.layout import Region


def _make_columns(
    columns: list[tuple[int, int]],
    *,
    width: int = 1600,
    height: int = 800,
    row_height: int = 30,
    row_gap: int = 20,
    top: int = 40,
    bottom: int = 40,
) -> Image.Image:
    """Render filled bars at given x-ranges; simulates columns of inky text.

    Each tuple in *columns* is (x_start, x_end) — a column block of full-width
    rows extending top→bottom. Overlapping / nested ranges work.
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    usable = height - top - bottom
    for y in range(top, height - bottom, row_height + row_gap):
        for x1, x2 in columns:
            draw.rectangle([x1, y, x2, y + row_height], fill="black")
    return img


def test_single_column_no_split():
    # One wide ink block, no internal gutters
    img = _make_columns([(80, 1500)], width=1600)
    assert detect_column_gutters(img) == []


def test_three_column_detects_two_gutters():
    # 3 ink blocks with 60-px gutters between them
    img = _make_columns(
        [(60, 500), (560, 1040), (1100, 1540)],
        width=1600,
    )
    gutters = detect_column_gutters(img)
    assert len(gutters) == 2, f"expected 2 gutters, got {gutters}"
    assert 500 < gutters[0] < 560
    assert 1040 < gutters[1] < 1100


def test_two_column_with_wide_gutter():
    img = _make_columns([(60, 700), (900, 1540)], width=1600)
    gutters = detect_column_gutters(img)
    assert len(gutters) == 1
    assert 700 < gutters[0] < 900


def test_too_narrow_image_skipped():
    img = Image.new("RGB", (200, 400), "white")
    assert detect_column_gutters(img) == []


def test_split_region_passes_through_when_no_gutters():
    img = _make_columns([(80, 1500)], width=1600)
    region = Region(
        index=3,
        label="table",
        task_type="table",
        score=0.9,
        bbox_2d=[0, 100, 1000, 800],
        polygon=[],
        crop=img,
    )
    out = split_region_by_columns(region)
    assert len(out) == 1
    assert out[0] is region


def test_split_region_produces_sub_regions():
    img = _make_columns(
        [(60, 500), (560, 1040), (1100, 1540)],
        width=1600,
    )
    region = Region(
        index=4,
        label="table",
        task_type="table",
        score=0.95,
        bbox_2d=[100, 200, 1100, 900],
        polygon=[],
        crop=img,
    )
    out = split_region_by_columns(region)
    assert len(out) == 3
    assert out[0].bbox_2d[0] == region.bbox_2d[0]
    assert out[-1].bbox_2d[2] == region.bbox_2d[2]
    for sub in out:
        assert sub.bbox_2d[1] == region.bbox_2d[1]
        assert sub.bbox_2d[3] == region.bbox_2d[3]
        assert sub.crop.width < region.crop.width


def _make_bands(bands, *, width=1600, row_height=30, row_gap=20, band_gap=140):
    """Stack N horizontal bands of inky text separated by blank bands."""
    heights = [sum(row_height + row_gap for _ in band) for band in bands]
    total = sum(heights) + band_gap * (len(bands) - 1) + 100
    img = Image.new("RGB", (width, total), "white")
    draw = ImageDraw.Draw(img)
    y = 50
    for band_idx, band in enumerate(bands):
        for (x1, x2) in band:
            bar_y = y
            for _ in band:
                draw.rectangle([x1, bar_y, x2, bar_y + row_height], fill="black")
                bar_y += row_height + row_gap
        y += heights[band_idx] + band_gap
    return img


def test_detect_row_gutters_two_bands():
    # Two horizontal bands of content separated by a big blank stripe
    img = Image.new("RGB", (1200, 1000), "white")
    d = ImageDraw.Draw(img)
    for y in range(40, 380, 50):
        d.rectangle([60, y, 1100, y + 30], fill="black")
    for y in range(620, 960, 50):
        d.rectangle([60, y, 1100, y + 30], fill="black")
    gutters = detect_row_gutters(img)
    assert len(gutters) == 1
    assert 400 < gutters[0] < 600


def test_split_region_by_rows_produces_bands():
    img = Image.new("RGB", (1200, 1000), "white")
    d = ImageDraw.Draw(img)
    for y in range(40, 380, 50):
        d.rectangle([60, y, 1100, y + 30], fill="black")
    for y in range(620, 960, 50):
        d.rectangle([60, y, 1100, y + 30], fill="black")
    region = Region(
        index=2,
        label="table",
        task_type="table",
        score=0.9,
        bbox_2d=[100, 200, 1300, 1200],
        polygon=[],
        crop=img,
    )
    out = split_region_by_rows(region)
    assert len(out) == 2
    assert out[0].bbox_2d[1] == region.bbox_2d[1]
    assert out[-1].bbox_2d[3] == region.bbox_2d[3]
    for sub in out:
        assert sub.bbox_2d[0] == region.bbox_2d[0]
        assert sub.bbox_2d[2] == region.bbox_2d[2]
        assert sub.crop.height < region.crop.height


def test_split_region_2d_skips_non_table():
    img = Image.new("RGB", (1200, 1000), "white")
    d = ImageDraw.Draw(img)
    for y in range(40, 960, 50):
        d.rectangle([60, y, 1100, y + 30], fill="black")
    region = Region(
        index=0,
        label="text",
        task_type="text",
        score=0.9,
        bbox_2d=[0, 0, 1200, 1000],
        polygon=[],
        crop=img,
    )
    out = split_region_2d(region)
    assert len(out) == 1


def test_split_multi_column_regions_reorders_by_reading_order():
    """Sub-regions should come out sorted by (y, x) top-to-bottom, left-to-right."""
    img = Image.new("RGB", (1200, 1000), "white")
    d = ImageDraw.Draw(img)
    # Two bands, each with two columns
    for y in range(40, 380, 50):
        d.rectangle([60, y, 520, y + 30], fill="black")
        d.rectangle([680, y, 1100, y + 30], fill="black")
    for y in range(620, 960, 50):
        d.rectangle([60, y, 520, y + 30], fill="black")
        d.rectangle([680, y, 1100, y + 30], fill="black")
    region = Region(
        index=5,
        label="table",
        task_type="table",
        score=0.95,
        bbox_2d=[0, 0, 1200, 1000],
        polygon=[],
        crop=img,
    )
    out = split_multi_column_regions([region])
    assert len(out) == 4, f"expected 2 bands x 2 cols = 4, got {len(out)}"
    # Top-left → top-right → bottom-left → bottom-right
    tops = [r.bbox_2d[1] for r in out]
    assert tops == sorted(tops)
    # Within same band, x should be sorted
    assert out[0].bbox_2d[0] < out[1].bbox_2d[0]
    assert out[2].bbox_2d[0] < out[3].bbox_2d[0]
