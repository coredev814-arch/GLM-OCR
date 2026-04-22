"""Generic 2D splitter for wide/tall table regions.

Why this exists: PP-DocLayoutV3 sometimes merges visually parallel form columns
*and* multiple form sections into a single region. Handing the whole blob to
GLM-OCR at once leads to dropped content, malformed HTML, and truncated
tables. We slice the region on both axes:

1. Row-split: find horizontal bands of low ink density between sections.
2. Column-split each band into parallel columns.

The signal on both axes is the same: smoothed ink density, local minima via
scipy.signal.find_peaks with a prominence filter, edge-margin suppression.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from PIL import Image

from .layout import Region

logger = logging.getLogger(__name__)


def _detect_dips_1d(
    profile: np.ndarray,
    *,
    min_separation: int,
    prominence: float,
    edge_margin: int,
    max_dips: int,
) -> List[int]:
    """Return positions of local minima in a 1D density profile."""
    try:
        from scipy.signal import find_peaks
    except ImportError:
        logger.warning("scipy not available; skipping split")
        return []

    length = len(profile)
    if length < 100:
        return []

    search = profile[edge_margin : length - edge_margin]
    if search.size < 100:
        return []

    inverted = search.max() - search
    peaks, _ = find_peaks(inverted, distance=min_separation, prominence=prominence)
    if peaks.size == 0:
        return []
    if peaks.size > max_dips:
        logger.debug("split: %d candidate dips exceed max=%d, abandoning", peaks.size, max_dips)
        return []
    return [int(p + edge_margin) for p in peaks]


def _density_profile(binary: np.ndarray, axis: int, *, smoothing_frac: float = 0.0125) -> np.ndarray:
    """Normalized ink-density profile along *axis* (0=vertical, 1=horizontal).

    `smoothing_frac` controls the moving-average kernel size relative to profile
    length. 0.0125 ≈ 1/80 — tight, good for column detection. Row detection
    wants more aggressive smoothing (~0.05) so inter-row gaps within a section
    get averaged away and only inter-section gaps survive as minima.
    """
    pixel_count = binary.shape[axis]
    summed = binary.sum(axis=axis) / (255.0 * pixel_count)
    length = len(summed)
    kernel = max(10, int(length * smoothing_frac))
    return np.convolve(summed, np.ones(kernel) / kernel, mode="same")


def _binarize(crop: Image.Image) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        logger.warning("opencv not available; skipping split")
        return None
    gray = np.array(crop.convert("L"))
    if gray.size == 0:
        return None
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def detect_column_gutters(
    crop: Image.Image,
    *,
    min_separation_frac: float = 0.20,
    prominence_frac: float = 0.3,
    edge_margin_frac: float = 0.12,
    max_gutters: int = 4,
) -> List[int]:
    """Return x-coordinates of column-break local minima (sorted left→right)."""
    binary = _binarize(crop)
    if binary is None:
        return []
    height, width = binary.shape
    if width < 300:
        return []
    profile = _density_profile(binary, axis=0)  # sum down columns
    if profile.max() <= 0:
        return []
    return _detect_dips_1d(
        profile,
        min_separation=max(100, int(width * min_separation_frac)),
        prominence=max(0.01, float(profile.mean()) * prominence_frac),
        edge_margin=int(width * edge_margin_frac),
        max_dips=max_gutters,
    )


def detect_row_gutters(
    crop: Image.Image,
    *,
    min_separation_frac: float = 0.08,
    prominence_frac: float = 0.4,
    edge_margin_frac: float = 0.04,
    max_gutters: int = 8,
    smoothing_frac: float = 0.05,
) -> List[int]:
    """Return y-coordinates of horizontal-band breaks (sorted top→bottom).

    More aggressive smoothing than column detection: inter-row gaps within a
    single section get averaged out, leaving only inter-section dips.
    """
    binary = _binarize(crop)
    if binary is None:
        return []
    height, width = binary.shape
    if height < 300:
        return []
    profile = _density_profile(binary, axis=1, smoothing_frac=smoothing_frac)
    if profile.max() <= 0:
        return []
    return _detect_dips_1d(
        profile,
        min_separation=max(60, int(height * min_separation_frac)),
        prominence=max(0.01, float(profile.mean()) * prominence_frac),
        edge_margin=int(height * edge_margin_frac),
        max_dips=max_gutters,
    )


def split_region_by_columns(
    region: Region,
    *,
    min_column_width: int = 80,
    min_split_width: int = 400,
) -> List[Region]:
    """Split a region at detected column breaks; pass through if none found."""
    if region.crop.width < min_split_width:
        return [region]
    gutters = detect_column_gutters(region.crop)
    if not gutters:
        return [region]
    return _slice_region(region, gutters, axis="x", min_slice=min_column_width)


def split_region_by_rows(
    region: Region,
    *,
    min_row_height: int = 60,
    min_split_height: int = 400,
) -> List[Region]:
    """Split a region at detected horizontal-band breaks; pass through if none."""
    if region.crop.height < min_split_height:
        return [region]
    gutters = detect_row_gutters(region.crop)
    if not gutters:
        return [region]
    return _slice_region(region, gutters, axis="y", min_slice=min_row_height)


def _slice_region(
    region: Region,
    gutters: List[int],
    *,
    axis: str,
    min_slice: int,
) -> List[Region]:
    """Common slicing implementation for column or row split."""
    crop_w, crop_h = region.crop.width, region.crop.height
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = region.bbox_2d
    bbox_w = bbox_x2 - bbox_x1
    bbox_h = bbox_y2 - bbox_y1

    if axis == "x":
        boundaries = [0] + gutters + [crop_w]
    else:
        boundaries = [0] + gutters + [crop_h]

    sub_regions: List[Region] = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if hi - lo < min_slice:
            continue
        if axis == "x":
            sub_crop = region.crop.crop((lo, 0, hi, crop_h))
            sub_bbox = [
                bbox_x1 + int(bbox_w * lo / crop_w),
                bbox_y1,
                bbox_x1 + int(bbox_w * hi / crop_w),
                bbox_y2,
            ]
        else:
            sub_crop = region.crop.crop((0, lo, crop_w, hi))
            sub_bbox = [
                bbox_x1,
                bbox_y1 + int(bbox_h * lo / crop_h),
                bbox_x2,
                bbox_y1 + int(bbox_h * hi / crop_h),
            ]
        sub_regions.append(
            Region(
                index=region.index,
                label=region.label,
                task_type=region.task_type,
                score=region.score,
                bbox_2d=sub_bbox,
                polygon=[],
                crop=sub_crop,
            )
        )

    if len(sub_regions) < 2:
        return [region]
    logger.info(
        "split: region %d (%s) %s-split into %d sub-regions at %s",
        region.index, region.label, axis, len(sub_regions), gutters,
    )
    return sub_regions


def split_region_2d(region: Region) -> List[Region]:
    """Row-split first, then column-split each band. Table regions only."""
    if region.task_type != "table":
        return [region]
    bands = split_region_by_rows(region)
    if len(bands) == 1:
        return split_region_by_columns(region)
    out: List[Region] = []
    for band in bands:
        out.extend(split_region_by_columns(band))
    return out


def split_multi_column_regions(regions: List[Region]) -> List[Region]:
    """Apply 2D splitting to each region; return renumbered reading-order list.

    Sub-regions are ordered primarily by y (top→bottom band), then by x
    (left→right column within the band).
    """
    expanded: List[Region] = []
    for region in regions:
        expanded.extend(split_region_2d(region))
    expanded.sort(key=lambda r: (r.bbox_2d[1], r.bbox_2d[0]))
    for i, region in enumerate(expanded):
        region.index = i
    return expanded
