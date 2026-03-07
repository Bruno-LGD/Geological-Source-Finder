from __future__ import annotations

from typing import Any

import pandas as pd

from gsf.constants import (
    CELL_STYLE,
    COLORS,
    EUCL_MODERATE,
    EUCL_STRONG,
    EUCL_VERY_STRONG,
    GEO_CLOSE_KM,
    GEO_MODERATE_KM,
    GEO_VERY_CLOSE_KM,
    AITCH_STEP,
    CELL_COLORS,
    CELL_OVERFLOW,
    MARKER_COLORS,
    MARKER_OVERFLOW,
)


def highlight_geo_dist(s: pd.Series) -> list[str]:
    """Apply background color to Geo Dist cells."""
    styles = []
    for val in s:
        try:
            num = float(str(val).replace(" km", ""))
        except (ValueError, TypeError, AttributeError):
            styles.append("")
            continue
        if num < GEO_VERY_CLOSE_KM:
            color = COLORS["very_strong"]
        elif num < GEO_CLOSE_KM:
            color = COLORS["strong"]
        elif num < GEO_MODERATE_KM:
            color = COLORS["moderate"]
        else:
            color = COLORS["weak"]
        styles.append(
            f"background-color: {color}; {CELL_STYLE};"
        )
    return styles


def _color_dist_cell(
    val: Any,  # noqa: ANN401
    thresholds: tuple[float, float, float],
) -> str:
    """Apply background color to a distance cell."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    t1, t2, t3 = thresholds
    if v < t1:
        color = COLORS["very_strong"]
    elif v < t2:
        color = COLORS["strong"]
    elif v < t3:
        color = COLORS["moderate"]
    else:
        color = COLORS["weak"]
    return (
        f"background-color: {color}; {CELL_STYLE};"
        " border-radius: 0px"
    )


def color_eucl_dist(val: Any) -> str:  # noqa: ANN401
    """Apply background color to Eucl Dist cells."""
    return _color_dist_cell(
        val, (EUCL_VERY_STRONG, EUCL_STRONG, EUCL_MODERATE),
    )


def color_aitch_with_thresholds(
    val: Any,  # noqa: ANN401
    mode_key: str = "trace",
) -> str:
    """Apply paired-color background using fixed step per mode."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    step = AITCH_STEP.get(mode_key, 0.8)
    color = CELL_OVERFLOW
    for i, c in enumerate(CELL_COLORS, start=1):
        if v < i * step:
            color = c
            break
    return (
        f"background-color: {color}; {CELL_STYLE};"
        " border-radius: 0px"
    )


def aitch_to_marker_color(
    val: float, mode_key: str = "trace",
) -> str:
    """Map Aitchison distance to a marker color."""
    step = AITCH_STEP.get(mode_key, 0.8)
    for i, color in enumerate(MARKER_COLORS, start=1):
        if val < i * step:
            return color
    return MARKER_OVERFLOW
