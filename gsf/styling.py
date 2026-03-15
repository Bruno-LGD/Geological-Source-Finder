from __future__ import annotations

from typing import Any

import pandas as pd

from gsf.constants import (
    ALR5_AITCH_MODERATE,
    ALR5_AITCH_STRONG,
    ALR5_AITCH_VERY_STRONG,
    CELL_STYLE,
    COLORS,
    CONF_COLORS,
    CONF_HIGH,
    CONF_MODERATE,
    CONF_VERY_HIGH,
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


def _text_color_for_bg(hex_bg: str) -> str:
    """Return white or dark text depending on background luminance."""
    h = hex_bg.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#FAFAF7" if luminance < 0.55 else "#2C2825"


def highlight_geo_dist(s: pd.Series) -> list[str]:
    """Apply background color to Geo Dist cells (right-aligned)."""
    styles = []
    for val in s:
        try:
            num = float(str(val).replace(" km", ""))
        except (ValueError, TypeError, AttributeError):
            styles.append("text-align: right")
            continue
        if num < GEO_VERY_CLOSE_KM:
            color = COLORS["very_strong"]
        elif num < GEO_CLOSE_KM:
            color = COLORS["strong"]
        elif num < GEO_MODERATE_KM:
            color = COLORS["moderate"]
        else:
            color = COLORS["weak"]
        text_c = _text_color_for_bg(color)
        styles.append(
            f"background-color: {color};"
            f" color: {text_c}; {CELL_STYLE};"
            " text-align: right"
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
    text_c = _text_color_for_bg(color)
    return (
        f"background-color: {color};"
        f" color: {text_c}; {CELL_STYLE}"
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
    text_c = _text_color_for_bg(color)
    return (
        f"background-color: {color};"
        f" color: {text_c}; {CELL_STYLE}"
    )


# Per-zone gradient pairs for ALR-5 (earth-tone palette)
# Each zone transitions from lighter to darker within its band.
_ALR5_ZONE_GRADIENTS: list[
    tuple[float, float, tuple[int, int, int], tuple[int, int, int]]
] = [
    # (zone_start, zone_end, light_rgb, dark_rgb)
    (0.0, ALR5_AITCH_VERY_STRONG, (106, 156, 168), (52, 98, 112)),
    (
        ALR5_AITCH_VERY_STRONG,
        ALR5_AITCH_STRONG,
        (146, 176, 118), (82, 118, 58),
    ),
    (
        ALR5_AITCH_STRONG,
        ALR5_AITCH_MODERATE,
        (214, 186, 132), (168, 132, 62),
    ),
    (
        ALR5_AITCH_MODERATE,
        float("inf"),
        (186, 118, 112), (138, 62, 52),
    ),
]


def alr5_zone_color(v: float) -> str:
    """Return hex color for ALR-5 distance using per-zone gradient."""
    for z_start, z_end, c_light, c_dark in _ALR5_ZONE_GRADIENTS:
        if v < z_end or z_end == float("inf"):
            width = (
                z_end - z_start
                if z_end != float("inf")
                else _ALR5_ZONE_GRADIENTS[-2][1]
            )
            t = min((v - z_start) / width, 1.0) if width > 0 else 0.0
            r = int(c_light[0] + t * (c_dark[0] - c_light[0]))
            g = int(c_light[1] + t * (c_dark[1] - c_light[1]))
            b = int(c_light[2] + t * (c_dark[2] - c_light[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    _, _, _, c = _ALR5_ZONE_GRADIENTS[-1]
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def color_aitch_alr5_gradient(val: Any) -> str:  # noqa: ANN401
    """Per-zone brightness gradient coloring for ALR-5 Aitch Dist."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    hex_bg = alr5_zone_color(v)
    text_c = _text_color_for_bg(hex_bg)
    return (
        f"background-color: {hex_bg};"
        f" color: {text_c}; {CELL_STYLE}"
    )


def color_confidence(val: Any) -> str:  # noqa: ANN401
    """Apply background color to a Conf cell (4 bands matching Excel)."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v > CONF_VERY_HIGH:
        color = CONF_COLORS["very_high"]
    elif v >= CONF_HIGH:
        color = CONF_COLORS["high"]
    elif v >= CONF_MODERATE:
        color = CONF_COLORS["moderate"]
    else:
        color = CONF_COLORS["low"]
    text_c = _text_color_for_bg(color)
    return (
        f"background-color: {color};"
        f" color: {text_c};"
        f" {CELL_STYLE}"
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
