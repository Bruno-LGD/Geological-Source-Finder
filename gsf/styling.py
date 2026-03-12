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
    CONF_LOW,
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


# Per-zone gradient pairs for ALR-5: (light_color, dark_color)
# Each zone keeps its identity (blue/green/peach/coral) while darker
# shades indicate worse matches within the zone.
_ALR5_ZONE_GRADIENTS: list[
    tuple[float, float, tuple[int, int, int], tuple[int, int, int]]
] = [
    # (zone_start, zone_end, light_rgb, dark_rgb)
    (0.0, ALR5_AITCH_VERY_STRONG, (173, 216, 230), (70, 130, 180)),
    (
        ALR5_AITCH_VERY_STRONG,
        ALR5_AITCH_STRONG,
        (144, 238, 144),
        (46, 139, 87),
    ),
    (
        ALR5_AITCH_STRONG,
        ALR5_AITCH_MODERATE,
        (255, 218, 185),
        (210, 105, 30),
    ),
    (
        ALR5_AITCH_MODERATE,
        float("inf"),
        (240, 128, 128),
        (178, 34, 34),
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
    return (
        f"background-color: {alr5_zone_color(v)}; {CELL_STYLE};"
        " border-radius: 0px"
    )


def color_confidence(val: Any) -> str:  # noqa: ANN401
    """Apply background color to a Conf cell."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v >= CONF_VERY_HIGH:
        color = CONF_COLORS["very_high"]
        text_color = "white"
    elif v >= CONF_HIGH:
        color = CONF_COLORS["high"]
        text_color = "black"
    elif v >= CONF_MODERATE:
        color = CONF_COLORS["moderate"]
        text_color = "black"
    elif v >= CONF_LOW:
        color = CONF_COLORS["low"]
        text_color = "black"
    else:
        color = CONF_COLORS["very_low"]
        text_color = "black"
    return (
        f"background-color: {color};"
        f" color: {text_color};"
        " border: 1px solid gray;"
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
