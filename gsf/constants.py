from __future__ import annotations

# Epsilon for replacing zero/invalid ratio values before log transform
EPSILON = 1e-10

# Metadata columns that precede ratio data in the Excel sheets
METADATA_COLUMNS = {
    "n", "Accession #", "Site", "Region", "Type", "Lithology",
}

# Aitchison distance thresholds for color coding
AITCH_VERY_STRONG = 1.0
AITCH_STRONG = 2.0
AITCH_MODERATE = 3.0

# Euclidean distance thresholds for color coding
EUCL_VERY_STRONG = 2.0
EUCL_STRONG = 4.0
EUCL_MODERATE = 6.0

# ALR-5 configuration
ALR5_ELEMENTS = ["Ni", "Cr", "Y", "Nb", "Sr", "Zr"]
ALR5_DENOMINATOR = "Zr"
ALR5_NUMERATORS = ["Ni", "Cr", "Y", "Nb", "Sr"]
ALR5_RATIO_NAMES = [
    "ln(Ni/Zr)", "ln(Cr/Zr)", "ln(Y/Zr)",
    "ln(Nb/Zr)", "ln(Sr/Zr)",
]
ALR5_D = 6  # number of parts in the sub-composition

# ALR-5 Aitchison thresholds (scaled by sqrt(5)/sqrt(16))
ALR5_AITCH_VERY_STRONG = 0.56
ALR5_AITCH_STRONG = 1.12
ALR5_AITCH_MODERATE = 1.68

# Geographical distance thresholds (km) for color coding
GEO_VERY_CLOSE_KM = 60
GEO_CLOSE_KM = 120
GEO_MODERATE_KM = 200

# Distance rounding precision for display
GEO_DIST_ROUND_TO_KM = 10

# Default / max number of results
DEFAULT_TOP_N = 20
MAX_TOP_N = 100

# ---------------------------------------------------------------------------
# Academic earth-tone color palette
# ---------------------------------------------------------------------------

# Color scheme for threshold visualization (4-tier earth tones)
COLORS = {
    "very_strong": "#4A7C8A",  # Slate teal
    "strong": "#7A9A5E",       # Sage green
    "moderate": "#C9A96E",     # Sandstone
    "weak": "#A65D57",         # Terra cotta
}

CELL_STYLE = "color: #2C2825; font-weight: 500; border: 1px solid #2C2825; text-align: right"

# Mode labels for column headers
MODE_LABELS: dict[str, str] = {
    "alr5": "ALR-5",
    "trace": "Trace",
    "all": "All",
}

# Canonical mode ordering (ALR-5 first = default sort)
MODE_ORDER: list[str] = ["alr5", "trace", "all"]


def aitch_col(mode_key: str) -> str:
    """Return the Aitchison distance column name for a mode."""
    return f"Aitch ({MODE_LABELS[mode_key]})"


# Lithology color map for scatter plots (muted geological tones)
LITHOLOGY_COLOR_MAP: dict[str, str] = {
    "Ophite-THOL": "#B5453A",   # Muted brick red
    "Ophite-ALK": "#7A3230",    # Dark garnet
    "Amphibolite": "#4A6FA5",   # Steel blue
    "Eclogite": "#8B6FAD",      # Muted purple
    "Metabasite": "#5E8C61",    # Forest sage
    "Basalt-ALK": "#3C3C3C",    # Charcoal
    "Gabbro-THOL": "#8A8A8A",   # Medium gray
    "Gabbro-CALC": "#B8B0A2",   # Warm light gray
    "Gabbro-ALK": "#6B6460",    # Warm dark gray
    "Metagabbro": "#7AAD7D",    # Light sage
    "Sillimanite": "#D4A03C",   # Golden amber
}

# Region color map for scatter plots
REGION_COLOR_MAP: dict[str, str] = {
    "Upper Guadalquivir": "#C9873A",  # Amber ochre
    "Southeast": "#4A6FA5",           # Steel blue
    "Subbaetic": "#B5453A",           # Brick red
    "Middle Guadalquivir": "#5E8C61", # Forest sage
    "Lower Guadalquivir": "#7B6B99",  # Muted lavender
}


def aitch_thresholds(
    mode_key: str,
) -> tuple[float, float, float]:
    """Return Aitchison thresholds for the given mode."""
    if mode_key == "alr5":
        return (
            ALR5_AITCH_VERY_STRONG,
            ALR5_AITCH_STRONG,
            ALR5_AITCH_MODERATE,
        )
    return (AITCH_VERY_STRONG, AITCH_STRONG, AITCH_MODERATE)


def has_euclidean(mode_key: str) -> bool:
    """Return True if the mode includes Euclidean distance."""
    return mode_key != "alr5"


# Paired-color step size per mode
AITCH_STEP: dict[str, float] = {"alr5": 0.5, "trace": 1.0, "all": 1.5}

# Map marker colors — one per band (earth-tone palette)
MARKER_COLORS = [
    "#4A7C8A",       # band 1 — slate teal
    "#7A9A5E",       # band 2 — sage green
    "#C9A96E",       # band 3 — sandstone
]
MARKER_OVERFLOW = "#A65D57"  # band 4+ — terra cotta

# Table cell background colors (earth-tone palette)
CELL_COLORS = [
    "#4A7C8A",  # band 1 — slate teal
    "#7A9A5E",  # band 2 — sage green
    "#C9A96E",  # band 3 — sandstone
]
CELL_OVERFLOW = "#A65D57"  # band 4+ — terra cotta

# Excel fill hex (same palette, no # prefix)
EXCEL_COLORS = [c.lstrip("#") for c in CELL_COLORS]
EXCEL_OVERFLOW = CELL_OVERFLOW.lstrip("#")

# Confidence score band thresholds and colors
CONF_VERY_HIGH = 75   # > 75%
CONF_HIGH = 60        # 60-75%
CONF_MODERATE = 45    # 45-60%
CONF_LOW = 45         # kept for import compat; styling uses 4 bands

CONF_COLORS: dict[str, str] = {
    "very_high": "#3D7A5F",  # Muted forest green
    "high": "#7A9A5E",       # Sage
    "moderate": "#C9A96E",   # Sandstone
    "low": "#A65D57",        # Terra cotta
}

CONF_EXCEL_COLORS: dict[str, str] = {
    k: v.lstrip("#") for k, v in CONF_COLORS.items()
}

# Map extent bounds for southern Iberia
MAP_BOUNDS = {
    "west":  -9.8,
    "east":   1.5,
    "south": 33.5,
    "north": 41.5,
}

# ---------------------------------------------------------------------------
# Plotly chart styling (publication-quality defaults)
# ---------------------------------------------------------------------------

PLOTLY_FONT = {
    "family": "Source Sans 3, Source Sans Pro, Helvetica, Arial, sans-serif",
    "color": "#2C2825",
    "size": 13,
}

PLOTLY_LAYOUT = {
    "font": PLOTLY_FONT,
    "title": {
        "font": {
            "family": "Crimson Pro, Georgia, serif",
            "size": 18,
            "color": "#2C2825",
        },
        "x": 0.0,
        "xanchor": "left",
    },
    "paper_bgcolor": "#FAFAF7",
    "plot_bgcolor": "#FAFAF7",
    "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
    "legend": {
        "font": {"size": 11},
        "bgcolor": "rgba(250,250,247,0.9)",
        "bordercolor": "#D4CCC0",
        "borderwidth": 1,
    },
}

PLOTLY_AXIS_STYLE = {
    "showline": True,
    "linecolor": "#2C2825",
    "linewidth": 1,
    "ticks": "outside",
    "tickfont": {"color": "#2C2825", "size": 11},
    "title_font": {"size": 13, "color": "#3D3530"},
    "showgrid": True,
    "gridcolor": "#E8E2D8",
    "gridwidth": 0.5,
    "griddash": "dot",
    "zeroline": False,
    "mirror": False,
}
