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

# Color scheme for threshold visualization
COLORS = {
    "very_strong": "#ADD8E6",  # Light blue
    "strong": "#90EE90",       # Light green
    "moderate": "#FFDAB9",     # Peach
    "weak": "#F08080",         # Light coral
}

CELL_STYLE = "color: black; border: 1px solid gray"

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


# Lithology color map for scatter plots
LITHOLOGY_COLOR_MAP: dict[str, str] = {
    "Ophite-THOL": "#FF0000",
    "Ophite-ALK": "#8B0000",
    "Amphibolite": "#0000FF",
    "Eclogite": "#EE82EE",
    "Metabasite": "#00FF00",
    "Basalt-ALK": "#000000",
    "Gabbro-THOL": "#808080",
    "Gabbro-CALC": "#D3D3D3",
    "Gabbro-ALK": "#696969",
    "Metagabbro": "#90EE90",
}

# Region color map for scatter plots
REGION_COLOR_MAP: dict[str, str] = {
    "Upper Guadalquivir": "orange",
    "Southeast": "dodgerblue",
    "Subbaetic": "red",
    "Middle Guadalquivir": "limegreen",
    "Lower Guadalquivir": "mediumpurple",
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

# Map marker colors — one per band, matching Excel conditional formatting
MARKER_COLORS = [
    "lightskyblue",    # band 1 — blue
    "mediumseagreen",  # band 2 — green
    "goldenrod",       # band 3 — yellow / gold
]
MARKER_OVERFLOW = "tomato"  # band 4+ — red

# Table cell background colors — matching Excel conditional formatting
CELL_COLORS = [
    "#ADD8E6",  # band 1 — light blue
    "#90EE90",  # band 2 — light green
    "#FFDAB9",  # band 3 — peach
]
CELL_OVERFLOW = "#F08080"  # band 4+ — light coral

# Excel fill hex (same palette, no # prefix)
EXCEL_COLORS = [c.lstrip("#") for c in CELL_COLORS]
EXCEL_OVERFLOW = CELL_OVERFLOW.lstrip("#")

# Map extent bounds for southern Iberia
MAP_BOUNDS = {
    "west":  -9.8,
    "east":   1.5,
    "south": 33.5,
    "north": 41.5,
}
