from __future__ import annotations

import os
import logging
from datetime import datetime
from io import BytesIO
from collections import Counter
from typing import Any

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from openpyxl.styles import PatternFill, Font, Border, Side

# -----------------------------
# Page Configuration (must be first Streamlit call)
# -----------------------------
st.set_page_config(
    page_title="GSF - Geology Source Finder",
    page_icon="\U0001FAA8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
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


# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(
    element_mode: str = "trace",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required Excel files from a 'Data' folder.

    Args:
        element_mode: "trace" for 16 ratio columns,
            "all" for 22 ratio columns.

    Returns:
        (artefact_df, geology_df, geo_coords_df, arch_coords_df)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_dirs = [
        os.path.join(base_dir, "Data"),
        os.path.join(base_dir, "..", "Data"),
    ]

    if element_mode == "all":
        filenames = {
            "artefact": "AXEs metabasite data (All Elements).xlsx",
            "geology": "Geology samples data (All Elements).xlsx",
            "coords": "Coordinates sheet.xlsx",
        }
    else:
        filenames = {
            "artefact": "AXEs metabasite data (Trace Elements).xlsx",
            "geology": "Geology samples data (Trace Elements).xlsx",
            "coords": "Coordinates sheet.xlsx",
        }

    found_data_dir = None
    for d in possible_dirs:
        paths = [os.path.join(d, f) for f in filenames.values()]
        if all(os.path.exists(p) for p in paths):
            found_data_dir = d
            logger.info("Data found in directory: %s", d)
            break

    if not found_data_dir:
        raise FileNotFoundError(
            f"Required files not found in: {possible_dirs}\n"
            "Place the Excel files in a 'Data' folder "
            "next to this script."
        )

    try:
        artefact_df = pd.read_excel(
            os.path.join(found_data_dir, filenames["artefact"]),
            sheet_name="AXEs Ratios",
        )
        geology_df = pd.read_excel(
            os.path.join(found_data_dir, filenames["geology"]),
            sheet_name="Geology ratios",
        )
        coords_sheets = pd.read_excel(
            os.path.join(found_data_dir, filenames["coords"]),
            sheet_name=[
                "Geology Samples Coord",
                "Archaeology Sites Coord",
            ],
        )
        geo_coords_df = coords_sheets["Geology Samples Coord"]
        arch_coords_df = coords_sheets["Archaeology Sites Coord"]
    except Exception as e:
        logger.exception("Error loading Excel files.")
        st.error(
            "Error loading Excel files. "
            "Check file names and sheet names."
        )
        raise e

    artefact_df["Accession #"] = (
        artefact_df["Accession #"].astype(str)
    )
    geology_df["Accession #"] = (
        geology_df["Accession #"].astype(str)
    )
    geo_coords_df["Accession #"] = (
        geo_coords_df["Accession #"].astype(str)
    )

    # Longitude values in the coordinate sheets are inconsistently stored:
    # most Spanish sites use positive values (e.g. 3.47 instead of -3.47)
    # while some sites (e.g. Vila Nova de São Pedro) are already negative.
    # All sites are in the Iberian Peninsula (west of the prime meridian),
    # so force all longitudes to negative with -abs() to normalise both cases.
    geo_coords_df["Longitude"] = geo_coords_df["Longitude"].abs() * -1
    arch_coords_df["Longitude"] = arch_coords_df["Longitude"].abs() * -1

    return artefact_df, geology_df, geo_coords_df, arch_coords_df


# -----------------------------
# Ratio Column Extraction
# -----------------------------
def _get_ratio_columns(sample: pd.Series) -> list[str]:
    """Extract ratio column names, excluding metadata and NaN."""
    ratio_data = sample.drop(
        labels=[c for c in METADATA_COLUMNS if c in sample.index],
    )
    return list(ratio_data.dropna().index)


def _count_total_ratio_columns(df: pd.DataFrame) -> int:
    """Count total ratio columns (all non-metadata columns)."""
    return len(
        [c for c in df.columns if c not in METADATA_COLUMNS]
    )


# -----------------------------
# Vectorized Distance Calculations
# -----------------------------
def _compute_distances_vectorized(
    query_ratios: np.ndarray, target_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Aitchison and Euclidean distances.

    Uses numpy broadcasting between one query vector
    and a matrix of target vectors.

    The Aitchison distance uses the Centered Log-Ratio transform:
        CLR(x) = log(x) - mean(log(x))
        d_A(x, y) = ||CLR(x) - CLR(y)||_2
    """
    q = np.where(
        np.isfinite(query_ratios) & (query_ratios > 0),
        query_ratios, EPSILON,
    )
    t = np.where(
        np.isfinite(target_matrix) & (target_matrix > 0),
        target_matrix, EPSILON,
    )

    log_q = np.log(q)
    clr_q = log_q - np.mean(log_q)

    log_t = np.log(t)
    clr_t = log_t - np.mean(log_t, axis=1, keepdims=True)

    aitchison_dists = np.linalg.norm(clr_t - clr_q, axis=1)
    euclidean_dists = np.linalg.norm(
        target_matrix - query_ratios, axis=1,
    )

    return aitchison_dists, euclidean_dists


def _compute_geodesic(
    origin: tuple[float, float] | None,
    destination: tuple[float, float] | None,
) -> str:
    """Compute geodesic distance in km, formatted as a string."""
    if origin is None or destination is None:
        return "Unknown"
    try:
        dist_km = geodesic(origin, destination).km
        rounded = int(
            round(dist_km / GEO_DIST_ROUND_TO_KM)
            * GEO_DIST_ROUND_TO_KM
        )
        return f"{rounded} km"
    except (ValueError, TypeError):
        return "Unknown"


# -----------------------------
# Unified Match Function
# -----------------------------
def get_top_matches(
    query_sample: pd.Series,
    target_df: pd.DataFrame,
    query_coords_df: pd.DataFrame,
    target_coords_df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    direction: str = "artefact_to_geology",
) -> pd.DataFrame:
    """Find the top-N matching samples by compositional similarity.

    Computes Aitchison and Euclidean distances (vectorized),
    sorts by [Aitchison, Euclidean], takes top_n, then computes
    geodesic distances only for those results.
    """
    ratio_cols = _get_ratio_columns(query_sample)
    if not ratio_cols:
        return pd.DataFrame()

    if direction == "artefact_to_geology":
        keep_cols = [
            "Lithology", "Accession #", "Site", "Region",
        ]
    else:
        keep_cols = ["Accession #", "Site", "Region"]

    available_cols = [
        c for c in keep_cols + ratio_cols
        if c in target_df.columns
    ]
    aligned_df = target_df[available_cols].dropna(  # pyright: ignore[reportCallIssue]
        subset=ratio_cols,
    )

    if aligned_df.empty:
        return pd.DataFrame()

    query_series = pd.Series(pd.to_numeric(
        query_sample[ratio_cols], errors="coerce",
    )).fillna(EPSILON)
    query_vector = np.asarray(query_series, dtype=float)
    target_matrix = np.asarray(
        aligned_df[ratio_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(EPSILON),
        dtype=float,
    )

    aitch_dists, eucl_dists = _compute_distances_vectorized(
        query_vector, target_matrix,
    )

    results_df = aligned_df[keep_cols].copy()
    results_df = results_df.reset_index(drop=True)
    results_df["Aitch Dist"] = np.round(aitch_dists, 2)
    results_df["Eucl Dist"] = np.round(eucl_dists, 2)

    results_df = results_df.sort_values(  # pyright: ignore[reportCallIssue]
        by=["Aitch Dist", "Eucl Dist"],
    ).head(top_n)
    results_df.reset_index(drop=True, inplace=True)
    results_df.index += 1

    # Compute geodesic distances only for top_n results
    if direction == "artefact_to_geology":
        site_match = query_coords_df[
            query_coords_df["Site"] == query_sample["Site"]
        ]
        query_coords: tuple[float, float] | None = (
            (
                float(site_match.iloc[0]["Latitude"]),
                float(site_match.iloc[0]["Longitude"]),
            )
            if not site_match.empty
            else None
        )
        target_coord_lookup: dict[
            str, tuple[float, float]
        ] = {}
        for _, row in target_coords_df.iterrows():
            target_coord_lookup[str(row["Accession #"])] = (
                float(row["Latitude"]),
                float(row["Longitude"]),
            )
        results_df["Geo Dist"] = (
            results_df["Accession #"].apply(
                lambda acc: _compute_geodesic(
                    query_coords,
                    target_coord_lookup.get(str(acc)),
                )
            )
        )
    else:
        acc_match = query_coords_df[
            query_coords_df["Accession #"]
            == str(query_sample["Accession #"])
        ]
        query_coords: tuple[float, float] | None = (
            (
                float(acc_match.iloc[0]["Latitude"]),
                float(acc_match.iloc[0]["Longitude"]),
            )
            if not acc_match.empty
            else None
        )
        target_coord_lookup = {}
        for _, row in target_coords_df.iterrows():
            target_coord_lookup[str(row["Site"])] = (
                float(row["Latitude"]),
                float(row["Longitude"]),
            )
        results_df["Geo Dist"] = results_df["Site"].apply(
            lambda site: _compute_geodesic(
                query_coords,
                target_coord_lookup.get(site),
            )
        )

    if direction == "artefact_to_geology":
        results_df = results_df.rename(columns={
            "Accession #": "Geo Acc #",
            "Site": "Geo Site",
        })
    else:
        results_df = results_df.rename(columns={
            "Accession #": "Artefact Acc #",
            "Site": "Artefact Site",
            "Region": "Artefact Region",
        })

    return results_df


# -----------------------------
# Table Styling Functions
# -----------------------------
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


def color_aitch_dist(val: Any) -> str:  # noqa: ANN401
    """Apply background color to Aitch Dist cells."""
    return _color_dist_cell(
        val, (AITCH_VERY_STRONG, AITCH_STRONG, AITCH_MODERATE),
    )


def color_eucl_dist(val: Any) -> str:  # noqa: ANN401
    """Apply background color to Eucl Dist cells."""
    return _color_dist_cell(
        val, (EUCL_VERY_STRONG, EUCL_STRONG, EUCL_MODERATE),
    )


# -----------------------------
# Excel Export with Formatting
# -----------------------------
def _get_fill_for_value(
    val: Any,  # noqa: ANN401
    thresholds: tuple[float, float, float],
) -> PatternFill | None:
    """Return a PatternFill for a numeric value."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return None
    t1, t2, t3 = thresholds
    if v < t1:
        return PatternFill(
            start_color="ADD8E6", fill_type="solid",
        )
    if v < t2:
        return PatternFill(
            start_color="90EE90", fill_type="solid",
        )
    if v < t3:
        return PatternFill(
            start_color="FFDAB9", fill_type="solid",
        )
    return PatternFill(
        start_color="F08080", fill_type="solid",
    )


def _get_geo_fill(val: Any) -> PatternFill | None:  # noqa: ANN401
    """Return a PatternFill for a Geo Dist string value."""
    try:
        num = float(str(val).replace(" km", ""))
    except (ValueError, TypeError, AttributeError):
        return None
    return _get_fill_for_value(
        num,
        (GEO_VERY_CLOSE_KM, GEO_CLOSE_KM, GEO_MODERATE_KM),
    )


def _create_styled_excel(
    results_df: pd.DataFrame,
    query_accession: str,
    query_site: str,
    direction: str,
) -> bytes:
    """Create a styled Excel workbook with color-coded cells."""
    output = BytesIO()
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:  # pyright: ignore[reportArgumentType]
        results_df.to_excel(
            writer, sheet_name="Results",
            index=True, startrow=2,
        )
        ws = writer.sheets["Results"]

        # Header metadata
        ws["A1"] = (
            f"Query: {query_accession} — {query_site}"
        )
        ws["A1"].font = Font(bold=True, size=12)
        dir_label = (
            "Artefact -> Geology"
            if direction == "artefact_to_geology"
            else "Geology -> Artefact"
        )
        ws["A2"] = (
            f"Direction: {dir_label} | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        # Find column letters for distance columns
        # row where pandas writes headers
        # (0-indexed startrow=2 -> Excel row 3)
        header_row = 3
        col_map: dict[str, int] = {}
        for cell in ws[header_row]:
            if cell.value in (
                "Aitch Dist", "Eucl Dist", "Geo Dist",
            ):
                col_map[cell.value] = cell.column

        aitch_thresholds = (
            AITCH_VERY_STRONG, AITCH_STRONG, AITCH_MODERATE,
        )
        eucl_thresholds = (
            EUCL_VERY_STRONG, EUCL_STRONG, EUCL_MODERATE,
        )

        # Apply fills to data rows
        for row in ws.iter_rows(
            min_row=header_row + 1, max_row=ws.max_row,
        ):
            for cell in row:
                cell.border = thin_border
                if cell.column == col_map.get("Aitch Dist"):
                    fill = _get_fill_for_value(
                        cell.value, aitch_thresholds,
                    )
                    if fill:
                        cell.fill = fill
                elif cell.column == col_map.get("Eucl Dist"):
                    fill = _get_fill_for_value(
                        cell.value, eucl_thresholds,
                    )
                    if fill:
                        cell.fill = fill
                elif cell.column == col_map.get("Geo Dist"):
                    fill = _get_geo_fill(cell.value)
                    if fill:
                        cell.fill = fill

        # Style header row
        for cell in ws[header_row]:
            cell.font = Font(bold=True)
            cell.border = thin_border

    return output.getvalue()


# -----------------------------
# UI Display Functions
# -----------------------------
def display_legend() -> None:
    """Display color-coded threshold explanations."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Aitchison Distance** (compositional)")
        st.markdown(
            f"- :blue[< {AITCH_VERY_STRONG}] "
            "-- Very strong match\n"
            f"- :green[{AITCH_VERY_STRONG} - "
            f"{AITCH_STRONG}] -- Strong match\n"
            f"- :orange[{AITCH_STRONG} - "
            f"{AITCH_MODERATE}] -- Moderate match\n"
            f"- :red[> {AITCH_MODERATE}] -- Weak match"
        )
    with col2:
        st.markdown("**Euclidean Distance** (geometric)")
        st.markdown(
            f"- :blue[< {EUCL_VERY_STRONG}] "
            "-- Very strong match\n"
            f"- :green[{EUCL_VERY_STRONG} - "
            f"{EUCL_STRONG}] -- Strong match\n"
            f"- :orange[{EUCL_STRONG} - "
            f"{EUCL_MODERATE}] -- Moderate match\n"
            f"- :red[> {EUCL_MODERATE}] -- Weak match"
        )
    with col3:
        st.markdown("**Geographical Distance**")
        st.markdown(
            f"- :blue[< {GEO_VERY_CLOSE_KM} km] "
            "-- Very close\n"
            f"- :green[{GEO_VERY_CLOSE_KM} - "
            f"{GEO_CLOSE_KM} km] -- Close\n"
            f"- :orange[{GEO_CLOSE_KM} - "
            f"{GEO_MODERATE_KM} km] -- Moderate distance\n"
            f"- :red[> {GEO_MODERATE_KM} km] -- Distant"
        )


def display_results_table(
    results_df: pd.DataFrame,
    query_accession: str,
    direction: str,
    query_site: str = "",
) -> None:
    """Style and display the results table with downloads."""
    styled_df = results_df.style.format(
        {"Aitch Dist": "{:.2f}", "Eucl Dist": "{:.2f}"},
    )
    styled_df = (
        styled_df.apply(  # pyright: ignore[reportAttributeAccessIssue]
            highlight_geo_dist, subset=["Geo Dist"],
        )
        .map(color_aitch_dist, subset=["Aitch Dist"])
        .map(color_eucl_dist, subset=["Eucl Dist"])
    )
    border_style = ("border", "1px solid gray")
    padding = ("padding", "4px")
    no_radius = ("border-radius", "0px")
    styled_df = styled_df.set_table_styles([
        {
            "selector": "table",
            "props": [
                ("border-collapse", "collapse"),
                border_style,
            ],
        },
        {
            "selector": "th",
            "props": [border_style, padding, no_radius],
        },
        {
            "selector": "td",
            "props": [border_style, padding, no_radius],
        },
    ])
    st.table(styled_df)

    direction_label = (
        "art2geo"
        if direction == "artefact_to_geology"
        else "geo2art"
    )
    base_filename = (
        f"GSF_{query_accession}"
        f"_{direction_label}_top{len(results_df)}"
    )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv_data = (
            results_df.to_csv(index=False).encode("utf-8")
        )
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"{base_filename}.csv",
            mime="text/csv",
        )
    with dl_col2:
        excel_data = _create_styled_excel(
            results_df, query_accession,
            query_site, direction,
        )
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"{base_filename}.xlsx",
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
        )


def display_scatter_plot(results_df: pd.DataFrame) -> None:
    """Create a scatter plot of Geo Dist vs. Aitch Dist."""
    plot_df = results_df[
        results_df["Geo Dist"] != "Unknown"
    ].copy()
    if plot_df.empty:
        st.warning(
            "No rows with known geographical distance to plot."
        )
        return

    plot_df["Geo Dist Num"] = plot_df["Geo Dist"].apply(  # pyright: ignore[reportAttributeAccessIssue]
        lambda v: float(str(v).replace(" km", ""))
    )

    hover_cols: dict[str, bool] = {}
    for col in [
        "Geo Acc #", "Geo Site",
        "Artefact Acc #", "Artefact Site",
    ]:
        if col in plot_df.columns:
            hover_cols[col] = True

    color_col = None
    color_map = None
    if "Lithology" in plot_df.columns:
        color_col = "Lithology"
        color_map = LITHOLOGY_COLOR_MAP
    elif "Artefact Region" in plot_df.columns:
        color_col = "Artefact Region"
        color_map = REGION_COLOR_MAP

    fig = px.scatter(
        plot_df,
        x="Geo Dist Num",
        y="Aitch Dist",
        color=color_col,
        hover_data=hover_cols,
        title=(
            "Scatter Plot of Geographical Distance "
            "vs. Aitchison Distance"
        ),
        color_discrete_map=color_map if color_map else {},
    )

    for ref in [
        GEO_VERY_CLOSE_KM, GEO_CLOSE_KM, GEO_MODERATE_KM,
    ]:
        fig.add_vline(
            x=ref, line_dash="dash",
            line_color="black", opacity=0.5,
        )
    for ref in [
        AITCH_VERY_STRONG, AITCH_STRONG, AITCH_MODERATE,
    ]:
        fig.add_hline(
            y=ref, line_dash="dash",
            line_color="black", opacity=0.5,
        )

    fig.update_traces(
        marker={
            "size": 20,
            "line": {"color": "black", "width": 2},
            "opacity": 0.8,
        },
    )
    fig.update_layout(
        template=None,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "black"},
        xaxis_title="Geographical Distance (km)",
        yaxis_title="Aitchison Distance",
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        legend_title=color_col if color_col else "Legend",
    )
    for ax_update in (fig.update_xaxes, fig.update_yaxes):
        ax_update(
            rangemode="tozero",
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickfont={"color": "black"},
            showgrid=False,
            zeroline=False,
        )
    st.plotly_chart(fig, use_container_width=True)



def _aitch_to_marker_color(val: float) -> str:
    """Map Aitchison distance value to a marker color (0.5-step bands)."""
    if val < 0.5:
        return "lightskyblue"
    if val < 1.0:
        return "steelblue"
    if val < 1.5:
        return "limegreen"
    if val < 2.0:
        return "gold"
    if val < 2.5:
        return "orange"
    if val < 3.0:
        return "tomato"
    return "darkred"


def display_results_map(
    results_df: pd.DataFrame,
    query_coords: tuple[float, float] | None,
    target_coords_df: pd.DataFrame,
    query_label: str,
    direction: str,
) -> None:
    """Display an interactive map with query and match locations."""
    acc_col = (
        "Geo Acc #"
        if direction == "artefact_to_geology"
        else "Artefact Acc #"
    )
    site_col = (
        "Geo Site"
        if direction == "artefact_to_geology"
        else "Artefact Site"
    )

    map_points: list[dict[str, Any]] = []
    if direction == "artefact_to_geology":
        for _, row in results_df.iterrows():
            if row["Geo Dist"] == "Unknown":
                continue
            acc = str(row[acc_col])
            coord_row = target_coords_df[
                target_coords_df["Accession #"].astype(str)
                == acc
            ]
            if not coord_row.empty:
                map_points.append({
                    "lat": coord_row.iloc[0]["Latitude"],
                    "lon": coord_row.iloc[0]["Longitude"],
                    "label": f"{acc} - {row[site_col]}",
                    "aitch": row["Aitch Dist"],
                })
    else:
        for _, row in results_df.iterrows():
            if row["Geo Dist"] == "Unknown":
                continue
            site = row[site_col]
            coord_row = target_coords_df[
                target_coords_df["Site"] == site
            ]
            if not coord_row.empty:
                map_points.append({
                    "lat": coord_row.iloc[0]["Latitude"],
                    "lon": coord_row.iloc[0]["Longitude"],
                    "label": f"{row[acc_col]} - {site}",
                    "aitch": row["Aitch Dist"],
                })

    if not map_points and query_coords is None:
        return

    fig = go.Figure()

    # Match points colored by Aitchison distance
    if map_points:
        map_df = pd.DataFrame(map_points)
        map_df["marker_color"] = map_df["aitch"].apply(
            _aitch_to_marker_color,
        )
        map_df["hover_text"] = map_df.apply(
            lambda r: (
                f"{r['label']}<br>Aitch: {r['aitch']:.2f}"
            ),
            axis=1,
        )

        # Black shadow trace behind all match points — simulates borders
        # (Scattermapbox marker.line is not supported)
        fig.add_trace(go.Scattermapbox(
            lat=map_df["lat"].tolist(),
            lon=map_df["lon"].tolist(),
            mode="markers",
            marker={"size": 16, "color": "black"},
            showlegend=False,
            hoverinfo="skip",
        ))

        # One trace per color group for legend entries.
        # Add traces worst-first so that better matches render
        # on top when multiple samples share the same coordinates.
        # legendrank keeps the legend ordered best-first.
        color_labels = [
            ("lightskyblue", 1, "< 0.5 (very strong)"),
            ("steelblue",   2, "0.5–1.0 (very strong)"),
            ("limegreen", 3, "1.0–1.5 (strong)"),
            ("gold",      4, "1.5–2.0 (strong)"),
            ("orange",    5, "2.0–2.5 (moderate)"),
            ("tomato",    6, "2.5–3.0 (moderate)"),
            ("darkred",   7, "> 3.0 (weak)"),
        ]
        for color, rank, label in reversed(color_labels):
            subset = map_df[
                map_df["marker_color"] == color
            ]
            if subset.empty:
                continue
            fig.add_trace(go.Scattermapbox(
                lat=subset["lat"],
                lon=subset["lon"],
                mode="markers",
                marker={"size": 12, "color": color},
                name=label,
                text=subset["hover_text"],
                legendrank=rank,
            ))

    # Query point — added last so it always renders on top.
    # White halo then black fill so it is distinct from match markers.
    if query_coords:
        fig.add_trace(go.Scattermapbox(
            lat=[query_coords[0]],
            lon=[query_coords[1]],
            mode="markers",
            marker={"size": 21, "color": "white"},
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scattermapbox(
            lat=[query_coords[0]],
            lon=[query_coords[1]],
            mode="markers",
            marker={"size": 15, "color": "black"},
            name=query_label,
            text=[query_label],
            legendrank=0,
        ))

    # Scale bar — ArcGIS-style alternating bar 0-50-100-200 km
    # Bottom-left corner of map, Atlantic Ocean west of Strait of Gibraltar
    # At 35.1°N: 111.32 × cos(35.1°) ≈ 91.1 km/° lon
    _sb_lat    = 35.1
    _sb_lon0   = -9.7
    _sb_dlon   = 50 / 91.1          # ≈ 0.549° per 50 km
    _sb_lon50  = _sb_lon0 + _sb_dlon
    _sb_lon100 = _sb_lon0 + 2 * _sb_dlon
    _sb_lon200 = _sb_lon0 + 4 * _sb_dlon
    _sb_lw     = 10   # bar thickness in screen pixels

    # Black base bar (0 → 200 km)
    fig.add_trace(go.Scattermapbox(
        lat=[_sb_lat, _sb_lat], lon=[_sb_lon0, _sb_lon200],
        mode="lines", line={"width": _sb_lw, "color": "#000000"},
        showlegend=False, hoverinfo="skip",
    ))
    # White cutout for 50-100 km segment — slightly narrower to preserve black outline
    fig.add_trace(go.Scattermapbox(
        lat=[_sb_lat, _sb_lat], lon=[_sb_lon50, _sb_lon100],
        mode="lines", line={"width": _sb_lw - 4, "color": "#ffffff"},
        showlegend=False, hoverinfo="skip",
    ))
    # Distance labels — layout annotations always render on top of map tiles
    # Paper x: linear lon→(lon-west)/(east-west), west=-9.8, east=1.5 (range=11.3)
    _lon_to_x = lambda lon: (lon + 9.8) / 11.3
    _sb_lbl_y = 0.04   # paper y for km numbers (just above bar)
    _sb_km_y  = 0.07   # paper y for "Kilometers"
    for _km, _llon in [(0, _sb_lon0), (50, _sb_lon50),
                       (100, _sb_lon100), (200, _sb_lon200)]:
        fig.add_annotation(
            x=_lon_to_x(_llon), y=_sb_lbl_y,
            xref="paper", yref="paper",
            text=str(_km), showarrow=False,
            font={"size": 11, "color": "black"},
            xanchor="center", yanchor="bottom",
        )
    fig.add_annotation(
        x=_lon_to_x((_sb_lon0 + _sb_lon200) / 2), y=_sb_km_y,
        xref="paper", yref="paper",
        text="Kilometers", showarrow=False,
        font={"size": 11, "color": "black"},
        xanchor="center", yanchor="bottom",
    )

    # Map extent: southern Iberia focus matching ArcGIS layout
    # East stops at Spanish coast (no Balearics), Morocco clearly visible below
    _MAP_BOUNDS = {
        "west":  -9.8,   # Portugal's Atlantic coast with small ocean gap
        "east":   1.5,   # southeastern Spanish coast (~0-1°E)
        "south": 33.5,   # Morocco well south of Melilla
        "north": 41.5,   # just past Madrid (~40.4°N)
    }
    fig.update_layout(
        mapbox={
            "style": "white-bg",
            "layers": [
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/"
                        "services/NatGeo_World_Map/MapServer/tile"
                        "/{z}/{y}/{x}"
                    ],
                    "sourceattribution": (
                        "Tiles © Esri — National Geographic, Esri,"
                        " DeLorme, NAVTEQ, USGS, NRCAN, GEBCO, NOAA"
                    ),
                }
            ],
            "bounds": _MAP_BOUNDS,
            "center": {"lat": 37.5, "lon": -4.15},  # fallback
            "zoom": 6,                               # fallback
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title="Geographical Distribution of Matches",
        height=750,
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Radar / Spider Chart
# -----------------------------
def display_radar_chart(
    query_sample: pd.Series,
    results_df: pd.DataFrame,
    target_df: pd.DataFrame,
    query_accession: str,
    direction: str,
) -> None:
    """Display a radar chart comparing ratio profiles."""
    ratio_cols = _get_ratio_columns(query_sample)
    if len(ratio_cols) < 3:
        st.info("Not enough ratio columns for a radar chart.")
        return

    # Build options for match selection
    acc_col = (
        "Geo Acc #"
        if direction == "artefact_to_geology"
        else "Artefact Acc #"
    )
    site_col = (
        "Geo Site"
        if direction == "artefact_to_geology"
        else "Artefact Site"
    )

    match_options: list[str] = []
    for _, row in results_df.iterrows():
        match_options.append(
            f"{row[acc_col]} - {row[site_col]}"
        )

    if not match_options:
        return

    selected = st.multiselect(
        "Select matches to compare on radar chart:",
        match_options,
        default=match_options[:min(3, len(match_options))],
        max_selections=8,
        key="radar_select",
    )

    if not selected:
        return

    # Collect ratio values
    query_values = np.asarray(pd.Series(pd.to_numeric(
        query_sample[ratio_cols], errors="coerce",
    )).fillna(EPSILON), dtype=float)
    all_values = [query_values]
    labels = [f"Query: {query_accession}"]

    for sel in selected:
        acc = sel.split(" - ")[0]
        match_rows = target_df[
            target_df["Accession #"] == acc
        ]
        if match_rows.empty:
            continue
        match_sample = match_rows.iloc[0]
        match_vals = np.asarray(pd.Series(pd.to_numeric(
            match_sample[ratio_cols], errors="coerce",
        )).fillna(EPSILON), dtype=float)
        all_values.append(match_vals)
        labels.append(sel)

    if len(all_values) < 2:
        st.info("No valid match data found for radar chart.")
        return

    # Normalize: log10 then min-max across all compared samples
    matrix = np.array(all_values)
    log_matrix = np.log10(np.clip(matrix, EPSILON, None))
    col_min = log_matrix.min(axis=0)
    col_max = log_matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normalized = (log_matrix - col_min) / col_range

    # Build radar chart
    fig = go.Figure()
    theta = [*list(ratio_cols), ratio_cols[0]]

    chart_colors = [
        "red", "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f",
    ]
    for i, (vals, label) in enumerate(
        zip(normalized, labels, strict=False)
    ):
        r = [*list(vals), vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            name=label,
            line={
                "width": 3 if i == 0 else 2,
                "color": chart_colors[
                    i % len(chart_colors)
                ],
            },
            fill="none",
        ))

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True, "range": [0, 1],
            },
        },
        showlegend=True,
        title="Ratio Profile Comparison (log-normalized)",
        margin={"l": 60, "r": 60, "t": 50, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Query Execution Helpers
# -----------------------------
def _get_query_coords(
    query_sample: pd.Series,
    coords_df: pd.DataFrame,
    direction: str,
) -> tuple[float, float] | None:
    """Look up the coordinates for a query sample."""
    if direction == "artefact_to_geology":
        match = coords_df[
            coords_df["Site"] == query_sample["Site"]
        ]
    else:
        match = coords_df[
            coords_df["Accession #"].astype(str)
            == str(query_sample["Accession #"])
        ]
    if not match.empty:
        return (
            float(match.iloc[0]["Latitude"]),
            float(match.iloc[0]["Longitude"]),
        )
    return None


def _resolve_sample(
    accession_number: str,
    source_df: pd.DataFrame,
    label_prefix: str,
    key_suffix: str = "",
) -> pd.Series | None:
    """Find a sample by accession number."""
    matches = source_df[
        source_df["Accession #"] == accession_number
    ]
    if matches.empty:
        st.error(
            f"No {label_prefix} found with "
            f"Accession # {accession_number}"
        )
        return None
    if len(matches) == 1:
        return matches.iloc[0]
    st.info(
        f"Multiple {label_prefix}s found for "
        f"Accession # {accession_number}. "
        "Please select a site:"
    )
    sites = list(matches["Site"].unique())  # pyright: ignore[reportAttributeAccessIssue]
    selected_site = str(st.selectbox(
        "Select a site",
        sites,
        key=f"site_select_{key_suffix}",
    ))
    site_mask = matches["Site"].str.lower() == selected_site.lower()  # pyright: ignore[reportAttributeAccessIssue]
    return matches[site_mask].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]


def _execute_query(
    accession_number: str,
    query_mode: str,
    top_n: int,
    selected_regions: list[str],
    artefact_df: pd.DataFrame,
    geology_df: pd.DataFrame,
    geo_coords_df: pd.DataFrame,
    arch_coords_df: pd.DataFrame,
) -> None:
    """Execute a single query and display results."""
    if query_mode == "Artefact \u2192 Geology":
        direction = "artefact_to_geology"
        source_df = artefact_df
        target_df = geology_df
        query_coords_df = arch_coords_df
        target_coords_df = geo_coords_df
        label_prefix = "artefact"
        region_col = "Region"
    else:
        direction = "geology_to_artefact"
        source_df = geology_df
        target_df = artefact_df
        query_coords_df = geo_coords_df
        target_coords_df = arch_coords_df
        label_prefix = "geology sample"
        region_col = "Artefact Region"

    sample = _resolve_sample(
        accession_number, source_df,
        label_prefix, key_suffix="single",
    )
    if sample is None:
        return

    # Build header
    if "Region" in sample.index and bool(
        pd.notnull(sample.get("Region"))
    ):
        site_info = (
            f"{sample['Site']} ({sample['Region']})"
        )
    else:
        site_info = str(sample.get("Site", "Unknown Site"))
    st.subheader(
        f"Results for Accession # "
        f"{sample['Accession #']} - {site_info}"
    )

    # Data quality indicator
    total_ratios = _count_total_ratio_columns(source_df)
    valid_ratios = len(_get_ratio_columns(sample))
    if valid_ratios < total_ratios:
        st.warning(
            f"Data Quality: {valid_ratios}/{total_ratios}"
            " ratio columns available. "
            f"Missing {total_ratios - valid_ratios} "
            "values - results based on fewer dimensions."
        )
    else:
        st.info(
            f"Data Quality: {valid_ratios}/{total_ratios}"
            " ratio columns available (complete)."
        )

    # Get top matches
    results_df = get_top_matches(
        sample, target_df, query_coords_df,
        target_coords_df, top_n, direction,
    )

    # Apply region filter
    if (
        selected_regions
        and not results_df.empty
        and region_col in results_df.columns
    ):
        results_df = pd.DataFrame(
            results_df[
                results_df[region_col].isin(selected_regions)
            ]
        )

    if results_df.empty:
        st.warning(
            "No matching results found. "
            "Try adjusting the region filter "
            "or increasing the number of results."
        )
        return

    # Display results
    with st.expander(
        "Distance Interpretation Guide", expanded=False,
    ):
        display_legend()

    display_results_table(
        results_df, accession_number,
        direction, query_site=site_info,
    )
    display_radar_chart(
        sample, results_df, target_df,
        accession_number, direction,
    )
    display_scatter_plot(results_df)

    query_coords = _get_query_coords(
        sample, query_coords_df, direction,
    )
    query_label = (
        f"Query: {sample['Accession #']} "
        f"- {sample['Site']}"
    )
    display_results_map(
        results_df, query_coords,
        target_coords_df, query_label, direction,
    )


# -----------------------------
# Batch Query
# -----------------------------
def _execute_batch_query(
    artefact_df: pd.DataFrame,
    geology_df: pd.DataFrame,
    geo_coords_df: pd.DataFrame,
    arch_coords_df: pd.DataFrame,
) -> None:
    """Batch query: upload CSV, run matching for each."""
    uploaded_file = st.file_uploader(
        "Upload a CSV with accession numbers",
        type=["csv"],
        key="batch_upload",
    )
    if uploaded_file is None:
        st.info(
            "Upload a CSV file containing a column "
            "of accession numbers."
        )
        return

    input_df = pd.read_csv(uploaded_file)
    if input_df.empty:
        st.error("The uploaded CSV is empty.")
        return

    acc_column = st.selectbox(
        "Select the column containing accession numbers:",
        input_df.columns,
        key="batch_col",
    )
    query_mode = st.radio("Query Direction:", [
        "Artefact \u2192 Geology",
        "Geology \u2192 Artefact",
    ], key="batch_mode")
    top_n = st.number_input(
        "Top N per query:",
        min_value=1, max_value=MAX_TOP_N,
        value=DEFAULT_TOP_N, step=1,
        key="batch_topn",
    )

    if not st.button("Run Batch", key="batch_run"):
        return

    if query_mode == "Artefact \u2192 Geology":
        direction = "artefact_to_geology"
        source_df = artefact_df
        target_df = geology_df
        query_coords_df = arch_coords_df
        target_coords_df = geo_coords_df
    else:
        direction = "geology_to_artefact"
        source_df = geology_df
        target_df = artefact_df
        query_coords_df = geo_coords_df
        target_coords_df = arch_coords_df

    accession_numbers = (
        input_df[acc_column]
        .astype(str).str.strip()
        .dropna().unique().tolist()
    )
    if not accession_numbers:
        st.error(
            "No valid accession numbers found "
            "in the selected column."
        )
        return

    all_results: list[pd.DataFrame] = []
    errors: list[str] = []
    progress = st.progress(0, text="Processing batch...")
    total = len(accession_numbers)

    for i, acc in enumerate(accession_numbers):
        progress.progress(
            (i + 1) / total,
            text=f"Processing {acc} ({i + 1}/{total})",
        )
        matches = source_df[
            source_df["Accession #"] == acc
        ]
        if matches.empty:
            errors.append(acc)
            continue
        sample = matches.iloc[0]
        result = get_top_matches(
            sample, target_df, query_coords_df,
            target_coords_df, top_n, direction,
        )
        if not result.empty:
            result.insert(0, "Query Acc #", acc)
            all_results.append(result)

    progress.empty()

    if errors:
        st.warning(
            f"{len(errors)} accession number(s) not found: "
            f"{', '.join(errors[:20])}"
        )

    if not all_results:
        st.error(
            "No results found for any accession number "
            "in the batch."
        )
        return

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.index += 1

    # Frequency summary
    acc_col = (
        "Geo Acc #"
        if direction == "artefact_to_geology"
        else "Artefact Acc #"
    )
    site_col = (
        "Geo Site"
        if direction == "artefact_to_geology"
        else "Artefact Site"
    )

    st.subheader("Source Frequency Summary")
    st.markdown(
        "Geological sources ranked by how often they "
        "appear across all queried samples:"
    )

    freq_df = (
        combined_df.groupby([acc_col, site_col])
        .agg(
            Count=("Query Acc #", "nunique"),
            Avg_Aitch=("Aitch Dist", "mean"),
            Min_Aitch=("Aitch Dist", "min"),
        )
        .sort_values(  # pyright: ignore[reportCallIssue]
            ["Count", "Avg_Aitch"],
            ascending=[False, True],
        )
        .reset_index()
    )
    freq_df["Avg_Aitch"] = np.round(
        freq_df["Avg_Aitch"], 2,
    )
    freq_df["Min_Aitch"] = np.round(
        freq_df["Min_Aitch"], 2,
    )
    st.dataframe(freq_df, use_container_width=True)

    # Combined results
    with st.expander("All Individual Results", expanded=False):
        st.dataframe(combined_df, use_container_width=True)

    # Downloads
    base_filename = (
        f"GSF_batch_{len(accession_numbers)}queries"
    )
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download combined results as CSV",
            combined_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base_filename}.csv",
            mime="text/csv",
            key="batch_csv",
        )
    with dl2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:  # pyright: ignore[reportArgumentType]
            freq_df.to_excel(
                writer,
                sheet_name="Frequency Summary",
                index=False,
            )
            combined_df.to_excel(
                writer,
                sheet_name="All Results",
                index=False,
            )
        xl_mime = (
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        )
        st.download_button(
            "Download as Excel",
            output.getvalue(),
            file_name=f"{base_filename}.xlsx",
            mime=xl_mime,
            key="batch_xlsx",
        )

    n_success = len(accession_numbers) - len(errors)
    st.success(
        f"Batch complete: {n_success} successful queries, "
        f"{len(all_results)} results total."
    )


# -----------------------------
# Comparison Mode
# -----------------------------
def _execute_comparison(
    artefact_df: pd.DataFrame,
    geology_df: pd.DataFrame,
    geo_coords_df: pd.DataFrame,
    arch_coords_df: pd.DataFrame,
) -> None:
    """Enter 2+ accession numbers to find shared sources."""
    st.markdown(
        "Enter 2 or more accession numbers to find "
        "**shared geological sources** that match "
        "multiple artefacts."
    )

    query_mode = st.radio("Query Direction:", [
        "Artefact \u2192 Geology",
        "Geology \u2192 Artefact",
    ], key="comp_mode")
    top_n = st.number_input(
        "Top N per sample:",
        min_value=1, max_value=MAX_TOP_N,
        value=DEFAULT_TOP_N, step=1,
        key="comp_topn",
    )

    acc_text = st.text_area(
        "Enter accession numbers "
        "(comma or newline separated):",
        height=100,
        key="comp_input",
        placeholder="e.g. 3003, 3004, 3005",
    )

    if not st.button("Compare", key="comp_run"):
        return

    # Parse accession numbers
    raw = acc_text.replace(",", "\n").strip()
    accession_numbers = [
        a.strip() for a in raw.split("\n") if a.strip()
    ]

    if len(accession_numbers) < 2:
        st.warning(
            "Please enter at least 2 accession numbers "
            "to compare."
        )
        return

    if query_mode == "Artefact \u2192 Geology":
        direction = "artefact_to_geology"
        source_df = artefact_df
        target_df = geology_df
        query_coords_df = arch_coords_df
        target_coords_df = geo_coords_df
    else:
        direction = "geology_to_artefact"
        source_df = geology_df
        target_df = artefact_df
        query_coords_df = geo_coords_df
        target_coords_df = arch_coords_df

    acc_col = (
        "Geo Acc #"
        if direction == "artefact_to_geology"
        else "Artefact Acc #"
    )
    site_col = (
        "Geo Site"
        if direction == "artefact_to_geology"
        else "Artefact Site"
    )

    # Run queries
    all_results: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for acc in accession_numbers:
        matches = source_df[
            source_df["Accession #"] == acc
        ]
        if matches.empty:
            errors.append(acc)
            continue
        sample = matches.iloc[0]
        result = get_top_matches(
            sample, target_df, query_coords_df,
            target_coords_df, top_n, direction,
        )
        if not result.empty:
            all_results[acc] = result

    if errors:
        st.warning(f"Not found: {', '.join(errors)}")

    if len(all_results) < 2:
        st.error(
            "Need results from at least 2 accession "
            "numbers to compare."
        )
        return

    # Find shared sources
    source_counter: Counter[tuple[str, str]] = Counter()
    source_aitch: dict[
        tuple[str, str], dict[str, float]
    ] = {}

    for query_acc, result_df in all_results.items():
        for _, row in result_df.iterrows():
            source_key = (
                str(row[acc_col]), str(row[site_col]),
            )
            source_counter[source_key] += 1
            if source_key not in source_aitch:
                source_aitch[source_key] = {}
            source_aitch[source_key][query_acc] = float(
                row["Aitch Dist"]
            )

    # Build comparison table
    shared_sources = {
        k: v for k, v in source_counter.items() if v >= 2
    }

    if not shared_sources:
        st.info(
            "No shared geological sources found across "
            "the queried samples in their top-N results. "
            "Try increasing the number of results."
        )
        # Still show individual results
        for acc, result_df in all_results.items():
            with st.expander(
                f"Results for {acc}", expanded=False,
            ):
                st.dataframe(
                    result_df, use_container_width=True,
                )
        return

    st.subheader(
        f"Shared Sources ({len(shared_sources)} found)"
    )
    st.markdown(
        "Sources appearing in the top-N results of "
        "2 or more queried samples:"
    )

    query_accs = list(all_results.keys())
    comp_rows: list[dict[str, Any]] = []
    for (src_acc, src_site), count in sorted(
        shared_sources.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        row_data: dict[str, Any] = {
            acc_col: src_acc,
            site_col: src_site,
            "Shared Count": count,
        }
        aitch_values = source_aitch.get(
            (src_acc, src_site), {},
        )
        for q_acc in query_accs:
            row_data[f"Aitch ({q_acc})"] = (
                aitch_values.get(q_acc, None)
            )
        # Add lithology if available
        if direction == "artefact_to_geology":
            for result_df in all_results.values():
                lit_row = result_df[
                    result_df[acc_col] == src_acc
                ]
                if (
                    not lit_row.empty
                    and "Lithology" in lit_row.columns
                ):
                    row_data["Lithology"] = (
                        lit_row.iloc[0]["Lithology"]
                    )
                    break
        comp_rows.append(row_data)

    comp_df = pd.DataFrame(comp_rows)

    # Sort by shared count desc, then average Aitchison dist
    aitch_cols = [
        c for c in comp_df.columns
        if c.startswith("Aitch (")
    ]
    if aitch_cols:
        comp_df["Avg Aitch"] = np.round(
            comp_df[aitch_cols].mean(axis=1), 2,
        )
        comp_df = comp_df.sort_values(  # pyright: ignore[reportCallIssue]
            ["Shared Count", "Avg Aitch"],
            ascending=[False, True],
        )
    comp_df.reset_index(drop=True, inplace=True)
    comp_df.index += 1

    # Highlight rows matching ALL queried samples
    def _highlight_shared(
        row: pd.Series,
    ) -> list[str]:
        if row["Shared Count"] == len(all_results):
            return [
                "background-color: #D4EDDA"
            ] * len(row)
        return [""] * len(row)

    styled = comp_df.style.apply(
        _highlight_shared, axis=1,
    )
    avg_cols = (
        ["Avg Aitch"]
        if "Avg Aitch" in comp_df.columns
        else []
    )
    for col in aitch_cols + avg_cols:
        styled = styled.map(
            color_aitch_dist, subset=[col],
        )

    st.table(styled)
    st.caption(
        "Green rows = sources matching ALL queried samples."
    )

    # Individual results in expanders
    for acc, result_df in all_results.items():
        with st.expander(
            f"Full results for {acc}", expanded=False,
        ):
            st.dataframe(
                result_df, use_container_width=True,
            )

    # Download
    csv_data = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download comparison as CSV",
        csv_data,
        file_name=(
            f"GSF_comparison_{len(query_accs)}samples.csv"
        ),
        mime="text/csv",
        key="comp_csv",
    )


# -----------------------------
# Main Application
# -----------------------------
def main() -> None:
    """Application entry point."""
    # Render sidebar to get settings (element mode)
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            **GSF - Geology Source Finder** matches
            archaeological artefacts to potential geological
            sources using trace element ratio comparison.

            **Metrics used:**
            - **Aitchison Distance**: Compositional distance
              using centered log-ratio (CLR) transform.
            - **Euclidean Distance**: Standard geometric
              distance between ratio vectors.
            - **Geographical Distance**: Geodesic distance
              between site coordinates in kilometres.
            """
        )
        st.divider()
        st.header("Settings")
        element_mode = st.radio(
            "Element Mode:",
            [
                "Trace Elements (16 ratios)",
                "All Elements (22 ratios)",
            ],
            help=(
                "Trace Elements uses 16 trace-element "
                "ratios. All Elements adds 6 "
                "major-element ratios (22 total)."
            ),
        )
        mode_key = (
            "trace" if "Trace" in element_mode else "all"
        )

    st.title("GSF - Geology Source Finder")

    # Load data with selected element mode
    try:
        (
            artefact_df, geology_df,
            geo_coords_df, arch_coords_df,
        ) = load_data(mode_key)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error("Error loading data.")
        st.exception(e)
        return

    # Show dataset counts in sidebar
    with st.sidebar:
        st.divider()
        st.caption(
            f"Dataset: {len(artefact_df)} artefacts, "
            f"{len(geology_df)} geology samples"
        )

    # Tab layout
    tab_single, tab_batch, tab_compare = st.tabs([
        "Single Query",
        "Batch Query",
        "Comparison",
    ])

    # --- Tab 1: Single Query ---
    with tab_single:
        query_mode = st.radio("Select Query Mode:", [
            "Artefact \u2192 Geology",
            "Geology \u2192 Artefact",
        ], key="single_mode")

        # Autocomplete accession number selection
        options_df = (
            artefact_df
            if query_mode == "Artefact \u2192 Geology"
            else geology_df
        )

        options = sorted(
            options_df.apply(
                lambda r: (
                    f"{r['Accession #']} - {r['Site']}"
                ),
                axis=1,
            ).unique()
        )

        selected = st.selectbox(
            "Select an accession number:",
            options=["", *options],
            format_func=lambda x: (
                "Type to search..." if x == "" else x
            ),
            key="single_acc",
        )
        accession_number = (
            selected.split(" - ")[0].strip()
            if selected
            else ""
        )

        top_n = st.number_input(
            "Number of top results:",
            min_value=1,
            max_value=MAX_TOP_N,
            value=DEFAULT_TOP_N,
            step=1,
            key="single_topn",
        )

        # Region filter
        all_regions = sorted(
            set(artefact_df["Region"].dropna().unique())
            | set(geology_df["Region"].dropna().unique())
        )
        selected_regions = st.multiselect(
            "Filter by region(s) (optional):",
            all_regions,
            key="single_regions",
        )

        # Execute query
        if accession_number:
            _execute_query(
                accession_number, query_mode,
                top_n, selected_regions,
                artefact_df, geology_df,
                geo_coords_df, arch_coords_df,
            )

    # --- Tab 2: Batch Query ---
    with tab_batch:
        _execute_batch_query(
            artefact_df, geology_df,
            geo_coords_df, arch_coords_df,
        )

    # --- Tab 3: Comparison Mode ---
    with tab_compare:
        _execute_comparison(
            artefact_df, geology_df,
            geo_coords_df, arch_coords_df,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred.")
        st.exception(e)
