from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import streamlit as st

from gsf.constants import (
    ALR5_DENOMINATOR,
    ALR5_ELEMENTS,
    ALR5_NUMERATORS,
    ALR5_RATIO_NAMES,
    METADATA_COLUMNS,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def _load_alr5_df(
    filepath: str, sheet_name: str,
) -> pd.DataFrame:
    """Load ppm data and compute ALR-5 coordinates.

    Reads the raw ppm sheet, extracts 6 elements
    (Ni, Cr, Y, Nb, Sr, Zr) and computes ln(X/Zr)
    for each numerator. Missing or non-positive values
    are left as NaN so that incomplete ALR coordinates
    are excluded from distance calculations rather than
    being replaced with extreme outliers.
    """
    raw = pd.read_excel(filepath, sheet_name=sheet_name)

    meta_cols = [
        c for c in METADATA_COLUMNS if c in raw.columns
    ]

    missing = [
        e for e in ALR5_ELEMENTS if e not in raw.columns
    ]
    if missing:
        raise ValueError(
            f"Missing elements {missing} in sheet "
            f"'{sheet_name}' of {filepath}"
        )

    result = raw[meta_cols].copy()
    elem_vals: dict[str, pd.Series] = {}
    for elem in ALR5_ELEMENTS:
        vals = pd.to_numeric(raw[elem], errors="coerce")
        # Non-positive values -> NaN; avoids log-space
        # outliers (ln(e/x) ~ -25) that dominate distances
        vals = vals.where(vals > 0, np.nan)
        elem_vals[elem] = vals

    zr = elem_vals[ALR5_DENOMINATOR]
    for num, name in zip(
        ALR5_NUMERATORS, ALR5_RATIO_NAMES,
    ):
        # NaN in numerator or denominator propagates to NaN
        result[name] = np.log(elem_vals[num] / zr)

    result["Accession #"] = (
        result["Accession #"].astype(str)
    )
    return result


@st.cache_data(show_spinner=False)
def load_data(
    element_mode: str = "trace",
    external_data_dir: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required Excel files from a 'Data' folder.

    Args:
        element_mode: "trace" for 16 ratio columns,
            "all" for 22 ratio columns,
            "alr5" for 5 ALR log-ratio coordinates.
        external_data_dir: optional path to an external
            data folder (e.g. the PhD/Data directory).
            Checked first; falls back to the local
            Data/ folder if empty or files not found.

    Returns:
        (artefact_df, geology_df, geo_coords_df, arch_coords_df)
    """
    # Build search directories: external first, then local
    search_dirs: list[str] = []
    if external_data_dir:
        search_dirs.append(external_data_dir)
        sub = os.path.join(
            external_data_dir, "Main database",
        )
        if os.path.isdir(sub):
            search_dirs.append(sub)
    search_dirs.extend([
        os.path.join(_PROJECT_ROOT, "Data"),
        os.path.join(_PROJECT_ROOT, "..", "Data"),
    ])

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

    # Resolve each file independently across search dirs
    # (PhD folder splits files: Main database/ vs Data/)
    resolved: dict[str, str] = {}
    for key, fname in filenames.items():
        for d in search_dirs:
            candidate = os.path.join(d, fname)
            if os.path.exists(candidate):
                resolved[key] = candidate
                logger.info(
                    "Resolved %s → %s", fname, candidate,
                )
                break

    missing = [
        filenames[k] for k in filenames if k not in resolved
    ]
    if missing:
        raise FileNotFoundError(
            f"Required files not found: {missing}\n"
            f"Searched in: {search_dirs}\n"
            "Place the Excel files in a 'Data' folder "
            "next to this script, or configure an "
            "external data source in the sidebar."
        )

    try:
        if element_mode == "alr5":
            artefact_df = _load_alr5_df(
                resolved["artefact"], "AXEs ppm data",
            )
            geology_df = _load_alr5_df(
                resolved["geology"], "Geology ppm data",
            )
        else:
            artefact_df = pd.read_excel(
                resolved["artefact"],
                sheet_name="AXEs Ratios",
            )
            geology_df = pd.read_excel(
                resolved["geology"],
                sheet_name="Geology ratios",
            )
        coords_sheets = pd.read_excel(
            resolved["coords"],
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
    # while some sites (e.g. Vila Nova de Sao Pedro) are already negative.
    # All sites are in the Iberian Peninsula (west of the prime meridian),
    # so force all longitudes to negative with -abs() to normalise both cases.
    geo_coords_df["Longitude"] = geo_coords_df["Longitude"].abs() * -1
    arch_coords_df["Longitude"] = arch_coords_df["Longitude"].abs() * -1

    return artefact_df, geology_df, geo_coords_df, arch_coords_df


@st.cache_data(show_spinner=False)
def load_photo_mapping() -> pd.DataFrame:
    """Load artefact-to-photo filename mapping from CSV."""
    for d in [
        os.path.join(_PROJECT_ROOT, "Data"),
        os.path.join(_PROJECT_ROOT, "..", "Data"),
    ]:
        path = os.path.join(d, "photo_mapping.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8")
            df["Accession #"] = df["Accession #"].astype(str).str.strip()
            return df
    return pd.DataFrame()
