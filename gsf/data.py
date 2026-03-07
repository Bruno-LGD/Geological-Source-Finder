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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required Excel files from a 'Data' folder.

    Args:
        element_mode: "trace" for 16 ratio columns,
            "all" for 22 ratio columns,
            "alr5" for 5 ALR log-ratio coordinates.

    Returns:
        (artefact_df, geology_df, geo_coords_df, arch_coords_df)
    """
    possible_dirs = [
        os.path.join(_PROJECT_ROOT, "Data"),
        os.path.join(_PROJECT_ROOT, "..", "Data"),
    ]

    if element_mode == "all":
        filenames = {
            "artefact": "AXEs metabasite data (All Elements).xlsx",
            "geology": "Geology samples data (All Elements).xlsx",
            "coords": "Coordinates sheet.xlsx",
        }
    else:
        # Both "trace" and "alr5" use the Trace Elements files
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
        if element_mode == "alr5":
            artefact_df = _load_alr5_df(
                os.path.join(
                    found_data_dir, filenames["artefact"],
                ),
                "AXEs ppm data",
            )
            geology_df = _load_alr5_df(
                os.path.join(
                    found_data_dir, filenames["geology"],
                ),
                "Geology ppm data",
            )
        else:
            artefact_df = pd.read_excel(
                os.path.join(
                    found_data_dir, filenames["artefact"],
                ),
                sheet_name="AXEs Ratios",
            )
            geology_df = pd.read_excel(
                os.path.join(
                    found_data_dir, filenames["geology"],
                ),
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
