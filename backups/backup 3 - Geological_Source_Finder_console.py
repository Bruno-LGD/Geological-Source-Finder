import os
import sys
import logging
from typing import Tuple, List

import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean
from geopy.distance import geodesic
import streamlit as st
import plotly.express as px

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Data Loading Function
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tries to load the required Excel files from two possible locations:
      1) A "Data" folder in the same directory as this script (GitHub structure),
      2) A "Data" folder in the parent directory (for local use).

    Expected files (with these exact names):
      - "AXEs metabasite data (Trace Elements).xlsx" (sheet: "AXEs Ratios")
      - "Geology samples data (Trace Elements).xlsx" (sheet: "Geology ratios")
      - "Coordinates sheet.xlsx" (sheet: "Geology Samples Coord" and "Archaeology Sites Coord")

    Returns:
      A tuple of DataFrames: (artefact_df, geology_df, geo_coords_df, arch_coords_df)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # List of directories to check:
    possible_dirs = [
        os.path.join(base_dir, "Data"),        # Data subfolder in the same directory as the script
        os.path.join(base_dir, "..", "Data")     # Data folder in the parent directory
    ]

    # Updated filenames (note the corrected name for coordinates file)
    filenames = {
        "artefact": "AXEs metabasite data (Trace Elements).xlsx",
        "geology":  "Geology samples data (Trace Elements).xlsx",
        "coords":   "Coordinates sheet.xlsx"  # corrected: singular "sheet"
    }

    found_data_dir = None
    for d in possible_dirs:
        artefact_path = os.path.join(d, filenames["artefact"])
        geology_path  = os.path.join(d, filenames["geology"])
        coords_path   = os.path.join(d, filenames["coords"])
        if all(os.path.exists(p) for p in [artefact_path, geology_path, coords_path]):
            found_data_dir = d
            logger.info(f"Data found in directory: {d}")
            break

    if not found_data_dir:
        raise FileNotFoundError(
            f"Required files not found in any of these directories:\n{possible_dirs}\n"
            "Please place the Excel files in a 'Data' folder either in the same directory as this script (for GitHub) or in the parent directory (for local use)."
        )

    try:
        artefact_df = pd.read_excel(
            os.path.join(found_data_dir, filenames["artefact"]),
            sheet_name="AXEs Ratios"
        )
        geology_df = pd.read_excel(
            os.path.join(found_data_dir, filenames["geology"]),
            sheet_name="Geology ratios"
        )
        geo_coords_df = pd.read_excel(
            os.path.join(found_data_dir, filenames["coords"]),
            sheet_name="Geology Samples Coord"
        )
        arch_coords_df = pd.read_excel(
            os.path.join(found_data_dir, filenames["coords"]),
            sheet_name="Archaeology Sites Coord"
        )
    except Exception as e:
        logger.exception("Error loading Excel files.")
        st.error("Error loading Excel files. Check file names and sheet names.")
        raise e

    # Ensure accession numbers are strings
    artefact_df['Accession #'] = artefact_df['Accession #'].astype(str)
    geology_df['Accession #'] = geology_df['Accession #'].astype(str)
    geo_coords_df['Accession #'] = geo_coords_df['Accession #'].astype(str)

    return artefact_df, geology_df, geo_coords_df, arch_coords_df

# -----------------------------
# Aitchison Distance Function
# -----------------------------
def aitchison_distance(x, y) -> float:
    """
    Calculate the Aitchison distance between two compositional vectors.
    """
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    x = np.nan_to_num(x, nan=1e-10, posinf=1e-10, neginf=1e-10)
    y = np.nan_to_num(y, nan=1e-10, posinf=1e-10, neginf=1e-10)
    log_x = np.log(x / np.prod(x)**(1/len(x)))
    log_y = np.log(y / np.prod(y)**(1/len(y)))
    return norm(log_x - log_y)

# -----------------------------
# Main Calculation Function
# -----------------------------
def get_top_20_matches(
    artefact: pd.Series,
    geology_df: pd.DataFrame,
    geo_coords_df: pd.DataFrame,
    arch_coords_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For a given artefact, compute and return a DataFrame of the top 20 geological matches.
    Uses fixed column indexing (artefact.dropna().iloc[6:]) for the ratio columns.
    """
    available_artefact_ratios = artefact.dropna().iloc[6:]
    ratio_cols = list(available_artefact_ratios.index)
    required_cols = ["Lithology", "Accession #", "Site", "Region"] + ratio_cols
    aligned_geology_df = geology_df[required_cols].dropna()

    distances = []
    if artefact["Site"] in arch_coords_df["Site"].values:
        coords_row = arch_coords_df[arch_coords_df["Site"] == artefact["Site"]].iloc[0]
        artefact_coords = (coords_row["Latitude"], coords_row["Longitude"])
    else:
        artefact_coords = None

    for _, geo_sample in aligned_geology_df.iterrows():
        geo_ratios = geo_sample[ratio_cols].values
        aitch_dist = aitchison_distance(available_artefact_ratios.values, geo_ratios)
        eucl_dist  = euclidean(available_artefact_ratios.values, geo_ratios)
        coords_row = geo_coords_df[geo_coords_df["Accession #"] == geo_sample["Accession #"]]
        if artefact_coords and not coords_row.empty:
            coords_row = coords_row.iloc[0]
            try:
                dist_km = geodesic(artefact_coords, (coords_row["Latitude"], coords_row["Longitude"])).km
                geo_distance = int(dist_km)
            except (ValueError, TypeError):
                geo_distance = "Unknown"
        else:
            geo_distance = "Unknown"
        distances.append([
            geo_sample["Lithology"],
            geo_sample["Accession #"],
            geo_sample["Site"],
            aitch_dist,
            eucl_dist,
            geo_distance
        ])

    results_df = pd.DataFrame(
        distances,
        columns=["Lithology", "Geo Acc #", "Geo Site", "Aitch Dist", "Eucl Dist", "Geo Dist"]
    )
    results_df = results_df.sort_values(by=["Aitch Dist", "Eucl Dist"]).head(20)
    results_df[["Aitch Dist", "Eucl Dist"]] = results_df[["Aitch Dist", "Eucl Dist"]].round(2)
    results_df.reset_index(drop=True, inplace=True)
    results_df.index += 1

    def fix_geo_dist(x):
        if pd.notnull(x) and x != "Unknown":
            return f"{int(round(x/10)*10)} km"
        return x

    results_df["Geo Dist"] = results_df["Geo Dist"].apply(fix_geo_dist)
    return results_df

# -----------------------------
# Table Styling Functions
# -----------------------------
def highlight_geo_dist(s: pd.Series) -> List[str]:
    """Apply background color to Geo Dist cells based on numeric bins."""
    styles = []
    for val in s:
        try:
            num = float(val.replace(" km", ""))
        except:
            styles.append("")
            continue
        if num < 60:
            styles.append("background-color: #ADD8E6; color: black; border: 1px solid gray;")
        elif num < 120:
            styles.append("background-color: #90EE90; color: black; border: 1px solid gray;")
        elif num < 200:
            styles.append("background-color: #FFDAB9; color: black; border: 1px solid gray;")
        else:
            styles.append("background-color: #F08080; color: black; border: 1px solid gray;")
    return styles

def color_aitch_dist(val) -> str:
    """Apply background color to Aitch Dist cells based on numeric bins."""
    try:
        v = float(val)
    except:
        return ""
    if v < 1:
        return "background-color: #ADD8E6; color: black; border: 1px solid gray; border-radius: 0px"
    elif v < 2:
        return "background-color: #90EE90; color: black; border: 1px solid gray; border-radius: 0px"
    elif v < 3:
        return "background-color: #FFDAB9; color: black; border: 1px solid gray; border-radius: 0px"
    else:
        return "background-color: #F08080; color: black; border: 1px solid gray; border-radius: 0px"

# -----------------------------
# Helper Functions for UI Display
# -----------------------------
def display_results_table(results_df: pd.DataFrame) -> None:
    """Style and display the results table along with a CSV download button."""
    styled_df = results_df.style.format({"Aitch Dist": "{:.2f}", "Eucl Dist": "{:.2f}"})
    styled_df = styled_df.apply(highlight_geo_dist, subset=["Geo Dist"])\
                             .applymap(color_aitch_dist, subset=["Aitch Dist"])
    styled_df = styled_df.set_table_styles([
        {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "1px solid gray")]},
        {"selector": "th",    "props": [("border", "1px solid gray"), ("padding", "4px"), ("border-radius", "0px")]},
        {"selector": "td",    "props": [("border", "1px solid gray"), ("padding", "4px"), ("border-radius", "0px")]}
    ])
    st.table(styled_df)

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name="results.csv",
        mime="text/csv"
    )

def display_scatter_plot(results_df: pd.DataFrame) -> None:
    """
    Create and display a scatter plot of Geo Dist vs. Aitch Dist.
    Configures the plot with white background, black text, partially transparent markers,
    and forces axes to start at zero. Rows with "Unknown" Geo Dist are excluded.
    """
    # Filter out rows with unknown Geo Dist
    plot_df = results_df[results_df["Geo Dist"] != "Unknown"].copy()
    plot_df["Geo Dist Num"] = plot_df["Geo Dist"].astype(str).str.replace(" km", "", regex=False).astype(float)

    lithology_color_map = {
        "Eclogite": "#EE82EE",      # bright purple
        "Amphibolite": "darkblue",
        "Metabasite": "green",
        "Ophite-THOL": "#FF0000",    # bright red
        "Ophite-ALK": "#FF7F7F",     # light red
        "Basalt-ALK": "#000000",     # black
        "Gabbro-ALK": "#A9A9A9",      # dark grey
        "Gabbro-THOL": "#808080",     # mid grey
        "Gabbro-CALC": "#D3D3D3",     # light grey
        "Metagabbro": "#90EE90"       # light green
    }
    lithology_order = [
        "Eclogite", "Amphibolite", "Metabasite",
        "Ophite-THOL", "Ophite-ALK", "Basalt-ALK",
        "Gabbro-ALK", "Gabbro-THOL", "Gabbro-CALC",
        "Metagabbro"
    ]

    fig = px.scatter(
        plot_df,
        x="Geo Dist Num",
        y="Aitch Dist",
        color="Lithology",
        category_orders={"Lithology": lithology_order},
        color_discrete_map=lithology_color_map,
        hover_data={"Geo Site": True, "Geo Acc #": True},
        title="Scatter Plot of Geographical Distance vs. Aitchison Distance"
    )
    for ref in [60, 120, 200]:
        fig.add_vline(x=ref, line_dash="dash", line_color="black", opacity=0.5)
    for ref in [1, 2, 3]:
        fig.add_hline(y=ref, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_traces(
        marker=dict(
            size=20,
            line=dict(color="black", width=2),
            opacity=0.8
        )
    )
    fig.update_layout(
        template=None,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        xaxis_title="Geographical Distance (km)",
        yaxis_title="Aitchison Distance",
        margin=dict(l=40, r=40, t=40, b=40),
        legend_title="Lithology"
    )
    fig.update_xaxes(
        rangemode="tozero",
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickfont=dict(color="black"),
        showgrid=False,
        zeroline=False
    )
    fig.update_yaxes(
        rangemode="tozero",
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickfont=dict(color="black"),
        showgrid=False,
        zeroline=False
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Main Application
# -----------------------------
def main() -> None:
    st.title("GSF - Geology Source Finder")
    accession_number = st.text_input("Enter the accession number of the artefact:")

    try:
        artefact_df, geology_df, geo_coords_df, arch_coords_df = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error("Error loading data.")
        st.exception(e)
        return

    if accession_number:
        artefacts = artefact_df[artefact_df["Accession #"] == accession_number]
        if artefacts.empty:
            st.error(f"No artefact found with Accession # {accession_number}")
        elif len(artefacts) == 1:
            artefact = artefacts.iloc[0]
            if "Region" in artefact.index and pd.notnull(artefact["Region"]):
                site_info = f"{artefact['Site']} ({artefact['Region']})"
            else:
                site_info = artefact.get("Site", "Unknown Site")
            st.subheader(f"Results for Accession # {artefact['Accession #']} - {site_info}")

            results_df = get_top_20_matches(artefact, geology_df, geo_coords_df, arch_coords_df)
            if not results_df.empty:
                display_results_table(results_df)
                display_scatter_plot(results_df)
        else:
            st.info(f"Multiple artefacts found for Accession # {accession_number}. Please select a site:")
            sites = artefacts["Site"].unique()
            selected_site = st.selectbox("Select a site", sites)
            selected_artefact = artefacts[artefacts["Site"].str.lower() == selected_site.lower()].iloc[0]
            st.subheader(f"Results for Accession # {selected_artefact['Accession #']} - {selected_site}")
            results_df = get_top_20_matches(selected_artefact, geology_df, geo_coords_df, arch_coords_df)
            if not results_df.empty:
                display_results_table(results_df)
                display_scatter_plot(results_df)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred.")
        st.exception(e)
