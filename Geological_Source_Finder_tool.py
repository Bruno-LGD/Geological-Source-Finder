import os
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
      1) A "Data" subfolder in the same directory as this script (for GitHub deployment),
      2) A "Data" folder in the parent directory (for local use).

    Expected files (with these exact names):
      - "AXEs metabasite data (Trace Elements).xlsx" (sheet: "AXEs Ratios")
      - "Geology samples data (Trace Elements).xlsx" (sheet: "Geology ratios")
      - "Coordinates sheet.xlsx" (sheets: "Geology Samples Coord" and "Archaeology Sites Coord")

    Returns:
      A tuple of DataFrames: (artefact_df, geology_df, geo_coords_df, arch_coords_df)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Directories to check in order
    possible_dirs = [
        os.path.join(base_dir, "Data"),
        os.path.join(base_dir, "..", "Data")
    ]

    filenames = {
        "artefact": "AXEs metabasite data (Trace Elements).xlsx",
        "geology":  "Geology samples data (Trace Elements).xlsx",
        "coords":   "Coordinates sheet.xlsx"  # note: singular "sheet"
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
        artefact_df = pd.read_excel(os.path.join(found_data_dir, filenames["artefact"]), sheet_name="AXEs Ratios")
        geology_df = pd.read_excel(os.path.join(found_data_dir, filenames["geology"]), sheet_name="Geology ratios")
        geo_coords_df = pd.read_excel(os.path.join(found_data_dir, filenames["coords"]), sheet_name="Geology Samples Coord")
        arch_coords_df = pd.read_excel(os.path.join(found_data_dir, filenames["coords"]), sheet_name="Archaeology Sites Coord")
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
    Replaces any NaNs/Inf with 1e-10, as in your original approach.
    """
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    x = np.nan_to_num(x, nan=1e-10, posinf=1e-10, neginf=1e-10)
    y = np.nan_to_num(y, nan=1e-10, posinf=1e-10, neginf=1e-10)
    log_x = np.log(x / np.prod(x)**(1/len(x)))
    log_y = np.log(y / np.prod(y)**(1/len(y)))
    return norm(log_x - log_y)

# -----------------------------
# Query Functions
# -----------------------------
def get_top_20_matches(
    artefact: pd.Series,
    geology_df: pd.DataFrame,
    geo_coords_df: pd.DataFrame,
    arch_coords_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For a given artefact, compute and return a DataFrame of the top 20 matching geology samples.
    Uses fixed column indexing (artefact.dropna().iloc[6:]) for the ratio columns.
    Returns a DataFrame with columns:
      Lithology, Geo Acc #, Geo Site, Region, Aitch Dist, Eucl Dist, Geo Dist.
    """

    # Force ratio columns to numeric, replacing invalid values with 1e-10
    available_artefact_ratios = pd.to_numeric(
        artefact.dropna().iloc[6:], errors='coerce'
    ).fillna(1e-10)
    available_artefact_ratios = np.nan_to_num(available_artefact_ratios, nan=1e-10, posinf=1e-10, neginf=1e-10)

    ratio_cols = list(pd.to_numeric(artefact.dropna().iloc[6:], errors='coerce').fillna(1e-10).index)

    required_cols = ["Lithology", "Accession #", "Site", "Region"] + ratio_cols
    aligned_geology_df = geology_df[required_cols].dropna()

    distances = []

    if artefact["Site"] in arch_coords_df["Site"].values:
        coords_row = arch_coords_df[arch_coords_df["Site"] == artefact["Site"]].iloc[0]
        artefact_coords = (coords_row["Latitude"], coords_row["Longitude"])
    else:
        artefact_coords = None

    for _, geo_sample in aligned_geology_df.iterrows():
        geo_ratios = pd.to_numeric(geo_sample[ratio_cols], errors='coerce').fillna(1e-10)
        geo_ratios = np.nan_to_num(geo_ratios, nan=1e-10, posinf=1e-10, neginf=1e-10)

        aitch_dist = aitchison_distance(available_artefact_ratios, geo_ratios)
        eucl_dist = euclidean(available_artefact_ratios, geo_ratios)

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
            geo_sample["Region"],
            round(aitch_dist, 2),
            round(eucl_dist, 2),
            geo_distance
        ])

    results_df = pd.DataFrame(
        distances,
        columns=["Lithology", "Geo Acc #", "Geo Site", "Region", "Aitch Dist", "Eucl Dist", "Geo Dist"]
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

def get_top_20_matches_for_geology(
    geology_sample: pd.Series,
    artefact_df: pd.DataFrame,
    arch_coords_df: pd.DataFrame,
    geo_coords_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For a given geology sample, compute and return a DataFrame of the top 20 matching artefacts.
    Uses fixed column indexing (geology_sample.dropna().iloc[6:]) for the ratio columns.
    Returns a DataFrame with columns:
      Artefact Acc #, Artefact Site, Artefact Region, Aitch Dist, Eucl Dist, Geo Dist.
    """

    available_geology_ratios = pd.to_numeric(
        geology_sample.dropna().iloc[6:], errors='coerce'
    ).fillna(1e-10)
    available_geology_ratios = np.nan_to_num(available_geology_ratios, nan=1e-10, posinf=1e-10, neginf=1e-10)

    ratio_cols = list(pd.to_numeric(
        geology_sample.dropna().iloc[6:], errors='coerce'
    ).fillna(1e-10).index)

    required_cols = ["Accession #", "Site", "Region"] + ratio_cols
    filtered_artefact_df = artefact_df[required_cols].dropna()

    distances = []
    geo_coords_row = geo_coords_df[geo_coords_df["Accession #"] == geology_sample["Accession #"]]
    if not geo_coords_row.empty:
        geo_coords_row = geo_coords_row.iloc[0]
        geology_coords = (geo_coords_row["Latitude"], geo_coords_row["Longitude"])
    else:
        geology_coords = None

    for _, artefact in filtered_artefact_df.iterrows():
        artefact_ratios = pd.to_numeric(artefact[ratio_cols], errors='coerce').fillna(1e-10)
        artefact_ratios = np.nan_to_num(artefact_ratios, nan=1e-10, posinf=1e-10, neginf=1e-10)

        aitch_dist = aitchison_distance(available_geology_ratios, artefact_ratios)
        eucl_dist = euclidean(available_geology_ratios, artefact_ratios)

        artefact_coords_row = arch_coords_df[arch_coords_df["Site"] == artefact["Site"]]
        if not artefact_coords_row.empty and geology_coords:
            artefact_coords_row = artefact_coords_row.iloc[0]
            try:
                dist_km = geodesic(geology_coords, (artefact_coords_row["Latitude"], artefact_coords_row["Longitude"])).km
                geo_distance = int(dist_km)
            except (ValueError, TypeError):
                geo_distance = "Unknown"
        else:
            geo_distance = "Unknown"

        distances.append([
            artefact["Accession #"],
            artefact["Site"],
            artefact.get("Region", "Unknown"),
            round(aitch_dist, 2),
            round(eucl_dist, 2),
            geo_distance
        ])

    results_df = pd.DataFrame(
        distances,
        columns=["Artefact Acc #", "Artefact Site", "Artefact Region", "Aitch Dist", "Eucl Dist", "Geo Dist"]
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
# UI Display Functions
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

###############################
# FIXED COLOR MAPS FOR LITHOLOGY & REGION
###############################
# Region color map
region_color_map = {
    "Upper Guadalquivir": "orange",
    "Southeast": "dodgerblue",
    "Subbaetic": "red",
    "Middle Guadalquivir": "limegreen",
    "Lower Guadalquivir": "mediumpurple"
}

# Lithology color map
lithology_color_map = {
    "Ophite-THOL": "#FF0000",       # bright red
    "Ophite-ALK": "#8B0000",        # dark red
    "Amphibolite": "#0000FF",       # bright blue
    "Eclogite": "#EE82EE",          # bright purple
    "Metabasite": "#00FF00",        # bright green
    "Basalt-ALK": "#000000",        # black
    "Gabbro-THOL": "#808080",       # mid grey
    "Gabbro-CALC": "#D3D3D3",       # light grey
    "Gabbro-ALK": "#696969",        # dark grey
    "Metagabbro": "#90EE90"         # light green
}

def display_scatter_plot(results_df: pd.DataFrame) -> None:
    """
    Create and display a scatter plot of Geo Dist vs. Aitch Dist.
    Dynamically determines which columns to use for hover_data,
    so that 'ValueError: hover_data_0...' doesn't occur.
    Also uses fixed color maps for lithologies or regions.
    """
    plot_df = results_df[results_df["Geo Dist"] != "Unknown"].copy()
    if plot_df.empty:
        st.warning("No rows with known geographical distance to plot.")
        return

    plot_df["Geo Dist Num"] = plot_df["Geo Dist"].astype(str).str.replace(" km", "", regex=False).astype(float)

    # Build hover_data dynamically
    hover_cols = {}
    for possible_col in ["Geo Acc #", "Geo Site", "Artefact Acc #", "Artefact Site"]:
        if possible_col in plot_df.columns:
            hover_cols[possible_col] = True

    # Decide color column if "Lithology" or "Artefact Region" exist
    color_col = None
    color_map = None
    if "Lithology" in plot_df.columns:
        color_col = "Lithology"
        color_map = lithology_color_map
    elif "Artefact Region" in plot_df.columns:
        color_col = "Artefact Region"
        color_map = region_color_map

    fig = px.scatter(
        plot_df,
        x="Geo Dist Num",
        y="Aitch Dist",
        color=color_col,
        hover_data=hover_cols,
        title="Scatter Plot of Geographical Distance vs. Aitchison Distance",
        color_discrete_map=(color_map if color_map else {})
    )

    # Reference lines
    for ref in [60, 120, 200]:
        fig.add_vline(x=ref, line_dash="dash", line_color="black", opacity=0.5)
    for ref in [1, 2, 3]:
        fig.add_hline(y=ref, line_dash="dash", line_color="black", opacity=0.5)

    # Marker style
    fig.update_traces(
        marker=dict(
            size=20,
            line=dict(color="black", width=2),
            opacity=0.8
        )
    )
    # Layout
    fig.update_layout(
        template=None,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        xaxis_title="Geographical Distance (km)",
        yaxis_title="Aitchison Distance",
        margin=dict(l=40, r=40, t=40, b=40),
        legend_title=color_col if color_col else "Legend"
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

    # Place "Enter the accession number" at the very top
    accession_number = st.text_input("Enter the accession number:")

    # Radio button for selecting query mode
    query_mode = st.radio("Select Query Mode:", ["Artefact -> Geology", "Geology -> Artefact"])

    # Let user choose how many top results to display
    top_n = st.number_input("Number of top results:", min_value=1, max_value=100, value=20, step=1)

    try:
        artefact_df, geology_df, geo_coords_df, arch_coords_df = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error("Error loading data.")
        st.exception(e)
        return

    # Multi-select region filter
    all_regions = sorted(set(artefact_df["Region"].dropna().unique()) | set(geology_df["Region"].dropna().unique()))
    selected_regions = st.multiselect("Filter by region(s) (optional):", all_regions)

    if accession_number:
        accession_number = str(accession_number)
        if query_mode == "Artefact -> Geology":
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

                # Retrieve top 20, then slice by top_n
                results_df = get_top_20_matches(artefact, geology_df, geo_coords_df, arch_coords_df)
                if len(results_df) > top_n:
                    results_df = results_df.head(top_n)

                if selected_regions:
                    results_df = results_df[results_df["Region"].isin(selected_regions)]

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
                if len(results_df) > top_n:
                    results_df = results_df.head(top_n)

                if selected_regions:
                    results_df = results_df[results_df["Region"].isin(selected_regions)]

                if not results_df.empty:
                    display_results_table(results_df)
                    display_scatter_plot(results_df)

        else:
            # Geology -> Artefact
            geology_samples = geology_df[geology_df["Accession #"] == accession_number]
            if geology_samples.empty:
                st.error(f"No geology sample found with Accession # {accession_number}")
            elif len(geology_samples) == 1:
                geology_sample = geology_samples.iloc[0]
                st.subheader(f"Results for Geology Sample Accession # {geology_sample['Accession #']} - {geology_sample['Site']}")
                results_df = get_top_20_matches_for_geology(geology_sample, artefact_df, arch_coords_df, geo_coords_df)
                if len(results_df) > top_n:
                    results_df = results_df.head(top_n)

                if selected_regions:
                    results_df = results_df[results_df["Artefact Region"].isin(selected_regions)]

                if not results_df.empty:
                    display_results_table(results_df)
                    display_scatter_plot(results_df)
            else:
                st.info(f"Multiple geology samples found for Accession # {accession_number}. Please select one:")
                sites = geology_samples["Site"].unique()
                selected_site = st.selectbox("Select a site", sites)
                geology_sample = geology_samples[geology_samples["Site"].str.lower() == selected_site.lower()].iloc[0]
                st.subheader(f"Results for Geology Sample Accession # {geology_sample['Accession #']} - {selected_site}")

                results_df = get_top_20_matches_for_geology(geology_sample, artefact_df, arch_coords_df, geo_coords_df)
                if len(results_df) > top_n:
                    results_df = results_df.head(top_n)

                if selected_regions:
                    results_df = results_df[results_df["Artefact Region"].isin(selected_regions)]

                if not results_df.empty:
                    display_results_table(results_df)
                    display_scatter_plot(results_df)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred.")
        st.exception(e)
