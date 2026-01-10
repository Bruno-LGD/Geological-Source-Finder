import os
import sys
import logging
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from numpy.linalg import norm
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
def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load required Excel files from the specified directory.
    Expected files:
      - "AXEs metabasite data (Trace elem-AI).xlsx" (sheet: "AXEs Ratios")
      - "Geology samples data (Trace elem-AI).xlsx" (sheet: "Geology ratios")
      - "Coordinates sheet.xlsx" (sheets: "Geology Samples Coord" and "Archaeology Sites Coord")
    """
    artefact_path = os.path.join(data_dir, "AXEs metabasite data (Trace elem-AI).xlsx")
    geology_path = os.path.join(data_dir, "Geology samples data (Trace elem-AI).xlsx")
    coordinates_path = os.path.join(data_dir, "Coordinates sheet.xlsx")
    
    for file_path in [artefact_path, geology_path, coordinates_path]:
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            st.error(f"Required file not found: {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    try:
        artefact_df = pd.read_excel(artefact_path, sheet_name="AXEs Ratios")
        geology_df = pd.read_excel(geology_path, sheet_name="Geology ratios")
        geo_coords_df = pd.read_excel(coordinates_path, sheet_name="Geology Samples Coord")
        arch_coords_df = pd.read_excel(coordinates_path, sheet_name="Archaeology Sites Coord")
    except Exception as e:
        logger.exception("Error loading Excel files.")
        st.error("Error loading Excel files. Please check file and sheet names.")
        raise e

    # Ensure accession numbers are read as strings
    artefact_df['Accession #'] = artefact_df['Accession #'].astype(str)
    geology_df['Accession #'] = geology_df['Accession #'].astype(str)
    geo_coords_df['Accession #'] = geo_coords_df['Accession #'].astype(str)
    arch_coords_df['Accession #'] = arch_coords_df['Accession #'].astype(str)
    return artefact_df, geology_df, geo_coords_df, arch_coords_df

# -----------------------------
# Aitchison Distance Function (Original)
# -----------------------------
def aitchison_distance(x, y):
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    x = np.nan_to_num(x, nan=1e-10, posinf=1e-10, neginf=1e-10)
    y = np.nan_to_num(y, nan=1e-10, posinf=1e-10, neginf=1e-10)
    log_x = np.log(x / np.prod(x) ** (1 / len(x)))
    log_y = np.log(y / np.prod(y) ** (1 / len(y)))
    return norm(log_x - log_y)

# -----------------------------
# Main Matching Function (Using Original Ratio Selection)
# -----------------------------
def get_top_20_matches(artefact: pd.Series,
                       artefact_df: pd.DataFrame,
                       geology_df: pd.DataFrame,
                       geo_coords_df: pd.DataFrame,
                       arch_coords_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given an artefact row, compute the top 20 geological matches.
    It uses all available ratio values from column 7 onward from the artefact 
    and then aligns the geology samples to only those columns.
    
    Returns a DataFrame with columns:
      [Lithology, Accession #, Site, Aitch Dist, Eucl Dist, Geographical Distance (km), Region]
    All distances are rounded to 2 decimal places.
    """
    # Use the original method: take all available ratio columns (from column 7 onward)
    available_artefact_ratios = artefact.dropna().iloc[6:]
    ratio_columns = list(available_artefact_ratios.index)
    
    # Align geology samples: use metadata columns plus these ratio columns; drop any sample missing a value in any of these ratio columns.
    aligned_geology_df = geology_df[['Lithology', 'Accession #', 'Site', 'Region'] + ratio_columns].dropna()
    
    # Get artefact coordinates (from Archaeology Sites Coord)
    if artefact['Site'] in arch_coords_df['Site'].values:
        artefact_coords = arch_coords_df[arch_coords_df['Site'] == artefact['Site']].iloc[0][['Latitude', 'Longitude']]
    else:
        artefact_coords = None
    
    distances = []
    for _, geo_sample in aligned_geology_df.iterrows():
        geo_ratios = geo_sample[ratio_columns].values.astype(float)
        aitch_dist = aitchison_distance(available_artefact_ratios.values.astype(float), geo_ratios)
        eucl_dist = euclidean(available_artefact_ratios.values.astype(float), geo_ratios)
        
        # Calculate geographical distance (Haversine) using geopy
        geo_coords = geo_coords_df[geo_coords_df['Accession #'] == geo_sample['Accession #']]
        if artefact_coords is not None and not geo_coords.empty:
            geo_coords = geo_coords.iloc[0]
            try:
                geo_distance = int(geodesic(
                    (artefact_coords['Latitude'], artefact_coords['Longitude']),
                    (geo_coords['Latitude'], geo_coords['Longitude'])
                ).km)
            except (ValueError, TypeError):
                geo_distance = "Unknown"
        else:
            geo_distance = "Unknown"
        
        distances.append([geo_sample['Lithology'],
                          geo_sample['Accession #'],
                          geo_sample['Site'],
                          aitch_dist,
                          eucl_dist,
                          geo_distance,
                          geo_sample['Region']])
    
    results_df = pd.DataFrame(distances, 
        columns=["Lithology", "Accession #", "Site", "Aitch Dist", "Eucl Dist", "Geographical Distance (km)", "Region"])
    
    results_df = results_df.sort_values(by=["Aitch Dist", "Eucl Dist"]).head(20)
    results_df[['Aitch Dist', 'Eucl Dist']] = results_df[['Aitch Dist', 'Eucl Dist']].round(2)
    results_df.reset_index(drop=True, inplace=True)
    results_df.index += 1
    return results_df

# -----------------------------
# (Optional) Table and Chart Styling Functions
# -----------------------------
def color_geo_dist(val):
    try:
        v = float(val.replace(" km", "")) if isinstance(val, str) and " km" in val else float(val)
    except:
        return ""
    if v < 60:
        return "background-color: #ADD8E6; color: black; border: 1px solid gray; border-radius: 0px"
    elif v < 120:
        return "background-color: #90EE90; color: black; border: 1px solid gray; border-radius: 0px"
    elif v < 200:
        return "background-color: #FFDAB9; color: black; border: 1px solid gray; border-radius: 0px"
    else:
        return "background-color: #F08080; color: black; border: 1px solid gray; border-radius: 0px"

def color_aitch_dist(val):
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
# Streamlit Sidebar: Data Directory Selection
# -----------------------------
st.sidebar.header("Data Directory")
default_data_dir = r"C:\Users\bruno\OneDrive\Archaeology\PhD\AI\Data"
change_dir = st.sidebar.checkbox("Change Data Directory", value=False)
if change_dir:
    data_dir = st.sidebar.text_input("Enter Data Directory:", default_data_dir)
else:
    data_dir = default_data_dir

try:
    artefact_df, geology_df, geo_coords_df, arch_coords_df = load_data(data_dir)
except Exception:
    st.stop()

# -----------------------------
# Main UI
# -----------------------------
st.title("Geological Source Finder Tool")
accession_number = st.text_input("Enter the accession number of the artefact:")

if accession_number:
    artefacts = artefact_df[artefact_df['Accession #'] == accession_number]
    if artefacts.empty:
        st.error(f"No artefact found with Accession # {accession_number}")
    elif len(artefacts) == 1:
        artefact = artefacts.iloc[0]
        st.subheader(f"Results for Accession # {artefact['Accession #']} - {artefact.get('Site', 'Unknown Site')}")
        results = get_top_20_matches(artefact, artefact_df, geology_df, geo_coords_df, arch_coords_df)
        st.write("### Top 20 Geological Matches")
        st.dataframe(results)
        # (Optional: You can add table styling and a chart as in previous versions.)
    else:
        st.info(f"Multiple artefacts found for Accession # {accession_number}. Please select one:")
        st.write(artefacts[["Site"]])
        selected_site = st.text_input("Enter the site for the artefact:")
        if selected_site:
            selected = artefacts[artefacts["Site"].str.lower() == selected_site.lower()]
            if selected.empty:
                st.error(f"No artefact found with Accession # {accession_number} from site {selected_site}")
            else:
                artefact = selected.iloc[0]
                st.subheader(f"Results for Accession # {artefact['Accession #']} - {artefact['Site']}")
                results = get_top_20_matches(artefact, artefact_df, geology_df, geo_coords_df, arch_coords_df)
                st.write("### Top 20 Geological Matches")
                st.dataframe(results)
