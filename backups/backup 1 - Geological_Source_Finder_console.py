
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from numpy.linalg import norm
from geopy.distance import geodesic
from tabulate import tabulate

# Set working directory to the folder containing the script and data
os.chdir(r"C:\Users\bruno\OneDrive\Archaeology\PhD\Geology Source Finder Python tool")

# Load artefact data
artefact_df = pd.read_excel('AXEs metabasite data (Trace elem-AI).xlsx', sheet_name='AXEs Ratios')

# Load geological sample data
geology_df = pd.read_excel('Geology samples data (Trace elem-AI).xlsx', sheet_name='Geology ratios')

# Load coordinates data
geo_coords_df = pd.read_excel('Coordinates sheet.xlsx', sheet_name='Geology Samples Coord')
arch_coords_df = pd.read_excel('Coordinates sheet.xlsx', sheet_name='Archaeology Sites Coord')

# Ensure that accession numbers in both artefact and geological data are treated as strings
artefact_df['Accession #'] = artefact_df['Accession #'].astype(str)
geology_df['Accession #'] = geology_df['Accession #'].astype(str)
geo_coords_df['Accession #'] = geo_coords_df['Accession #'].astype(str)

# Aitchison Distance Function
def aitchison_distance(x, y):
    x = pd.to_numeric(x, errors='coerce')  
    y = pd.to_numeric(y, errors='coerce')
    x = np.nan_to_num(x, nan=1e-10, posinf=1e-10, neginf=1e-10)
    y = np.nan_to_num(y, nan=1e-10, posinf=1e-10, neginf=1e-10)
    log_x = np.log(x / np.prod(x) ** (1 / len(x)))
    log_y = np.log(y / np.prod(y) ** (1 / len(y)))
    return norm(log_x - log_y)

# Function to get the top 20 matches for a specific artefact
def get_top_20_matches(artefact):
    distances = []
    available_artefact_ratios = artefact.dropna().iloc[6:]  # Remove metadata and keep only available ratios

    # Align geological samples with available artefact ratios, keeping metadata intact
    aligned_geology_df = geology_df[['Lithology', 'Accession #', 'Site', 'Region'] + list(available_artefact_ratios.index)].dropna()

    # Get artefact coordinates
    if artefact['Site'] in arch_coords_df['Site'].values:
        artefact_coords = arch_coords_df[arch_coords_df['Site'] == artefact['Site']].iloc[0][['Latitude', 'Longitude']]
    else:
        artefact_coords = None  # Mark as missing coordinates

    # Compare the artefact against geological samples
    for _, geo_sample in aligned_geology_df.iterrows():
        geo_ratios = geo_sample[available_artefact_ratios.index].values  # Get corresponding geological ratios

        # Calculate distances
        aitch_dist = aitchison_distance(available_artefact_ratios.values, geo_ratios)
        eucl_dist = euclidean(available_artefact_ratios.values, geo_ratios)

        # Get geographical distance (Haversine distance)
        geo_coords = geo_coords_df[geo_coords_df['Accession #'] == geo_sample['Accession #']]
        if artefact_coords is not None and not geo_coords.empty:
            geo_coords = geo_coords.iloc[0]
            geo_distance = int(
                geodesic(
                    (artefact_coords['Latitude'], artefact_coords['Longitude']),
                    (geo_coords['Latitude'], geo_coords['Longitude'])
                ).km
            )
        else:
            geo_distance = 'Unknown'

        # Append results, keeping metadata
        distances.append([geo_sample['Lithology'], geo_sample['Accession #'], geo_sample['Site'], aitch_dist, eucl_dist, geo_distance])

    # Create a DataFrame with the results
    results_df = pd.DataFrame(distances, columns=['Lithology', 'Geo Acc #', 'Geo Site', 'Aitch Dist', 'Eucl Dist', 'Harv Dist'])

    # Sort the DataFrame by Aitchison distance first, and then by Euclidean distance
    results_df = results_df.sort_values(by=['Aitch Dist', 'Eucl Dist']).head(20)

    # Round the distances to 2 decimal places for better readability
    results_df[['Aitch Dist', 'Eucl Dist']] = results_df[['Aitch Dist', 'Eucl Dist']].round(2)

    # Reset the index and add 1 to start from 1
    results_df.reset_index(drop=True, inplace=True)
    results_df.index += 1

    # Print the results using tabulate with custom formatting
    table = tabulate(results_df, headers='keys', tablefmt='grid', showindex=True, numalign="right", stralign="center")
    print(f"\nTop 20 Geological Matches for Accession # {artefact['Accession #']} :")
    print(table)

# Main function to keep asking for accession numbers
def main():
    while True:
        accession_number = input("\nEnter the accession number of the artefact (or type 'exit' to quit): ").strip()
        if accession_number.lower() == 'exit':
            print("Exiting the program.")
            break

        # Find all artefacts with the given accession number
        artefacts = artefact_df[artefact_df['Accession #'] == accession_number]

        if artefacts.empty:
            print(f"No artefact found with Accession # {accession_number}")
        elif len(artefacts) == 1:
            # If only one artefact found, process it directly
            artefact = artefacts.iloc[0]
            get_top_20_matches(artefact)
        else:
            # If multiple artefacts are found, ask which site to process
            print(f"Multiple artefacts found for Accession # {accession_number}:")
            print(artefacts[['Site']])
            
            while True:
                site = input("Please specify the site for the artefact: ").strip()
                # Filter for the selected site
                selected_artefact = artefacts[artefacts['Site'].str.lower() == site.lower()]
                
                if selected_artefact.empty:
                    print(f"No artefact found with Accession # {accession_number} from site {site}. Please try again.")
                else:
                    get_top_20_matches(selected_artefact.iloc[0])
                    break

if __name__ == "__main__":
    main()
