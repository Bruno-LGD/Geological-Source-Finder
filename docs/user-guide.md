# User Guide

## Getting Started

### Launching the Application

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501` in your default browser.

### Sidebar Settings

Before running queries, configure settings in the left sidebar:

- **Element Mode**: Choose between "Trace Elements" (16 ratios, more robust) or "All Elements"
  (22 ratios, more comprehensive). The current dataset statistics are displayed below the selector.
- **About Section**: Expandable panel explaining the methodology and distance metrics.

## Single Query

### Step-by-Step

1. **Select Query Direction**
   - *Artefact to Geology*: Find geological sources matching an artefact
   - *Geology to Artefact*: Find artefacts matching a geological sample

2. **Choose a Sample**
   - Use the searchable dropdown to find your sample by accession number
   - If duplicate accession numbers exist, a secondary selector appears

3. **Set Number of Matches**
   - Adjust the "Top N matches" slider (1-100, default 20)
   - More matches provide broader context but may include weaker matches

4. **Filter by Region** (optional)
   - Restrict results to specific geographic regions
   - Useful when you have prior knowledge of likely source areas

5. **Run Query**
   - Click "Find Matches" to execute

### Understanding Results

#### Results Table

Each row represents a potential match, color-coded by match quality:

| Color       | Meaning           |
|-------------|-------------------|
| Light Blue  | Very strong match |
| Light Green | Strong match      |
| Peach       | Moderate match    |
| Light Coral | Weak match        |

Columns include:

- **Rank**: Position in match quality order
- **Accession #**: Target sample identifier
- **Site**: Sample location name
- **Region**: Geographic region
- **Lithology**: Rock type
- **Aitchison Dist.**: Primary compositional distance
- **Euclidean Dist.**: Secondary distance metric
- **Geo Dist. (km)**: Physical distance between locations

#### Radar Chart

Compares the geochemical profile (ratio values) of your query sample against selected matches.
Each axis represents a different element ratio, normalized to a log scale for visibility.

- Select up to 8 matches to compare
- Similar profile shapes suggest compositional affinity
- Useful for visual confirmation of statistical matches

#### Scatter Plot

Plots matches with:

- **X-axis**: Geographic distance (km)
- **Y-axis**: Aitchison distance
- Points colored by **lithology** or **region**
- Reference lines indicate threshold boundaries

Ideal matches cluster in the bottom-left corner (low compositional and geographic distance).

#### Map

Interactive map showing:

- **Red marker**: Your query sample location
- **Green markers**: Very strong matches (Aitchison < 1.0)
- **Lime markers**: Strong matches (1.0-2.0)
- **Orange markers**: Moderate matches (2.0-3.0)
- **Dark red markers**: Weak matches (> 3.0)

Hover over markers for sample details.

### Downloading Results

- **CSV**: Plain text format, suitable for further analysis
- **Excel**: Styled workbook with color-coded cells matching the on-screen table

## Batch Query

### Preparing Your CSV

Create a CSV file with at least one column containing accession numbers. Example:

```csv
Accession
ACC-001
ACC-002
ACC-003
```

### Running a Batch Query

1. Upload your CSV file
2. Select the column containing accession numbers
3. Choose query direction and number of top matches
4. Click "Run Batch Query"

### Batch Results

- **Frequency Summary**: Ranks geological sources by how many times they appear as a top match across
  all queried samples. Sources appearing frequently may represent primary procurement areas.
- **Combined Results Table**: Full results for all queries, downloadable as CSV or Excel.
- **Error Report**: Lists any accession numbers not found in the database.

## Comparison Mode

### Purpose

Compare two or more artefacts to find geological sources that match multiple artefacts.
This can reveal:

- Common procurement areas
- Trade network hubs
- Workshop connections between artefacts

### Running a Comparison

1. Enter two or more accession numbers (comma-separated or one per line)
2. Set the number of top matches per sample
3. Click "Compare"

### Comparison Results

- **Shared Sources Table**: Lists geological sources found in the top matches of two or more queried artefacts
  - **Shared Count**: Number of queried artefacts this source matches
  - **Avg Aitchison**: Average compositional distance across all matching queries
  - Green highlighting indicates sources matching *all* queried artefacts
- **Individual Results**: Expandable panels showing the full match list for each queried artefact

## Tips

- **Start with Trace Elements mode** for initial screening - it's more robust against alteration
- **Use All Elements mode** to refine results when trace elements show ambiguous matches
- **Check the radar chart** to visually confirm that a statistical match has a similar geochemical profile
- **Geographic distance is contextual** - a compositionally strong match at a great distance may
  indicate long-distance exchange rather than a bad match
- **In batch mode**, sources with high frequency counts across many artefacts are the most reliable identifications
- **In comparison mode**, universal matches (green rows) are the strongest candidates for a common source

## Data Quality Warnings

The application displays warnings when:

- A sample has missing ratio values (some ratios will be excluded from computation)
- An accession number from a batch/comparison query is not found in the database
- Coordinates are unavailable for geographic distance calculation (shown as "Unknown")
