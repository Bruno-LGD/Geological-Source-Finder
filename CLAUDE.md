# GSF - Geology Source Finder

## Project Overview

A Streamlit web application for matching archaeological artefacts to potential geological sources
using trace element ratio analysis. Built for archaeological provenance studies of metabasite
stone axes from southern Spain.

## Tech Stack

- **Framework:** Streamlit (wide layout, custom theme)
- **Language:** Python 3.x
- **Key Libraries:** pandas, numpy, scipy, geopy, plotly, openpyxl

## Project Structure

```text
Geology-Source-Finder/
├── app.py                 # Main application (all logic in one file)
├── run.py                 # Launch helper (calls streamlit run app.py)
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit theme & server config
├── Data/
│   ├── AXEs metabasite data (All Elements).xlsx
│   ├── AXEs metabasite data (Trace Elements).xlsx
│   ├── Geology samples data (All Elements).xlsx
│   ├── Geology samples data (Trace Elements).xlsx
│   └── Coordinates sheet.xlsx
└── docs/                  # Documentation
```

## Architecture

The entire application is in `app.py` (~1,315 lines), organized into sections:

1. **Constants & Config** (lines 19-92) - Thresholds, color schemes, metadata columns
2. **Data Loading** (lines 98-167) - Cached Excel loading with `@st.cache_data`
3. **Distance Functions** (lines 173-314) - Aitchison (CLR transform), Euclidean, Geodesic
4. **Excel Export** (lines 371-453) - Styled workbook generation with color-coded fills
5. **Visualization** (lines 459-795) - Tables, radar charts, scatter plots, maps
6. **Query Execution** (lines 801-909) - Single, batch, and comparison modes
7. **Main UI** (lines 1199-1308) - Sidebar + three-tab interface

## Key Concepts

- **Aitchison Distance**: Primary metric. Uses Centered Log-Ratio (CLR) transform for compositional data.
  Lower = better match.
- **Element Modes**: "Trace Elements" (16 ratios) or "All Elements" (22 ratios including major elements)
- **Query Directions**: Artefact→Geology or Geology→Artefact matching
- **Metadata Columns**: n, Accession #, Site, Region, Type, Lithology (excluded from calculations)

## Distance Thresholds

| Metric     | Very Strong | Strong   | Moderate |
|------------|-------------|----------|----------|
| Aitchison  | < 1.0       | < 2.0    | < 3.0    |
| Euclidean  | < 2.0       | < 4.0    | < 6.0    |
| Geographic | < 60 km     | < 120 km | < 200 km |

## Data Files

- **AXEs files**: Archaeological artefact ratio data (sheet: "AXEs Ratios")
- **Geology files**: Reference geology sample ratios (sheet: "Geology ratios")
- **Coordinates sheet**: Two sheets - "Geology Samples Coord" and "Archaeology Sites Coord"

## Running

```bash
streamlit run app.py
# or
python run.py
```

## Development Notes

- All distance calculations are vectorized with NumPy for performance
- Geodesic distances are only computed for top-N results (lazy evaluation)
- Zero/invalid ratio values are replaced with epsilon (1e-10) before CLR transform
- The app resolves file paths flexibly (checks current dir and parent dir)
- Private functions are prefixed with underscore
