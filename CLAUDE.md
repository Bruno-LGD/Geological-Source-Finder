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
├── app.py                 # Thin entry point (set_page_config + imports gsf.main)
├── run.py                 # Launch helper (calls streamlit run app.py)
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit theme & server config
├── gsf/                   # Main application package
│   ├── __init__.py        # Re-exports main()
│   ├── constants.py       # All constants, thresholds, color maps, small helpers
│   ├── config.py          # JSON config persistence (app settings)
│   ├── auth.py            # OAuth device code flow, token management, Graph API URLs
│   ├── data.py            # Excel/CSV loading (@st.cache_data)
│   ├── distances.py       # Vectorized Aitchison, Euclidean, geodesic calculations
│   ├── photos.py          # OneDrive/local photo resolution and caching
│   ├── matching.py        # get_top_matches, get_top_matches_multimode
│   ├── styling.py         # HTML/CSS color functions for distance cells
│   ├── export.py          # Excel workbook generation with openpyxl formatting
│   ├── viz.py             # Display functions (table, map, scatter, radar, photos, legend)
│   ├── queries.py         # Single, batch, comparison query execution
│   └── app.py             # main() — sidebar config, tabs, data loading orchestration
├── Data/
│   ├── AXEs metabasite data (All Elements).xlsx
│   ├── AXEs metabasite data (Trace Elements).xlsx
│   ├── Geology samples data (All Elements).xlsx
│   ├── Geology samples data (Trace Elements).xlsx
│   └── Coordinates sheet.xlsx
└── docs/                  # Documentation
```

## Architecture

The application is organized as the `gsf/` Python package with 12 focused modules:

1. **constants.py** - Thresholds, color schemes, metadata columns, mode labels
2. **config.py** - JSON config persistence (photos folder, OneDrive settings)
3. **auth.py** - Microsoft Graph OAuth2 device code flow, token management
4. **data.py** - Cached Excel/CSV loading with `@st.cache_data`
5. **distances.py** - Aitchison (CLR transform), ALR-5, Euclidean, Geodesic
6. **photos.py** - OneDrive Graph API image download, local photo resolution
7. **matching.py** - Single-mode and multi-mode top-N matching logic
8. **styling.py** - HTML/CSS color functions for distance cells in tables
9. **export.py** - Styled Excel workbook generation with openpyxl
10. **viz.py** - Tables, radar charts, scatter plots, maps, photo viewers
11. **queries.py** - Single, batch, and comparison query execution
12. **app.py** - Main UI: sidebar + three-tab interface

Module dependency graph (acyclic):

```text
constants, config, auth          (leaf modules — no gsf imports)
data, distances                  → constants
photos                           → auth
styling, export                  → constants
matching                         → constants, distances
viz                              → constants, styling, export, photos, distances
queries                          → constants, matching, viz, photos, distances
gsf/app                          → constants, config, auth, data, queries
```

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
