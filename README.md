# GSF - Geology Source Finder

An interactive web application for matching archaeological artefacts to potential geological sources
using trace element ratio analysis. Designed for provenance studies of metabasite stone axes from
southern Spain.

## Features

- **Single Query** - Select an artefact or geology sample and find its closest matches based on geochemical composition
- **Batch Query** - Upload a CSV of accession numbers to process multiple samples at once with
  frequency-ranked results
- **Comparison Mode** - Compare multiple artefacts to find shared geological sources, revealing
  potential trade patterns or workshop connections
- **Interactive Visualizations** - Color-coded results tables, radar/spider charts, scatter plots, and geographic maps
- **Excel & CSV Export** - Download styled, color-coded spreadsheets of results

## How It Works

The tool compares geochemical profiles using three distance metrics:

| Metric                  | What It Measures                                                  | Interpretation                                     |
|-------------------------|-------------------------------------------------------------------|----------------------------------------------------|
| **Aitchison Distance**  | Compositional similarity using Centered Log-Ratio (CLR) transform | Primary matching metric; < 1.0 = very strong match |
| **Euclidean Distance**  | Raw magnitude difference between ratio profiles                   | Secondary confirmation metric                      |
| **Geographic Distance** | Physical distance between sample locations (geodesic)             | Contextual; < 60 km = very close                   |

Samples can be compared using either **16 trace element ratios** or **22 ratios** (trace + major elements).

## Installation

### Prerequisites

- Python 3.9 or higher

### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Geology-Source-Finder
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Launch the Application

```bash
streamlit run app.py
```

Or use the helper script:

```bash
python run.py
```

The application opens in your browser at `http://localhost:8501`.

### Single Query

1. Choose the query direction: **Artefact to Geology** or **Geology to Artefact**
2. Select a sample from the dropdown (searchable by accession number)
3. Adjust the number of top matches (1-100)
4. Optionally filter results by region
5. View results as a color-coded table, radar chart, scatter plot, or map

### Batch Query

1. Prepare a CSV file with a column containing accession numbers
2. Upload the file and select the accession column
3. Choose the query direction and number of top matches
4. Results include a frequency summary ranking geological sources by how often they appear as matches

### Comparison Mode

1. Enter two or more accession numbers
2. The tool finds top matches for each and identifies **shared geological sources** - those appearing in multiple queries
3. Sources matching all queried artefacts are highlighted in green

## Data

The application expects Excel files in the `Data/` directory:

| File                                          | Contents                                                  |
|-----------------------------------------------|-----------------------------------------------------------|
| `AXEs metabasite data (Trace Elements).xlsx`  | Artefact ratio data (16 trace element ratios)             |
| `AXEs metabasite data (All Elements).xlsx`    | Artefact ratio data (22 ratios including major elements)  |
| `Geology samples data (Trace Elements).xlsx`  | Geology reference sample ratios (16 ratios)               |
| `Geology samples data (All Elements).xlsx`    | Geology reference sample ratios (22 ratios)               |
| `Coordinates sheet.xlsx`                      | GPS coordinates for geology samples and archaeology sites |

## Dependencies

| Package   | Version   | Purpose                          |
|-----------|-----------|----------------------------------|
| streamlit | >= 1.42.0 | Web application framework        |
| pandas    | >= 2.2.0  | Data manipulation                |
| numpy     | >= 2.1.0  | Vectorized numerical computation |
| scipy     | >= 1.14.0 | Scientific computing             |
| geopy     | >= 2.4.0  | Geodesic distance calculations   |
| plotly    | >= 6.0.0  | Interactive charts and maps      |
| openpyxl  | >= 3.1.0  | Excel file export with styling   |

## Project Structure

```text
Geology-Source-Finder/
├── app.py                 # Main application
├── run.py                 # Launch helper script
├── requirements.txt       # Python dependencies
├── CLAUDE.md              # Project context for AI assistants
├── .streamlit/
│   └── config.toml        # Theme and server configuration
├── Data/                  # Excel datasets (not tracked in git)
│   └── ...
└── docs/
    ├── methodology.md     # Distance metrics and statistical methods
    └── user-guide.md      # Detailed usage instructions
```

## License

This project is part of academic research. Contact the author for usage terms.
