# Methodology

## Overview

The Geology Source Finder uses compositional data analysis (CoDA) to compare the geochemical profiles
of archaeological artefacts against a reference database of geological samples. The goal is to identify
the most likely geological source for a given artefact based on trace element ratios.

## Distance Metrics

### Aitchison Distance (Primary Metric)

Aitchison distance is the standard metric for compositional data - data that represents parts of a whole,
such as element concentrations or ratios. Unlike Euclidean distance, it respects the compositional nature
of the data and is scale-invariant.

**Implementation:**

1. **Centered Log-Ratio (CLR) Transform**: For a composition vector `x`:

   ```text
   CLR(x) = [log(x₁) - mean(log(x)), log(x₂) - mean(log(x)), ..., log(xₙ) - mean(log(x))]
   ```

2. **Distance Calculation**: The Aitchison distance between compositions `x` and `y`:

   ```text
   d_A(x, y) = ||CLR(x) - CLR(y)||₂
   ```

   This is the Euclidean norm of the difference between CLR-transformed vectors.

**Edge Case Handling:**

- Zero or invalid values are replaced with epsilon (1e-10) before log transformation
- NaN values in ratio columns are dropped before computation
- Only ratio columns present in both query and target samples are used

**Interpretation Thresholds:**

| Distance   | Interpretation                                |
|------------|-----------------------------------------------|
| < 1.0      | Very strong match - highly likely same source |
| 1.0 - 2.0  | Strong match - probable same source           |
| 2.0 - 3.0  | Moderate match - possible relationship        |
| > 3.0      | Weak match - unlikely same source             |

### Euclidean Distance (Secondary Metric)

Standard geometric distance between ratio vectors, used as a secondary confirmation metric:

```text
d_E(x, y) = sqrt(sum((xᵢ - yᵢ)²))
```

**Interpretation Thresholds:**

| Distance   | Interpretation    |
|------------|-------------------|
| < 2.0      | Very strong match |
| 2.0 - 4.0  | Strong match      |
| 4.0 - 6.0  | Moderate match    |
| > 6.0      | Weak match        |

### Geographic Distance (Contextual Metric)

Geodesic distance between sample locations using the WGS-84 ellipsoid model, computed via the `geopy`
library. This provides archaeological context rather than a similarity measure.

**Interpretation Thresholds:**

| Distance      | Interpretation                                |
|---------------|-----------------------------------------------|
| < 60 km       | Very close - strong geographical association  |
| 60 - 120 km   | Close - plausible transport distance          |
| 120 - 200 km  | Moderate - significant but possible transport |
| > 200 km      | Distant - long-distance exchange likely       |

## Element Modes

### Trace Elements (16 ratios)

Uses ratios derived from trace elements only. Trace elements are typically more diagnostic for
provenance studies because they are less affected by weathering and post-depositional processes.

### All Elements (22 ratios)

Includes both trace element and major element ratios. Provides more data points for comparison
but may be more sensitive to alteration processes.

## Matching Algorithm

1. **Column Alignment**: Identify ratio columns common to both query and target datasets
2. **NaN Filtering**: Drop samples/columns with missing values
3. **Vectorized Distance Computation**: Calculate Aitchison and Euclidean distances for all targets
   simultaneously using NumPy broadcasting
4. **Ranking**: Sort by Aitchison distance (primary), then Euclidean distance (secondary)
5. **Top-N Selection**: Return the N closest matches
6. **Geodesic Computation**: Calculate geographic distances only for the top-N results
   (lazy evaluation for performance)

## Batch Processing

When processing multiple samples:

- Each sample is matched independently against the full reference database
- Results are aggregated into a frequency table ranking geological sources by how often they appear
  as top matches
- This reveals patterns across collections - frequently matched sources may represent primary
  procurement areas

## Comparison Analysis

When comparing multiple artefacts:

- Top-N matches are computed for each artefact independently
- The intersection of match sets identifies shared geological sources
- Sources appearing in matches for all queried artefacts are highlighted as universal matches
- Shared sources are ranked by match count (descending) and average Aitchison distance (ascending)

## References

- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman and Hall.
- Egozcue, J.J., Pawlowsky-Glahn, V. (2005). Groups of parts and their balances in
  compositional data analysis. *Mathematical Geology*, 37(7), 795-828.
