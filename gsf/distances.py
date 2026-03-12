from __future__ import annotations

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from gsf.constants import EPSILON, GEO_DIST_ROUND_TO_KM, METADATA_COLUMNS


def get_ratio_columns(sample: pd.Series) -> list[str]:
    """Extract ratio column names, excluding metadata and NaN."""
    ratio_data = sample.drop(
        labels=[c for c in METADATA_COLUMNS if c in sample.index],
    )
    return list(ratio_data.dropna().index)


def count_total_ratio_columns(df: pd.DataFrame) -> int:
    """Count total ratio columns (all non-metadata columns)."""
    return len(
        [c for c in df.columns if c not in METADATA_COLUMNS]
    )


def compute_distances_vectorized(
    query_ratios: np.ndarray, target_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Aitchison and Euclidean distances.

    Uses numpy broadcasting between one query vector
    and a matrix of target vectors.

    The Aitchison distance uses the Centered Log-Ratio transform:
        CLR(x) = log(x) - mean(log(x))
        d_A(x, y) = ||CLR(x) - CLR(y)||_2
    """
    q = np.where(
        np.isfinite(query_ratios) & (query_ratios > 0),
        query_ratios, EPSILON,
    )
    t = np.where(
        np.isfinite(target_matrix) & (target_matrix > 0),
        target_matrix, EPSILON,
    )

    log_q = np.log(q)
    clr_q = log_q - np.mean(log_q)

    log_t = np.log(t)
    clr_t = log_t - np.mean(log_t, axis=1, keepdims=True)

    aitchison_dists = np.linalg.norm(clr_t - clr_q, axis=1)
    euclidean_dists = np.linalg.norm(
        target_matrix - query_ratios, axis=1,
    )

    return aitchison_dists, euclidean_dists


def compute_alr5_aitchison_vectorized(
    query_alr: np.ndarray, target_matrix: np.ndarray,
) -> np.ndarray:
    """Compute Aitchison distance from ALR coordinates.

    Formula: d_A^2 = sum(d_i^2) - (sum(d_i))^2 / D
    where d_i = alr_i(x) - alr_i(y),
    D = k + 1 (k ALR coordinates + Zr reference element).

    D is inferred from input shape so partial compositions
    (when some elements are missing) are handled correctly.
    ALR values are already log-ratios; no log transform needed.
    """
    k = query_alr.shape[0]   # number of ALR coordinates
    D = k + 1                # k numerators + Zr denominator
    diffs = target_matrix - query_alr
    sum_sq = np.sum(diffs ** 2, axis=1)
    sq_sum = np.sum(diffs, axis=1) ** 2
    d_a_sq = np.maximum(sum_sq - sq_sum / D, 0.0)
    return np.sqrt(d_a_sq)


def compute_geodesic(
    origin: tuple[float, float] | None,
    destination: tuple[float, float] | None,
) -> str:
    """Compute geodesic distance in km, formatted as a string."""
    if origin is None or destination is None:
        return "Unknown"
    try:
        dist_km = geodesic(origin, destination).km
        rounded = int(
            round(dist_km / GEO_DIST_ROUND_TO_KM)
            * GEO_DIST_ROUND_TO_KM
        )
        return f"{rounded} km"
    except (ValueError, TypeError):
        return "Unknown"
