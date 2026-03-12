"""Confidence scoring for GSF match results.

Quantifies the reliability of each artefact-to-source match as a
percentage (0–100 %) based on three components:

1. Base Distance Score  — exponential decay from Aitchison distance
2. Agreement & Rank Consistency — cross-mode tier and rank agreement
3. Ambiguity / Separation — rank, gap, density, geographic coherence

See the confidence-scoring skill documentation for full details.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gsf.constants import AITCH_STEP, aitch_col

# ---------------------------------------------------------------------------
# Decay constants for base distance score (exponential decay)
# ---------------------------------------------------------------------------
DECAY_CONSTANTS: dict[str, float] = {
    "alr5": 0.7,
    "trace": 0.35,
    "all": 0.23,
}

# "Good" Aitchison thresholds per mode (for tier agreement)
GOOD_THRESHOLDS: dict[str, float] = {
    "alr5": 1.0,
    "trace": 2.0,
    "all": 3.0,
}

# ---------------------------------------------------------------------------
# Mode weight tables for base distance score
# ---------------------------------------------------------------------------
_BASE_WEIGHT_MAP: dict[frozenset[str], dict[str, float]] = {
    frozenset({"alr5", "trace", "all"}): {
        "alr5": 0.50, "trace": 0.30, "all": 0.20,
    },
    frozenset({"alr5", "trace"}): {
        "alr5": 0.60, "trace": 0.40,
    },
    frozenset({"alr5", "all"}): {
        "alr5": 0.65, "all": 0.35,
    },
    frozenset({"trace", "all"}): {
        "trace": 0.60, "all": 0.40,
    },
}

# Ambiguity cross-mode weights (normalised by available modes)
AMBIGUITY_WEIGHTS: dict[str, float] = {
    "alr5": 0.30, "trace": 0.40, "all": 0.30,
}


# ===================================================================
# Component helpers
# ===================================================================

def _base_distance_score(distance: float, mode: str) -> float:
    """Convert Aitchison distance to 0–100 via exponential decay."""
    k = DECAY_CONSTANTS.get(mode, 0.35)
    return 100.0 * math.exp(-k * distance)


def _get_base_weights(
    available: list[str],
) -> dict[str, float]:
    key = frozenset(available)
    if key in _BASE_WEIGHT_MAP:
        return _BASE_WEIGHT_MAP[key]
    if len(available) == 1:
        return {available[0]: 1.0}
    w = 1.0 / len(available)
    return {m: w for m in available}


# --- Agreement --------------------------------------------------------

def _tier_agreement(
    distances: dict[str, float],
) -> float:
    modes = list(distances.keys())
    if len(modes) < 2:
        return 0.0

    all_good = all(
        distances[m] < GOOD_THRESHOLDS.get(m, 2.0)
        for m in modes
    )
    if all_good:
        return 100.0

    all_acceptable = all(
        distances[m] < 2.0 * GOOD_THRESHOLDS.get(m, 2.0)
        for m in modes
    )
    if all_acceptable:
        return 75.0

    # Check for disagreement (one excellent, another poor)
    scores = [
        _base_distance_score(distances[m], m) for m in modes
    ]
    if max(scores) > 70 and min(scores) < 35:
        return 40.0 * 0.5  # penalty

    return 40.0


def _rank_consistency(ranks: dict[str, int]) -> float:
    if len(ranks) < 2:
        return 0.0
    vals = list(ranks.values())
    mean_r = sum(vals) / len(vals)
    max_r = max(vals)
    if mean_r <= 2.0 and max_r <= 3:
        return 100.0
    if mean_r <= 3.0 and max_r <= 5:
        return 90.0
    if mean_r <= 5.0 and max_r <= 10:
        return 75.0
    if mean_r <= 10.0:
        return 55.0
    if mean_r <= 20.0:
        return 35.0
    return 15.0


# --- Ambiguity sub-factors --------------------------------------------

def _rank_score(rank: int) -> float:
    if rank == 1:
        return 100.0
    if rank == 2:
        return 85.0
    if rank == 3:
        return 65.0
    if rank <= 5:
        return 45.0
    if rank <= 10:
        return 25.0
    return 10.0


def _gap_score(d1: float, d2: float) -> float:
    if d1 <= 0:
        return 100.0
    gap = (d2 - d1) / d1
    if gap >= 0.25:
        return 100.0
    if gap >= 0.10:
        return 70.0
    if gap >= 0.03:
        return 40.0
    return 15.0


def _density_score(count: int) -> float:
    if count <= 5:
        return 100.0
    if count <= 10:
        return 80.0
    if count <= 20:
        return 55.0
    if count <= 35:
        return 35.0
    return 15.0


def _geo_coherence_score(
    target_acc: str,
    competitors: list[str],
    metadata_df: pd.DataFrame,
) -> float:
    """Score geographic coherence among nearby competitors."""
    if not competitors:
        return 50.0

    target_row = metadata_df[
        metadata_df["Accession #"] == target_acc
    ]
    if target_row.empty:
        return 50.0

    target_site = str(target_row.iloc[0].get("Site", ""))
    target_region = str(target_row.iloc[0].get("Region", ""))

    scores: list[float] = []
    for comp_acc in competitors:
        comp_row = metadata_df[
            metadata_df["Accession #"] == comp_acc
        ]
        if comp_row.empty:
            scores.append(10.0)
            continue
        comp_site = str(comp_row.iloc[0].get("Site", ""))
        comp_region = str(comp_row.iloc[0].get("Region", ""))
        if comp_site == target_site:
            scores.append(100.0)
        elif comp_region == target_region:
            scores.append(60.0)
        else:
            scores.append(10.0)

    avg = sum(scores) / len(scores) if scores else 50.0
    if avg >= 90:
        return 100.0
    if avg >= 75:
        return 85.0
    if avg >= 60:
        return 70.0
    if avg >= 45:
        return 55.0
    if avg >= 30:
        return 40.0
    return 20.0


def _ambiguity_per_mode(
    target_acc: str,
    mode: str,
    full_dist_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> float:
    """Compute ambiguity score for one mode's distance vector."""
    sorted_df = full_dist_df.sort_values(
        "distance",
    ).reset_index(drop=True)

    # Rank of the target among all candidates
    idx = sorted_df.index[
        sorted_df["Accession #"] == target_acc
    ]
    rank = int(idx[0]) + 1 if len(idx) > 0 else len(sorted_df)

    # Gap between #1 and #2
    if len(sorted_df) >= 2:
        d1 = float(sorted_df.iloc[0]["distance"])
        d2 = float(sorted_df.iloc[1]["distance"])
    else:
        d1, d2 = 0.0, 0.0

    # Density: count within 1 step-size of best match
    step = AITCH_STEP.get(mode, 1.0)
    best_dist = float(sorted_df.iloc[0]["distance"])
    density_count = int(
        (sorted_df["distance"] <= best_dist + step).sum()
    )

    # Competitors within 1 step-size for geographic coherence
    nearby = sorted_df[
        sorted_df["distance"] <= best_dist + step
    ]
    competitors = [
        str(a) for a in nearby["Accession #"].values
        if str(a) != target_acc
    ]

    return (
        0.25 * _rank_score(rank)
        + 0.30 * _gap_score(d1, d2)
        + 0.20 * _density_score(density_count)
        + 0.25 * _geo_coherence_score(
            target_acc, competitors, metadata_df,
        )
    )


# ===================================================================
# Main entry point
# ===================================================================

def compute_row_confidence(
    target_acc: str,
    mode_distances: dict[str, float],
    mode_ranks: dict[str, int],
    full_dist_data: dict[str, pd.DataFrame],
    metadata_df: pd.DataFrame,
) -> int:
    """Compute confidence score for a single result row.

    Returns an integer 0–100.
    """
    available = [
        m for m in mode_distances
        if not math.isnan(mode_distances[m])
    ]
    if not available:
        return 0

    n = len(available)

    # 1. Base Distance Score
    weights = _get_base_weights(available)
    base = sum(
        weights[m] * _base_distance_score(mode_distances[m], m)
        for m in available
    )

    # 2. Agreement & Rank Consistency
    if n >= 2:
        tier = _tier_agreement(
            {m: mode_distances[m] for m in available},
        )
        rank_c = _rank_consistency(
            {m: mode_ranks[m] for m in available
             if m in mode_ranks},
        )
        agreement = 0.5 * tier + 0.5 * rank_c
    else:
        agreement = 0.0

    # 3. Ambiguity / Separation
    amb_scores: dict[str, float] = {}
    for mode in available:
        if mode in full_dist_data:
            amb_scores[mode] = _ambiguity_per_mode(
                target_acc, mode,
                full_dist_data[mode], metadata_df,
            )
    if amb_scores:
        aw = {
            m: AMBIGUITY_WEIGHTS.get(m, 1.0 / 3)
            for m in amb_scores
        }
        total_w = sum(aw.values())
        ambiguity = sum(
            aw[m] * amb_scores[m] / total_w
            for m in amb_scores
        )
    else:
        ambiguity = 25.0

    # Final weighted combination
    if n >= 3:
        conf = (
            0.35 * base + 0.30 * agreement
            + 0.35 * ambiguity
        )
    elif n == 2:
        conf = (
            0.35 * base + 0.25 * agreement
            + 0.40 * ambiguity
        )
    else:
        conf = 0.55 * base + 0.45 * ambiguity

    return max(0, min(100, round(conf)))


def add_confidence_column(
    results_df: pd.DataFrame,
    full_dist_data: dict[str, pd.DataFrame],
    metadata_df: pd.DataFrame,
    enabled_modes: list[str],
    acc_col: str = "Accession #",
) -> pd.DataFrame:
    """Insert a 'Conf' column into *results_df*.

    The column is placed just before 'Geo Dist' if it exists,
    otherwise at the end.
    """
    scores: list[int] = []

    for _, row in results_df.iterrows():
        target_acc = str(row[acc_col])

        mode_distances: dict[str, float] = {}
        mode_ranks: dict[str, int] = {}

        for mode in enabled_modes:
            col = aitch_col(mode)
            if col not in row.index:
                continue
            try:
                d = float(row[col])
            except (ValueError, TypeError):
                continue
            mode_distances[mode] = d

            # Derive rank from full distance data
            if mode in full_dist_data:
                fdd = full_dist_data[mode]
                sorted_accs = (
                    fdd.sort_values("distance")["Accession #"]
                    .astype(str).tolist()
                )
                try:
                    mode_ranks[mode] = (
                        sorted_accs.index(target_acc) + 1
                    )
                except ValueError:
                    mode_ranks[mode] = len(sorted_accs)

        scores.append(compute_row_confidence(
            target_acc, mode_distances, mode_ranks,
            full_dist_data, metadata_df,
        ))

    df = results_df.copy()
    cols = list(df.columns)
    insert_pos = (
        cols.index("Geo Dist")
        if "Geo Dist" in cols
        else len(cols)
    )
    df.insert(insert_pos, "Conf", scores)
    return df
