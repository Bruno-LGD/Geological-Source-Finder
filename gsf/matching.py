from __future__ import annotations

import numpy as np
import pandas as pd

from gsf.confidence import add_confidence_column
from gsf.constants import DEFAULT_TOP_N, EPSILON, aitch_col
from gsf.distances import (
    compute_alr5_aitchison_vectorized,
    compute_distances_vectorized,
    compute_geodesic,
    get_ratio_columns,
)


def get_top_matches(
    query_sample: pd.Series,
    target_df: pd.DataFrame,
    query_coords_df: pd.DataFrame,
    target_coords_df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    direction: str = "artefact_to_geology",
    mode_key: str = "trace",
) -> pd.DataFrame:
    """Find the top-N matching samples by compositional similarity.

    Computes Aitchison and Euclidean distances (vectorized),
    sorts by [Aitchison, Euclidean], takes top_n, then computes
    geodesic distances only for those results.
    """
    ratio_cols = get_ratio_columns(query_sample)
    if not ratio_cols:
        return pd.DataFrame()

    if direction == "artefact_to_geology":
        keep_cols = [
            "Lithology", "Accession #", "Site", "Region",
        ]
    else:
        keep_cols = ["Accession #", "Site", "Region"]

    available_cols = [
        c for c in keep_cols + ratio_cols
        if c in target_df.columns
    ]
    aligned_df = target_df[available_cols].dropna(  # pyright: ignore[reportCallIssue]
        subset=ratio_cols,
    )

    if aligned_df.empty:
        return pd.DataFrame()

    query_series = pd.Series(pd.to_numeric(
        query_sample[ratio_cols], errors="coerce",
    )).fillna(EPSILON)
    query_vector = np.asarray(query_series, dtype=float)
    target_matrix = np.asarray(
        aligned_df[ratio_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(EPSILON),
        dtype=float,
    )

    results_df = aligned_df[keep_cols].copy()
    results_df = results_df.reset_index(drop=True)

    if mode_key == "alr5":
        aitch_dists = compute_alr5_aitchison_vectorized(
            query_vector, target_matrix,
        )
        results_df["Aitch Dist"] = np.round(aitch_dists, 2)
        results_df = results_df.sort_values(
            by=["Aitch Dist"],
        ).head(top_n)
    else:
        aitch_dists, eucl_dists = (
            compute_distances_vectorized(
                query_vector, target_matrix,
            )
        )
        results_df["Aitch Dist"] = np.round(aitch_dists, 2)
        results_df["Eucl Dist"] = np.round(eucl_dists, 2)
        results_df = results_df.sort_values(  # pyright: ignore[reportCallIssue]
            by=["Aitch Dist", "Eucl Dist"],
        ).head(top_n)
    results_df.reset_index(drop=True, inplace=True)
    results_df.index += 1

    # Compute geodesic distances only for top_n results
    if direction == "artefact_to_geology":
        site_match = query_coords_df[
            query_coords_df["Site"] == query_sample["Site"]
        ]
        query_coords: tuple[float, float] | None = (
            (
                float(site_match.iloc[0]["Latitude"]),
                float(site_match.iloc[0]["Longitude"]),
            )
            if not site_match.empty
            else None
        )
        target_coord_lookup: dict[
            str, tuple[float, float]
        ] = {}
        for _, row in target_coords_df.iterrows():
            target_coord_lookup[str(row["Accession #"])] = (
                float(row["Latitude"]),
                float(row["Longitude"]),
            )
        results_df["Geo Dist"] = (
            results_df["Accession #"].apply(
                lambda acc: compute_geodesic(
                    query_coords,
                    target_coord_lookup.get(str(acc)),
                )
            )
        )
    else:
        acc_match = query_coords_df[
            query_coords_df["Accession #"]
            == str(query_sample["Accession #"])
        ]
        query_coords: tuple[float, float] | None = (
            (
                float(acc_match.iloc[0]["Latitude"]),
                float(acc_match.iloc[0]["Longitude"]),
            )
            if not acc_match.empty
            else None
        )
        target_coord_lookup = {}
        for _, row in target_coords_df.iterrows():
            target_coord_lookup[str(row["Site"])] = (
                float(row["Latitude"]),
                float(row["Longitude"]),
            )
        results_df["Geo Dist"] = results_df["Site"].apply(
            lambda site: compute_geodesic(
                query_coords,
                target_coord_lookup.get(site),
            )
        )

    if direction == "artefact_to_geology":
        results_df = results_df.rename(columns={
            "Accession #": "Geo Acc #",
            "Site": "Geo Site",
        })
    else:
        results_df = results_df.rename(columns={
            "Accession #": "Artefact Acc #",
            "Site": "Artefact Site",
            "Region": "Artefact Region",
        })

    return results_df


def get_top_matches_multimode(
    query_samples: dict[str, pd.Series],
    target_dfs: dict[str, pd.DataFrame],
    query_coords_df: pd.DataFrame,
    target_coords_df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    direction: str = "artefact_to_geology",
    enabled_modes: list[str] | None = None,
    sort_mode: str = "alr5",
) -> pd.DataFrame:
    """Find top-N matches using multiple distance modes.

    Computes Aitchison distance for each enabled mode,
    merges by Accession #, sorts by sort_mode's distance.
    """
    if enabled_modes is None:
        enabled_modes = list(query_samples.keys())

    if direction == "artefact_to_geology":
        keep_cols = [
            "Lithology", "Accession #", "Site", "Region",
        ]
    else:
        keep_cols = ["Accession #", "Site", "Region"]

    # Build base metadata from sort_mode's target_df
    base_target = target_dfs[sort_mode]
    available_keep = [
        c for c in keep_cols if c in base_target.columns
    ]
    base_df = (
        base_target[available_keep]
        .drop_duplicates(subset=["Accession #"])
        .copy()
        .reset_index(drop=True)
    )

    # Compute Aitchison distances for each mode
    # Also store full (unrounded) distances for confidence scoring
    full_dist_data: dict[str, pd.DataFrame] = {}

    for mode in enabled_modes:
        query_sample = query_samples[mode]
        target_df = target_dfs[mode]
        col_name = aitch_col(mode)

        ratio_cols = get_ratio_columns(query_sample)
        if not ratio_cols:
            base_df[col_name] = np.nan
            continue

        avail_cols = [
            c for c in ["Accession #"] + ratio_cols
            if c in target_df.columns
        ]
        aligned_df = target_df[avail_cols].dropna(
            subset=ratio_cols,
        )

        if aligned_df.empty:
            base_df[col_name] = np.nan
            continue

        query_series = pd.Series(pd.to_numeric(
            query_sample[ratio_cols], errors="coerce",
        )).fillna(EPSILON)
        query_vector = np.asarray(query_series, dtype=float)
        target_matrix = np.asarray(
            aligned_df[ratio_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(EPSILON),
            dtype=float,
        )

        if mode == "alr5":
            dists = compute_alr5_aitchison_vectorized(
                query_vector, target_matrix,
            )
        else:
            dists, _ = compute_distances_vectorized(
                query_vector, target_matrix,
            )

        # Save full distances for confidence scoring
        full_dist_data[mode] = pd.DataFrame({
            "Accession #": (
                aligned_df["Accession #"].astype(str).values
            ),
            "distance": dists,
        })

        dist_df = pd.DataFrame({
            "Accession #": aligned_df["Accession #"].values,
            col_name: np.round(dists, 2),
        })
        base_df = base_df.merge(
            dist_df, on="Accession #", how="left",
        )

    # Sort by primary mode's distance
    sort_col = aitch_col(sort_mode)
    base_df = base_df.dropna(subset=[sort_col])
    base_df = base_df.sort_values(
        by=[sort_col],
    ).head(top_n)
    base_df.reset_index(drop=True, inplace=True)
    base_df.index += 1

    # Compute geodesic distances for top results (before confidence,
    # so the proximity component can use Geo Dist)
    if direction == "artefact_to_geology":
        # Use any mode's query_sample for site lookup
        q_sample = query_samples[sort_mode]
        site_match = query_coords_df[
            query_coords_df["Site"] == q_sample["Site"]
        ]
        query_coords: tuple[float, float] | None = (
            (
                float(site_match.iloc[0]["Latitude"]),
                float(site_match.iloc[0]["Longitude"]),
            )
            if not site_match.empty
            else None
        )
        target_coord_lookup: dict[
            str, tuple[float, float]
        ] = {}
        for _, row in target_coords_df.iterrows():
            target_coord_lookup[str(row["Accession #"])] = (
                float(row["Latitude"]),
                float(row["Longitude"]),
            )
        base_df["Geo Dist"] = (
            base_df["Accession #"].apply(
                lambda acc: compute_geodesic(
                    query_coords,
                    target_coord_lookup.get(str(acc)),
                )
            )
        )
    else:
        q_sample = query_samples[sort_mode]
        acc_match = query_coords_df[
            query_coords_df["Accession #"]
            == str(q_sample["Accession #"])
        ]
        query_coords = (
            (
                float(acc_match.iloc[0]["Latitude"]),
                float(acc_match.iloc[0]["Longitude"]),
            )
            if not acc_match.empty
            else None
        )
        target_coord_lookup = {}
        for _, row in target_coords_df.iterrows():
            target_coord_lookup[str(row["Site"])] = (
                float(row["Latitude"]),
                float(row["Longitude"]),
            )
        base_df["Geo Dist"] = base_df["Site"].apply(
            lambda site: compute_geodesic(
                query_coords,
                target_coord_lookup.get(site),
            )
        )

    # Add confidence column (after Geo Dist, so proximity can use it)
    metadata_cols = ["Accession #", "Site", "Region"]
    avail_meta = [
        c for c in metadata_cols
        if c in base_target.columns
    ]
    metadata_df = (
        base_target[avail_meta]
        .drop_duplicates(subset=["Accession #"])
        .copy()
    )
    metadata_df["Accession #"] = (
        metadata_df["Accession #"].astype(str)
    )
    base_df = add_confidence_column(
        base_df, full_dist_data, metadata_df,
        enabled_modes, acc_col="Accession #",
        geo_dist_col="Geo Dist",
    )

    if direction == "artefact_to_geology":
        base_df = base_df.rename(columns={
            "Accession #": "Geo Acc #",
            "Site": "Geo Site",
        })
    else:
        base_df = base_df.rename(columns={
            "Accession #": "Artefact Acc #",
            "Site": "Artefact Site",
            "Region": "Artefact Region",
        })

    return base_df
