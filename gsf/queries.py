from __future__ import annotations

from collections import Counter
from functools import partial
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from gsf.constants import (
    DEFAULT_TOP_N,
    MAX_TOP_N,
    MODE_LABELS,
    aitch_col,
)
from gsf.distances import count_total_ratio_columns, get_ratio_columns
from gsf.matching import get_top_matches_multimode
from gsf.photos import get_artefact_images, prefetch_artefact_images
from gsf.styling import (
    color_aitch_with_thresholds,
    color_aitch_alr5_gradient,
)
from gsf.viz import (
    display_artefact_photos,
    display_legend,
    display_radar_chart,
    display_results_map,
    display_results_table,
    display_scatter_plot,
)


def _get_query_coords(
    query_sample: pd.Series,
    coords_df: pd.DataFrame,
    direction: str,
) -> tuple[float, float] | None:
    """Look up the coordinates for a query sample."""
    if direction == "artefact_to_geology":
        match = coords_df[
            coords_df["Site"] == query_sample["Site"]
        ]
    else:
        match = coords_df[
            coords_df["Accession #"].astype(str)
            == str(query_sample["Accession #"])
        ]
    if not match.empty:
        return (
            float(match.iloc[0]["Latitude"]),
            float(match.iloc[0]["Longitude"]),
        )
    return None


def _resolve_sample(
    accession_number: str,
    source_df: pd.DataFrame,
    label_prefix: str,
    key_suffix: str = "",
) -> pd.Series | None:
    """Find a sample by accession number."""
    matches = source_df[
        source_df["Accession #"] == accession_number
    ]
    if matches.empty:
        st.error(
            f"No {label_prefix} found with "
            f"Accession # {accession_number}"
        )
        return None
    if len(matches) == 1:
        return matches.iloc[0]
    st.info(
        f"Multiple {label_prefix}s found for "
        f"Accession # {accession_number}. "
        "Please select a site:"
    )
    sites = list(matches["Site"].unique())  # pyright: ignore[reportAttributeAccessIssue]
    selected_site = str(st.selectbox(
        "Select a site",
        sites,
        key=f"site_select_{key_suffix}",
    ))
    site_mask = matches["Site"].str.lower() == selected_site.lower()  # pyright: ignore[reportAttributeAccessIssue]
    return matches[site_mask].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]


def execute_query(
    accession_number: str,
    query_mode: str,
    top_n: int,
    selected_regions: list[str],
    mode_datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    geo_coords_df: pd.DataFrame,
    arch_coords_df: pd.DataFrame,
    enabled_modes: list[str],
    sort_mode: str = "alr5",
    photo_df: pd.DataFrame | None = None,
    photos_base_dir: str = "",
    access_token: str = "",
    share_url: str = "",
) -> None:
    """Execute a single query and display results."""
    if query_mode == "Artefact \u2192 Geology":
        direction = "artefact_to_geology"
        query_coords_df = arch_coords_df
        target_coords_df = geo_coords_df
        label_prefix = "artefact"
        region_col = "Region"
    else:
        direction = "geology_to_artefact"
        query_coords_df = geo_coords_df
        target_coords_df = arch_coords_df
        label_prefix = "geology sample"
        region_col = "Artefact Region"

    # Resolve query sample in sort mode for header info
    sort_art_df, sort_geo_df = mode_datasets[sort_mode]
    if direction == "artefact_to_geology":
        source_df = sort_art_df
    else:
        source_df = sort_geo_df

    sample = _resolve_sample(
        accession_number, source_df,
        label_prefix, key_suffix="single",
    )
    if sample is None:
        return

    # Start photo downloads early (runs in background)
    if (
        direction == "artefact_to_geology"
        and photo_df is not None
        and not photo_df.empty
        and (photos_base_dir or (access_token and share_url))
    ):
        prefetch_artefact_images(
            str(sample["Accession #"]),
            photo_df, photos_base_dir,
            access_token=access_token,
            share_url=share_url,
        )

    # Build header
    if "Region" in sample.index and bool(
        pd.notnull(sample.get("Region"))
    ):
        site_info = (
            f"{sample['Site']} ({sample['Region']})"
        )
    else:
        site_info = str(sample.get("Site", "Unknown Site"))
    st.subheader(
        f"Results for Accession # "
        f"{sample['Accession #']} - {site_info}"
    )

    # Data quality indicator for sort mode
    total_ratios = count_total_ratio_columns(source_df)
    valid_ratios = len(get_ratio_columns(sample))
    if valid_ratios < total_ratios:
        st.warning(
            f"Data Quality ({MODE_LABELS[sort_mode]}): "
            f"{valid_ratios}/{total_ratios}"
            " ratio columns available. "
            f"Missing {total_ratios - valid_ratios} "
            "values."
        )
    else:
        st.info(
            f"Data Quality ({MODE_LABELS[sort_mode]}): "
            f"{valid_ratios}/{total_ratios}"
            " ratio columns available (complete)."
        )

    # Resolve query samples and target dfs for all modes
    query_samples: dict[str, pd.Series] = {}
    target_dfs: dict[str, pd.DataFrame] = {}
    for mode in enabled_modes:
        art_df, geo_df = mode_datasets[mode]
        if direction == "artefact_to_geology":
            src, tgt = art_df, geo_df
        else:
            src, tgt = geo_df, art_df
        matches = src[
            src["Accession #"] == accession_number
        ]
        if not matches.empty:
            query_samples[mode] = matches.iloc[0]
            target_dfs[mode] = tgt

    if sort_mode not in query_samples:
        st.error("Query sample not found in sort mode data.")
        return

    # Get top matches across all modes (cached per query)
    dist_cache_key = (
        f"_dist_cache_{accession_number}_{direction}"
        f"_{sort_mode}_{top_n}"
        f"_{'_'.join(sorted(enabled_modes))}"
    )
    cached_results = st.session_state.get(dist_cache_key)
    if cached_results is not None:
        results_df = cached_results
    else:
        results_df = get_top_matches_multimode(
            query_samples, target_dfs,
            query_coords_df, target_coords_df,
            top_n, direction,
            enabled_modes=enabled_modes,
            sort_mode=sort_mode,
        )
        st.session_state[dist_cache_key] = results_df

    # Apply region filter
    if (
        selected_regions
        and not results_df.empty
        and region_col in results_df.columns
    ):
        results_df = pd.DataFrame(
            results_df[
                results_df[region_col].isin(selected_regions)
            ]
        )

    if results_df.empty:
        st.warning(
            "No matching results found. "
            "Try adjusting the region filter "
            "or increasing the number of results."
        )
        return

    # Display results
    with st.expander(
        "Distance Interpretation Guide", expanded=False,
    ):
        display_legend(enabled_modes=enabled_modes)

    display_results_table(
        results_df, accession_number,
        direction, query_site=site_info,
        enabled_modes=enabled_modes,
    )

    # Display artefact photographs (if any source configured)
    if (
        (photos_base_dir or (access_token and share_url))
        and photo_df is not None
        and not photo_df.empty
        and direction == "artefact_to_geology"
    ):
        display_artefact_photos(
            str(sample["Accession #"]),
            photo_df,
            photos_base_dir,
            access_token=access_token,
            share_url=share_url,
        )

    # Radar chart uses sort mode's data
    sort_target_df = target_dfs[sort_mode]
    display_radar_chart(
        query_samples[sort_mode], results_df,
        sort_target_df,
        accession_number, direction,
        mode_key=sort_mode,
    )
    display_scatter_plot(
        results_df, sort_mode=sort_mode,
    )

    query_coords = _get_query_coords(
        sample, query_coords_df, direction,
    )
    query_label = (
        f"Query: {sample['Accession #']} "
        f"- {sample['Site']}"
    )
    display_results_map(
        results_df, query_coords,
        target_coords_df, query_label, direction,
        sort_mode=sort_mode,
    )


def execute_batch_query(
    mode_datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    geo_coords_df: pd.DataFrame,
    arch_coords_df: pd.DataFrame,
    enabled_modes: list[str],
    sort_mode: str = "alr5",
) -> None:
    """Batch query: upload CSV, run matching for each."""
    uploaded_file = st.file_uploader(
        "Upload a CSV with accession numbers",
        type=["csv"],
        key="batch_upload",
    )
    if uploaded_file is None:
        st.info(
            "Upload a CSV file containing a column "
            "of accession numbers."
        )
        return

    input_df = pd.read_csv(uploaded_file)
    if input_df.empty:
        st.error("The uploaded CSV is empty.")
        return

    acc_column = st.selectbox(
        "Select the column containing accession numbers:",
        input_df.columns,
        key="batch_col",
    )
    query_mode = st.radio("Query Direction:", [
        "Artefact \u2192 Geology",
        "Geology \u2192 Artefact",
    ], key="batch_mode")
    top_n = st.number_input(
        "Top N per query:",
        min_value=1, max_value=MAX_TOP_N,
        value=DEFAULT_TOP_N, step=1,
        key="batch_topn",
    )

    if not st.button("Run Batch", key="batch_run"):
        return

    if query_mode == "Artefact \u2192 Geology":
        direction = "artefact_to_geology"
        query_coords_df_b = arch_coords_df
        target_coords_df = geo_coords_df
    else:
        direction = "geology_to_artefact"
        query_coords_df_b = geo_coords_df
        target_coords_df = arch_coords_df

    # Use sort mode source_df for accession lookup
    sort_art_df, sort_geo_df = mode_datasets[sort_mode]
    source_df = (
        sort_art_df
        if direction == "artefact_to_geology"
        else sort_geo_df
    )

    accession_numbers = (
        input_df[acc_column]
        .astype(str).str.strip()
        .dropna().unique().tolist()
    )
    if not accession_numbers:
        st.error(
            "No valid accession numbers found "
            "in the selected column."
        )
        return

    all_results: list[pd.DataFrame] = []
    errors: list[str] = []
    progress = st.progress(0, text="Processing batch...")
    total = len(accession_numbers)

    for i, acc in enumerate(accession_numbers):
        progress.progress(
            (i + 1) / total,
            text=f"Processing {acc} ({i + 1}/{total})",
        )
        matches = source_df[
            source_df["Accession #"] == acc
        ]
        if matches.empty:
            errors.append(acc)
            continue

        # Resolve per-mode query samples
        query_samples: dict[str, pd.Series] = {}
        target_dfs: dict[str, pd.DataFrame] = {}
        for mode in enabled_modes:
            art_df, geo_df = mode_datasets[mode]
            if direction == "artefact_to_geology":
                src, tgt = art_df, geo_df
            else:
                src, tgt = geo_df, art_df
            m = src[src["Accession #"] == acc]
            if not m.empty:
                query_samples[mode] = m.iloc[0]
                target_dfs[mode] = tgt

        if sort_mode not in query_samples:
            errors.append(acc)
            continue

        result = get_top_matches_multimode(
            query_samples, target_dfs,
            query_coords_df_b, target_coords_df,
            top_n, direction,
            enabled_modes=enabled_modes,
            sort_mode=sort_mode,
        )
        if not result.empty:
            result.insert(0, "Query Acc #", acc)
            all_results.append(result)

    progress.empty()

    if errors:
        st.warning(
            f"{len(errors)} accession number(s) not found: "
            f"{', '.join(errors[:20])}"
        )

    if not all_results:
        st.error(
            "No results found for any accession number "
            "in the batch."
        )
        return

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.index += 1

    sort_col = aitch_col(sort_mode)

    # Frequency summary
    acc_col = (
        "Geo Acc #"
        if direction == "artefact_to_geology"
        else "Artefact Acc #"
    )
    site_col = (
        "Geo Site"
        if direction == "artefact_to_geology"
        else "Artefact Site"
    )

    st.subheader("Source Frequency Summary")
    st.markdown(
        "Geological sources ranked by how often they "
        "appear across all queried samples:"
    )

    freq_df = (
        combined_df.groupby([acc_col, site_col])
        .agg(
            Count=("Query Acc #", "nunique"),
            Avg_Aitch=(sort_col, "mean"),
            Min_Aitch=(sort_col, "min"),
        )
        .sort_values(  # pyright: ignore[reportCallIssue]
            ["Count", "Avg_Aitch"],
            ascending=[False, True],
        )
        .reset_index()
    )
    freq_df["Avg_Aitch"] = np.round(
        freq_df["Avg_Aitch"], 2,
    )
    freq_df["Min_Aitch"] = np.round(
        freq_df["Min_Aitch"], 2,
    )
    styled_freq = freq_df.style.format(
        {"Avg_Aitch": "{:.2f}", "Min_Aitch": "{:.2f}"},
    )
    if sort_mode == "alr5":
        styled_freq = styled_freq.map(
            color_aitch_alr5_gradient,
            subset=["Avg_Aitch", "Min_Aitch"],
        )
    else:
        aitch_styler = partial(
            color_aitch_with_thresholds,
            mode_key=sort_mode,
        )
        styled_freq = styled_freq.map(
            aitch_styler,
            subset=["Avg_Aitch", "Min_Aitch"],
        )
    st.dataframe(styled_freq, use_container_width=True)

    # Combined results
    with st.expander("All Individual Results", expanded=False):
        st.dataframe(combined_df, use_container_width=True)

    # Downloads
    base_filename = (
        f"GSF_batch_{len(accession_numbers)}queries"
    )
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download combined results as CSV",
            combined_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base_filename}.csv",
            mime="text/csv",
            key="batch_csv",
        )
    with dl2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:  # pyright: ignore[reportArgumentType]
            freq_df.to_excel(
                writer,
                sheet_name="Frequency Summary",
                index=False,
            )
            combined_df.to_excel(
                writer,
                sheet_name="All Results",
                index=False,
            )
        xl_mime = (
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        )
        st.download_button(
            "Download as Excel",
            output.getvalue(),
            file_name=f"{base_filename}.xlsx",
            mime=xl_mime,
            key="batch_xlsx",
        )

    n_success = len(accession_numbers) - len(errors)
    st.success(
        f"Batch complete: {n_success} successful queries, "
        f"{len(all_results)} results total."
    )


def execute_comparison(
    mode_datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    geo_coords_df: pd.DataFrame,
    arch_coords_df: pd.DataFrame,
    enabled_modes: list[str],
    sort_mode: str = "alr5",
    photo_df: pd.DataFrame | None = None,
    photos_base_dir: str = "",
    access_token: str = "",
    share_url: str = "",
) -> None:
    """Enter 2+ accession numbers to find shared sources."""
    st.markdown(
        "Enter 2 or more accession numbers to find "
        "**shared geological sources** that match "
        "multiple artefacts."
    )

    query_mode = st.radio("Query Direction:", [
        "Artefact \u2192 Geology",
        "Geology \u2192 Artefact",
    ], key="comp_mode")
    top_n = st.number_input(
        "Top N per sample:",
        min_value=1, max_value=MAX_TOP_N,
        value=DEFAULT_TOP_N, step=1,
        key="comp_topn",
    )

    acc_text = st.text_area(
        "Enter accession numbers "
        "(comma or newline separated):",
        height=100,
        key="comp_input",
        placeholder="e.g. 3003, 3004, 3005",
    )

    if not st.button("Compare", key="comp_run"):
        return

    # Parse accession numbers
    raw = acc_text.replace(",", "\n").strip()
    accession_numbers = [
        a.strip() for a in raw.split("\n") if a.strip()
    ]

    if len(accession_numbers) < 2:
        st.warning(
            "Please enter at least 2 accession numbers "
            "to compare."
        )
        return

    if query_mode == "Artefact \u2192 Geology":
        direction = "artefact_to_geology"
        query_coords_df_c = arch_coords_df
        target_coords_df = geo_coords_df
    else:
        direction = "geology_to_artefact"
        query_coords_df_c = geo_coords_df
        target_coords_df = arch_coords_df

    sort_col = aitch_col(sort_mode)

    # Use sort mode for source lookup
    sort_art_df, sort_geo_df = mode_datasets[sort_mode]
    source_df = (
        sort_art_df
        if direction == "artefact_to_geology"
        else sort_geo_df
    )

    acc_col = (
        "Geo Acc #"
        if direction == "artefact_to_geology"
        else "Artefact Acc #"
    )
    site_col = (
        "Geo Site"
        if direction == "artefact_to_geology"
        else "Artefact Site"
    )

    # Run queries
    all_results: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for acc in accession_numbers:
        matches = source_df[
            source_df["Accession #"] == acc
        ]
        if matches.empty:
            errors.append(acc)
            continue

        # Resolve per-mode query samples
        query_samples: dict[str, pd.Series] = {}
        target_dfs: dict[str, pd.DataFrame] = {}
        for mode in enabled_modes:
            art_df, geo_df = mode_datasets[mode]
            if direction == "artefact_to_geology":
                src, tgt = art_df, geo_df
            else:
                src, tgt = geo_df, art_df
            m = src[src["Accession #"] == acc]
            if not m.empty:
                query_samples[mode] = m.iloc[0]
                target_dfs[mode] = tgt

        if sort_mode not in query_samples:
            errors.append(acc)
            continue

        result = get_top_matches_multimode(
            query_samples, target_dfs,
            query_coords_df_c, target_coords_df,
            top_n, direction,
            enabled_modes=enabled_modes,
            sort_mode=sort_mode,
        )
        if not result.empty:
            all_results[acc] = result

    if errors:
        st.warning(f"Not found: {', '.join(errors)}")

    if len(all_results) < 2:
        st.error(
            "Need results from at least 2 accession "
            "numbers to compare."
        )
        return

    # Display photos of compared artefacts
    if (
        (photos_base_dir or (access_token and share_url))
        and photo_df is not None
        and not photo_df.empty
        and direction == "artefact_to_geology"
    ):
        photo_cols = st.columns(
            min(len(all_results), 4)
        )
        for i, acc in enumerate(all_results):
            with photo_cols[i % min(len(all_results), 4)]:
                images = get_artefact_images(
                    acc, photo_df, photos_base_dir,
                    access_token, share_url,
                )
                if images:
                    try:
                        st.image(
                            images[0],
                            caption=f"Acc # {acc}",
                            use_container_width=True,
                        )
                    except Exception:
                        st.caption(f"Acc # {acc}")

    # Find shared sources (using sort mode's distance)
    source_counter: Counter[tuple[str, str]] = Counter()
    source_aitch: dict[
        tuple[str, str], dict[str, float]
    ] = {}

    for query_acc, result_df in all_results.items():
        for _, row in result_df.iterrows():
            source_key = (
                str(row[acc_col]), str(row[site_col]),
            )
            source_counter[source_key] += 1
            if source_key not in source_aitch:
                source_aitch[source_key] = {}
            source_aitch[source_key][query_acc] = float(
                row[sort_col]
            )

    # Build comparison table
    shared_sources = {
        k: v for k, v in source_counter.items() if v >= 2
    }

    if not shared_sources:
        st.info(
            "No shared geological sources found across "
            "the queried samples in their top-N results. "
            "Try increasing the number of results."
        )
        # Still show individual results
        for acc, result_df in all_results.items():
            with st.expander(
                f"Results for {acc}", expanded=False,
            ):
                st.dataframe(
                    result_df, use_container_width=True,
                )
        return

    st.subheader(
        f"Shared Sources ({len(shared_sources)} found)"
    )
    st.markdown(
        "Sources appearing in the top-N results of "
        "2 or more queried samples:"
    )

    query_accs = list(all_results.keys())
    comp_rows: list[dict[str, Any]] = []
    for (src_acc, src_site), count in sorted(
        shared_sources.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        row_data: dict[str, Any] = {
            acc_col: src_acc,
            site_col: src_site,
            "Shared Count": count,
        }
        aitch_values = source_aitch.get(
            (src_acc, src_site), {},
        )
        for q_acc in query_accs:
            row_data[f"Aitch ({q_acc})"] = (
                aitch_values.get(q_acc, None)
            )
        # Add lithology if available
        if direction == "artefact_to_geology":
            for result_df in all_results.values():
                lit_row = result_df[
                    result_df[acc_col] == src_acc
                ]
                if (
                    not lit_row.empty
                    and "Lithology" in lit_row.columns
                ):
                    row_data["Lithology"] = (
                        lit_row.iloc[0]["Lithology"]
                    )
                    break
        comp_rows.append(row_data)

    comp_df = pd.DataFrame(comp_rows)

    # Sort by shared count desc, then average Aitchison dist
    comp_aitch_cols = [
        c for c in comp_df.columns
        if c.startswith("Aitch (")
    ]
    if comp_aitch_cols:
        comp_df["Avg Aitch"] = np.round(
            comp_df[comp_aitch_cols].mean(axis=1), 2,
        )
        comp_df = comp_df.sort_values(  # pyright: ignore[reportCallIssue]
            ["Shared Count", "Avg Aitch"],
            ascending=[False, True],
        )
    comp_df.reset_index(drop=True, inplace=True)
    comp_df.index += 1

    # Highlight rows matching ALL queried samples
    def _highlight_shared(
        row: pd.Series,
    ) -> list[str]:
        if row["Shared Count"] == len(all_results):
            return [
                "background-color: #D4EDDA"
            ] * len(row)
        return [""] * len(row)

    styled = comp_df.style.apply(
        _highlight_shared, axis=1,
    )
    avg_cols = (
        ["Avg Aitch"]
        if "Avg Aitch" in comp_df.columns
        else []
    )
    for col in comp_aitch_cols + avg_cols:
        if sort_mode == "alr5":
            styled = styled.map(
                color_aitch_alr5_gradient, subset=[col],
            )
        else:
            aitch_color_fn = partial(
                color_aitch_with_thresholds,
                mode_key=sort_mode,
            )
            styled = styled.map(
                aitch_color_fn, subset=[col],
            )

    st.table(styled)
    st.caption(
        "Green rows = sources matching ALL queried samples."
    )

    # Individual results in expanders
    for acc, result_df in all_results.items():
        with st.expander(
            f"Full results for {acc}", expanded=False,
        ):
            st.dataframe(
                result_df, use_container_width=True,
            )

    # Download
    csv_data = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download comparison as CSV",
        csv_data,
        file_name=(
            f"GSF_comparison_{len(query_accs)}samples.csv"
        ),
        mime="text/csv",
        key="comp_csv",
    )
