from __future__ import annotations

import logging
import os
import time

import pandas as pd
import streamlit as st

from gsf.auth import (
    get_token_cache_path,
    get_valid_access_token,
    poll_for_token,
    save_token_cache,
    start_device_code_flow,
)
from gsf.config import load_config, save_config
from gsf.constants import (
    DEFAULT_TOP_N,
    MAX_TOP_N,
    MODE_LABELS,
)
from gsf.data import load_data, load_photo_mapping
from gsf.queries import (
    execute_batch_query,
    execute_comparison,
    execute_query,
)

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Application entry point."""
    # Render sidebar to get settings (element mode)
    with st.sidebar:
        st.header("About")
        st.markdown(
            "**GSF - Geology Source Finder** matches "
            "archaeological artefacts to potential "
            "geological sources using trace element "
            "ratio comparison."
        )
        st.divider()
        st.header("Element Modes")
        use_alr5 = st.checkbox(
            "ALR-5 (5 log-ratios)",
            value=True,
            key="mode_alr5",
            help=(
                "5 additive log-ratio coordinates "
                "from 6 key elements "
                "(Ni, Cr, Y, Nb, Sr, Zr)."
            ),
        )
        use_trace = st.checkbox(
            "Trace Elements (16 ratios)",
            value=True,
            key="mode_trace",
            help=(
                "16 trace-element ratios."
            ),
        )
        use_all = st.checkbox(
            "All Elements (22 ratios)",
            value=True,
            key="mode_all",
            help=(
                "22 ratios including 6 major-element "
                "ratios."
            ),
        )
        enabled_modes: list[str] = []
        if use_alr5:
            enabled_modes.append("alr5")
        if use_trace:
            enabled_modes.append("trace")
        if use_all:
            enabled_modes.append("all")
        if not enabled_modes:
            st.error(
                "Enable at least one element mode."
            )
            return
        sort_mode = enabled_modes[0]
        mode_labels_str = ", ".join(
            MODE_LABELS[m] for m in enabled_modes
        )
        st.caption(
            f"**Active:** {mode_labels_str} · "
            f"Sorted by {MODE_LABELS[sort_mode]}"
        )
        st.divider()
        saved_config = load_config()

        # --- Local photos folder ---
        saved_dir = (
            saved_config.get("photos_base_dir", "") or ""
        )
        photos_base_dir = st.text_input(
            "Local Photos Folder:",
            value=saved_dir,
            help=(
                "Local path to the folder containing "
                "artefact photographs (e.g. a OneDrive "
                "sync folder)."
            ),
            key="photos_dir",
        ) or ""
        if photos_base_dir != saved_dir:
            save_config(
                {**saved_config, "photos_base_dir": photos_base_dir}
            )

        # --- OneDrive (Graph API) ---
        st.markdown("---")
        st.markdown("**OneDrive (online)**")

        saved_share = (
            saved_config.get("onedrive_folder_url", "")
            or ""
        )
        share_url = st.text_input(
            "OneDrive Shared Folder URL:",
            value=saved_share,
            help=(
                "The 1drv.ms sharing link to the root "
                "photos folder in OneDrive."
            ),
            key="onedrive_url",
        ) or ""
        if share_url != saved_share:
            save_config(
                {**saved_config, "onedrive_folder_url": share_url}
            )

        saved_client = (
            saved_config.get("azure_client_id", "") or ""
        )
        client_id = st.text_input(
            "Azure App Client ID:",
            value=saved_client,
            help=(
                "Register a free Azure AD app at "
                "portal.azure.com, enable public client "
                "flows, add Files.Read permission, then "
                "paste the Application (client) ID here."
            ),
            key="azure_client_id",
            type="password",
        ) or ""
        if client_id != saved_client:
            save_config(
                {**saved_config, "azure_client_id": client_id}
            )

        # Auth status and sign-in
        access_token = ""
        if client_id and share_url:
            access_token = (
                get_valid_access_token(client_id) or ""
            )
            if access_token:
                st.success("OneDrive: signed in")
                if st.button(
                    "Sign out", key="graph_signout",
                ):
                    st.session_state.pop(
                        "graph_token_data", None,
                    )
                    path = get_token_cache_path()
                    if os.path.exists(path):
                        os.remove(path)
                    st.rerun()
            else:
                dc = st.session_state.get("device_code_flow")
                if dc:
                    st.info(
                        f"Go to **{dc['verification_uri']}** "
                        f"and enter code: "
                        f"**{dc['user_code']}**"
                    )
                    if time.time() > dc.get("expires_at", 0):
                        st.error("Code expired. Try again.")
                        st.session_state.pop(
                            "device_code_flow", None,
                        )
                    elif st.button(
                        "Check sign-in status",
                        key="graph_poll",
                    ):
                        try:
                            result = poll_for_token(
                                client_id, dc["device_code"],
                            )
                        except ValueError as exc:
                            st.error(str(exc))
                            st.session_state.pop(
                                "device_code_flow", None,
                            )
                            result = None
                        if result:
                            result["expires_at"] = (
                                time.time()
                                + result.get("expires_in", 3600)
                            )
                            st.session_state[
                                "graph_token_data"
                            ] = result
                            st.session_state.pop(
                                "device_code_flow", None,
                            )
                            save_token_cache(result)
                            st.rerun()
                        else:
                            st.warning(
                                "Not ready yet. Complete "
                                "sign-in in browser, then "
                                "check again."
                            )
                else:
                    if st.button(
                        "Sign in to OneDrive",
                        key="graph_signin",
                    ):
                        flow = start_device_code_flow(
                            client_id,
                        )
                        if flow:
                            flow["expires_at"] = (
                                time.time()
                                + flow.get("expires_in", 900)
                            )
                            st.session_state[
                                "device_code_flow"
                            ] = flow
                            st.rerun()
                        else:
                            st.error(
                                "Could not start sign-in. "
                                "Check the Client ID."
                            )

    st.title("GSF - Geology Source Finder")

    # Load data for all enabled modes
    mode_datasets: dict[
        str, tuple[pd.DataFrame, pd.DataFrame]
    ] = {}
    geo_coords_df: pd.DataFrame | None = None
    arch_coords_df: pd.DataFrame | None = None
    try:
        for mode in enabled_modes:
            art_df, geo_df, gc_df, ac_df = load_data(mode)
            mode_datasets[mode] = (art_df, geo_df)
            if geo_coords_df is None:
                geo_coords_df = gc_df
                arch_coords_df = ac_df
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error("Error loading data.")
        st.exception(e)
        return

    assert geo_coords_df is not None
    assert arch_coords_df is not None

    # Use sort mode's data for autocomplete and counts
    sort_art_df, sort_geo_df = mode_datasets[sort_mode]

    # Load photo mapping
    photo_df = load_photo_mapping()

    # Show dataset counts in sidebar
    with st.sidebar:
        st.divider()
        st.caption(
            f"Dataset: {len(sort_art_df)} artefacts, "
            f"{len(sort_geo_df)} geology samples"
        )
        if not photo_df.empty:
            st.caption(
                f"Photo mapping: {len(photo_df)} artefacts"
            )

    # Tab layout
    tab_single, tab_batch, tab_compare = st.tabs([
        "Single Query",
        "Batch Query",
        "Comparison",
    ])

    # --- Tab 1: Single Query ---
    with tab_single:
        query_mode = st.radio("Select Query Mode:", [
            "Artefact \u2192 Geology",
            "Geology \u2192 Artefact",
        ], key="single_mode")

        # Autocomplete accession number selection
        options_df = (
            sort_art_df
            if query_mode == "Artefact \u2192 Geology"
            else sort_geo_df
        )

        options = sorted(
            options_df.apply(
                lambda r: (
                    f"{r['Accession #']} - {r['Site']}"
                ),
                axis=1,
            ).unique()
        )

        selected = st.selectbox(
            "Select an accession number:",
            options=["", *options],
            format_func=lambda x: (
                "Type to search..." if x == "" else x
            ),
            key="single_acc",
        )
        accession_number = (
            selected.split(" - ")[0].strip()
            if selected
            else ""
        )

        top_n = st.number_input(
            "Number of top results:",
            min_value=1,
            max_value=MAX_TOP_N,
            value=DEFAULT_TOP_N,
            step=1,
            key="single_topn",
        )

        # Region filter
        all_regions = sorted(
            set(sort_art_df["Region"].dropna().unique())
            | set(sort_geo_df["Region"].dropna().unique())
        )
        selected_regions = st.multiselect(
            "Filter by region(s) (optional):",
            all_regions,
            key="single_regions",
        )

        # Execute query
        if accession_number:
            execute_query(
                accession_number, query_mode,
                top_n, selected_regions,
                mode_datasets,
                geo_coords_df, arch_coords_df,
                enabled_modes=enabled_modes,
                sort_mode=sort_mode,
                photo_df=photo_df,
                photos_base_dir=photos_base_dir,
                access_token=access_token,
                share_url=share_url,
            )

    # --- Tab 2: Batch Query ---
    with tab_batch:
        execute_batch_query(
            mode_datasets,
            geo_coords_df, arch_coords_df,
            enabled_modes=enabled_modes,
            sort_mode=sort_mode,
        )

    # --- Tab 3: Comparison Mode ---
    with tab_compare:
        execute_comparison(
            mode_datasets,
            geo_coords_df, arch_coords_df,
            enabled_modes=enabled_modes,
            sort_mode=sort_mode,
            photo_df=photo_df,
            photos_base_dir=photos_base_dir,
            access_token=access_token,
            share_url=share_url,
        )
