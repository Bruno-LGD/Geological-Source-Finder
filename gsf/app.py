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

# ---------------------------------------------------------------------------
# Global CSS for academic / scientific styling
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

/* Global typography */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', 'Source Sans Pro', sans-serif;
    color: #2C2825;
}

/* Main title */
h1 {
    font-family: 'Crimson Pro', 'Georgia', serif !important;
    font-weight: 700 !important;
    color: #2C2825 !important;
    letter-spacing: -0.02em !important;
    border-bottom: 2px solid #8B6914 !important;
    padding-bottom: 0.3em !important;
}

/* Section headers (h2) */
h2, [data-testid="stSubheader"] h2 {
    font-family: 'Crimson Pro', 'Georgia', serif !important;
    font-weight: 600 !important;
    color: #3D3530 !important;
    font-size: 1.5rem !important;
    border-bottom: 1px solid #D4CCC0 !important;
    padding-bottom: 0.2em !important;
    margin-top: 1.5em !important;
}

/* h3 subheadings */
h3 {
    font-family: 'Crimson Pro', 'Georgia', serif !important;
    font-weight: 600 !important;
    color: #4A443E !important;
    font-size: 1.2rem !important;
}

/* Sidebar headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Crimson Pro', 'Georgia', serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #5A4F45 !important;
    border-bottom: none !important;
}

/* Tab labels */
button[data-baseweb="tab"] {
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #8B6914 !important;
    color: #2C2825 !important;
}
button[data-baseweb="tab"][aria-selected="false"] {
    color: #7A7068 !important;
}

/* Expander headers */
[data-testid="stExpander"] summary span {
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    color: #4A443E !important;
}

/* Captions and small text */
[data-testid="stCaption"], .stCaption {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.8rem !important;
    color: #7A7068 !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] button {
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 500 !important;
    border: 1px solid #8B6914 !important;
    color: #8B6914 !important;
    background: transparent !important;
    border-radius: 3px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #8B6914 !important;
    color: #FAFAF7 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #D4CCC0 !important;
    border-radius: 4px !important;
}

/* Selectbox and inputs */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    border-color: #C4B9A8 !important;
    border-radius: 3px !important;
}

/* Info/warning/error boxes - more muted */
[data-testid="stAlert"] {
    border-radius: 3px !important;
    font-size: 0.9rem !important;
}

/* Dividers */
hr {
    border-color: #D4CCC0 !important;
}

/* Progress bar */
[role="progressbar"] > div {
    background-color: #8B6914 !important;
}
</style>
"""


def main() -> None:
    """Application entry point."""
    # Inject global CSS
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    # Render sidebar to get settings (element mode)
    with st.sidebar:
        # Branding block
        st.markdown(
            '<p style="font-family: \'Crimson Pro\', serif; '
            'font-size: 1.3rem; font-weight: 700; color: #2C2825; '
            'letter-spacing: -0.01em; margin-bottom: 0.1em;">'
            'GSF</p>'
            '<p style="font-family: \'Source Sans 3\', sans-serif; '
            'font-size: 0.82rem; color: #7A7068; line-height: 1.5; '
            'margin-top: 0;">'
            'Matching archaeological artefacts to potential geological '
            'sources using compositional distance analysis of trace '
            'element ratios.'
            '</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        # Analysis Configuration
        st.markdown("### Analysis Configuration")
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

        # --- Photo Configuration (collapsible) ---
        with st.expander("Photo Configuration", expanded=False):
            # Local photos folder
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

            # OneDrive (Graph API)
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

    # Branded header
    st.markdown(
        '<div style="margin-bottom: 0.5em;">'
        '<h1 style="margin-bottom: 0.1em;">'
        'Geology Source Finder</h1>'
        '<p style="font-family: \'Source Sans 3\', sans-serif; '
        'color: #7A7068; font-size: 0.95rem; margin-top: 0; '
        'font-style: italic;">'
        'Archaeological Provenance Analysis through '
        'Trace Element Geochemistry'
        '</p></div>',
        unsafe_allow_html=True,
    )

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
