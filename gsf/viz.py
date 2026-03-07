from __future__ import annotations

import base64
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from gsf.constants import (
    EPSILON,
    GEO_CLOSE_KM,
    GEO_MODERATE_KM,
    GEO_VERY_CLOSE_KM,
    LITHOLOGY_COLOR_MAP,
    MAP_BOUNDS,
    MODE_LABELS,
    MODE_ORDER,
    REGION_COLOR_MAP,
    AITCH_STEP,
    MARKER_COLORS,
    MARKER_OVERFLOW,
    aitch_col,
    aitch_thresholds,
)
from gsf.distances import get_ratio_columns
from gsf.export import create_styled_excel
from gsf.photos import get_artefact_images, to_image_bytes
from gsf.styling import (
    color_aitch_with_thresholds,
    highlight_geo_dist,
)


def _render_zoom_viewer(
    img_src: bytes,
    uid: str,
    height: int = 450,
) -> None:
    """Render an interactive pan/zoom viewer via HTML.

    img_src: image bytes (base64-encoded into the HTML so no
    cross-origin requests are needed from the iframe).
    """
    cid, iid = f"zc_{uid}", f"zi_{uid}"

    b64 = base64.b64encode(img_src).decode("ascii")
    img_attr = f'src="data:image/jpeg;base64,{b64}"'

    html = (
        f'<div id="{cid}" style="width:100%;height:{height}px;'
        "overflow:hidden;position:relative;background:#1a1a1a;"
        'cursor:grab;border-radius:6px;">'
        f'<img id="{iid}" {img_attr} '
        'style="position:absolute;max-width:none;'
        "image-orientation:from-image;"
        'transform-origin:0 0;" draggable="false">'
        "</div>"
        "<script>(function(){"
        f"const c=document.getElementById('{cid}'),"
        f"im=document.getElementById('{iid}');"
        "let s=1,tx=0,ty=0,drag=0,sx,sy;"
        "function up(){im.style.transform="
        "'translate('+tx+'px,'+ty+'px) scale('+s+')';}"
        "function fit(){"
        "let rx=c.clientWidth/im.naturalWidth,"
        "ry=c.clientHeight/im.naturalHeight;"
        "s=Math.min(rx,ry);"
        "tx=(c.clientWidth-im.naturalWidth*s)/2;"
        "ty=(c.clientHeight-im.naturalHeight*s)/2;up();}"
        "im.onload=function(){fit();};"
        "c.onwheel=function(e){e.preventDefault();"
        "let r=c.getBoundingClientRect(),"
        "mx=e.clientX-r.left,my=e.clientY-r.top,"
        "ps=s;s*=e.deltaY<0?1.15:0.87;"
        "s=Math.max(0.1,Math.min(30,s));"
        "tx=mx-(mx-tx)*(s/ps);ty=my-(my-ty)*(s/ps);up();};"
        "c.onmousedown=function(e){"
        "drag=1;sx=e.clientX-tx;sy=e.clientY-ty;"
        "c.style.cursor='grabbing';};"
        "c.onmousemove=function(e){"
        "if(!drag)return;tx=e.clientX-sx;"
        "ty=e.clientY-sy;up();};"
        "c.onmouseup=c.onmouseleave=function(){"
        "drag=0;c.style.cursor='grab';};"
        "c.ondblclick=function(){fit();};"
        "})();</script>"
    )
    components.html(html, height=height + 10)


def display_artefact_photos(
    accession: str,
    photo_df: pd.DataFrame,
    photos_base_dir: str = "",
    access_token: str = "",
    share_url: str = "",
) -> None:
    """Display artefact photographs from local files or OneDrive."""
    images = get_artefact_images(
        accession, photo_df, photos_base_dir,
        access_token, share_url,
    )
    if not images:
        # Diagnostic: show why no images were found
        row = photo_df[
            photo_df["Accession #"] == str(accession).strip()
        ] if not photo_df.empty else pd.DataFrame()
        if row.empty:
            st.caption(
                f"No photo mapping for accession {accession}"
            )
        else:
            err = st.session_state.get(
                "_graph_last_error", "",
            )
            st.warning(
                f"Photo mapping found but images failed to "
                f"load. {f'Last error: {err}' if err else ''}"
            )
        return
    with st.expander(
        f"Artefact Photographs ({len(images)} images)"
        " — scroll to zoom, drag to pan, "
        "double-click to reset",
        expanded=True,
    ):
        cols = st.columns(min(len(images), 3))
        for i, img_src in enumerate(images):
            with cols[i % 3]:
                img_bytes = to_image_bytes(img_src)
                if img_bytes:
                    _render_zoom_viewer(
                        img_bytes,
                        uid=f"{accession}_{i}",
                    )
                    st.caption(f"View {i + 1}")
                else:
                    st.caption(
                        f"View {i + 1}: not available"
                    )


def display_legend(
    enabled_modes: list[str] | None = None,
) -> None:
    """Display color-coded threshold explanations."""
    if enabled_modes is None:
        enabled_modes = list(MODE_ORDER)

    n_cols = len(enabled_modes) + 1  # +1 for Geo Dist
    cols = st.columns(n_cols)

    for i, mode in enumerate(enabled_modes):
        vs, s, m = aitch_thresholds(mode)
        label = (
            f"**Aitchison Distance** ({MODE_LABELS[mode]})"
        )
        with cols[i]:
            st.markdown(label)
            st.markdown(
                f"- :blue[< {vs}] "
                "-- Very strong match\n"
                f"- :green[{vs} - "
                f"{s}] -- Strong match\n"
                f"- :orange[{s} - "
                f"{m}] -- Moderate match\n"
                f"- :red[> {m}] -- Weak match"
            )

    with cols[-1]:
        st.markdown("**Geographical Distance**")
        st.markdown(
            f"- :blue[< {GEO_VERY_CLOSE_KM} km] "
            "-- Very close\n"
            f"- :green[{GEO_VERY_CLOSE_KM} - "
            f"{GEO_CLOSE_KM} km] -- Close\n"
            f"- :orange[{GEO_CLOSE_KM} - "
            f"{GEO_MODERATE_KM} km] -- Moderate distance\n"
            f"- :red[> {GEO_MODERATE_KM} km] -- Distant"
        )


def display_results_table(
    results_df: pd.DataFrame,
    query_accession: str,
    direction: str,
    query_site: str = "",
    enabled_modes: list[str] | None = None,
) -> None:
    """Style and display the results table with downloads."""
    if enabled_modes is None:
        enabled_modes = list(MODE_ORDER)

    styled_df = results_df.style

    # Format and color each enabled mode's Aitch column
    for mode in enabled_modes:
        col = aitch_col(mode)
        if col not in results_df.columns:
            continue
        styled_df = styled_df.format(
            "{:.2f}", subset=[col],
        )
        styler = partial(
            color_aitch_with_thresholds,
            mode_key=mode,
        )
        styled_df = styled_df.map(styler, subset=[col])

    styled_df = styled_df.apply(  # pyright: ignore[reportAttributeAccessIssue]
        highlight_geo_dist, subset=["Geo Dist"],
    )
    border_style = ("border", "1px solid gray")
    padding = ("padding", "4px")
    no_radius = ("border-radius", "0px")
    styled_df = styled_df.set_table_styles([
        {
            "selector": "table",
            "props": [
                ("border-collapse", "collapse"),
                border_style,
            ],
        },
        {
            "selector": "th",
            "props": [border_style, padding, no_radius],
        },
        {
            "selector": "td",
            "props": [border_style, padding, no_radius],
        },
    ])
    st.table(styled_df)

    direction_label = (
        "art2geo"
        if direction == "artefact_to_geology"
        else "geo2art"
    )
    base_filename = (
        f"GSF_{query_accession}"
        f"_{direction_label}_top{len(results_df)}"
    )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv_data = (
            results_df.to_csv(index=False).encode("utf-8")
        )
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"{base_filename}.csv",
            mime="text/csv",
        )
    with dl_col2:
        excel_data = create_styled_excel(
            results_df, query_accession,
            query_site, direction,
            enabled_modes=enabled_modes,
        )
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"{base_filename}.xlsx",
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
        )


def display_scatter_plot(
    results_df: pd.DataFrame,
    sort_mode: str = "alr5",
) -> None:
    """Create a scatter plot of Geo Dist vs. sort mode Aitch."""
    sort_col = aitch_col(sort_mode)
    if sort_col not in results_df.columns:
        return

    plot_df = results_df[
        results_df["Geo Dist"] != "Unknown"
    ].copy()
    if plot_df.empty:
        st.warning(
            "No rows with known geographical distance to plot."
        )
        return

    plot_df["Geo Dist Num"] = plot_df["Geo Dist"].apply(  # pyright: ignore[reportAttributeAccessIssue]
        lambda v: float(str(v).replace(" km", ""))
    )

    hover_cols: dict[str, bool] = {}
    for col in [
        "Geo Acc #", "Geo Site",
        "Artefact Acc #", "Artefact Site",
    ]:
        if col in plot_df.columns:
            hover_cols[col] = True

    color_col = None
    color_map = None
    if "Lithology" in plot_df.columns:
        color_col = "Lithology"
        color_map = LITHOLOGY_COLOR_MAP
    elif "Artefact Region" in plot_df.columns:
        color_col = "Artefact Region"
        color_map = REGION_COLOR_MAP

    fig = px.scatter(
        plot_df,
        x="Geo Dist Num",
        y=sort_col,
        color=color_col,
        hover_data=hover_cols,
        title=(
            "Scatter Plot of Geographical Distance "
            f"vs. {sort_col}"
        ),
        color_discrete_map=color_map if color_map else {},
    )

    for ref in [
        GEO_VERY_CLOSE_KM, GEO_CLOSE_KM, GEO_MODERATE_KM,
    ]:
        fig.add_vline(
            x=ref, line_dash="dash",
            line_color="black", opacity=0.5,
        )
    for ref in aitch_thresholds(sort_mode):
        fig.add_hline(
            y=ref, line_dash="dash",
            line_color="black", opacity=0.5,
        )

    fig.update_traces(
        marker={
            "size": 20,
            "line": {"color": "black", "width": 2},
            "opacity": 0.8,
        },
    )
    fig.update_layout(
        template=None,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "black"},
        xaxis_title="Geographical Distance (km)",
        yaxis_title=sort_col,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        legend_title=color_col if color_col else "Legend",
    )
    for ax_update in (fig.update_xaxes, fig.update_yaxes):
        ax_update(
            rangemode="tozero",
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickfont={"color": "black"},
            showgrid=False,
            zeroline=False,
        )
    st.plotly_chart(fig, use_container_width=True)


def display_results_map(
    results_df: pd.DataFrame,
    query_coords: tuple[float, float] | None,
    target_coords_df: pd.DataFrame,
    query_label: str,
    direction: str,
    sort_mode: str = "alr5",
) -> None:
    """Display an interactive map with query and match locations."""
    from gsf.styling import aitch_to_marker_color

    sort_col = aitch_col(sort_mode)
    if sort_col not in results_df.columns:
        return

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

    map_points: list[dict[str, Any]] = []
    if direction == "artefact_to_geology":
        for _, row in results_df.iterrows():
            if row["Geo Dist"] == "Unknown":
                continue
            acc = str(row[acc_col])
            coord_row = target_coords_df[
                target_coords_df["Accession #"].astype(str)
                == acc
            ]
            if not coord_row.empty:
                map_points.append({
                    "lat": coord_row.iloc[0]["Latitude"],
                    "lon": coord_row.iloc[0]["Longitude"],
                    "label": f"{acc} - {row[site_col]}",
                    "aitch": row[sort_col],
                })
    else:
        for _, row in results_df.iterrows():
            if row["Geo Dist"] == "Unknown":
                continue
            site = row[site_col]
            coord_row = target_coords_df[
                target_coords_df["Site"] == site
            ]
            if not coord_row.empty:
                map_points.append({
                    "lat": coord_row.iloc[0]["Latitude"],
                    "lon": coord_row.iloc[0]["Longitude"],
                    "label": f"{row[acc_col]} - {site}",
                    "aitch": row[sort_col],
                })

    if not map_points and query_coords is None:
        return

    fig = go.Figure()

    # Invisible anchors at map-extent corners so fitbounds always
    # shows the full southern-Iberia region regardless of data spread
    fig.add_trace(go.Scattermapbox(
        lat=[MAP_BOUNDS["south"], MAP_BOUNDS["north"]],
        lon=[MAP_BOUNDS["west"], MAP_BOUNDS["east"]],
        mode="markers",
        marker={"size": 1, "opacity": 0},
        showlegend=False,
        hoverinfo="skip",
    ))

    # Match points colored by Aitchison distance
    if map_points:
        map_df = pd.DataFrame(map_points)
        map_df["marker_color"] = map_df["aitch"].apply(
            lambda v: aitch_to_marker_color(v, sort_mode),
        )
        map_df["hover_text"] = map_df.apply(
            lambda r: (
                f"{r['label']}<br>Aitch: {r['aitch']:.2f}"
            ),
            axis=1,
        )

        # Black shadow trace behind all match points — simulates borders
        # (Scattermapbox marker.line is not supported)
        fig.add_trace(go.Scattermapbox(
            lat=map_df["lat"].tolist(),
            lon=map_df["lon"].tolist(),
            mode="markers",
            marker={"size": 16, "color": "black"},
            showlegend=False,
            hoverinfo="skip",
        ))

        # One trace per color group for legend entries.
        # Add traces worst-first so that better matches render
        # on top when multiple samples share the same coordinates.
        # legendrank keeps the legend ordered best-first.
        step = AITCH_STEP.get(sort_mode, 0.8)
        def _fmt(v: float) -> str:
            return f"{v:.1f}" if v == int(v) else f"{v:.1f}"
        color_labels = [
            (MARKER_COLORS[i], i + 1,
             f"{_fmt(i * step)}–{_fmt((i + 1) * step)}")
            for i in range(len(MARKER_COLORS))
        ] + [
            (MARKER_OVERFLOW, len(MARKER_COLORS) + 1,
             f"> {_fmt(len(MARKER_COLORS) * step)}"),
        ]
        for color, rank, label in reversed(color_labels):
            subset = map_df[
                map_df["marker_color"] == color
            ]
            if subset.empty:
                continue
            fig.add_trace(go.Scattermapbox(
                lat=subset["lat"],
                lon=subset["lon"],
                mode="markers",
                marker={"size": 12, "color": color},
                name=label,
                text=subset["hover_text"],
                legendrank=rank,
            ))

    # Query point — added last so it always renders on top.
    # White halo then black fill so it is distinct from match markers.
    if query_coords:
        fig.add_trace(go.Scattermapbox(
            lat=[query_coords[0]],
            lon=[query_coords[1]],
            mode="markers",
            marker={"size": 21, "color": "white"},
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scattermapbox(
            lat=[query_coords[0]],
            lon=[query_coords[1]],
            mode="markers",
            marker={"size": 15, "color": "black"},
            name=query_label,
            text=[query_label],
            legendrank=0,
        ))

    # Map layout
    fig.update_layout(
        mapbox={
            "style": "white-bg",
            "layers": [
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/"
                        "services/NatGeo_World_Map/MapServer/tile"
                        "/{z}/{y}/{x}"
                    ],
                    "sourceattribution": (
                        "Tiles © Esri — National Geographic, Esri,"
                        " DeLorme, NAVTEQ, USGS, NRCAN, GEBCO, NOAA"
                    ),
                }
            ],
            "bounds": MAP_BOUNDS,
            "center": {"lat": 38.0, "lon": -4.0},   # fallback
            "zoom": 5.5,                              # fallback
        },
        height=700,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title="Geographical Distribution of Matches",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scale bar overlay — pure HTML/CSS positioned over the bottom-left of the map
    st.markdown(
        """
        <div style="
            position: relative;
            margin-top: -90px;
            margin-left: 20px;
            margin-bottom: 50px;
            z-index: 1000;
            width: 220px;
            pointer-events: none;
        ">
            <div style="text-align: center; font-size: 11px; font-weight: bold;
                        color: #222; margin-bottom: 2px;">Kilometers</div>
            <div style="display: flex; height: 8px; border: 1px solid #222;">
                <div style="flex: 1; background: #222;"></div>
                <div style="flex: 1; background: #fff;"></div>
                <div style="flex: 1; background: #222;"></div>
                <div style="flex: 1; background: #fff;"></div>
            </div>
            <div style="position: relative; font-size: 11px; color: #222;
                        margin-top: 1px; height: 16px;">
                <span style="position: absolute; left: 0; transform: translateX(-50%);">0</span>
                <span style="position: absolute; left: 25%; transform: translateX(-50%);">50</span>
                <span style="position: absolute; left: 50%; transform: translateX(-50%);">100</span>
                <span style="position: absolute; left: 100%; transform: translateX(-50%);">200</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_radar_chart(
    query_sample: pd.Series,
    results_df: pd.DataFrame,
    target_df: pd.DataFrame,
    query_accession: str,
    direction: str,
    mode_key: str = "trace",
) -> None:
    """Display a radar chart comparing ratio profiles."""
    ratio_cols = get_ratio_columns(query_sample)
    if len(ratio_cols) < 3:
        st.info("Not enough ratio columns for a radar chart.")
        return

    # Build options for match selection
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

    match_options: list[str] = []
    for _, row in results_df.iterrows():
        match_options.append(
            f"{row[acc_col]} - {row[site_col]}"
        )

    if not match_options:
        return

    selected = st.multiselect(
        "Select matches to compare on radar chart:",
        match_options,
        default=match_options[:min(3, len(match_options))],
        max_selections=8,
        key=f"radar_select_{query_accession}_{mode_key}",
    )

    if not selected:
        return

    # Collect ratio values
    query_values = np.asarray(pd.Series(pd.to_numeric(
        query_sample[ratio_cols], errors="coerce",
    )).fillna(EPSILON), dtype=float)
    all_values = [query_values]
    labels = [f"Query: {query_accession}"]

    for sel in selected:
        acc = sel.split(" - ")[0]
        match_rows = target_df[
            target_df["Accession #"] == acc
        ]
        if match_rows.empty:
            continue
        match_sample = match_rows.iloc[0]
        match_vals = np.asarray(pd.Series(pd.to_numeric(
            match_sample[ratio_cols], errors="coerce",
        )).fillna(EPSILON), dtype=float)
        all_values.append(match_vals)
        labels.append(sel)

    if len(all_values) < 2:
        st.info("No valid match data found for radar chart.")
        return

    # Normalize: min-max across all compared samples
    matrix = np.array(all_values)
    if mode_key == "alr5":
        # ALR values are already log-scale; skip log10
        work = matrix
    else:
        work = np.log10(np.clip(matrix, EPSILON, None))
    col_min = work.min(axis=0)
    col_max = work.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normalized = (work - col_min) / col_range

    # Build radar chart
    fig = go.Figure()
    theta = [*list(ratio_cols), ratio_cols[0]]

    chart_colors = [
        "red", "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f",
    ]
    for i, (vals, label) in enumerate(
        zip(normalized, labels, strict=False)
    ):
        r = [*list(vals), vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            name=label,
            line={
                "width": 3 if i == 0 else 2,
                "color": chart_colors[
                    i % len(chart_colors)
                ],
            },
            fill="none",
        ))

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True, "range": [0, 1],
            },
        },
        showlegend=True,
        title="Ratio Profile Comparison (log-normalized)",
        margin={"l": 60, "r": 60, "t": 50, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)
