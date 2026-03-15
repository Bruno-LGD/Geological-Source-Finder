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
    COLORS,
    CONF_COLORS,
    CONF_HIGH,
    CONF_MODERATE,
    CONF_VERY_HIGH,
    EPSILON,
    GEO_CLOSE_KM,
    GEO_MODERATE_KM,
    GEO_VERY_CLOSE_KM,
    LITHOLOGY_COLOR_MAP,
    MAP_BOUNDS,
    MODE_LABELS,
    MODE_ORDER,
    PLOTLY_AXIS_STYLE,
    PLOTLY_LAYOUT,
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
    _text_color_for_bg,
    color_aitch_with_thresholds,
    color_confidence,
    highlight_geo_dist,
)


def _render_zoom_viewer(
    img_src: bytes,
    uid: str,
    height: int = 450,
) -> None:
    """Render an interactive pan/zoom viewer via HTML."""
    html_cache_key = f"_zoom_html_{uid}"
    html = st.session_state.get(html_cache_key)

    if html is None:
        cid, iid = f"zc_{uid}", f"zi_{uid}"

        b64 = base64.b64encode(img_src).decode("ascii")
        img_attr = f'src="data:image/jpeg;base64,{b64}"'

        html = (
            f'<div id="{cid}" style="width:100%;'
            f"height:{height}px;"
            "overflow:hidden;position:relative;"
            "background:#2C2825;"
            'cursor:grab;border-radius:4px;">'
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
            "tx=mx-(mx-tx)*(s/ps);"
            "ty=my-(my-ty)*(s/ps);up();};"
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
        st.session_state[html_cache_key] = html

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
        " \u2014 scroll to zoom, drag to pan, "
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


def _legend_color_swatch(color: str, label: str) -> str:
    """Return HTML for a small colored square + label."""
    text_c = _text_color_for_bg(color)
    return (
        f'<span style="display:inline-block;width:14px;height:14px;'
        f"background:{color};border:1px solid #8A8078;"
        f'border-radius:2px;vertical-align:middle;margin-right:5px;'
        f'"></span>'
        f'<span style="vertical-align:middle;font-size:0.85rem;'
        f'color:#2C2825;">{label}</span>'
    )


def display_legend(
    enabled_modes: list[str] | None = None,
) -> None:
    """Display color-coded threshold explanations as compact HTML."""
    if enabled_modes is None:
        enabled_modes = list(MODE_ORDER)

    sections: list[str] = []

    for mode in enabled_modes:
        vs, s, m = aitch_thresholds(mode)
        label = MODE_LABELS[mode]
        if mode == "alr5":
            title = f"Aitchison ({label}, gradient)"
        else:
            title = f"Aitchison ({label})"

        items = [
            _legend_color_swatch(COLORS["very_strong"], f"&lt; {vs}"),
            _legend_color_swatch(COLORS["strong"], f"{vs}\u2013{s}"),
            _legend_color_swatch(COLORS["moderate"], f"{s}\u2013{m}"),
            _legend_color_swatch(COLORS["weak"], f"&gt; {m}"),
        ]
        section = (
            f'<div style="margin-right:2em;margin-bottom:0.5em;">'
            f'<div style="font-weight:600;font-size:0.85rem;'
            f'margin-bottom:4px;color:#3D3530;">{title}</div>'
            + " &nbsp; ".join(items)
            + "</div>"
        )
        sections.append(section)

    # Confidence Score
    conf_items = [
        _legend_color_swatch(CONF_COLORS["very_high"], f"&gt; {CONF_VERY_HIGH}%"),
        _legend_color_swatch(CONF_COLORS["high"], f"{CONF_HIGH}\u2013{CONF_VERY_HIGH}%"),
        _legend_color_swatch(CONF_COLORS["moderate"], f"{CONF_MODERATE}\u2013{CONF_HIGH - 1}%"),
        _legend_color_swatch(CONF_COLORS["low"], f"&lt; {CONF_MODERATE}%"),
    ]
    sections.append(
        '<div style="margin-right:2em;margin-bottom:0.5em;">'
        '<div style="font-weight:600;font-size:0.85rem;'
        'margin-bottom:4px;color:#3D3530;">Confidence Score</div>'
        + " &nbsp; ".join(conf_items)
        + "</div>"
    )

    # Geographical Distance
    geo_items = [
        _legend_color_swatch(COLORS["very_strong"], f"&lt; {GEO_VERY_CLOSE_KM} km"),
        _legend_color_swatch(COLORS["strong"], f"{GEO_VERY_CLOSE_KM}\u2013{GEO_CLOSE_KM} km"),
        _legend_color_swatch(COLORS["moderate"], f"{GEO_CLOSE_KM}\u2013{GEO_MODERATE_KM} km"),
        _legend_color_swatch(COLORS["weak"], f"&gt; {GEO_MODERATE_KM} km"),
    ]
    sections.append(
        '<div style="margin-bottom:0.5em;">'
        '<div style="font-weight:600;font-size:0.85rem;'
        'margin-bottom:4px;color:#3D3530;">Geographical Distance</div>'
        + " &nbsp; ".join(geo_items)
        + "</div>"
    )

    html = (
        '<div style="display:flex;flex-wrap:wrap;gap:0.5em 1em;'
        'padding:0.6em 0;font-family:\'Source Sans 3\',sans-serif;">'
        + "".join(sections)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


_DISPLAY_RENAME = {
    "Geo Site": "Geo Area",
    "Aitch (ALR-5)": "ALR-5 D",
    "Aitch (Trace)": "Trace D",
    "Aitch (All)": "All D",
    "Geo Dist": "Hav D",
}


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

    display_df = results_df.rename(columns=_DISPLAY_RENAME)
    styled_df = display_df.style

    # Format and color each enabled mode's Aitch column
    for mode in enabled_modes:
        orig_col = aitch_col(mode)
        col = _DISPLAY_RENAME.get(orig_col, orig_col)
        if col not in display_df.columns:
            continue
        styled_df = styled_df.format(
            "{:.2f}", subset=[col],
        )
        styler = partial(
            color_aitch_with_thresholds,
            mode_key=mode,
        )
        styled_df = styled_df.map(styler, subset=[col])

    # Format and color the Conf column
    if "Conf" in display_df.columns:
        styled_df = styled_df.format(
            "{:.0f}%", subset=["Conf"],
        )
        styled_df = styled_df.map(
            color_confidence, subset=["Conf"],
        )

    styled_df = styled_df.apply(  # pyright: ignore[reportAttributeAccessIssue]
        highlight_geo_dist, subset=["Hav D"],
    )

    # Uniform width for all numeric distance/score columns (sized to widest: "Hav D")
    numeric_cols = [
        c for c in ["ALR-5 D", "Trace D", "All D", "Conf", "Hav D"]
        if c in display_df.columns
    ]
    if numeric_cols:
        styled_df = styled_df.set_properties(
            subset=numeric_cols,
            **{"width": "72px", "min-width": "72px", "max-width": "72px"},
        )

    # Build per-column-index header width rules for uniform numeric columns
    col_list = list(display_df.columns)
    numeric_header_styles = []
    for col_name in numeric_cols:
        if col_name in col_list:
            idx = col_list.index(col_name)
            numeric_header_styles.append({
                "selector": f"th.col_heading.col{idx}",
                "props": [
                    ("width", "72px"),
                    ("min-width", "72px"),
                    ("max-width", "72px"),
                ],
            })

    # Table styling: thin black borders on all cells, bold headers, autofit columns
    styled_df = styled_df.set_table_styles([
        {
            "selector": "",
            "props": [
                ("border-collapse", "collapse"),
                ("table-layout", "fixed"),
                ("width", "auto"),
                ("font-family", "'Source Sans 3', sans-serif"),
                ("font-size", "0.88rem"),
                ("border", "1px solid #2C2825"),
            ],
        },
        {
            "selector": "th",
            "props": [
                ("padding", "6px 8px"),
                ("font-weight", "bold"),
                ("text-align", "center"),
                ("white-space", "nowrap"),
                ("color", "#2C2825"),
                ("background-color", "#F2EDE4"),
                ("border", "1px solid #2C2825"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("padding", "5px 8px"),
                ("white-space", "nowrap"),
                ("border", "1px solid #2C2825"),
            ],
        },
        *numeric_header_styles,
    ])
    # Render as HTML directly for full CSS control
    table_html = styled_df.to_html()
    st.markdown(table_html, unsafe_allow_html=True)

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
    """Create a scatter plot of Aitch vs. Geo Dist."""
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
        x=sort_col,
        y="Geo Dist Num",
        color=color_col,
        hover_data=hover_cols,
        title=(
            f"{sort_col} vs. Geographical Distance"
        ),
        color_discrete_map=color_map if color_map else {},
    )

    # Compute axis ranges from data with padding
    x_min = float(plot_df[sort_col].min())
    x_max = float(plot_df[sort_col].max())
    x_pad = (x_max - x_min) * 0.15 or 0.1
    x_lo = max(0, x_min - x_pad)
    x_hi = x_max + x_pad

    y_max = float(plot_df["Geo Dist Num"].max())
    y_pad = max(y_max * 0.1, 10)
    y_hi = y_max + y_pad

    # Smooth 2D gradient background: subdivide both axes into
    # N strips and color each cell by its normalised distance
    # from the origin (0,0).  Bottom-left = teal, top-right = red.
    _N_STRIPS = 12
    _GRAD_STOPS = [
        (0.00, (55, 110, 130)),   # deep teal
        (0.20, (74, 135, 120)),
        (0.40, (100, 145, 75)),   # olive sage
        (0.55, (155, 155, 65)),   # yellow-green
        (0.70, (190, 150, 80)),   # sandstone
        (0.85, (175, 110, 75)),   # amber
        (1.00, (155, 70, 65)),    # brick red
    ]

    def _grad_color(t: float) -> str:
        t = max(0.0, min(1.0, t))
        for k in range(len(_GRAD_STOPS) - 1):
            t0, c0 = _GRAD_STOPS[k]
            t1, c1 = _GRAD_STOPS[k + 1]
            if t <= t1:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                r = int(c0[0] + f * (c1[0] - c0[0]))
                g = int(c0[1] + f * (c1[1] - c0[1]))
                b = int(c0[2] + f * (c1[2] - c0[2]))
                return f"#{r:02x}{g:02x}{b:02x}"
        c = _GRAD_STOPS[-1][1]
        return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

    x_step = (x_hi - x_lo) / _N_STRIPS
    y_step = y_hi / _N_STRIPS
    for xi in range(_N_STRIPS):
        for yi in range(_N_STRIPS):
            tx = (xi + 0.5) / _N_STRIPS
            ty = (yi + 0.5) / _N_STRIPS
            t = (0.7 * tx**2 + 0.3 * ty**2) ** 0.5
            fig.add_shape(
                type="rect",
                x0=x_lo + xi * x_step,
                x1=x_lo + (xi + 1) * x_step,
                y0=yi * y_step,
                y1=(yi + 1) * y_step,
                fillcolor=_grad_color(t),
                opacity=0.35,
                line_width=0, layer="below",
            )

    # Vertical gridlines every 0.05
    import math
    vline_start = math.ceil(x_lo / 0.05) * 0.05
    v = vline_start
    while v < x_hi:
        fig.add_vline(
            x=v, line_dash="dash",
            line_color="#8A8078", opacity=0.45,
        )
        v = round(v + 0.05, 4)

    # Horizontal gridlines at 50, 150, 250, ...
    for ref in range(50, int(y_hi) + 1, 100):
        if ref < y_hi:
            fig.add_hline(
                y=ref, line_dash="dash",
                line_color="#8A8078", opacity=0.45,
            )

    fig.update_traces(
        marker={
            "size": 12,
            "line": {"color": "#4A443E", "width": 1},
            "opacity": 0.9,
        },
    )

    # Axis style without default grid (we draw our own above)
    _no_grid_axis = {
        k: v for k, v in PLOTLY_AXIS_STYLE.items()
        if k != "showgrid" and not k.startswith("grid")
    }

    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title=sort_col,
        yaxis_title="Geographical Distance (km)",
        legend_title=color_col if color_col else "Legend",
        width=850,
    )
    fig.update_xaxes(
        range=[x_lo, x_hi],
        showgrid=False,
        **_no_grid_axis,
    )
    fig.update_yaxes(
        range=[0, y_hi],
        showgrid=False,
        **_no_grid_axis,
    )
    st.plotly_chart(fig, use_container_width=False)


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

    # Invisible anchors at map-extent corners
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

        # Black shadow trace behind all match points
        fig.add_trace(go.Scattermapbox(
            lat=map_df["lat"].tolist(),
            lon=map_df["lon"].tolist(),
            mode="markers",
            marker={"size": 16, "color": "#2C2825"},
            showlegend=False,
            hoverinfo="skip",
        ))

        # One trace per color group for legend entries
        step = AITCH_STEP.get(sort_mode, 0.8)
        def _fmt(v: float) -> str:
            return f"{v:.1f}" if v == int(v) else f"{v:.1f}"
        color_labels = [
            (MARKER_COLORS[i], i + 1,
             f"{_fmt(i * step)}\u2013{_fmt((i + 1) * step)}")
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

    # Query point
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
            marker={"size": 15, "color": "#2C2825"},
            name=query_label,
            text=[query_label],
            legendrank=0,
        ))

    # Map layout
    fig.update_layout(
        font=PLOTLY_LAYOUT["font"],
        title={
            "text": "Geographical Distribution of Matches",
            "font": {
                "family": "Crimson Pro, Georgia, serif",
                "size": 18,
                "color": "#2C2825",
            },
            "x": 0.0,
            "xanchor": "left",
        },
        legend=PLOTLY_LAYOUT["legend"],
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
                        "Tiles \u00a9 Esri \u2014 National Geographic, Esri,"
                        " DeLorme, NAVTEQ, USGS, NRCAN, GEBCO, NOAA"
                    ),
                }
            ],
            "bounds": MAP_BOUNDS,
            "center": {"lat": 38.0, "lon": -4.0},
            "zoom": 5.2,
        },
        height=650,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        paper_bgcolor="#FAFAF7",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"staticPlot": True},
    )

    # Scale bar overlay
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
                        color: #2C2825; margin-bottom: 2px;
                        font-family: 'Source Sans 3', sans-serif;">Kilometers</div>
            <div style="display: flex; height: 8px; border: 1px solid #2C2825;">
                <div style="flex: 1; background: #2C2825;"></div>
                <div style="flex: 1; background: #FAFAF7;"></div>
                <div style="flex: 1; background: #2C2825;"></div>
                <div style="flex: 1; background: #FAFAF7;"></div>
            </div>
            <div style="position: relative; font-size: 11px; color: #2C2825;
                        margin-top: 1px; height: 16px;
                        font-family: 'Source Sans 3', sans-serif;">
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
    """Display a deviation heatmap comparing match ratios to the query."""
    ratio_cols = get_ratio_columns(query_sample)
    if len(ratio_cols) < 3:
        st.info("Not enough ratio columns for comparison.")
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

    match_options: list[str] = []
    for _, row in results_df.iterrows():
        match_options.append(
            f"{row[acc_col]} - {row[site_col]}"
        )

    if not match_options:
        return

    selected = st.multiselect(
        "Select matches to compare:",
        match_options,
        default=match_options,
        key=f"radar_select_{query_accession}_{mode_key}",
    )

    if not selected:
        return

    # Collect raw query ratio values (keep NaN/zero info)
    query_raw = np.asarray(pd.Series(pd.to_numeric(
        query_sample[ratio_cols], errors="coerce",
    )), dtype=float)

    # Threshold: values below this are treated as "missing"
    MISSING_THRESH = 1e-6

    # Collect match ratio values
    match_labels: list[str] = []
    deviations: list[np.ndarray] = []
    missing_masks: list[np.ndarray] = []

    for sel in selected:
        acc = sel.split(" - ")[0]
        match_rows = target_df[
            target_df["Accession #"] == acc
        ]
        if match_rows.empty:
            continue
        match_sample = match_rows.iloc[0]
        match_raw = np.asarray(pd.Series(pd.to_numeric(
            match_sample[ratio_cols], errors="coerce",
        )), dtype=float)

        # Track where either value is missing
        missing = np.isnan(query_raw) | np.isnan(match_raw)
        if mode_key != "alr5":
            # For raw ratios, also flag zeros as missing
            missing |= (
                (np.abs(query_raw) < MISSING_THRESH)
                | (np.abs(match_raw) < MISSING_THRESH)
            )

        if mode_key == "alr5":
            # ALR values are already log-ratios: simple difference
            dev = np.where(missing, 0.0, match_raw - query_raw)
        else:
            # Raw ratios: log-ratio deviation ln(match/query)
            q_safe = np.clip(
                np.where(missing, 1.0, query_raw), EPSILON, None,
            )
            m_safe = np.clip(
                np.where(missing, 1.0, match_raw), EPSILON, None,
            )
            dev = np.log(m_safe / q_safe)
        deviations.append(dev)
        missing_masks.append(missing)
        match_labels.append(sel)

    if not deviations:
        st.info("No valid match data found for comparison.")
        return

    dev_matrix = np.array(deviations)
    miss_matrix = np.array(missing_masks)

    # Filter out columns where ALL samples have missing data
    valid_cols = ~miss_matrix.all(axis=0)
    if not valid_cols.any():
        st.info("No shared ratio data between query and matches.")
        return
    dev_matrix = dev_matrix[:, valid_cols]
    miss_matrix = miss_matrix[:, valid_cols]
    filtered_ratio_cols = [
        c for c, v in zip(ratio_cols, valid_cols) if v
    ]

    # Set missing cells to NaN so they show as blank/gray
    dev_matrix = np.where(miss_matrix, np.nan, dev_matrix)

    # Short ratio labels
    short_labels = [
        c.replace("ln(", "").rstrip(")")
        if c.startswith("ln(") else c
        for c in filtered_ratio_cols
    ]

    # Use absolute deviation for color scale
    abs_matrix = np.abs(dev_matrix)

    # Clamp scale to a sensible range (ignore outliers)
    finite_vals = abs_matrix[np.isfinite(abs_matrix)]
    if len(finite_vals) > 0:
        z_max = float(np.percentile(finite_vals, 95))
        z_max = max(z_max, 0.3)  # minimum visible range
    else:
        z_max = 1.0

    # Text labels: show signed value or "n/a" for missing
    text_matrix = np.where(
        np.isfinite(dev_matrix),
        np.char.mod("%+.2f", dev_matrix),
        "n/a",
    )

    # Build metadata for each selected match
    acc_col_res = (
        "Geo Acc #"
        if direction == "artefact_to_geology"
        else "Artefact Acc #"
    )
    site_col_res = (
        "Geo Site"
        if direction == "artefact_to_geology"
        else "Artefact Site"
    )
    region_col_res = "Region" if direction == "artefact_to_geology" else "Artefact Region"

    meta_rows: list[dict[str, str]] = []
    for i, sel in enumerate(match_labels):
        acc = sel.split(" - ")[0]
        res_row = results_df[
            results_df[acc_col_res] == acc
        ]
        if res_row.empty:
            meta_rows.append({
                "n": str(i + 1), "Lithology": "",
                "Acc #": acc, "Area": "", "Region": "",
            })
        else:
            r = res_row.iloc[0]
            meta_rows.append({
                "n": str(i + 1),
                "Lithology": str(r.get("Lithology", "")),
                "Acc #": str(r.get(acc_col_res, acc)),
                "Area": str(r.get(site_col_res, "")),
                "Region": str(r.get(region_col_res, "")),
            })

    # Color interpolation: blue (0) → green (mid) → amber → red (max)
    _COLOR_STOPS = [
        (0.0, (91, 163, 207)),    # blue
        (0.15, (168, 212, 230)),   # light blue
        (0.4, (74, 158, 74)),     # green
        (0.7, (212, 168, 58)),    # amber
        (1.0, (192, 57, 43)),     # red
    ]

    def _deviation_color(val: float, z_max_: float) -> str:
        t = min(abs(val) / z_max_, 1.0) if z_max_ > 0 else 0.0
        # Interpolate between stops
        for j in range(len(_COLOR_STOPS) - 1):
            t0, c0 = _COLOR_STOPS[j]
            t1, c1 = _COLOR_STOPS[j + 1]
            if t <= t1:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                r = int(c0[0] + f * (c1[0] - c0[0]))
                g = int(c0[1] + f * (c1[1] - c0[1]))
                b = int(c0[2] + f * (c1[2] - c0[2]))
                return f"#{r:02x}{g:02x}{b:02x}"
        c = _COLOR_STOPS[-1][1]
        return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

    # Build HTML table
    meta_cols = ["n", "Lithology", "Acc #", "Area", "Region"]
    hdr_style = (
        "padding:6px 8px;font-weight:bold;text-align:center;"
        "white-space:nowrap;color:#2C2825;background-color:#F2EDE4;"
        "border:1px solid #2C2825;"
    )
    td_style_base = (
        "padding:5px 8px;white-space:nowrap;"
        "border:1px solid #2C2825;text-align:center;"
    )
    meta_td_style = (
        "padding:5px 8px;white-space:nowrap;"
        "border:1px solid #2C2825;text-align:center;"
    )

    html_parts = [
        "<div style='overflow-x:auto;'>",
        "<table style='border-collapse:collapse;font-family:\"Source Sans 3\",sans-serif;"
        "font-size:0.88rem;border:1px solid #2C2825;'>",
    ]

    # Header row
    html_parts.append("<thead><tr>")
    for mc in meta_cols:
        html_parts.append(f"<th style='{hdr_style}'>{mc}</th>")
    for sl in short_labels:
        html_parts.append(f"<th style='{hdr_style}'>{sl}</th>")
    html_parts.append("</tr></thead>")

    # Data rows
    html_parts.append("<tbody>")
    for row_i in range(len(match_labels)):
        html_parts.append("<tr>")
        meta = meta_rows[row_i]
        for mc in meta_cols:
            val = meta.get(mc, "")
            html_parts.append(
                f"<td style='{meta_td_style}'>{val}</td>"
            )
        for col_j in range(len(short_labels)):
            dev_val = dev_matrix[row_i, col_j]
            abs_val = abs_matrix[row_i, col_j]
            if np.isfinite(dev_val):
                bg = _deviation_color(dev_val, z_max)
                txt = f"{dev_val:+.2f}"
                html_parts.append(
                    f"<td style='{td_style_base}"
                    f"background-color:{bg};color:#000000;'>"
                    f"{txt}</td>"
                )
            else:
                html_parts.append(
                    f"<td style='{td_style_base}"
                    f"background-color:#e0e0e0;color:#888;'>n/a</td>"
                )
        html_parts.append("</tr>")
    html_parts.append("</tbody></table></div>")

    st.markdown(
        f"**Ratio Deviation from Query: {query_accession}**"
        "  —  blue = close, red = far",
    )
    st.markdown("".join(html_parts), unsafe_allow_html=True)
