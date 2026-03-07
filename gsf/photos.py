from __future__ import annotations

import base64
import os

import pandas as pd
import requests
import streamlit as st
from urllib.parse import quote

from gsf.auth import GRAPH_BASE

# Path corrections: CSV relative paths -> actual disk paths
_PATH_PREFIXES: dict[str, str] = {
    "Montefrío/": "UGR assemblages/Montefrío/",
    "Los Millares/": "UGR assemblages/Los Millares/",
    "El Malagón/": "UGR assemblages/El Malagón/",
    "Museo Almería/": "Museo Almería pictures/",
}


def _encode_share_url(share_url: str) -> str:
    """Encode a OneDrive share URL into a Graph API sharing token.

    Format: 'u!' + base64url(share_url) with no padding.
    """
    encoded = base64.urlsafe_b64encode(
        share_url.encode("utf-8"),
    ).rstrip(b"=").decode("ascii")
    return f"u!{encoded}"


def _resolve_graph_path(relative_path: str) -> str:
    """Apply CSV-to-disk path corrections for Graph API."""
    for csv_prefix, disk_prefix in _PATH_PREFIXES.items():
        if relative_path.startswith(csv_prefix):
            return disk_prefix + relative_path[len(csv_prefix):]
    return relative_path


def _resolve_share_folder(
    access_token: str, share_url: str,
) -> tuple[str, str] | None:
    """Resolve a share URL to (drive_id, folder_path).

    Caches result in session_state.  Returns None on failure.
    """
    cache_key = "_share_resolved"
    cached = st.session_state.get(cache_key)
    if cached:
        return cached

    share_token = _encode_share_url(share_url)
    url = (
        f"{GRAPH_BASE}/shares/{share_token}/driveItem"
    )
    try:
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            params={
                "$select": "id,name,parentReference",
            },
            timeout=15,
        )
        if not resp.ok:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text[:200]
            st.session_state["_graph_last_error"] = (
                f"Share resolve HTTP {resp.status_code}: "
                f"{err_body}"
            )
            return None

        item = resp.json()
        drive_id = item["parentReference"]["driveId"]
        parent_path = item["parentReference"].get(
            "path", "",
        )
        # parent_path: "/drives/{id}/root:/some/path"
        if "/root:" in parent_path:
            base = parent_path.split(
                "/root:",
            )[1].lstrip("/")
        else:
            base = ""
        folder_name = item.get("name", "")
        folder_path = (
            f"{base}/{folder_name}" if base
            else folder_name
        )
        result = (drive_id, folder_path)
        st.session_state[cache_key] = result
        return result
    except Exception as exc:
        st.session_state["_graph_last_error"] = str(exc)
        return None


def _fetch_image_bytes_from_graph(
    access_token: str,
    share_url: str,
    relative_path: str,
) -> bytes | None:
    """Download image bytes from OneDrive via Graph API.

    Fetches server-side so images can be embedded as base64 --
    avoids browser CORS/CSP blocks when loading CDN URLs from
    within a Streamlit iframe.
    """
    resolved = _resolve_share_folder(
        access_token, share_url,
    )
    if not resolved:
        return None
    drive_id, folder_path = resolved

    full_path = f"{folder_path}/{relative_path}"
    encoded_path = quote(full_path, safe="/")
    url = (
        f"{GRAPH_BASE}/drives/{drive_id}"
        f"/root:/{encoded_path}:/content"
    )
    try:
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            timeout=30,
        )
        if not resp.ok:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text[:200]
            st.session_state["_graph_last_error"] = (
                f"HTTP {resp.status_code} for "
                f"{full_path}: {err_body}"
            )
            return None
        return resp.content
    except Exception as exc:
        st.session_state["_graph_last_error"] = str(exc)
        return None


def _get_cached_image_bytes(
    access_token: str,
    share_url: str,
    relative_path: str,
) -> bytes | None:
    """Fetch image bytes with session-level caching."""
    cache_key = f"imgbytes_{relative_path}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached

    data = _fetch_image_bytes_from_graph(
        access_token, share_url, relative_path,
    )
    if data:
        st.session_state[cache_key] = data
    return data


def _resolve_photo_path(
    base_dir: str, relative_path: str,
) -> str | None:
    """Resolve a CSV relative path to an actual file on disk."""
    # Apply known prefix corrections
    resolved = relative_path
    for csv_prefix, disk_prefix in _PATH_PREFIXES.items():
        if relative_path.startswith(csv_prefix):
            resolved = disk_prefix + relative_path[len(csv_prefix):]
            break

    full_path = os.path.join(base_dir, resolved)
    if os.path.isfile(full_path):
        return full_path

    # Fallback: try original path unchanged
    fallback = os.path.join(base_dir, relative_path)
    if os.path.isfile(fallback):
        return fallback

    return None


def get_artefact_images(
    accession: str,
    photo_df: pd.DataFrame,
    photos_base_dir: str = "",
    access_token: str = "",
    share_url: str = "",
) -> list[str | bytes]:
    """Return image sources for an accession's photos.

    Tries local files first (if photos_base_dir is set),
    falls back to Graph API (downloads bytes server-side).
    Returns list of:
    - str: local file path
    - bytes: image data
    """
    if photo_df.empty:
        return []
    row = photo_df[
        photo_df["Accession #"] == str(accession).strip()
    ]
    if row.empty:
        return []
    row = row.iloc[0]

    images: list[str | bytes] = []
    for col in [
        "Image_1", "Image_2", "Image_3",
        "Image_4", "Image_5", "Image_6",
    ]:
        val = row.get(col)
        if not (pd.notna(val) and str(val).strip()):
            continue
        relative = str(val).strip()

        # Try local path first
        if photos_base_dir:
            local = _resolve_photo_path(
                photos_base_dir, relative,
            )
            if local:
                images.append(local)
                continue

        # Try Graph API (download bytes server-side)
        if access_token and share_url:
            graph_path = _resolve_graph_path(relative)
            data = _get_cached_image_bytes(
                access_token, share_url, graph_path,
            )
            if data:
                images.append(data)

    return images


def to_image_bytes(img_src: str | bytes) -> bytes | None:
    """Convert an image source (file path or bytes) to bytes."""
    if isinstance(img_src, bytes):
        return img_src
    try:
        with open(img_src, "rb") as f:
            return f.read()
    except Exception:
        return None
