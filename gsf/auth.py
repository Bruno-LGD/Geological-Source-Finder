from __future__ import annotations

import json
import logging
import os
import time

import requests
import streamlit as st

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Microsoft Graph / OneDrive constants
_GRAPH_AUTHORITY = "https://login.microsoftonline.com/common"
_GRAPH_DEVICE_CODE_URL = f"{_GRAPH_AUTHORITY}/oauth2/v2.0/devicecode"
_GRAPH_TOKEN_URL = f"{_GRAPH_AUTHORITY}/oauth2/v2.0/token"
GRAPH_SCOPES = "Files.Read offline_access"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"


# --- Token Cache ---

def get_token_cache_path() -> str:
    """Return path to the OAuth token cache JSON file."""
    return os.path.join(_PROJECT_ROOT, "Data", "token_cache.json")


def _load_token_cache() -> dict:
    """Load cached OAuth tokens from disk."""
    path = get_token_cache_path()
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_token_cache(data: dict) -> None:
    """Save OAuth tokens to disk."""
    path = get_token_cache_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        logger.warning("Could not save token cache to %s", path)


# --- OAuth2 Device Code Flow ---

def start_device_code_flow(client_id: str) -> dict | None:
    """Request a device code from Microsoft identity platform.

    Returns dict with device_code, user_code, verification_uri,
    or None on failure.
    """
    try:
        resp = requests.post(
            _GRAPH_DEVICE_CODE_URL,
            data={
                "client_id": client_id,
                "scope": GRAPH_SCOPES,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("Device code request failed: %s", exc)
        return None


def poll_for_token(
    client_id: str, device_code: str,
) -> dict | None:
    """Single non-blocking poll to exchange device code for tokens.

    Returns token dict on success, None if still pending.
    Raises ValueError if the flow expired or was denied.
    """
    try:
        resp = requests.post(
            _GRAPH_TOKEN_URL,
            data={
                "client_id": client_id,
                "grant_type": (
                    "urn:ietf:params:oauth:grant-type:device_code"
                ),
                "device_code": device_code,
            },
            timeout=15,
        )
        data = resp.json()

        if "access_token" in data:
            return data

        error = data.get("error", "")
        if error in ("authorization_pending", "slow_down"):
            return None

        raise ValueError(
            data.get("error_description", f"Auth failed: {error}")
        )
    except requests.RequestException as exc:
        logger.error("Token poll failed: %s", exc)
        return None


def _refresh_access_token(
    client_id: str, refresh_token: str,
) -> dict | None:
    """Use refresh token to get a new access token."""
    try:
        resp = requests.post(
            _GRAPH_TOKEN_URL,
            data={
                "client_id": client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "scope": GRAPH_SCOPES,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("Token refresh failed: %s", exc)
        return None


def get_valid_access_token(client_id: str) -> str | None:
    """Return a valid Graph API access token, refreshing if needed.

    Checks session_state -> disk cache -> refresh token.
    Returns None if not authenticated.
    """
    token_data = st.session_state.get("graph_token_data")

    if not token_data:
        token_data = _load_token_cache()
        if token_data and "access_token" in token_data:
            st.session_state["graph_token_data"] = token_data
        else:
            return None

    # Check if token is still valid (5-min buffer)
    expires_at = token_data.get("expires_at", 0)
    if time.time() < expires_at - 300:
        return token_data.get("access_token")

    # Token expired — try refresh
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        return None

    new_data = _refresh_access_token(client_id, refresh_token)
    if not new_data:
        st.session_state.pop("graph_token_data", None)
        return None

    new_data["expires_at"] = (
        time.time() + new_data.get("expires_in", 3600)
    )
    st.session_state["graph_token_data"] = new_data
    save_token_cache(new_data)
    return new_data["access_token"]
