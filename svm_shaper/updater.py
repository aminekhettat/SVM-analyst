"""Auto-update support for SVM Analyst.

Checks the latest GitHub release, compares it with the running version, and
(when the user consents) downloads the new EXE and swaps it in-place via a
small batch helper that runs after the current process exits.

Public API
----------
fetch_latest_release()  – query GitHub; return (tag, exe_url) or None
is_update_available()   – return (tag, exe_url) if newer version exists
download_update()       – stream EXE to a local file with progress callbacks
apply_update()          – write replacement batch script and schedule restart

The module is intentionally free of PySide6 dependencies so it can be unit-
tested without a display.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from typing import Callable, Optional, Tuple

from svm_shaper import __version__

# CREATE_NO_WINDOW is only available on Windows; fall back to 0 elsewhere.
_CREATE_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_GITHUB_API = "https://api.github.com/repos/aminekhettat/SVM-analyst/releases/latest"
_EXE_ASSET = "svm-analyst.exe"
_USER_AGENT = "svm-analyst-updater"


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def _parse_version(v: str) -> tuple:
    """Convert *v* (``'v1.2.3'`` or ``'1.2.3'``) to a comparable tuple.

    Returns ``(0,)`` when the string cannot be parsed so comparisons still work.
    """
    try:
        return tuple(int(x) for x in v.lstrip("v").split("."))
    except ValueError:
        return (0,)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def fetch_latest_release(timeout: int = 8) -> Optional[Tuple[str, str]]:
    """Query the GitHub releases API for the latest release.

    Returns ``(tag_name, exe_download_url)`` when the latest release carries
    the ``svm-analyst.exe`` asset, otherwise ``None`` (network error, asset
    absent, or malformed response).
    """
    req = urllib.request.Request(
        _GITHUB_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": _USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    tag = data.get("tag_name", "")
    for asset in data.get("assets", []):
        if asset.get("name") == _EXE_ASSET:
            return tag, asset["browser_download_url"]
    return None


def is_update_available() -> Optional[Tuple[str, str]]:
    """Return ``(latest_version, download_url)`` when a newer release exists.

    Returns ``None`` when the running version is already the latest, when
    no ``svm-analyst.exe`` asset is found in the latest release, or when
    the network is unavailable.
    """
    result = fetch_latest_release()
    if result is None:
        return None
    latest, url = result
    if _parse_version(latest) > _parse_version(__version__):
        return latest, url
    return None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_update(
    url: str,
    dest_path: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Stream the asset at *url* to *dest_path*.

    Parameters
    ----------
    url:
        Direct download URL for the new EXE (from :func:`fetch_latest_release`).
    dest_path:
        Local file path where the download will be written.
    progress_callback:
        Optional callable ``(bytes_received, total_bytes)`` invoked after each
        chunk.  *total_bytes* is ``0`` when the server omits ``Content-Length``.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        received = 0
        with open(dest_path, "wb") as fh:
            while True:
                block = resp.read(65536)
                if not block:
                    break
                fh.write(block)
                received += len(block)
                if progress_callback is not None:
                    progress_callback(received, total)


# ---------------------------------------------------------------------------
# Apply (Windows EXE swap)
# ---------------------------------------------------------------------------


def apply_update(new_exe_path: str) -> None:
    """Replace the running EXE with *new_exe_path* and relaunch.

    Writes a temporary ``.bat`` file that:

    1. Polls ``tasklist`` until this process has exited.
    2. Moves the downloaded EXE over the current EXE (``move /Y``).
    3. Relaunches the new EXE.
    4. Deletes itself.

    Then spawns that batch file in a hidden ``cmd.exe`` window.

    Raises ``RuntimeError`` when not running as a frozen PyInstaller EXE
    (i.e. ``sys.frozen`` is not set), because replacement of a live Python
    interpreter is not safe.
    """
    if not getattr(sys, "frozen", False):
        raise RuntimeError(
            "apply_update() is only meaningful when running as a frozen EXE."
        )

    current_exe = sys.executable
    pid = os.getpid()

    bat_fd, bat_path = tempfile.mkstemp(suffix=".bat", prefix="svm_upd_")
    os.close(bat_fd)

    lines = [
        "@echo off",
        ":wait",
        f'tasklist /FI "PID eq {pid}" 2>NUL | find /I "{pid}" >NUL',
        "if not errorlevel 1 (",
        "    timeout /t 1 /nobreak >NUL",
        "    goto wait",
        ")",
        f'move /Y "{new_exe_path}" "{current_exe}"',
        f'start "" "{current_exe}"',
        'del "%~f0"',
    ]
    with open(bat_path, "w") as fh:
        fh.write("\r\n".join(lines))

    subprocess.Popen(
        ["cmd.exe", "/C", bat_path],
        creationflags=_CREATE_NO_WINDOW,
        close_fds=True,
    )
