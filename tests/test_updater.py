"""Tests for svm_shaper.updater

Features covered
----------------
  - _parse_version              : version string parsing and comparison
  - fetch_latest_release        : GitHub API query (mocked HTTP)
  - is_update_available         : version comparison against running build
  - download_update             : file download with progress callbacks
  - apply_update                : frozen-EXE guard and batch script content
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from svm_shaper.updater import (
    _EXE_ASSET,
    _GITHUB_API,
    _parse_version,
    apply_update,
    download_update,
    fetch_latest_release,
    is_update_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http_response(payload: bytes, content_length: int | None = None):
    """Return a mock context-manager that mimics urllib.request.urlopen."""
    resp = MagicMock()
    # Simulate chunked reading: one full block then b"" to signal EOF.
    resp.read.side_effect = [payload, b""]
    headers: dict[str, str] = {}
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    resp.headers = headers
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _github_payload(tag: str, assets: list[dict]) -> bytes:
    return json.dumps({"tag_name": tag, "assets": assets}).encode("utf-8")


# ---------------------------------------------------------------------------
# _parse_version
# ---------------------------------------------------------------------------

class TestParseVersion(unittest.TestCase):

    def test_simple_triplet(self):
        self.assertEqual(_parse_version("1.2.3"), (1, 2, 3))

    def test_v_prefix_stripped(self):
        self.assertEqual(_parse_version("v2.0.0"), (2, 0, 0))

    def test_single_segment(self):
        self.assertEqual(_parse_version("5"), (5,))

    def test_two_segments(self):
        self.assertEqual(_parse_version("v1.10"), (1, 10))

    def test_invalid_string_returns_zero(self):
        self.assertEqual(_parse_version("invalid"), (0,))

    def test_empty_string_returns_zero(self):
        self.assertEqual(_parse_version(""), (0,))

    def test_comparison_newer(self):
        self.assertGreater(_parse_version("v1.2.0"), _parse_version("v1.1.9"))

    def test_comparison_older(self):
        self.assertLess(_parse_version("v1.0.0"), _parse_version("v1.0.1"))

    def test_comparison_equal(self):
        self.assertEqual(_parse_version("v1.1.4"), _parse_version("1.1.4"))


# ---------------------------------------------------------------------------
# fetch_latest_release
# ---------------------------------------------------------------------------

class TestFetchLatestRelease(unittest.TestCase):

    def test_returns_tag_and_exe_url(self):
        assets = [
            {"name": _EXE_ASSET, "browser_download_url": "https://cdn/svm-analyst.exe"},
        ]
        payload = _github_payload("v9.9.9", assets)
        with patch("urllib.request.urlopen", return_value=_make_http_response(payload)):
            result = fetch_latest_release()
        self.assertEqual(result, ("v9.9.9", "https://cdn/svm-analyst.exe"))

    def test_returns_none_when_exe_asset_absent(self):
        assets = [
            {"name": "SVM-Analyst-9.9.9.zip", "browser_download_url": "https://cdn/zip"},
        ]
        payload = _github_payload("v9.9.9", assets)
        with patch("urllib.request.urlopen", return_value=_make_http_response(payload)):
            result = fetch_latest_release()
        self.assertIsNone(result)

    def test_returns_none_on_url_error(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            result = fetch_latest_release()
        self.assertIsNone(result)

    def test_returns_none_on_generic_exception(self):
        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            result = fetch_latest_release()
        self.assertIsNone(result)

    def test_returns_none_on_malformed_json(self):
        resp = MagicMock()
        resp.read.return_value = b"not json {"
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp):
            result = fetch_latest_release()
        self.assertIsNone(result)

    def test_uses_correct_api_url(self):
        assets = [{"name": _EXE_ASSET, "browser_download_url": "https://cdn/svm.exe"}]
        payload = _github_payload("v1.0.0", assets)
        with patch("urllib.request.urlopen", return_value=_make_http_response(payload)) as mock_open:
            fetch_latest_release()
        req = mock_open.call_args[0][0]
        self.assertEqual(req.full_url, _GITHUB_API)

    def test_sends_accept_header(self):
        assets = [{"name": _EXE_ASSET, "browser_download_url": "https://cdn/svm.exe"}]
        payload = _github_payload("v1.0.0", assets)
        with patch("urllib.request.urlopen", return_value=_make_http_response(payload)) as mock_open:
            fetch_latest_release()
        req = mock_open.call_args[0][0]
        self.assertIn("application/vnd.github", req.headers.get("Accept", ""))


# ---------------------------------------------------------------------------
# is_update_available
# ---------------------------------------------------------------------------

class TestIsUpdateAvailable(unittest.TestCase):

    def test_newer_tag_returns_tuple(self):
        with patch("svm_shaper.updater.fetch_latest_release",
                   return_value=("v99.0.0", "https://cdn/svm.exe")):
            result = is_update_available()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "v99.0.0")
        self.assertEqual(result[1], "https://cdn/svm.exe")

    def test_same_version_returns_none(self):
        from svm_shaper import __version__

        with patch("svm_shaper.updater.fetch_latest_release",
                   return_value=(f"v{__version__}", "https://cdn/svm.exe")):
            result = is_update_available()
        self.assertIsNone(result)

    def test_older_tag_returns_none(self):
        with patch("svm_shaper.updater.fetch_latest_release",
                   return_value=("v0.0.1", "https://cdn/svm.exe")):
            result = is_update_available()
        self.assertIsNone(result)

    def test_network_failure_returns_none(self):
        with patch("svm_shaper.updater.fetch_latest_release", return_value=None):
            result = is_update_available()
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# download_update
# ---------------------------------------------------------------------------

class TestDownloadUpdate(unittest.TestCase):

    def test_file_content_matches_response(self):
        content = b"fake exe binary data"
        with patch("urllib.request.urlopen", return_value=_make_http_response(content)):
            fd, path = tempfile.mkstemp(suffix=".exe")
            os.close(fd)
            try:
                download_update("https://cdn/svm.exe", path)
                with open(path, "rb") as fh:
                    self.assertEqual(fh.read(), content)
            finally:
                os.unlink(path)

    def test_progress_callback_receives_bytes(self):
        content = b"x" * 200_000
        calls: list[tuple[int, int]] = []

        with patch("urllib.request.urlopen",
                   return_value=_make_http_response(content, len(content))):
            fd, path = tempfile.mkstemp(suffix=".exe")
            os.close(fd)
            try:
                download_update(
                    "https://cdn/svm.exe",
                    path,
                    progress_callback=lambda r, t: calls.append((r, t)),
                )
            finally:
                os.unlink(path)

        self.assertTrue(len(calls) > 0, "progress_callback was never called")
        received_total = calls[-1][0]
        self.assertEqual(received_total, len(content))

    def test_progress_callback_total_zero_when_no_content_length(self):
        """When server omits Content-Length, total_bytes passed to callback is 0."""
        content = b"data"
        calls: list[tuple[int, int]] = []

        with patch("urllib.request.urlopen",
                   return_value=_make_http_response(content)):  # no content_length arg
            fd, path = tempfile.mkstemp(suffix=".exe")
            os.close(fd)
            try:
                download_update(
                    "https://cdn/svm.exe",
                    path,
                    progress_callback=lambda r, t: calls.append((r, t)),
                )
            finally:
                os.unlink(path)

        self.assertTrue(all(t == 0 for _, t in calls))

    def test_no_callback_does_not_raise(self):
        content = b"binary"
        with patch("urllib.request.urlopen", return_value=_make_http_response(content)):
            fd, path = tempfile.mkstemp(suffix=".exe")
            os.close(fd)
            try:
                download_update("https://cdn/svm.exe", path)  # callback=None
            finally:
                os.unlink(path)


# ---------------------------------------------------------------------------
# apply_update
# ---------------------------------------------------------------------------

class TestApplyUpdate(unittest.TestCase):

    def test_raises_when_not_frozen(self):
        """apply_update must raise when sys.frozen is absent / False."""
        with patch.object(sys, "frozen", False, create=True):
            with self.assertRaises(RuntimeError):
                apply_update("/tmp/svm-analyst-new.exe")

    def test_raises_without_frozen_attribute(self):
        """apply_update must raise when running as a plain Python script."""
        # Remove sys.frozen entirely
        frozen_backup = getattr(sys, "frozen", None)
        if hasattr(sys, "frozen"):
            delattr(sys, "frozen")
        try:
            with self.assertRaises(RuntimeError):
                apply_update("/tmp/svm-analyst-new.exe")
        finally:
            if frozen_backup is not None:
                sys.frozen = frozen_backup  # type: ignore[attr-defined]

    def test_creates_bat_script_with_pid(self):
        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", r"C:\App\svm-analyst.exe"),
            patch("os.getpid", return_value=12345),
            patch("subprocess.Popen") as mock_popen,
        ):
            fd, new_exe = tempfile.mkstemp(suffix=".exe")
            os.close(fd)
            try:
                apply_update(new_exe)

                mock_popen.assert_called_once()
                cmd_args = mock_popen.call_args[0][0]
                self.assertEqual(cmd_args[0], "cmd.exe")
                self.assertEqual(cmd_args[1], "/C")

                bat_path = cmd_args[2]
                self.assertTrue(bat_path.endswith(".bat"))

                with open(bat_path) as fh:
                    bat_content = fh.read()

                self.assertIn("12345", bat_content)
                self.assertIn("svm-analyst.exe", bat_content)
                self.assertIn("move /Y", bat_content)
                self.assertIn("start", bat_content)

                os.unlink(bat_path)
            finally:
                if os.path.exists(new_exe):
                    os.unlink(new_exe)

    def test_bat_references_current_exe_path(self):
        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", r"C:\MyApp\svm-analyst.exe"),
            patch("os.getpid", return_value=99),
            patch("subprocess.Popen"),
        ):
            fd, new_exe = tempfile.mkstemp(suffix=".exe")
            os.close(fd)
            args_captured: list = []
            with patch("subprocess.Popen", side_effect=lambda a, **kw: args_captured.append(a)):
                try:
                    apply_update(new_exe)
                finally:
                    if os.path.exists(new_exe):
                        os.unlink(new_exe)

            bat_path = args_captured[0][2]
            with open(bat_path) as fh:
                content = fh.read()
            self.assertIn(r"C:\MyApp\svm-analyst.exe", content)
            os.unlink(bat_path)

    def test_popen_uses_create_no_window_flag(self):
        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", r"C:\App\svm-analyst.exe"),
            patch("os.getpid", return_value=1),
            patch("subprocess.Popen") as mock_popen,
        ):
            fd, new_exe = tempfile.mkstemp(suffix=".exe")
            os.close(fd)
            try:
                apply_update(new_exe)
            finally:
                if os.path.exists(new_exe):
                    os.unlink(new_exe)

            kwargs = mock_popen.call_args[1]
            self.assertEqual(
                kwargs.get("creationflags"), subprocess.CREATE_NO_WINDOW
            )

            # clean up bat
            bat_path = mock_popen.call_args[0][0][2]
            if os.path.exists(bat_path):
                os.unlink(bat_path)


if __name__ == "__main__":
    unittest.main()
