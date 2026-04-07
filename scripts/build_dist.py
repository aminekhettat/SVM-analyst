"""Automated distribution build script for SVM Analyst.

Usage
-----
    python scripts/build_dist.py

Steps
-----
  1. Read version from svm_shaper/__init__.py.
  2. Generate application icon (scripts/generate_icon.py).
  3. Regenerate the user manual PDF (scripts/generate_user_manual_pdf.py).
  4. Regenerate Sphinx HTML documentation (docs/ → docs/_build/).
  5. Run PyInstaller to build dist/svm-analyst.exe.
  6. Assemble dist-release/SVM-Analyst-{version}/ with all required files.
  7. Compute SHA-256 checksums for all distributed files.
  8. Create dist-release/SVM-Analyst-{version}.zip.
  9. Print a summary of the output.

Distribution folder layout
--------------------------
    SVM-Analyst-{version}/
        svm-analyst.exe
        README.txt
        CHANGELOG.txt
        LICENSE.txt
        THIRD_PARTY_NOTICES.txt
        checksums.sha256
        docs/
            SVM-Analyst-User-Manual-{version}.pdf
        examples/
            default_config.json
            sample_waveform.csv
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_version() -> str:
    """Read __version__ from svm_shaper/__init__.py."""
    text = (ROOT / "svm_shaper" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("Cannot read __version__ from svm_shaper/__init__.py")
    return m.group(1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], *, step: str, env: dict | None = None) -> None:
    """Run a subprocess command, raising RuntimeError on failure."""
    merged_env = {**os.environ, **(env or {})}
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, env=merged_env)
    if result.returncode != 0:
        raise RuntimeError(f"{step} failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


def step_generate_icon() -> None:
    """Step 2: Generate svm-analyst.ico via scripts/generate_icon.py."""
    print("[2/8] Generating application icon ...")
    icon_script = ROOT / "scripts" / "generate_icon.py"
    _run([sys.executable, str(icon_script)], step="Icon generation")


def step_generate_user_manual() -> None:
    """Step 3: Regenerate the user manual PDF."""
    print("[3/8] Generating user manual PDF ...")
    manual_script = ROOT / "scripts" / "generate_user_manual_pdf.py"
    _run([sys.executable, str(manual_script)], step="User manual generation")


def step_generate_sphinx_docs() -> None:
    """Step 4: Regenerate Sphinx HTML documentation."""
    print("[4/8] Regenerating Sphinx HTML documentation ...")
    docs_src = str(ROOT / "docs")
    docs_build = str(ROOT / "docs" / "_build")
    _run(
        [sys.executable, "-m", "sphinx", docs_src, docs_build, "-b", "html", "-q"],
        step="Sphinx build",
    )
    print(f"  Output: {docs_build}")


def step_run_pyinstaller() -> None:
    """Step 5: Build the executable with PyInstaller."""
    print("[5/8] Running PyInstaller ...")
    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "svm-analyst.spec",
            "--clean",
            "--noconfirm",
        ],
        step="PyInstaller",
        env={"PYQTGRAPH_QT_LIB": "PySide6", "QT_API": "PySide6"},
    )
    exe = ROOT / "dist" / "svm-analyst.exe"
    if not exe.exists():
        raise RuntimeError(f"Expected executable not found: {exe}")
    size_mb = exe.stat().st_size / 1_048_576
    print(f"  Built:  {exe}  ({size_mb:.1f} MB)")


def _readme_text(version: str) -> str:
    return (
        f"SVM Analyst {version}\n"
        f"{'=' * (len('SVM Analyst ') + len(version))}\n"
        "Educational simulator for PWM and space-vector modulation (SVM).\n"
        "Designed for three-phase inverter analysis and harmonic comparison.\n"
        "\n"
        "QUICK START\n"
        "-----------\n"
        "1. Double-click svm-analyst.exe to launch the application.\n"
        "2. No installation or Python runtime required.\n"
        "3. Select a modulation method from the list on the left.\n"
        "4. Set machine and inverter parameters (PWM frequency, speed, voltage, etc.).\n"
        "5. Use the oscilloscope controls to pause, step, or export waveforms.\n"
        f"6. Open docs\\SVM-Analyst-User-Manual-{version}.pdf for full documentation.\n"
        "\n"
        "ACCESSIBILITY\n"
        "-------------\n"
        "All controls have accessible names and descriptions compatible with screen\n"
        "readers (NVDA, JAWS). Use Tab / Shift+Tab to navigate, arrow keys to adjust\n"
        "numeric controls, and Space to toggle buttons and checkboxes.\n"
        "\n"
        "REQUIREMENTS\n"
        "------------\n"
        "- Windows 10 or Windows 11 (64-bit).\n"
        "- Display resolution of at least 1280 x 720.\n"
        "- No Python installation required.\n"
        "\n"
        "KNOWN LIMITATIONS\n"
        "-----------------\n"
        "- The first launch may take a few seconds while libraries initialise.\n"
        "- Windows SmartScreen may prompt on first run; allow execution if appropriate.\n"
        "\n"
        "FILES INCLUDED\n"
        "--------------\n"
        "  svm-analyst.exe           Application executable\n"
        f"  docs/SVM-Analyst-User-Manual-{version}.pdf\n"
        "                            Full user manual\n"
        "  examples/default_config.json   Ready-to-load SVM configuration\n"
        "  examples/sample_waveform.csv   Sample exported waveform\n"
        "  CHANGELOG.txt             Version history\n"
        "  LICENSE.txt               Redistribution terms\n"
        "  THIRD_PARTY_NOTICES.txt   Open-source licences of bundled libraries\n"
        "  checksums.sha256          SHA-256 hashes for integrity verification\n"
        "\n"
        "LICENCE\n"
        "-------\n"
        "See LICENSE.txt for redistribution terms.\n"
        "Third-party library licences are listed in THIRD_PARTY_NOTICES.txt.\n"
        "\n"
        "Author:  Amine KHETTAT\n"
        "Project: https://github.com/aminekhettat/SVM-analyst\n"
    )


def _third_party_notices() -> str:
    # (name, default_version, license, author_url)
    packages = [
        ("PySide6", "LGPL-3.0 / GPL-2.0 / GPL-3.0", "Qt Group (https://www.qt.io/)"),
        ("numpy", "BSD 3-Clause", "NumPy Developers (https://numpy.org/)"),
        (
            "matplotlib",
            "PSF-compatible (matplotlib License)",
            "Matplotlib Development Team (https://matplotlib.org/)",
        ),
        (
            "pyqtgraph",
            "MIT License",
            "Luke Campagnola et al. (https://pyqtgraph.readthedocs.io/)",
        ),
        ("PyPDF2", "BSD 3-Clause", "Mathieu Fenniak et al."),
        (
            "Pillow",
            "HPND (MIT-compatible)",
            "Jeffrey A. Clark (Alex) et al. (https://python-pillow.org/)",
        ),
        ("scipy", "BSD 3-Clause", "SciPy Developers (https://scipy.org/)"),
    ]
    sep = "=" * 60
    lines = [
        "Third-Party Software Notices",
        sep,
        "",
        "SVM Analyst bundles the following open-source packages.",
        "Full licence texts are available in the source repository:",
        "  https://github.com/aminekhettat/SVM-analyst",
        "",
    ]
    for name, lic, author in packages:
        try:
            ver = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            ver = "unknown"
        lines += [
            f"Package : {name} {ver}",
            f"Licence : {lic}",
            f"Author  : {author}",
            "",
        ]
    lines += [
        sep,
        "IMPORTANT NOTICE – PySide6 / Qt",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "PySide6 is distributed under the LGPL 3.0 licence (or GPL).",
        "A copy of the LGPL 3.0 is available at:",
        "  https://www.gnu.org/licenses/lgpl-3.0.html",
        "Qt is a registered trademark of The Qt Company Ltd.",
        "",
    ]
    return "\n".join(lines)


def step_assemble_distribution(version: str) -> Path:
    """Step 6: Create the distribution folder."""
    print("[6/8] Assembling distribution folder ...")
    dist_dir = ROOT / "dist-release" / f"SVM-Analyst-{version}"
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True)

    # Executable
    shutil.copy2(ROOT / "dist" / "svm-analyst.exe", dist_dir / "svm-analyst.exe")
    print("  Copied: svm-analyst.exe")

    # User manual PDF
    docs_dest = dist_dir / "docs"
    docs_dest.mkdir()
    manual_name = f"SVM-Analyst-User-Manual-{version}.pdf"
    manual_src = ROOT / "docs" / manual_name
    if manual_src.exists():
        shutil.copy2(manual_src, docs_dest / manual_name)
        print(f"  Copied: docs/{manual_name}")
    else:
        print(f"  WARNING: user manual not found: {manual_src}")

    # License
    license_src = ROOT / "LICENSE"
    if license_src.exists():
        shutil.copy2(license_src, dist_dir / "LICENSE.txt")
        print("  Copied: LICENSE.txt")

    # Changelog
    changelog_src = ROOT / "CHANGELOG.txt"
    if changelog_src.exists():
        shutil.copy2(changelog_src, dist_dir / "CHANGELOG.txt")
        print("  Copied: CHANGELOG.txt")

    # Generated text files
    (dist_dir / "README.txt").write_text(_readme_text(version), encoding="utf-8")
    print("  Generated: README.txt")

    (dist_dir / "THIRD_PARTY_NOTICES.txt").write_text(
        _third_party_notices(), encoding="utf-8"
    )
    print("  Generated: THIRD_PARTY_NOTICES.txt")

    # Examples
    examples_dest = dist_dir / "examples"
    examples_dest.mkdir()
    examples_src = ROOT / "examples"
    if examples_src.exists():
        for src in sorted(examples_src.glob("*")):
            if src.is_file():
                shutil.copy2(src, examples_dest / src.name)
                print(f"  Copied: examples/{src.name}")

    return dist_dir


def step_checksums(dist_dir: Path) -> None:
    """Step 7: Compute SHA-256 checksums for all files in the distribution folder."""
    print("[7/8] Computing SHA-256 checksums ...")
    lines = [
        f"# SVM Analyst {dist_dir.name}  –  SHA-256 checksums",
        "",
    ]
    for f in sorted(dist_dir.rglob("*")):
        if f.is_file() and f.name != "checksums.sha256":
            rel = f.relative_to(dist_dir).as_posix()
            lines.append(f"{_sha256(f)}  {rel}")
    lines.append("")
    (dist_dir / "checksums.sha256").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {dist_dir / 'checksums.sha256'}")


def step_zip(dist_dir: Path, version: str) -> Path:
    """Step 8: Create the ZIP archive."""
    print("[8/8] Creating ZIP archive ...")
    zip_path = ROOT / "dist-release" / f"SVM-Analyst-{version}.zip"
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for f in sorted(dist_dir.rglob("*")):
            if f.is_file():
                arcname = Path(f"SVM-Analyst-{version}") / f.relative_to(dist_dir)
                zf.write(f, arcname)
    size_mb = zip_path.stat().st_size / 1_048_576
    print(f"  Archive: {zip_path}  ({size_mb:.1f} MB)")
    return zip_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full distribution build pipeline."""
    banner = "SVM Analyst  –  Distribution Build Script"
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))
    print()

    version = _read_version()
    print(f"[1/8] Version: {version}")
    print()

    step_generate_icon()
    step_generate_user_manual()
    step_generate_sphinx_docs()
    step_run_pyinstaller()
    dist_dir = step_assemble_distribution(version)
    step_checksums(dist_dir)
    zip_path = step_zip(dist_dir, version)

    total_files = sum(1 for f in dist_dir.rglob("*") if f.is_file())
    total_mb = (
        sum(f.stat().st_size for f in dist_dir.rglob("*") if f.is_file()) / 1_048_576
    )

    print()
    print("=" * len(banner))
    print("Build complete.")
    print(f"  Release folder : {dist_dir}")
    print(f"  ZIP archive    : {zip_path}")
    print(f"  Files          : {total_files}")
    print(f"  Uncompressed   : {total_mb:.1f} MB")
    print("=" * len(banner))


if __name__ == "__main__":
    main()
