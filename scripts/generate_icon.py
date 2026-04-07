"""Generate the SVM Analyst application icon.

Produces ``assets/svm-analyst.ico`` (multi-size: 16, 24, 32, 48, 64, 128, 256)
and ``assets/svm-analyst-256.png`` (reference PNG at full resolution).

Design:
  - Dark navy blue rounded-square background (matches Blind Systems branding).
  - SVM hexagon outline in bright cyan.
  - Three-phase sinusoidal reference curves in the center (red / green / blue).
  - Bold "SA" initials anchored to the bottom-right corner.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
OUTPUT_ICO = ASSETS_DIR / "svm-analyst.ico"
OUTPUT_PNG = ASSETS_DIR / "svm-analyst-256.png"

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------
_BG_COLOR = (13, 27, 42)  # dark navy   #0D1B2A
_HEX_COLOR = (0, 188, 212)  # cyan        #00BCD4
_RING_COLOR = (21, 101, 192)  # blue        #1565C0
_PH_A_COLOR = (255, 107, 107)  # coral red
_PH_B_COLOR = (106, 220, 106)  # lime green
_PH_C_COLOR = (116, 185, 255)  # sky blue
_TEXT_COLOR = (255, 255, 255)  # white


def _hex_vertices(
    cx: float, cy: float, r: float, offset_deg: float = 0.0
) -> list[tuple[float, float]]:
    """Return six vertices of a regular hexagon centred at (cx, cy) with radius r."""
    pts: list[tuple[float, float]] = []
    for i in range(6):
        a = math.radians(offset_deg + i * 60)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _draw_icon(size: int) -> Image.Image:
    """Render a single icon image at *size* × *size* pixels."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    s = float(size)
    cx = cy = s / 2.0

    # --- Rounded-square background -------------------------------------------
    corner_r = s * 0.20
    draw.rounded_rectangle(
        [0, 0, size - 1, size - 1],
        radius=corner_r,
        fill=(*_BG_COLOR, 255),
    )

    # --- Subtle outer ring (depth cue) ----------------------------------------
    ring_r = s * 0.44
    ring_w = max(1, round(s * 0.025))
    draw.ellipse(
        [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
        outline=(*_RING_COLOR, 160),
        width=ring_w,
    )

    # --- SVM Hexagon (drawn as individual edges for a clean thick outline) -----
    hex_r = s * 0.37
    hex_pts = _hex_vertices(cx, cy, hex_r, offset_deg=0.0)  # flat-top (0° = right)
    hex_w = max(1, round(s * 0.045))
    for i in range(6):
        p1 = hex_pts[i]
        p2 = hex_pts[(i + 1) % 6]
        draw.line([p1, p2], fill=(*_HEX_COLOR, 255), width=hex_w)

    # --- Three-phase sinusoidal waveforms (only for sizes ≥ 32 px) ------------
    if size >= 32:
        n = 300
        t = np.linspace(0, 1, n)
        wave_amp = s * 0.20  # peak-to-peak half-height in pixels
        x_start = cx - s * 0.30
        x_end = cx + s * 0.30
        wave_w = max(1, round(s * 0.025))

        for phase_deg, color in (
            (0, _PH_A_COLOR),
            (120, _PH_B_COLOR),
            (240, _PH_C_COLOR),
        ):
            vals = np.sin(2 * math.pi * t + math.radians(phase_deg))
            xs = x_start + t * (x_end - x_start)
            ys = cy - vals * wave_amp
            pts_list: list[tuple[float, float]] = [
                (float(x), float(y)) for x, y in zip(xs, ys)
            ]
            if len(pts_list) >= 2:
                draw.line(pts_list, fill=(*color, 200), width=wave_w)

    # --- Central dot (focal point) --------------------------------------------
    dot_r = max(2, round(s * 0.045))
    draw.ellipse(
        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
        fill=(*_HEX_COLOR, 255),
    )

    # --- "SA" initials (only for sizes ≥ 48 px) -------------------------------
    if size >= 48:
        font_size = max(8, round(s * 0.22))
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont
        for font_path in (
            "arialbd.ttf",
            "arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except OSError:
                pass
        else:
            font = ImageFont.load_default()

        text = "SA"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        margin = max(2, round(s * 0.05))
        tx = size - tw - margin
        ty = size - th - margin
        # Drop-shadow for legibility on any background
        draw.text((tx + 1, ty + 1), text, font=font, fill=(0, 0, 0, 180))
        draw.text((tx, ty), text, font=font, fill=(*_TEXT_COLOR, 255))

    return img


def generate_icon(output_ico: Path, output_png: Path | None = None) -> None:
    """Generate the .ico file at *output_ico* and optionally a PNG at *output_png*."""
    output_ico.parent.mkdir(parents=True, exist_ok=True)

    sizes: Sequence[int] = (256, 128, 64, 48, 32, 24, 16)
    images = [_draw_icon(s) for s in sizes]

    # Pillow's ICO format requires RGBA → RGBA preserved; append_images for multi-size
    images[0].save(
        output_ico,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=images[1:],
    )
    ico_kb = output_ico.stat().st_size // 1024
    print(f"Generated icon: {output_ico}  ({ico_kb} KB, {len(sizes)} sizes)")

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(output_png, format="PNG")
        print(f"Generated PNG:  {output_png}")


def main() -> None:
    """Entry-point: generate icon and reference PNG."""
    generate_icon(OUTPUT_ICO, OUTPUT_PNG)


if __name__ == "__main__":
    main()
