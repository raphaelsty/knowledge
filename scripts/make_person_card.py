"""
Render a 280x382 personality "card" PNG matching the existing template in
`web/img/people/`: transparent background, circular avatar at the top, bold
name in slate-900, gray subtitle in slate-500.

Template parameters were recovered by sampling the existing iacopo-poli.png
(see commit history). Anything not specified here is intentionally fixed so
new cards drop in next to the existing 60+ without visible drift.

Usage:
    uv run scripts/make_person_card.py --slug iacopo-poli \
        --name "Iacopo Poli" --role "LightOn · CTO" \
        --avatar-from-existing web/img/people/iacopo-poli.png

    uv run scripts/make_person_card.py --slug amelie-chatelain \
        --name "Amélie Chatelain" --role "LightOn · search & retrieval" \
        --avatar-url https://avatars.githubusercontent.com/u/58592892?v=4

The output is always `web/img/people/<slug>.png` (280x382 RGBA).
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CANVAS_W, CANVAS_H = 280, 382

# Avatar geometry: a 240 px circle centered horizontally, top edge at y=12.
AVATAR_CX, AVATAR_CY = 140, 132
AVATAR_RADIUS = 120

# Text colours sampled from the existing cards.
COLOR_NAME = (15, 23, 42, 255)  # slate-900
COLOR_SUBTITLE = (100, 116, 139, 255)  # slate-500

# Baselines tuned to land on the same rows as the existing PNGs.
NAME_BASELINE_Y = 295
SUBTITLE_BASELINE_Y = 335


def _load_font(
    candidates: list[tuple[str, int, int]],
) -> ImageFont.FreeTypeFont:
    """Try a list of (path, size, index) candidates in order; return the first that loads.

    `index` selects the face inside a .ttc collection (Helvetica.ttc / HelveticaNeue.ttc
    bundle the Regular / Bold / Italic / … variants at different indices). 0 for .ttf.
    """
    for path, size, index in candidates:
        try:
            return ImageFont.truetype(path, size, index=index)
        except (OSError, IndexError):
            continue
    raise RuntimeError(f"No usable font found from {candidates!r}")


def name_font() -> ImageFont.FreeTypeFont:
    return _load_font(
        [
            ("/System/Library/Fonts/HelveticaNeue.ttc", 22, 1),  # Helvetica Neue Bold
            ("/System/Library/Fonts/Helvetica.ttc", 22, 1),  # Helvetica Bold
            ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22, 0),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22, 0),
        ]
    )


def subtitle_font() -> ImageFont.FreeTypeFont:
    return _load_font(
        [
            ("/System/Library/Fonts/HelveticaNeue.ttc", 16, 0),  # Helvetica Neue Regular
            ("/System/Library/Fonts/Helvetica.ttc", 16, 0),  # Helvetica Regular
            ("/System/Library/Fonts/Supplemental/Arial.ttf", 16, 0),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16, 0),
        ]
    )


def _fetch_url(url: str) -> bytes:
    """Plain GET with a UA — GitHub returns 403 for the default Python UA."""
    req = urllib.request.Request(url, headers={"User-Agent": "knowledge-card-generator/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read()


def load_avatar(args: argparse.Namespace) -> Image.Image:
    """
    Return a square RGBA image suitable for circular masking. When pulling
    from an existing card we just extract the previously-masked circular
    region (so re-running on iacopo-poli.png preserves the same photo).
    """
    if args.avatar_url:
        data = _fetch_url(args.avatar_url)
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        # Crop to centered square then resize to the avatar diameter.
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        return img.resize((2 * AVATAR_RADIUS, 2 * AVATAR_RADIUS), Image.LANCZOS)

    if args.avatar_from_existing:
        src = Image.open(args.avatar_from_existing).convert("RGBA")
        # The template puts the circle at (AVATAR_CX, AVATAR_CY) with
        # AVATAR_RADIUS — re-crop that exact region.
        bbox = (
            AVATAR_CX - AVATAR_RADIUS,
            AVATAR_CY - AVATAR_RADIUS,
            AVATAR_CX + AVATAR_RADIUS,
            AVATAR_CY + AVATAR_RADIUS,
        )
        return src.crop(bbox)

    if args.avatar_file:
        img = Image.open(args.avatar_file).convert("RGBA")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        return img.resize((2 * AVATAR_RADIUS, 2 * AVATAR_RADIUS), Image.LANCZOS)

    raise SystemExit("Need one of --avatar-url / --avatar-from-existing / --avatar-file")


def render(args: argparse.Namespace) -> Image.Image:
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255, 255, 255, 0))

    # --- avatar (circular mask) ---
    avatar = load_avatar(args)
    mask = Image.new("L", avatar.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, avatar.size[0], avatar.size[1]), fill=255)
    # Antialias the mask edge so the circle doesn't look stepped.
    mask_aa = mask.resize((avatar.size[0] * 2, avatar.size[1] * 2), Image.LANCZOS).resize(avatar.size, Image.LANCZOS)
    canvas.paste(avatar, (AVATAR_CX - AVATAR_RADIUS, AVATAR_CY - AVATAR_RADIUS), mask_aa)

    # --- text ---
    draw = ImageDraw.Draw(canvas)
    nf = name_font()
    sf = subtitle_font()

    def draw_centered(y_baseline: int, text: str, font: ImageFont.FreeTypeFont, color):
        # Use anchor="ms" so y is the baseline and x is the horizontal center.
        draw.text((CANVAS_W // 2, y_baseline), text, font=font, fill=color, anchor="ms")

    draw_centered(NAME_BASELINE_Y, args.name, nf, COLOR_NAME)
    draw_centered(SUBTITLE_BASELINE_Y, args.role, sf, COLOR_SUBTITLE)

    return canvas


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True, help="Output filename stem (no .png)")
    p.add_argument("--name", required=True, help='Bold name line, e.g. "Iacopo Poli"')
    p.add_argument("--role", required=True, help='Subtitle, e.g. "LightOn · CTO"')
    p.add_argument("--avatar-url", help="Avatar URL to download (square crop applied)")
    p.add_argument(
        "--avatar-from-existing",
        help="Reuse the circular avatar already baked into another card PNG",
    )
    p.add_argument("--avatar-file", help="Local image file (square crop applied)")
    p.add_argument(
        "--out-dir",
        default="web/img/people",
        help="Output directory (default: web/img/people)",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.slug}.png"

    img = render(args)
    img.save(out_path, "PNG")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
