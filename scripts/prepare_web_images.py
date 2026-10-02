"""Prepare the site's images from the source art in `assets/img/`.

- `banner.png` is a black silhouette on white. White is keyed out
  (alpha = 255 minus brightness), so the page's red glow shows through the
  sky and the outlines in the crowd. It's saved as WebP with lossy alpha:
  45 KB instead of 241 KB lossless, with alpha off by under 1% on average
  (5% at most, on edge pixels).
- `zoroaster_1.png` has semi-transparent silhouettes and a glowing horizon
  over a transparent sky. It's flattened onto black, the night scene it reads
  as, and saved as lossy WebP at display size.

The originals stay in `assets/`, which isn't published. `web/img/` gets the
small versions that the site uses.

Usage:
    python scripts/prepare_web_images.py   (needs Pillow)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageOps

from scripts.build_site import WEB_DIR

ASSETS = Path("assets/img")
OUT = WEB_DIR / "img"
VISION_WIDTH = 1200


def key_out_white(image: Image.Image) -> Image.Image:
    """Black ink whose opacity is each pixel's darkness, so white becomes transparent."""
    gray = image.convert("L")
    ink = Image.new("L", gray.size, 0)
    return Image.merge("RGBA", (ink, ink, ink, ImageOps.invert(gray)))


def flatten(image: Image.Image, color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """The image over a solid color, without transparency."""
    rgba = image.convert("RGBA")
    base = Image.new("RGB", rgba.size, color)
    base.paste(rgba, mask=rgba.getchannel("A"))
    return base


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    banner = key_out_white(Image.open(ASSETS / "banner.png"))
    banner.save(OUT / "banner.webp", quality=85, alpha_quality=60, method=6)
    vision = flatten(Image.open(ASSETS / "zoroaster_1.png"))
    vision = vision.resize((VISION_WIDTH, round(vision.height * VISION_WIDTH / vision.width)), Image.LANCZOS)
    vision.save(OUT / "zoroaster_1.webp", quality=82, method=6)
    for path in sorted(OUT.glob("*.webp")):
        with Image.open(path) as im:
            print(f"{path}: {im.size[0]}x{im.size[1]}, {path.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
