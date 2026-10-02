"""Prepare the site's images from the source art in `assets/img/`.

- `banner.png` is a black silhouette on white. White is keyed out
  (alpha = 255 minus brightness), so the page's red glow shows through the
  sky and the outlines in the crowd. It's saved as WebP with lossy alpha:
  45 KB instead of 241 KB lossless, with alpha off by under 1% on average
  (5% at most, on edge pixels).
- `zoroaster_1.png` has semi-transparent silhouettes and a glowing horizon
  over a transparent sky. It's flattened onto black, the night scene it reads
  as, and saved as lossy WebP at display size.
- `faravahar_render.png` is `Faravahar.svg` from Wikimedia Commons (CC BY-SA
  3.0, Ploxhoi and Kevin McCormick), rendered once with headless Edge at 3x
  size on a transparent background. The SVG itself is 171 KB.
  - Its white parts become opaque and its black lines transparent
    (`light_mask`), so CSS can paint it in any color, with the stone showing
    through the lines. Saved as WebP with alpha.
  - The favicon is the same symbol in the site's red, on a transparent
    square.

The originals stay in `assets/`, which isn't published. `web/img/` gets the
small versions that the site uses.

Usage:
    python scripts/prepare_web_images.py   (needs Pillow)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageOps

from scripts.build_site import WEB_DIR

ASSETS = Path("assets/img")
OUT = WEB_DIR / "img"
VISION_WIDTH = 1200
EMBLEM_WIDTH = 480
FAVICON_SIZE = 64
RED = (212, 65, 46)  # --red in web/style.css


def key_out_white(image: Image.Image) -> Image.Image:
    """Black ink whose opacity is each pixel's darkness, so white becomes transparent."""
    gray = image.convert("L")
    ink = Image.new("L", gray.size, 0)
    return Image.merge("RGBA", (ink, ink, ink, ImageOps.invert(gray)))


def light_mask(image: Image.Image) -> Image.Image:
    """White ink whose opacity is each pixel's lightness times its alpha, so
    white parts stay and black lines and the background become transparent."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float64)
    out = np.full(rgba.shape, 255, dtype=np.uint8)
    out[..., 3] = np.round(rgba[..., :3].mean(axis=2) * rgba[..., 3] / 255)
    return Image.fromarray(out, "RGBA")


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
    emblem = light_mask(Image.open(ASSETS / "faravahar_render.png"))
    emblem = emblem.crop(emblem.getbbox())
    emblem = emblem.resize((EMBLEM_WIDTH, round(emblem.height * EMBLEM_WIDTH / emblem.width)), Image.LANCZOS)
    emblem.save(OUT / "faravahar.webp", quality=90, alpha_quality=90, method=6)
    icon = Image.new("RGBA", (FAVICON_SIZE, FAVICON_SIZE), (0, 0, 0, 0))
    small = emblem.resize((FAVICON_SIZE, round(emblem.height * FAVICON_SIZE / emblem.width)), Image.LANCZOS)
    red = Image.new("RGBA", small.size, RED + (255,))
    red.putalpha(small.getchannel("A"))
    icon.alpha_composite(red, (0, (FAVICON_SIZE - small.height) // 2))
    icon.save(OUT / "favicon.png", optimize=True)
    for path in sorted([*OUT.glob("*.webp"), *OUT.glob("*.png")]):
        with Image.open(path) as im:
            print(f"{path}: {im.size[0]}x{im.size[1]}, {path.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
