"""Generates synthetic test imagery for different terrain landscapes (Hills & Coast).

Allows testing the single-view depth and slope pipeline across:
1. Mountainous terrain (slopes, ridges, valleys).
2. Coastal island (water body masking, plateau elevation).
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_mountain_sample(size: int = 768) -> Image.Image:
    """Creates a realistic aerial mountain ridge tile."""
    img = Image.new("RGB", (size, size), (95, 90, 80))
    draw = ImageDraw.Draw(img)

    # Base noise terrain
    np.random.seed(42)
    noise = np.random.normal(0, 15, (size, size))
    base_arr = np.array(img, dtype=np.float32)
    base_arr += noise[:, :, None]
    base_arr = np.clip(base_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(base_arr)
    draw = ImageDraw.Draw(img)

    # Mountain ridges (high-contrast peaks with sunlit faces and shadows)
    ridges = [
        ([(80, 120), (220, 200), (380, 240), (520, 360), (680, 480)], 45, (160, 155, 140), (50, 48, 42)),
        ([(150, 450), (320, 520), (460, 480), (620, 620)], 35, (145, 140, 128), (42, 40, 36)),
        ([(350, 80), (480, 140), (620, 180)], 30, (170, 165, 150), (55, 52, 46)),
    ]

    for points, width, lit_col, shad_col in ridges:
        # Shadow side (south-east)
        for offset in range(width, 0, -4):
            shifted = [(p[0] + offset, p[1] + offset) for p in points]
            draw.line(shifted, fill=shad_col, width=int(offset * 1.5))
        # Lit side (north-west)
        for offset in range(width, 0, -4):
            shifted = [(p[0] - offset // 2, p[1] - offset // 2) for p in points]
            draw.line(shifted, fill=lit_col, width=int(offset * 1.2))
        # Sharp crest
        draw.line(points, fill=(210, 205, 195), width=3)

    # Blur for natural geological erosion
    img = img.filter(ImageFilter.GaussianBlur(radius=3.5))

    # Add green valley vegetation in lowlands
    arr = np.array(img, dtype=np.int16)
    lowland = (arr[:, :, 0] < 110) & (arr[:, :, 1] < 110)
    arr[lowland, 1] += 25  # Boost green in valleys
    arr[lowland, 0] -= 10
    arr[lowland, 2] -= 10
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def create_coastal_sample(size: int = 768) -> Image.Image:
    """Creates a coastal island surrounded by ocean."""
    # Deep blue ocean
    img = Image.new("RGB", (size, size), (22, 68, 140))
    draw = ImageDraw.Draw(img)

    # Shallow water reef turquoise rim
    center = (size // 2, size // 2)
    draw.ellipse([140, 140, 628, 628], fill=(35, 125, 175))

    # Sandy beaches
    draw.ellipse([180, 180, 588, 588], fill=(215, 198, 150))

    # Island plateau with dense greenery
    draw.ellipse([220, 220, 548, 548], fill=(42, 115, 52))

    # Interior hills
    draw.ellipse([280, 280, 488, 488], fill=(85, 130, 65))
    draw.ellipse([340, 330, 430, 440], fill=(130, 120, 95))

    img = img.filter(ImageFilter.GaussianBlur(radius=6.0))
    return img


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    mtn = create_mountain_sample(768)
    mtn_path = raw_dir / "mountain_ridge.png"
    mtn.save(mtn_path)
    print(f"Created: {mtn_path}")

    coast = create_coastal_sample(768)
    coast_path = raw_dir / "coastal_island.png"
    coast.save(coast_path)
    print(f"Created: {coast_path}")


if __name__ == "__main__":
    main()
