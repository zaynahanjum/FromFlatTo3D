#!/usr/bin/env python3
"""
create_sample_tile.py - Generates an aerial cityscape packed with skyscrapers and buildings.
Completely replaces the park with a dense high-rise financial district.
Features 20+ distinct buildings with varied elevations, tiered spires, and sharp shadows.
"""

import os
from PIL import Image, ImageDraw


def generate_sample_aerial_tile(output_path: str = "data/raw/sample_city.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 768, 768

    # 1. Base Ground (asphalt and urban concrete terrain)
    img = Image.new("RGB", (width, height), color=(180, 178, 172))
    draw = ImageDraw.Draw(img)

    # 2. Comprehensive Multi-Lane Road Grid
    road_color = (48, 52, 58)
    line_color = (220, 210, 130)

    # Major Boulevards dividing the 4 quadrants
    draw.rectangle([0, 355, 768, 395], fill=road_color)
    draw.rectangle([355, 0, 395, 768], fill=road_color)

    # Secondary Avenues
    draw.rectangle([0, 170, 768, 195], fill=road_color)
    draw.rectangle([0, 560, 768, 585], fill=road_color)
    draw.rectangle([170, 0, 195, 768], fill=road_color)
    draw.rectangle([560, 0, 585, 768], fill=road_color)

    # Dashed/continuous center road markings
    draw.line([(0, 375), (768, 375)], fill=line_color, width=2)
    draw.line([(375, 0), (375, 768)], fill=line_color, width=2)
    draw.line([(0, 182), (768, 182)], fill=line_color, width=1)
    draw.line([(0, 572), (768, 572)], fill=line_color, width=1)
    draw.line([(182, 0), (182, 768)], fill=line_color, width=1)
    draw.line([(572, 0), (572, 768)], fill=line_color, width=1)

    # 3. Dense Skyscraper & Building Catalog
    # Format: (x, y, width, height, roof_color, shadow_length, list_of_tiers)
    # Tier format: (tier_w, tier_h, tier_color)
    buildings = [
        # --- NW QUADRANT: NEW FINANCIAL DOWNTOWN (formerly the park) ---
        # Supertall Central Spire Tower (Simulated 160m)
        (50, 45, 85, 95, (230, 235, 245), 32, [(60, 70, (240, 245, 255)), (35, 40, (255, 255, 255))]),
        # Twin Office Tower A (130m)
        (225, 45, 75, 90, (215, 225, 235), 26, [(50, 60, (230, 238, 248))]),
        # Twin Office Tower B (130m)
        (225, 225, 75, 90, (215, 225, 235), 26, [(50, 60, (230, 238, 248))]),
        # Banking Headquarters (110m)
        (50, 225, 90, 85, (225, 220, 215), 22, [(65, 60, (240, 235, 230))]),
        # Mid-rise Office Block (75m)
        (60, 150, 70, 45, (200, 205, 210), 16, []),
        # Commercial Corner Tower (90m)
        (230, 150, 65, 50, (210, 215, 220), 18, []),

        # --- NE QUADRANT: CORPORATE & TECH DISTRICT ---
        # Mega High-Rise Complex (150m)
        (425, 45, 95, 105, (235, 235, 245), 30, [(70, 80, (245, 248, 255)), (45, 50, (255, 255, 255))]),
        # Tech Headquarters (120m)
        (615, 50, 85, 90, (220, 230, 240), 24, [(55, 60, (235, 240, 250))]),
        # Glass Office Tower (105m)
        (425, 225, 90, 85, (210, 220, 230), 22, [(60, 55, (225, 235, 245))]),
        # High-Rise Hotel (115m)
        (615, 220, 85, 95, (225, 220, 215), 24, [(55, 65, (238, 232, 228))]),
        # Infill Commercial Block (70m)
        (435, 165, 70, 40, (195, 200, 205), 15, []),
        # Infill Commercial Block (70m)
        (625, 160, 65, 40, (195, 200, 205), 15, []),

        # --- SW QUADRANT: HIGH-DENSITY URBAN RESIDENTIAL & MIXED-USE ---
        # Residential High-Rise Tower 1 (95m)
        (45, 425, 65, 65, (220, 195, 185), 20, [(45, 45, (235, 210, 200))]),
        # Residential High-Rise Tower 2 (95m)
        (130, 425, 65, 65, (215, 190, 180), 20, [(45, 45, (230, 205, 195))]),
        # Residential High-Rise Tower 3 (95m)
        (215, 425, 65, 65, (225, 200, 190), 20, [(45, 45, (240, 215, 205))]),
        # Stepped Apartment Block A (65m)
        (45, 615, 60, 60, (205, 180, 170), 14, []),
        # Stepped Apartment Block B (65m)
        (130, 615, 60, 60, (210, 185, 175), 14, []),
        # Stepped Apartment Block C (65m)
        (215, 615, 60, 60, (200, 175, 165), 14, []),
        # Logistics & Service Depot (45m)
        (50, 510, 100, 50, (185, 160, 150), 10, []),
        (180, 510, 100, 50, (180, 155, 145), 10, []),

        # --- SE QUADRANT: GRAND CONVENTION & ENTERTAINMENT COMPLEX ---
        # Grand Exhibition & Convention Hall (80m, broad landmark footprint)
        (425, 425, 160, 150, (220, 225, 230), 24, [(110, 100, (235, 240, 245)), (60, 50, (250, 252, 255))]),
        # Plaza Hotel Tower (110m)
        (615, 425, 80, 85, (225, 215, 210), 22, [(50, 55, (238, 228, 220))]),
        # Entertainment Arena Block (70m)
        (425, 610, 110, 85, (205, 210, 215), 16, [(70, 55, (220, 225, 230))]),
        # Commercial Retail High-Rise (85m)
        (565, 610, 80, 85, (215, 210, 205), 18, [(50, 55, (230, 225, 220))]),
    ]

    # Draw all buildings with realistic drop shadows (cast south-east towards 135 deg)
    shadow_color = (40, 42, 46)

    for (bx, by, bw, bh, roof_col, s_len, tiers) in buildings:
        # 1. Base drop shadow
        draw.rectangle([bx + s_len, by + s_len, bx + bw + s_len, by + bh + s_len], fill=shadow_color)
        # 2. Main building base
        draw.rectangle([bx, by, bx + bw, by + bh], fill=roof_col, outline=(85, 90, 95), width=2)

        # 3. Tiered upper levels & spires
        for (tw, th, tcol) in tiers:
            tx = bx + (bw - tw) // 2
            ty = by + (bh - th) // 2
            tier_shadow = s_len // 2
            # Tier drop shadow on rooftop
            draw.rectangle([tx + tier_shadow, ty + tier_shadow, tx + tw + tier_shadow, ty + th + tier_shadow], fill=(55, 58, 62))
            # Tier rooftop
            draw.rectangle([tx, ty, tx + tw, ty + th], fill=tcol, outline=(100, 105, 110), width=1)

    img.save(output_path)
    print(f"[+] Successfully generated skyscraper metropolis: {output_path} ({width}x{height}, {len(buildings)} buildings)")
    return output_path


if __name__ == "__main__":
    generate_sample_aerial_tile()
