#!/usr/bin/env python3
"""
Advanced DNA Helix Background Generator (Improved Version)
Usage: python generate_dna_bg.py
"""

import os
import sys

def generate_dna_background(filename='static/DNAimage.jpg'):
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import math
        import random
    except ImportError:
        print("\n⚠️ Installing Pillow...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow', '-q'])
        from PIL import Image, ImageDraw, ImageFilter
        import math
        import random

    width, height = 1400, 900  # Increased resolution

    print("📐 Creating canvas...")
    img = Image.new('RGB', (width, height), color=(10, 30, 60))
    draw = ImageDraw.Draw(img, 'RGBA')

    # 🌈 Smooth gradient (Sky Blue → Deep Blue)
    print("🎨 Creating gradient...")
    for y in range(height):
        ratio = y / height
        r = int(10 + 20 * ratio)
        g = int(80 + 100 * ratio)
        b = int(150 + 80 * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    # ✨ Add particles (background stars)
    print("✨ Adding particles...")
    for _ in range(200):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.ellipse((x, y, x+2, y+2), fill=(255, 255, 255, 120))

    # 🧬 DNA parameters
    center_x = width // 2
    amplitude = 120
    freq = 0.015

    left_strand = []
    right_strand = []

    print("🧬 Calculating helix...")
    for y in range(0, height, 2):
        x1 = center_x + amplitude * math.cos(y * freq)
        x2 = center_x + amplitude * math.cos(y * freq + math.pi)
        left_strand.append((int(x1), y))
        right_strand.append((int(x2), y))

    # 🌟 Draw strand function (multi-layer glow)
    def draw_glow_line(points, colors):
        for i in range(len(points)-1):
            x1, y1 = points[i]
            x2, y2 = points[i+1]

            # Outer glow
            draw.line([(x1, y1), (x2, y2)], fill=colors[0], width=14)
            # Mid glow
            draw.line([(x1, y1), (x2, y2)], fill=colors[1], width=8)
            # Core
            draw.line([(x1, y1), (x2, y2)], fill=colors[2], width=3)

    print("✨ Drawing strands...")
    draw_glow_line(left_strand, [
        (0, 255, 255, 80),     # cyan glow
        (0, 200, 255, 160),
        (255, 255, 255, 220)
    ])

    draw_glow_line(right_strand, [
        (180, 0, 255, 80),     # violet glow
        (220, 120, 255, 160),
        (255, 255, 255, 220)
    ])

    # 🔗 Draw base-pair connections
    print("🔗 Drawing base pairs...")
    for i in range(0, len(left_strand), 10):
        x1, y1 = left_strand[i]
        x2, y2 = right_strand[i]

        color = random.choice([
            (255, 100, 100, 180),   # A-T
            (100, 255, 100, 180),   # G-C
            (100, 200, 255, 180)
        ])

        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

    # 🌫️ Apply slight blur for realism
    print("🌫️ Applying blur...")
    img = img.filter(ImageFilter.GaussianBlur(0.5))

    # Ensure folder exists
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

    print(f"💾 Saving {filename}...")
    img.save(filename, 'JPEG', quality=95)

    print("\n✅ DONE! High-quality DNA background generated.")
    return True


if __name__ == '__main__':
    filename = 'static/DNAimage.jpg'
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    print("\n" + "="*50)
    print("🧬 Advanced DNA Background Generator")
    print("="*50 + "\n")

    generate_dna_background(filename)

    print("\n🚀 Now run: python app.py\n")