"""Synthetic exam paper and attacker content for the demo."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_DIRS = [r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"]


def font(size: int, bold: bool = False):
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold
             else ["arial.ttf", "DejaVuSans.ttf", "Arial.ttf"])
    for d in _FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


QUESTIONS = [
    ("Q1.", "Prove that sqrt(2) is irrational.", "[4]"),
    ("Q2.", "Find the roots of 3x^2 - 5x + 2 = 0.", "[3]"),
    ("Q3.", "A ladder 10 m long reaches a window 8 m high.", "[3]"),
    ("",    "Find the distance of its foot from the wall.", ""),
    ("Q4.", "Evaluate: sin 30 + cos 60 - tan 45.", "[2]"),
    ("Q5.", "In a class of 40, the mean mark is 62. If one", "[4]"),
    ("",    "mark of 80 was read as 40, find the true mean.", ""),
    ("Q6.", "Construct a triangle with sides 5, 6 and 7 cm", "[4]"),
    ("",    "and draw its circumcircle.", ""),
    ("Q7.", "Prove: the tangent at any point of a circle is", "[5]"),
    ("",    "perpendicular to the radius through it.", ""),
]

# Question region in the default layout (x0, y0, x1, y1) as fractions of the page.
DEFAULT_ROI_FRAC = (0.05, 0.34, 0.95, 0.965)
# Where question Q5 sits, and a blank strip under the last question.
Q5_FRAC = (0.05, 0.575, 0.95, 0.66)
BLANK_FRAC = (0.08, 0.85, 0.92, 0.95)


def exam_paper(width: int = 720, height: int = 1040) -> np.ndarray:
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    s = width / 720

    # Header with a coloured seal
    cx, cy, r = int(70 * s), int(70 * s), int(42 * s)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 70, 140), outline=(200, 160, 40), width=int(5 * s))
    d.text((cx, cy), "SEB", fill="white", font=font(int(22 * s), True), anchor="mm")
    d.text((int(130 * s), int(38 * s)), "STATE EXAMINATION BOARD", fill=(20, 40, 90), font=font(int(30 * s), True))
    d.text((int(130 * s), int(80 * s)), "Secondary School Certificate - 2026", fill="black", font=font(int(20 * s)))
    d.line([(int(30 * s), int(130 * s)), (width - int(30 * s), int(130 * s))], fill=(20, 40, 90), width=int(3 * s))

    d.text((int(40 * s), int(150 * s)), "Subject: MATHEMATICS (041)", fill="black", font=font(int(22 * s), True))
    d.text((width - int(40 * s), int(150 * s)), "Max. Marks: 25", fill="black", font=font(int(20 * s)), anchor="ra")
    d.text((int(40 * s), int(185 * s)), "Time: 1 hour 30 min", fill="black", font=font(int(20 * s)))
    d.text((width - int(40 * s), int(185 * s)), "Set: A", fill="black", font=font(int(20 * s)), anchor="ra")

    d.rectangle([int(40 * s), int(225 * s), width - int(40 * s), int(320 * s)], outline=(120, 120, 120), width=2)
    instr = ["General Instructions:", "1. All questions are compulsory.",
             "2. Use of calculators is not permitted.", "3. Marks are indicated against each question."]
    for i, line in enumerate(instr):
        d.text((int(55 * s), int(232 * s + i * 21 * s)), line, fill=(60, 60, 60),
               font=font(int(17 * s), i == 0))

    y = int(370 * s)
    for num, text, marks in QUESTIONS:
        if num:
            y += int(12 * s)
            d.text((int(50 * s), y), num, fill="black", font=font(int(21 * s), True))
        d.text((int(100 * s), y), text, fill="black", font=font(int(21 * s)))
        if marks:
            d.text((width - int(50 * s), y), marks, fill=(90, 90, 90), font=font(int(19 * s)), anchor="ra")
        y += int(36 * s)

    d.text((width // 2, height - int(25 * s)), "*** END OF QUESTION PAPER ***", fill=(90, 90, 90),
           font=font(int(16 * s)), anchor="mm")
    return np.asarray(img)


def fake_notice(shape, rect, text="NOTICE: Q5 is CANCELLED.\nAll students get full marks.") -> np.ndarray:
    """White image with attacker text inside `rect` (used by the forgery attack)."""
    H, W, C = shape
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(180, 0, 0), width=3)
    size = max(10, min(int((y1 - y0) / 3.2), int((x1 - x0) / 16)))
    d.multiline_text(((x0 + x1) // 2, (y0 + y1) // 2), text, fill=(180, 0, 0),
                     font=font(size, True), anchor="mm", align="center", spacing=6)
    arr = np.asarray(img)
    if C == 1:
        arr = np.asarray(img.convert("L"))[:, :, None]
    return arr
