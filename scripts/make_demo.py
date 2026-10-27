"""Make docs/demo.png by running the real `locate` command on a drawn screenshot.

The script draws a simple settings window with three buttons, crops the
"Apply" button as the template, then runs
`python -m screen_vision locate ... --out docs/demo.png` and prints its output.

Run from the repository root:
  python scripts/make_demo.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FONT = cv2.FONT_HERSHEY_SIMPLEX
APPLY_BUTTON = (406, 316, 92, 36)  # left, top, width, height of the button as drawn


def button(img, left, top, width, height, label, fill, text_color):
    cv2.rectangle(img, (left, top), (left + width, top + height), fill, -1)
    cv2.rectangle(img, (left, top), (left + width, top + height), (150, 150, 150), 1)
    (text_w, text_h), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    cv2.putText(img, label, (left + (width - text_w) // 2, top + (height + text_h) // 2), FONT, 0.55, text_color, 1, cv2.LINE_AA)


def field(img, left, top, label, value):
    cv2.putText(img, label, (left, top + 20), FONT, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.rectangle(img, (left + 90, top), (left + 470, top + 30), (255, 255, 255), -1)
    cv2.rectangle(img, (left + 90, top), (left + 470, top + 30), (170, 170, 170), 1)
    cv2.putText(img, value, (left + 98, top + 20), FONT, 0.5, (30, 30, 30), 1, cv2.LINE_AA)


def draw_screen():
    """A 640x400 'screenshot' (BGR): desktop, one window, a form and three buttons."""
    img = np.full((400, 640, 3), (120, 96, 52), dtype=np.uint8)
    cv2.rectangle(img, (40, 30), (600, 370), (242, 242, 242), -1)
    cv2.rectangle(img, (40, 30), (600, 64), (64, 64, 64), -1)
    cv2.putText(img, "Settings", (56, 54), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for i, color in enumerate([(87, 95, 255), (46, 189, 254), (64, 201, 39)]):
        cv2.circle(img, (540 + i * 20, 47), 6, color, -1, cv2.LINE_AA)
    field(img, 70, 90, "Name", "Sample User")
    field(img, 70, 140, "Email", "user@example.com")
    field(img, 70, 190, "Server", "https://example.com")
    cv2.rectangle(img, (160, 245), (176, 261), (255, 255, 255), -1)
    cv2.rectangle(img, (160, 245), (176, 261), (120, 120, 120), 1)
    cv2.line(img, (163, 253), (167, 258), (40, 120, 40), 2, cv2.LINE_AA)
    cv2.line(img, (167, 258), (174, 248), (40, 120, 40), 2, cv2.LINE_AA)
    cv2.putText(img, "Send me updates", (186, 258), FONT, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    button(img, 306, 316, 92, 36, "Cancel", (230, 230, 230), (30, 30, 30))
    button(img, *APPLY_BUTTON, "Apply", (230, 230, 230), (30, 30, 30))
    button(img, 506, 316, 72, 36, "OK", (200, 120, 30), (255, 255, 255))
    return img


def main():
    screen = draw_screen()
    left, top, width, height = APPLY_BUTTON
    template = screen[top : top + height + 1, left : left + width + 1].copy()
    DOCS.mkdir(exist_ok=True)
    cv2.imwrite(str(DOCS / "demo-template.png"), template)
    with tempfile.TemporaryDirectory() as tmp:
        shot = Path(tmp) / "screenshot.png"
        cv2.imwrite(str(shot), screen)
        cmd = [sys.executable, "-m", "screen_vision", "locate", str(shot), str(DOCS / "demo-template.png"), "--out", str(DOCS / "demo.png")]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"wrote {DOCS / 'demo.png'} and {DOCS / 'demo-template.png'}")


if __name__ == "__main__":
    main()
