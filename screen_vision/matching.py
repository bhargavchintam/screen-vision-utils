"""Find where a small image (a template) appears on a screenshot.

This uses OpenCV template matching with normalised cross-correlation
(`cv2.TM_CCOEFF_NORMED`) on grayscale copies of both images, as the original
detect-image-coordinates-from-window-screenshot script did. The score is
between -1 and 1; an exact copy of the template scores close to 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Match:
    """Best match of a template on a screenshot, in screenshot pixels."""

    left: int
    top: int
    width: int
    height: int
    score: float

    @property
    def center(self) -> tuple[int, int]:
        """Middle of the box, for example to click on it."""
        return self.left + self.width // 2, self.top + self.height // 2

    def describe(self) -> str:
        x, y = self.center
        return (
            f"left={self.left} top={self.top} width={self.width} height={self.height} "
            f"center_x={x} center_y={y} score={self.score:.4f}"
        )


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"expected a grayscale, BGR or BGRA image, got shape {image.shape}")


def locate(screenshot: np.ndarray, template: np.ndarray) -> Match:
    """Return the best match of `template` on `screenshot`.

    Both images are numpy arrays as OpenCV loads them (BGR, BGRA or
    grayscale). The template must fit inside the screenshot.
    """
    screen, needle = _gray(screenshot), _gray(template)
    if needle.size == 0:
        raise ValueError("template is empty")
    height, width = needle.shape[:2]
    if height > screen.shape[0] or width > screen.shape[1]:
        raise ValueError(f"template ({width}x{height}) is larger than the screenshot ({screen.shape[1]}x{screen.shape[0]})")
    result = cv2.matchTemplate(screen, needle, cv2.TM_CCOEFF_NORMED)
    # A flat (single-colour) template has no variance, which can give NaN scores.
    result = np.nan_to_num(result, nan=-1.0, posinf=1.0, neginf=-1.0)
    _, score, _, (left, top) = cv2.minMaxLoc(result)
    return Match(left=int(left), top=int(top), width=int(width), height=int(height), score=float(score))


def read_image(path: str | Path) -> np.ndarray:
    """Load an image with OpenCV; raise FileNotFoundError if it cannot be read."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def locate_file(screenshot_path: str | Path, template_path: str | Path) -> Match:
    """Same as `locate`, with both images read from files."""
    return locate(read_image(screenshot_path), read_image(template_path))


def draw_match(image: np.ndarray, match: Match, color: tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Return a copy of `image` with the match box and its score drawn on it."""
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3].copy()
    top_left = (match.left, match.top)
    bottom_right = (match.left + match.width - 1, match.top + match.height - 1)
    cv2.rectangle(out, top_left, bottom_right, color, thickness)
    label = f"score {match.score:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_top = match.top - text_h - baseline - 4
    if label_top < 0:  # no room above the box: put the label below it
        label_top = min(match.top + match.height + 2, out.shape[0] - text_h - baseline - 2)
    cv2.rectangle(out, (match.left, label_top), (match.left + text_w + 6, label_top + text_h + baseline + 2), color, -1)
    cv2.putText(out, label, (match.left + 3, label_top + text_h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
