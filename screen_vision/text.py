"""Find text and its boxes in an image with the Google Cloud Vision API.

This needs the optional `google-cloud-vision` package
(`pip install "screen-vision-utils[text]"`) and a Google Cloud service-account
key. Point GOOGLE_APPLICATION_CREDENTIALS at the key file; the Vision client
reads that variable itself. Never commit the key file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CREDENTIALS_ENV = "GOOGLE_APPLICATION_CREDENTIALS"


class SetupError(RuntimeError):
    """The text command cannot run: missing package or credentials."""


@dataclass(frozen=True)
class TextBox:
    """One piece of detected text and the corners of its box, in pixels."""

    text: str
    vertices: tuple[tuple[int, int], ...]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(left, top, width, height) of the smallest box around the vertices."""
        xs = [x for x, _ in self.vertices]
        ys = [y for _, y in self.vertices]
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)

    @property
    def center(self) -> tuple[float, float]:
        """Mean of the vertices, as the original detect_text returned."""
        xs = [x for x, _ in self.vertices]
        ys = [y for _, y in self.vertices]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def describe(self) -> str:
        left, top, width, height = self.bbox
        x, y = self.center
        return f"text={self.text!r} left={left} top={top} width={width} height={height} center_x={x:.1f} center_y={y:.1f}"


def boxes_from_annotations(annotations: Iterable) -> list[TextBox]:
    """Turn Vision `text_annotations` into TextBox objects.

    The first annotation is the whole text block of the image; the rest are
    single words. Only the words are returned.
    """
    boxes = []
    for i, annotation in enumerate(annotations):
        if i == 0:
            continue
        vertices = tuple((int(v.x), int(v.y)) for v in annotation.bounding_poly.vertices)
        if vertices:
            boxes.append(TextBox(text=annotation.description, vertices=vertices))
    return boxes


def find_text(boxes: Iterable[TextBox], word: str) -> list[TextBox]:
    """Boxes whose text is exactly `word` (surrounding spaces ignored)."""
    return [box for box in boxes if box.text.strip() == word.strip()]


def check_credentials() -> Path:
    """Return the key file path from GOOGLE_APPLICATION_CREDENTIALS, or raise SetupError."""
    value = os.environ.get(CREDENTIALS_ENV)
    if not value:
        raise SetupError(
            f"{CREDENTIALS_ENV} is not set. Point it at your Google Cloud service-account key file, "
            f"for example: export {CREDENTIALS_ENV}=/path/to/key.json (keep the key out of git)."
        )
    path = Path(value).expanduser()
    if not path.is_file():
        raise SetupError(f"{CREDENTIALS_ENV} points to {path}, which is not a file.")
    return path


def _vision():
    try:
        from google.cloud import vision
    except ImportError as err:
        raise SetupError('google-cloud-vision is not installed: pip install "screen-vision-utils[text]"') from err
    return vision


def detect_text(path: str | Path, client=None) -> list[TextBox]:
    """Detect the words in an image file and return their boxes.

    `client` is a `vision.ImageAnnotatorClient`; one is created when it is
    not given (this reads GOOGLE_APPLICATION_CREDENTIALS).
    """
    content = Path(path).read_bytes()
    if client is None:
        check_credentials()
    vision = _vision()
    if client is None:
        client = vision.ImageAnnotatorClient()
    response = client.text_detection(image=vision.Image(content=content))
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")
    return boxes_from_annotations(response.text_annotations)
