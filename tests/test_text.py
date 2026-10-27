"""Tests for the Vision text helpers. No network calls and no credentials are used."""

from types import SimpleNamespace

import pytest

from screen_vision import text
from screen_vision.cli import main


def annotation(description, corners):
    vertices = [SimpleNamespace(x=x, y=y) for x, y in corners]
    return SimpleNamespace(description=description, bounding_poly=SimpleNamespace(vertices=vertices))


ANNOTATIONS = [
    annotation("Cancel Apply OK", [(10, 10), (300, 10), (300, 40), (10, 40)]),  # whole text block
    annotation("Cancel", [(10, 10), (70, 10), (70, 40), (10, 40)]),
    annotation("Apply", [(120, 12), (180, 12), (180, 38), (120, 38)]),
    annotation("OK", [(260, 10), (300, 10), (300, 40), (260, 40)]),
]


def test_boxes_skip_the_whole_text_block():
    boxes = text.boxes_from_annotations(ANNOTATIONS)
    assert [b.text for b in boxes] == ["Cancel", "Apply", "OK"]


def test_box_geometry():
    apply = text.boxes_from_annotations(ANNOTATIONS)[1]
    assert apply.bbox == (120, 12, 60, 26)
    assert apply.center == (150.0, 25.0)
    assert apply.describe() == "text='Apply' left=120 top=12 width=60 height=26 center_x=150.0 center_y=25.0"


def test_find_text_matches_whole_words():
    boxes = text.boxes_from_annotations(ANNOTATIONS)
    assert [b.text for b in text.find_text(boxes, " OK ")] == ["OK"]
    assert text.find_text(boxes, "App") == []


def test_missing_credentials_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(text.CREDENTIALS_ENV, raising=False)
    image = tmp_path / "screen.png"
    image.write_bytes(b"not used")
    assert main(["text", str(image)]) == 2
    assert text.CREDENTIALS_ENV in capsys.readouterr().err


def test_credentials_must_point_to_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv(text.CREDENTIALS_ENV, str(tmp_path / "missing.json"))
    with pytest.raises(text.SetupError, match="not a file"):
        text.check_credentials()


def test_detect_text_with_a_fake_client(tmp_path):
    vision = pytest.importorskip("google.cloud.vision")
    image = tmp_path / "screen.png"
    image.write_bytes(b"\x89PNG fake bytes")

    class FakeClient:
        def text_detection(self, image):
            assert image.content == b"\x89PNG fake bytes"
            return vision.AnnotateImageResponse(
                text_annotations=[
                    {"description": "OK", "bounding_poly": {"vertices": [{"x": 0, "y": 0}]}},
                    {"description": "OK", "bounding_poly": {"vertices": [{"x": 5, "y": 6}, {"x": 15, "y": 6}, {"x": 15, "y": 16}, {"x": 5, "y": 16}]}},
                ]
            )

    boxes = text.detect_text(image, client=FakeClient())
    assert [(b.text, b.bbox) for b in boxes] == [("OK", (5, 6, 10, 10))]
