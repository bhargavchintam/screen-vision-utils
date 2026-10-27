"""Tests for template matching. Every image is made here with numpy and OpenCV."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from screen_vision import Match, draw_match, locate, locate_file
from screen_vision.cli import main

ROOT = Path(__file__).resolve().parents[1]


def fake_screen(width=480, height=320, seed=0):
    """A window-like screenshot: noisy background, a title bar and three buttons."""
    rng = np.random.default_rng(seed)
    screen = rng.integers(200, 240, size=(height, width, 3), dtype=np.uint8)
    cv2.rectangle(screen, (0, 0), (width - 1, 28), (120, 60, 20), -1)
    cv2.putText(screen, "Settings", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for i, label in enumerate(["Cancel", "Apply", "OK"]):
        x = 60 + i * 130
        cv2.rectangle(screen, (x, height - 70), (x + 100, height - 30), (90, 90, 90), 2)
        cv2.putText(screen, label, (x + 14, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return screen


def crop(image, left, top, width, height):
    return image[top : top + height, left : left + width].copy()


def test_finds_a_cropped_button_exactly():
    screen = fake_screen()
    template = crop(screen, 188, 245, 110, 50)  # the "Apply" button
    match = locate(screen, template)
    assert (match.left, match.top, match.width, match.height) == (188, 245, 110, 50)
    assert match.score > 0.99


@pytest.mark.parametrize("left,top", [(0, 0), (480 - 40, 320 - 30), (0, 320 - 30), (480 - 40, 0)])
def test_finds_templates_at_the_corners(left, top):
    screen = fake_screen(seed=1)
    match = locate(screen, crop(screen, left, top, 40, 30))
    assert (match.left, match.top) == (left, top)


def test_grayscale_and_bgra_inputs_work():
    screen = fake_screen(seed=2)
    template = crop(screen, 50, 100, 60, 40)
    gray = locate(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY), cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
    bgra = locate(cv2.cvtColor(screen, cv2.COLOR_BGR2BGRA), template)
    assert (gray.left, gray.top) == (bgra.left, bgra.top) == (50, 100)


def test_a_template_that_is_not_there_scores_low():
    screen = fake_screen(seed=3)
    other = np.random.default_rng(99).integers(0, 255, size=(40, 60, 3), dtype=np.uint8)
    assert locate(screen, other).score < 0.5


def test_template_larger_than_the_screenshot_is_rejected():
    screen = fake_screen(width=100, height=80)
    with pytest.raises(ValueError, match="larger than the screenshot"):
        locate(screen, fake_screen(width=120, height=60))


def test_center_and_description():
    match = Match(left=10, top=20, width=30, height=40, score=0.98765)
    assert match.center == (25, 40)
    assert match.describe() == "left=10 top=20 width=30 height=40 center_x=25 center_y=40 score=0.9877"


def test_locate_file_and_missing_file(tmp_path):
    screen = fake_screen(seed=4)
    cv2.imwrite(str(tmp_path / "screen.png"), screen)
    cv2.imwrite(str(tmp_path / "button.png"), crop(screen, 318, 245, 110, 50))
    match = locate_file(tmp_path / "screen.png", tmp_path / "button.png")
    assert (match.left, match.top) == (318, 245)
    with pytest.raises(FileNotFoundError):
        locate_file(tmp_path / "missing.png", tmp_path / "button.png")


def test_draw_match_marks_the_box_and_keeps_the_input():
    screen = fake_screen(seed=5)
    before = screen.copy()
    match = locate(screen, crop(screen, 60, 245, 110, 50))
    out = draw_match(screen, match, color=(0, 0, 255))
    assert np.array_equal(screen, before)
    assert tuple(out[match.top + match.height // 2, match.left]) == (0, 0, 255)
    assert tuple(out[match.top + match.height - 1, match.left + match.width // 2]) == (0, 0, 255)


def test_cli_prints_the_box_and_writes_the_annotated_image(tmp_path, capsys):
    screen = fake_screen(seed=6)
    cv2.imwrite(str(tmp_path / "screen.png"), screen)
    cv2.imwrite(str(tmp_path / "button.png"), crop(screen, 188, 245, 110, 50))
    out = tmp_path / "found" / "demo.png"
    code = main(["locate", str(tmp_path / "screen.png"), str(tmp_path / "button.png"), "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.startswith("left=188 top=245 width=110 height=50 center_x=243 center_y=270 score=")
    written = cv2.imread(str(out))
    assert written.shape == screen.shape
    assert tuple(written[245 + 25, 188]) == (0, 0, 255)


def test_cli_exits_1_below_min_score(tmp_path, capsys):
    cv2.imwrite(str(tmp_path / "screen.png"), fake_screen(seed=7))
    cv2.imwrite(str(tmp_path / "other.png"), np.random.default_rng(1).integers(0, 255, size=(40, 60, 3), dtype=np.uint8))
    code = main(["locate", str(tmp_path / "screen.png"), str(tmp_path / "other.png")])
    assert code == 1
    assert "no match" in capsys.readouterr().err


def test_cli_exits_2_for_a_missing_file(tmp_path, capsys):
    code = main(["locate", str(tmp_path / "nope.png"), str(tmp_path / "nope.png")])
    assert code == 2
    assert "could not read image" in capsys.readouterr().err


def test_python_dash_m_entry_point(tmp_path):
    screen = fake_screen(seed=8)
    cv2.imwrite(str(tmp_path / "screen.png"), screen)
    cv2.imwrite(str(tmp_path / "button.png"), crop(screen, 60, 245, 110, 50))
    result = subprocess.run(
        [sys.executable, "-m", "screen_vision", "locate", str(tmp_path / "screen.png"), str(tmp_path / "button.png")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("left=60 top=245 ")
