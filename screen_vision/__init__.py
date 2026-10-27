"""Small helpers to find things on a screenshot.

- `locate`: find a template image with OpenCV template matching.
- `screen_vision.text.detect_text`: find words and their boxes with the
  Google Cloud Vision API (optional extra: `screen-vision-utils[text]`).
"""

from .matching import Match, draw_match, locate, locate_file, read_image

__all__ = ["Match", "draw_match", "locate", "locate_file", "read_image"]
__version__ = "0.1.0"
