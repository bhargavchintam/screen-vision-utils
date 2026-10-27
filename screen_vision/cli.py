"""Command line: python -m screen_vision {locate,text} ...

Exit codes: 0 when something was found, 1 when nothing matched, 2 for usage
or setup errors (missing file, package or credentials).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import matching


def _locate(args: argparse.Namespace) -> int:
    try:
        screenshot = matching.read_image(args.screenshot)
        template = matching.read_image(args.template)
        match = matching.locate(screenshot, template)
    except (FileNotFoundError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.out:
        import cv2

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), matching.draw_match(screenshot, match)):
            print(f"error: could not write {out}", file=sys.stderr)
            return 2
    if match.score < args.min_score:
        print(f"no match: best score {match.score:.4f} is below --min-score {args.min_score}", file=sys.stderr)
        return 1
    print(match.describe())
    return 0


def _text(args: argparse.Namespace) -> int:
    from . import text

    if not Path(args.image).is_file():
        print(f"error: could not read image: {args.image}", file=sys.stderr)
        return 2
    try:
        boxes = text.detect_text(args.image)
    except text.SetupError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.find is not None:
        boxes = text.find_text(boxes, args.find)
        if not boxes:
            print(f"no match: {args.find!r} was not found in {args.image}", file=sys.stderr)
            return 1
    for box in boxes:
        print(box.describe())
    return 0


def build_parser(prog: str = "python -m screen_vision") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Find an image or text on a screenshot.")
    commands = parser.add_subparsers(dest="command", required=True)

    find = commands.add_parser("locate", help="find a template image on a screenshot (OpenCV template matching)")
    find.add_argument("screenshot", help="the full screenshot")
    find.add_argument("template", help="the small image to look for")
    find.add_argument("--out", help="write a copy of the screenshot with the found box drawn on it")
    find.add_argument(
        "--min-score", type=float, default=0.9, help="treat a best score below this as no match (default: 0.9, scores go up to 1)"
    )
    find.set_defaults(run=_locate)

    words = commands.add_parser("text", help="list the words in an image and their boxes (Google Cloud Vision)")
    words.add_argument("image")
    words.add_argument("--find", metavar="WORD", help="print only the boxes whose text is exactly WORD")
    words.set_defaults(run=_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    # The `screen-vision` console script and `python -m screen_vision` run the same code.
    prog = "screen-vision" if Path(sys.argv[0]).name == "screen-vision" else "python -m screen_vision"
    args = build_parser(prog).parse_args(argv)
    return args.run(args)
