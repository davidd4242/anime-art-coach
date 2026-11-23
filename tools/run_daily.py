"""Run the released kernel on a new output directory; retain pass provenance."""
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import redraw


def render(source, output, passes):
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError("Input must be an image file.")
    output.mkdir(parents=True, exist_ok=False)
    for number in range(1, passes + 1):
        folder = output / f"pass-{number}"
        redraw.run(source, folder)
        source = folder / "result.png"
    return source


def arguments(parser):
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="new directory")
    parser.add_argument("--passes", type=int, choices=(1, 2), default=1)
