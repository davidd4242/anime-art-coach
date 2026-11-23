"""Codex entry point. Render one image with the approved daily settings."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from run_daily import arguments, render

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    args = parser.parse_args()
    print(render(args.input, args.output, args.passes))
