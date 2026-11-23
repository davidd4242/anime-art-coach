"""Windows desktop launcher for the existing daily renderer and DSH evidence mode."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def select_image():
    # Use the system file dialog. The embedded Python distribution omits Tk.
    command = """
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Filter = 'Images (*.png;*.jpg;*.jpeg;*.webp)|*.png;*.jpg;*.jpeg;*.webp|All files (*.*)|*.*'
if ($d.ShowDialog() -eq 'OK') { [Console]::Write($d.FileName) }
$d.Dispose()
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", command],
        check=True, stdout=subprocess.PIPE, encoding="utf-8-sig",
    )
    return Path(result.stdout) if result.stdout else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--mode", choices=("codex", "dsh"), default="codex")
    parser.add_argument("--passes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, help="a new output directory")
    parser.add_argument("--crop", nargs=4, action="append", type=int,
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"), default=[])
    args = parser.parse_args()
    if args.crop and args.mode != "dsh":
        parser.error("--crop requires --mode dsh")
    source = args.input if args.input is not None else select_image()
    if source is None:
        return 0
    if not source.is_file():
        parser.error(f"Input file not found: {source}")
    unique = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output = args.output or ROOT / "runs" / unique
    command = [sys.executable, "-B", "-X", "utf8", str(ROOT / "adapters" / args.mode / "run.py"),
               str(source.resolve()), str(output.resolve()), "--passes", str(args.passes)]
    for box in args.crop:
        command.extend(["--crop", *map(str, box)])
    return subprocess.run(command).returncode


if __name__ == "__main__":
    sys.exit(main())
