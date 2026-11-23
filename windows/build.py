"""Build the x64 portable ZIP from committed project files and verified Windows wheels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "anime-art-coach-windows-x64/"
PTH = b"python312.zip\r\n.\r\nLib/site-packages\r\n../scripts\r\nimport site\r\n"


def dependency(item, cache):
    path = cache / item["filename"]
    if not path.exists():
        with urlopen(item["url"], timeout=120) as reply:
            data = reply.read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Download digest mismatch: {path.name}")
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"Cache digest mismatch: {path.name}")
    return path


def add_distribution(archive, src, subdir):
    with ZipFile(src) as packed:
        for entry in packed.infolist():
            if entry.is_dir():
                continue
            name = entry.filename
            if ".data/" in name or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"Unexpected archive layout: {name}")
            data = PTH if name == "python312._pth" else packed.read(entry)
            archive.writestr(PREFIX + subdir + name, data)


def build(output, cache):
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / "windows/runtime-lock.json").read_text())
    python = dependency(spec["python"], cache)
    wheels = [dependency(row, cache) for row in spec["wheels"]]
    names = subprocess.check_output(["git", "ls-tree", "-rz", "--name-only", "HEAD"], cwd=ROOT)
    with NamedTemporaryFile(dir=output.parent, delete=False) as temp:
        staged = Path(temp.name)
    try:
        with ZipFile(staged, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name in names.decode().split("\0"):
                if not name or name.startswith("."):
                    continue
                data = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
                target = Path(name).name if name in ("windows/Run.cmd", "windows/Review.cmd") else name
                archive.writestr(PREFIX + target, data)
            add_distribution(archive, python, "runtime/")
            for wheel in wheels:
                add_distribution(archive, wheel, "runtime/Lib/site-packages/")
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.cache.resolve())
