"""Smoke-test the actual portable launcher with two passes and Unicode paths."""
from pathlib import Path
import json
import subprocess
import sys
from tempfile import TemporaryDirectory
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def verify():
    with TemporaryDirectory(prefix="art coach ") as temp:
        work = Path(temp)
        source = work / "案例 1.png"
        source.write_bytes((ROOT / "examples" / "momo.png").read_bytes())
        for mode in ("codex", "dsh"):
            out = work / mode
            command = [sys.executable, "-B", "-X", "utf8", str(ROOT / "windows/launch.py"),
                       str(source), "--mode", mode, "--passes", "2", "--output", str(out)]
            if mode == "dsh":
                command.extend(["--crop", "595", "1011", "655", "1049"])
            subprocess.run(command, check=True)
            previous = source
            for n in (1, 2):
                folder = out / f"pass-{n}"
                report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                assert Path(report["input"]).resolve() == previous.resolve()
                previous = folder / "result.png"
                with Image.open(source) as original, Image.open(previous) as result:
                    assert result.size == original.size
                    assert result.getchannel("A").tobytes() == original.getchannel("A").tobytes()
                assert (folder / "debug_detail_protect.png").is_file()
            if mode == "dsh":
                evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
                assert evidence["alpha_equal"] and (out / evidence["crops"][0]["file"]).is_file()
                for n in (1, 2):
                    with Image.open(work / "codex" / f"pass-{n}/result.png") as a:
                        with Image.open(out / f"pass-{n}/result.png") as b:
                            assert a.tobytes() == b.tobytes()
    print("PASS: Unicode paths, daily render, DSH evidence, alpha and both passes")


if __name__ == "__main__":
    verify()
