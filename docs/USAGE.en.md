# Install, render, and review

[简体中文](USAGE.md) · English

For Windows 10/11 x64, the portable archive includes Python and the daily CPU renderer. [Windows instructions](../windows/README.en.md)

## Install the Python edition

Run the following from the extracted project root on macOS or Linux. The reference installation used macOS arm64 and Python 3.9.6. On Windows PowerShell, create the environment with `py -m venv .venv` and invoke `.\.venv\Scripts\python.exe` for the commands below.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import cv2, numpy, skimage, PIL; print(cv2.__version__)"
```

If pip reports that no distribution matches, check your Python version, CPU architecture, and package index. For a missing import, install with the same interpreter that will run the renderer. Keep paths with spaces in quotes. A transparent original PNG retains edge coverage.

## First render

Choose the entry point for your edition. Each creates a new output directory. If a run fails partway through, keep that directory for diagnosis and retry in a fresh one.

```sh
# Codex
python adapters/codex/run.py examples/momo.png runs/case1
# DSH
python adapters/dsh/run.py examples/starfish.png runs/case2 --crop 475 715 651 1070
```

Open `pass-1/result.png` under your output directory. Compare the whole figure at equal height, then inspect the details.

| Output in each pass directory | How to use it |
|---|---|
| `layer_flat.png`, `layer_shadow.png`, `layer_highlight.png`, `layer_lines.png` | Inspect base color, multiply shadow, screen highlights and alpha-composited ink. The matching input layers are in `original_layers/`. |
| `debug_detail_protect.png` | White indicates high preservation weight. |
| `debug_materials.png`, `debug_linetiers.png` | Check material and line classification. |
| `report.json` | Check source, parameters, protection mode, and duration. Review local paths before sharing. |

## Repeat a pass and compare

```sh
python adapters/dsh/run.py examples/momo.png runs/two --passes 2 --crop 595 1011 655 1049 --crop 592 402 647 462 --crop 370 350 437 425
```

For Codex, use its entry point and omit the crop arguments. Pass two reads the first result. Keep `result.png`, its report, and its protection mask together. One pass is the daily default. Two passes expose cumulative losses. In a blind comparison, randomize the full figures and allow a tie before revealing the corresponding crops.

## Adjust a setting

`python scripts/redraw.py --help` lists the options. The direct CLI accepts `-o`; give it a fresh directory. The daily `artist` preset uses strength 1, automatic detail protection, and denoising off. Try a lower strength from the original if a surface looks too flat.

```sh
python scripts/redraw.py examples/starfish.png -o runs/gentler --strength 0.65
```

For explicit protection, pass a source-sized grayscale PNG with `--detail-mask`. White retains original detail and black requests no extra protection. Supplying a mask replaces automatic detail selection. Keep a default render for comparison.

The optional Case 1 collar treatment reads a matching daily render, checks its identity, and exports `result_momo_collar.png` beside it. Run it after the Case 1 example above.

```sh
python scripts/momo_collar.py runs/case1/pass-1
```

## Optional denoising

Use a separate worker environment. The pinned upstream list at `scripts/vendor/gpt_anime_denoise/requirements-macos-arm64-py312.txt` is for macOS arm64 with Python 3.12. For other platforms use the upstream installation instructions. L0 runs without model weights.

```sh
python scripts/redraw.py examples/momo.png -o runs/l0 --denoise l0_0.001 --denoise-python .denoise-venv/bin/python
```

Try `l0_0.0005` for a gentler adjustment. Inspect faint fabric texture and small decorations afterward. For `esrgan60_l0`, download the listed weight under its original filename and pass its directory with `--denoise-models`. The worker verifies its SHA-256. [Upstream links, filename, and checksum](THIRD_PARTY.en.md)

## Judge the result

Size, alpha, and the number of surviving marks are measurable. Whether a highlight works depends on the material, illumination, and purpose of the stroke. A rim light can describe volume and a metal glint can identify an eyelet. Inspect those roles before choosing a preferred render. [Step-by-step DSH review](../adapters/dsh/OPERATIONS.en.md)
