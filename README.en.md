# Anime Art Coach: local refinement for anime illustrations

[简体中文](README.md) · English

**Soften a distracting highlight without flattening the knee.** Anime Art Coach is a local post-processing and visual review tool for anime illustrations. Give it an existing PNG or JPEG. It returns a full-size PNG and separate base-color, shadow, highlight, and line layers so you can see what changed.

Run it directly from Python, or ask Codex or DeepSeek Harness (DSH) to run it, inspect a region, and compare the results. The default renderer runs on your CPU and needs no API key. It neither trains a model nor generates new images from text.

## Download and run

| Edition | Use it when |
| --- | --- |
| [Windows x64 portable](https://github.com/davidd4242/anime-art-coach/releases/download/v1.0.1/anime-art-coach-windows-x64.zip) | You want to drop in a picture without installing Python separately |
| [Codex edition](https://github.com/davidd4242/anime-art-coach/releases/download/v1.0.1/anime-art-coach-codex.zip) | You want to run it from the terminal or with Codex and inspect the render, layers, protection mask, and report |
| [DeepSeek Harness edition](https://github.com/davidd4242/anime-art-coach/releases/download/v1.0.1/anime-art-coach-dsh.zip) | You want a location grid, numerical change readings, and aligned crops for a detail-by-detail review |

These links point to v1.0.1. [All releases and checksums](https://github.com/davidd4242/anime-art-coach/releases)

**Windows 10/11 (64-bit Intel or AMD):** Extract the entire portable ZIP. Double-click `Run.cmd` to select an image, or drop an image onto it. Use `Review.cmd` to generate the grid and crops as well. Find the render at `runs\<run ID>\pass-1\result.png`. [Windows guide](windows/README.en.md)

**macOS / Linux:** Open a terminal in the extracted project directory. Use `input.png` for the original, or substitute its path. For transparent illustrations, use the original PNG if available.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

# Run the command for the edition you downloaded
python adapters/codex/run.py input.png runs/first
python adapters/dsh/run.py input.png runs/first
```

Open `runs/first/pass-1/result.png`. Compare the full images at the same display height before zooming in. One pass is the default; use a fresh output directory for each run. For a lighter touch, try `python scripts/redraw.py input.png -o runs/gentler --strength 0.65`. [Full usage and options](docs/USAGE.en.md)

## What changed, and what stayed

**Leg highlight, knee, and garter.** The first pass selected two of five highlight candidates. The second selected none. Those counts tell you where the rule acted, not whether the picture improved. Check whether you can still read the change of plane at the knee, the shadow on the rear leg, and the gold motifs on the garter. Left to right: original, earlier baseline, current first pass, and current second pass.

![Aligned regression crops showing the leg highlight, knee and garter motifs](examples/en/iteration/case-2-regression.png)

**Collar, lacing, and rabbit badge.** An early version softened the shading but also faded two pink strokes on the badge’s face. The current version restores them with a reviewed mask specific to this source. That shows the protection mechanism at work; automatic detection on unfamiliar artwork is a separate question. [Open the four-column detail comparison](examples/en/iteration/case-1-regression.png)

For the full figures, see the [collar example](examples/en/momo-compare.png) and [leg example](examples/en/starfish-compare.png). In those presentation images, the left side is the original transparent PNG and the right side is the finished image supplied by the user, resized and JPEG-compressed. The right side is not direct renderer output or a pixel-level test reference.

## How the renderer works

1. **Estimate regions and line types.** Alpha, color, brightness, and neighboring pixels help estimate materials and distinguish silhouettes, material boundaries, and lines within a single material.
2. **Separate the paint layers.** Base color, multiply shadows, screened highlights, and ink are handled separately. A narrow highlight can be softened without applying the same change to a nearby cast shadow.
3. **Check for details that may have faded.** The renderer compares a provisional composite with the original and uses protection weights to preserve thin lines, low-contrast marks, and small motifs. It saves both the result and the evidence needed to inspect it.

In `pass-1`, `result.png` is the finished render, `layer_*.png` contains the four layer images, and `debug_detail_protect.png` visualizes detail protection. `report.json` includes settings and local paths; check it before sharing.

[Layers, rules, and iteration](docs/METHOD.en.md) · [Shadow decomposition](examples/maps/case-1-source-shadow.png) · [Line classifications](examples/maps/case-2-line-tiers.png) · [Separate monocular relative-depth example](examples/maps/case-2-depth-view.png). The depth example is a separate experiment; everyday rendering uses 2D layers.

## Review with an assistant

Try this: “Look at the full figure first, then compare the aligned crops. Check three things: is the change of plane at the knee still readable, is the rear leg still in shadow, and are the garter motifs intact? Cite the crop for each judgment. If you cannot tell, mark it for human review.”

The DSH edition accepts `--crop left top right bottom` in source-image pixels and writes `grid.png`, `evidence.json`, and aligned crops. A large numerical difference helps locate a change; it cannot tell you which version looks better. Online review is optional. Image access and usage charges depend on your DSH configuration.

[Codex operations guide](adapters/codex/OPERATIONS.en.md) · [DSH operations guide](adapters/dsh/OPERATIONS.en.md)

## What’s verified, and where to be careful

- Both public source images were run for two passes, with checks for dimensions, alpha, and full-size RGBA pixel hashes. The v1.0.1 portable archive also passed [Windows x64 CI](https://github.com/davidd4242/anime-art-coach/actions/runs/45244488990) using its bundled interpreter.
- A bright edge can describe form in strong backlighting or describe the material on polished metal. On unfamiliar artwork, compare the default result with the original before deciding whether to lower the strength or use a local mask.
- Samples help developers revise rules and check whether they generalize; adding images does not train the model. Both adapters use the same renderer, but different assistants may assess the crops differently. The project has not benchmarked the online vision capabilities of commercial models.

[Validation and reproduction](docs/VALIDATION.en.md) · [Third-party components, optional weights, and artwork notes](docs/THIRD_PARTY.en.md)
