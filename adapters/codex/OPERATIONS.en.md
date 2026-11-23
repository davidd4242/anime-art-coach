# Codex operating guide

[简体中文](OPERATIONS.md) · English

Open the extracted project root in Codex and ask the assistant to read this guide and the [usage guide](../../docs/USAGE.en.md). The Python renderer works locally without an image-generation service.

1. Identify the original input and any supplied final image. Feed the original to the renderer.
2. Install `requirements.txt` with the interpreter you will use. Try `python adapters/codex/run.py examples/momo.png runs/codex-first`. Choose a new output folder.
3. Open `pass-1/result.png` and show both full figures at the same height. Inspect facial contours, hair locks, material edges, light-colored icons, and overlap shadows.
4. To check cumulative losses, run `python adapters/codex/run.py examples/momo.png runs/codex-repeat --passes 2`. Look at both reports, protection masks, and matching source-coordinate crops. Measure coordinates from the top left.
5. Link to the results. Separate improvements, ties, and regressions. If an ornament breaks, name it, record the crop, identify the first failing pass, and save the settings. Use a new directory for experiments.
6. When asked for the local collar treatment on Case 1, run `python scripts/momo_collar.py runs/codex-first/pass-1`. The variation is exported alongside the daily result.

**Example request**

> Process examples/starfish.png into runs/star-check with the daily settings. Run two passes and retain ornament protection. Show the whole figure first, then compare the stars on the garter, the leg highlights, and the knee shadow at matching coordinates. Report what survived and what needs a human decision.

For a batch, give each input a separate output directory and reserve whole works for evaluation. [How rules evolve](../../docs/METHOD.en.md)
