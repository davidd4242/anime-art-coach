# DeepSeek Harness review guide

[简体中文](OPERATIONS.md) · English

Give this guide to the reviewing assistant. The renderer applies the rules. The reviewer locates small marks, checks their condition, and states what the available evidence supports.

## First run

1. Install the Python dependencies from the project root. For online review, check that `dsh --help` identifies DeepSeek Harness with a headless profile. The local renderer can run independently. [Installation guide](../../docs/USAGE.en.md)
2. Try the following on the 997 × 1578 Case 1 original. Each box is measured from the top left as left, top, right, bottom. The right and bottom bounds are exclusive.

```sh
python adapters/dsh/run.py examples/momo.png runs/review --passes 2 \
  --crop 595 1011 655 1049 --crop 592 402 647 462 --crop 370 350 437 425
```

3. Open `runs/review/grid.png`, then `evidence.json`. Each crop sheet shows the original, pass one, and pass two from left to right. The crop files follow the argument order.
4. To ask your configured DSH provider for an English review, use the command below. It submits a read-only task and may incur provider usage. Image access depends on the tools in your profile. For a fully local review, inspect the files yourself.

```sh
(cd runs/review && sh ../../adapters/dsh/review.sh en)
```

If the DSH launcher cannot find its runtime, open the result directory in the desktop application and use the prompt below. If system Node and an existing DSH installation are available, check the package’s `lib/bin.js --help` using Node and set `DSH_CLI_JS` to that file’s absolute path.

## Give the reviewer an observable task

1. **Establish provenance.** The original files are `examples/momo.png` and `examples/starfish.png`. Files containing `delivered` are supplied finished images. If provenance is unclear, report the filenames and sizes and ask for clarification.
2. **Choose two to four checks.** Give each mark a count, location, and material. For example, count the two pink strokes on the rabbit badge and trace the openings in the collar. For shading, check whether the ridge on the knee survives as the long highlights soften.
3. **Locate matching crops.** Use the grid, then specify a box on the original. To convert preview coordinates, multiply each axis by its original-to-preview size ratio. Without explicit boxes, the script chooses up to three cells by mean change. A tiny badge may contribute little to a cell average.
4. **Check integrity.** Read `size`, `passes`, and `alpha_equal` in the evidence file. Read `params` and `denoise` in the report. In a two-pass run, check that the second input is `pass-1/result.png`. Resolve size or alpha errors before interpreting the image.
5. **Compare individual strokes.** Trace the bandage’s ear tips and face, the marks on the badge, and the connected black lacing. A change to the cloth visible through an opening and a broken strap are different observations. For a highlight, identify the material and nearby shadow. Mark ambiguous evidence as “needs human review.”
6. **Read the numbers.** `mean_abs_rgb` is the mean absolute displayed channel change from the original to the last pass, on a common gray background for visible pixels, on a scale of 0–255. Large changes may come from a useful shading adjustment or a damaged ornament. Inspect the crop to tell which.
7. **Report a judgment.** State what is visible at full size first, then list each detail and the first pass where it changes. A tie is valid. Preserve the tested settings unless asked to change them.

**Example task**

> Read evidence.json and report the image size. For each crop, record coordinates, marks in the original, pass-one result, pass-two result, and the file supporting your claim. Inspect the bandage face, pink badge marks, and black lacing. If you cannot view a crop, report only the measurements and file location. Give a full-figure preference only after viewing the full figures.

When opinions differ, ask each reviewer to locate the exact stroke, its direction, and the nearby shadow. Show the matched crops to a human at equal magnification. For model comparisons, use the held-out protocol in the [illustrated method](../../docs/METHOD.en.md).
