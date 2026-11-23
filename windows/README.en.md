# Windows x64 portable edition

[简体中文](README.md) · English

[Download the Windows x64 ZIP](https://github.com/davidd4242/anime-art-coach/releases/download/v1.0.1/anime-art-coach-windows-x64.zip)

For Windows 10/11 on a 64-bit Intel or AMD processor. The package contains CPython, the CPU rendering dependencies, and the shared Codex/DSH renderer. Extract the entire ZIP to a writable directory such as `C:\ArtCoach`.

## Render or review

1. Double-click **Run.cmd** in the extracted root and select the original PNG or JPG. You can also drag one image onto the launcher.
2. Open `runs\<run ID>\pass-1\result.png`. Layers and the report are beside it.
3. For a grid, change measurements, and aligned crops, use **Review.cmd** instead. Open `grid.png`, `evidence.json`, and `crop-01.png` in the new run directory.

## PowerShell or assistant access

Run these from the extracted root:

```powershell
.\runtime\python.exe -X utf8 .\windows\launch.py ".\examples\momo.png" --output ".\runs\case1"
.\runtime\python.exe -X utf8 .\windows\launch.py ".\examples\momo.png" --mode dsh --passes 2 --crop 595 1011 655 1049 --output ".\runs\review-case1"
```

Pass two reads the result of pass one. Choose a new output name on each invocation or omit `--output` for an automatically allocated directory. Rendering, grid generation, and crops work offline. For assistant instructions, see the [usage guide](../docs/USAGE.en.md), [Codex guide](../adapters/codex/OPERATIONS.en.md), or [DSH guide](../adapters/dsh/OPERATIONS.en.md).

## Verify or troubleshoot

- Run `.\runtime\python.exe -B -X utf8 .\windows\verify.py` for a local smoke test. It uses an input path with spaces and Chinese characters, checks both entry points, two passes, original size, alpha, the mask file, and three-column crops. Success prints PASS.
- If the runtime cannot be found, extract all files first. A short writable path avoids deeply nested directories.
- For a cv2 DLL load failure, check CPU architecture and the Microsoft Visual C++ x64 runtime. Windows N/KN may also need the Media Feature Pack. [Microsoft runtime guide](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- Optional neural denoising needs a compatible Windows worker environment and weights. See the [optional denoising instructions](../docs/USAGE.en.md#optional-denoising). The Python license is in `runtime\LICENSE.txt`. Dependency licenses ship in package or dist-info directories.
