# Windows 便携版

简体中文 · [English](README.en.md)

[下载 Windows x64 ZIP](https://github.com/davidd4242/anime-art-coach/releases/download/v1.0.1/anime-art-coach-windows-x64.zip)

适用 Windows 10 / 11 x64（Intel 或 AMD 64 位）。默认绘图和复核在本机 CPU 上运行。便携包包含 Python、数值库、图像库，以及 Codex / DSH 共用的日常渲染内核。使用前请把 ZIP 全部解压到自己的可写目录，如 `C:\ArtCoach`。

## 最快出图

1. 双击包根目录的 **Run.cmd**，在系统选图窗口选择原始 PNG 或 JPG。也可以把一张图片拖到 Run.cmd 上。透明立绘优先用原始 PNG。
2. 窗口显示本次处理的路径。打开包内 `runs\本次编号\pass-1\result.png` 查看成图。同目录还有图层与运行报告。
3. 想看网格、裁片和变化读数时，双击 **Review.cmd** 重新选图。打开这次运行目录中的 `grid.png`、`evidence.json`、`crop-01.png` 等。两个启动器每次新建输出目录。

## 从 PowerShell 或助手调用

在解压目录打开 PowerShell。下例使用随包附带的原图。

```powershell
.\runtime\python.exe -X utf8 .\windows\launch.py ".\examples\momo.png" --output ".\runs\case1"
.\runtime\python.exe -X utf8 .\windows\launch.py ".\examples\momo.png" --mode dsh --passes 2 --crop 595 1011 655 1049 --output ".\runs\review-case1"
```

第二遍从第一遍的实际输出继续处理。再次运行请换新的输出目录。省略 `--output` 会自动分配目录。可先看 `docs\USAGE.md`，再让 Codex 或 DeepSeek Harness 按对应的 `adapters\…\OPERATIONS.md` 使用上述本地命令。默认渲染、网格和裁片可离线运行。在线模型评图由各自工具另行配置。

## 自检与常见问题

- 可在包根目录执行 `.\runtime\python.exe -B -X utf8 .\windows\verify.py`。它会在临时目录内实际处理带中文及空格路径的样图，并检查两种入口、连续两遍、原尺寸、透明通道、保护蒙版和三栏裁片。成功时显示 PASS。
- 若提示找不到 runtime，请先右键 ZIP 选择“全部解压”。建议放在路径较短的可写目录，尽量避免层层嵌套。
- 若导入 cv2 提示 DLL 加载失败，请检查系统架构，以及 Microsoft Visual C++ x64 运行库。Windows N / KN 版还可能需要系统的 Media Feature Pack。[微软运行库说明](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- 此包提供日常 CPU 流程。可选神经网络降噪需要另配兼容的 Windows Python 工作环境及权重，详见 `docs\USAGE.md`。Python 许可在 `runtime\LICENSE.txt`，依赖的许可随对应的 dist-info 或包目录保留。
