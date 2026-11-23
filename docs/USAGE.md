# 安装、处理和检查

简体中文 · [English](USAGE.en.md)

Windows 10 / 11 x64 可以直接下载内置运行环境的便携包，解压后双击 Run.cmd。下载链接和两种启动方式见[Windows 使用说明](../windows/README.md)。

**环境。** 本次从公开 PyPI 安装四项直接依赖，在 macOS arm64、Python 3.9.6 的干净环境中验证。建议使用能安装这些 wheel 的 64 位 Python。Windows PowerShell 可以先运行 `py -m venv .venv`，再用 `.\.venv\Scripts\python.exe` 代替下文中的 `python`。目录名有空格时加引号。运行位置是解压后的项目根目录。

## 1. 安装并确认解释器

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import cv2, numpy, skimage, PIL; print(cv2.__version__)"
```

若出现 No matching distribution，先确认 Python 版本、处理器架构和 pip 的下载源。若出现 ModuleNotFoundError，用同一个 python 重新执行 `-m pip install`。默认渲染走本地 CPU，输入可以是 PNG 或 JPG。透明图优先使用原始 PNG。

## 2. 渲染一次

```sh
# Codex 包
python adapters/codex/run.py examples/momo.png runs/momo
# DSH 包
python adapters/dsh/run.py examples/starfish.png runs/star --crop 475 715 651 1070
```

第一次用其中一条即可。两种入口要求使用新的输出目录，以免旧结果混入。运行中断后保留目录供诊断，改用新目录重试。原始文件、工作目录、成品文件分开放。

| 文件（在 pass-1 中） | 用途 |
|---|---|
| result.png | 按原输入尺寸输出的成图。先缩到相同显示高度看整张。 |
| layer_flat / layer_shadow / layer_highlight / layer_lines.png | 平涂色、乘法阴影、滤色高光、透明描线；四者按此顺序合成。 |
| debug_detail_protect.png | 本次保留原稿细节的权重。白色表示高权重。 |
| debug_materials.png / debug_linetiers.png | 检查材质归类和线条分级。 |
| report.json | 记录参数、区域、所用保护模式与处理时间。报告含本机输入及运行路径，对外分享前检查。 |

## 3. 连续处理与盲测

```sh
python adapters/dsh/run.py examples/momo.png runs/two --passes 2 --crop 595 1011 655 1049 --crop 592 402 647 462 --crop 370 350 437 425
```

Codex 版用对应的 run.py 和 `--passes 2`，去掉裁片参数。第二遍读取第一遍的实际 result.png。保留其相邻的 report.json 和蒙版，以便程序核对并沿用保护范围。两遍用于检验重复处理，平时默认一遍。审美盲测先看等高整图，随机编号，允许选“一样”；选完再看相同编号的局部。

## 4. 需要改参数时

两个包都可以直接调用 `python scripts/redraw.py --help`。直接入口使用 `-o 输出目录`，请给它新目录。默认 `--preset artist`、强度 1、自动装饰保护，额外降噪关闭。若整图压平过头，先单独降低强度，例如：

```sh
python scripts/redraw.py examples/starfish.png -o runs/star-light --strength 0.65
```

局部指定保留区域可用与源图同尺寸的灰度 PNG 和 `--detail-mask`，白色保留、黑色不额外保留。此蒙版会替代自动细节选择。先留默认图，再对照局部。陌白茉茉的单图选项另行使用：

```sh
python scripts/momo_collar.py runs/momo/pass-1
```

输出为同目录的 result_momo_collar.png。它校验原始比 V 人物 PNG 和日常结果，再在蓝色衣领指定范围补回一部分原有阴影。其他人物沿用通用算法。

## 5. 可选降噪

把额外依赖安装在独立环境。包内有上游依赖清单 `scripts/vendor/gpt_anime_denoise/requirements-macos-arm64-py312.txt`，对应 macOS arm64 和 Python 3.12；其他系统按上游平台说明安装。L0 模式无需模型权重。示例假定已创建并安装好 .denoise-venv。

```sh
python scripts/redraw.py examples/momo.png -o runs/momo-l0 --denoise l0_0.001 --denoise-python .denoise-venv/bin/python
```

可先试 `l0_0.0005`，再看纹理与饰物是否一起变淡。若选 `esrgan60_l0`，另从第三方说明中的官方发行页取权重，保持原文件名，使用 `--denoise-models 权重目录`。内核会核对 SHA-256。缺权重或依赖时会报错，默认流程无需下载模型或设置 API 密钥。

## 6. 结果怎么判

前后尺寸、透明通道和装饰数量是可核对的事实；“更好看”需要在具体画面上判断。亮面人类画师也会使用，尤其是强逆光皮肤和镜面金属。先检查高光属于什么材质、是否承担光向和体积信息。只凭“高光多”就一律处理，会损伤原画。弱视觉模型的逐步程序见 DSH 包中的操作书。
