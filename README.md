# 降低ai味—用算法教导你的大模型学会画画

给模型一张动漫立绘，让它用本地 Python 工具调整描线、衣褶、皮肤亮纹和色块，再用原图检查装饰有没有丢。适合已有成图的后期处理。构图、姿势和角色设计由输入图提供。

[下载 Codex 版](https://github.com/davidd4242/anime-art-coach/releases/latest/download/anime-art-coach-codex.zip) · [下载 DSH 版](https://github.com/davidd4242/anime-art-coach/releases/latest/download/anime-art-coach-dsh.zip)

| 选哪个 | 运行之后得到什么 |
|---|---|
| Codex | 默认成图、分层 PNG、保护蒙版、运行报告；可用自然语言让 Codex 执行和复核。 |
| DSH（DeepSeek Harness） | 同一内核的成图，另有坐标网格、变化读数和同位置裁片。适合让视觉能力较弱的模型逐项检查。 |

## 先看两个案例

左栏来自对话中最初提交的透明 PNG，右栏来自作者指定展示的最终交付 JPG。两栏在同尺寸白底上展示，局部采用同一位置、同一倍率。交付 JPG 已缩放和压缩；复现内核的像素验证使用原尺寸 PNG。

### 陌白茉茉

看宽袖和领口蓝布的明暗，再看黑色交叉带中的几个孔。布料可以简化，孔的形状和交叠关系仍然要读得清。

![陌白茉茉原图与收到的成品对照](examples/momo-compare.png)

### 海星猫耳

看前腿上的长条高光、膝盖转面，再看袜口的金色装饰。亮纹在这张图上可以概括，金色小图案本身就是角色设计的一部分。

![海星猫耳原图与收到的成品对照](examples/starfish-compare.png)

## 第一次使用

解压一个对应版本，进入包含 `requirements.txt` 的目录。以下命令适用于 macOS / Linux。

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

在 Codex 版中执行 `python adapters/codex/run.py examples/momo.png runs/first`。在 DSH 版中执行 `python adapters/dsh/run.py examples/momo.png runs/first --crop 595 1011 655 1049`。打开 `runs/first/pass-1/result.png`。再次运行时给一个新输出目录。

[详细操作手册](docs/USAGE.md) · [算法原理与样本迭代](docs/METHOD.md) · [验证记录与示例来源](docs/VALIDATION.md) · [第三方组件](docs/THIRD_PARTY.md)

## 模型能学到什么

工作流把渲染决定写成可执行规则，把复核拆成可观察的任务。两种助手在同一环境对同一输入执行同一配置，可得到相同的渲染结果。模型判断哪种画法更合适仍然需要单独评测。本项目没有 DeepSeek V4.1 Flash 与 Astra 6.2 的识图等效性测试，也没有对这两种模型做权重训练。

按下载版本打开 `adapters/codex/OPERATIONS.md` 或 `adapters/dsh/OPERATIONS.md`，内含可直接交给模型的指令、操作步骤和判断范例。
