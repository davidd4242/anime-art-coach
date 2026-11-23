# Codex 操作书

简体中文 · [English](OPERATIONS.en.md)

在 Codex 打开解压后的根目录，让它读本文件和 docs/USAGE.md。模型负责解释操作和核对结果，Python 内核负责渲染。无需为渲染配置图像生成服务。

1. 确认用户给的是源图还是历史成品。只把原图交给渲染器。
2. 用当前解释器安装 requirements.txt。先运行 `python adapters/codex/run.py examples/momo.png runs/codex-first`。给新文件夹，保留源文件。
3. 打开 pass-1/result.png，看与原图同高的全身对照。观察脸部外缘、发束和衣料内线的强弱，检查浅色图标和交叠阴影。
4. 需要验证累计损失时，用 `python adapters/codex/run.py examples/momo.png runs/codex-repeat --passes 2`。检查两遍的 report.json、装饰蒙版和原位置裁片。源图坐标按左上原点记录。
5. 以实际输出作答，附本地文件链接。区分可见改善、持平、退化。若装饰破损，记录物件、坐标、首次退化遍数和参数；调参另存结果。
6. 若用户明确要求保留陌白领口阴影，另用 `python scripts/momo_collar.py runs/codex-first/pass-1`，展示额外局部版本。日常出图留在 result.png。

复制给 Codex 的示例请求：

> 运行本目录中的日常算法，用 examples/starfish.png 作原图，输出 runs/star-check。连续跑两遍，保持装饰保护。先给我看整图，再按同一坐标比较袜口星纹、腿部高光和膝盖投影；说明哪些细节保住、哪些结论需我判断。

如果需要让 AI 连续复核多张图，每张图单独建目录。开发样本与验收样本按作品分开，具体办法见 docs/METHOD.md。
