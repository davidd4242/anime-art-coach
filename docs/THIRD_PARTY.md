# 第三方组件与素材范围

默认处理依赖 NumPy、OpenCV、scikit-image 和 Pillow。使用 pip 从各自发行渠道安装，各依赖适用其自身条款。

可选降噪使用 CXP-2024 的 gpt-anime-denoise v1.5.0，随包保留上游 denoise.py、MIT 许可及特定平台的依赖清单。ESRGAN 权重通过上游发行渠道单独取得，文件名为 RealESRGAN_x4plus_anime_6B.pth，期望 SHA-256 为 f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da。包内保留第三方许可；下载权重仅在显式选用该模式时需要。[降噪发行页](https://github.com/CXP-2024/gpt-anime-denoise/releases) · [Real-ESRGAN 权重](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)

公开展示素材仅为用户指定的陌白茉茉和海星猫耳。项目包含两张原始图的去元数据像素副本、两张用户交付图的去元数据像素副本、两张全身对照图、基于这两个案例制作的局部识别与调试示意图，以及一张服务于已审阅陌白原图的灰度保护蒙版。展示图权利归各自权利人；第三方代码的许可证仅适用于相应组件。训练营卡面、参考图库、来源压缩包及模型账号配置没有作为发行材料。
