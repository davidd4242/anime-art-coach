# 验证过程与复跑

## 先核对素材身份

| 角色 | 公开原稿 | 交付成品来源 | 用途 |
|---|---|---|---|
| 陌白茉茉（抬手比 V） | examples/momo.png，997 × 1578 RGBA | examples/momo-delivered.png，来自 808 × 1280 JPG | 检查猫脸、兔子、黑色交叉带，以及局部阴影选项 |
| 海星猫耳 | examples/starfish.png，1024 × 1536 RGBA | examples/starfish-delivered.png，来自 853 × 1280 JPG | 检查腿部亮纹、膝盖和海洋装饰 |

原稿取自对话工作目录。两份存档的文件 SHA-256 分别与当时上传的原始 PNG 相符。发布时以解码像素重新保存，移除图像元数据。右栏用用户提供的成品做展示；它已经缩放并经过 JPEG 压缩，不用作渲染输入或逐像素回归基准。

## 日常内核回归

在 macOS arm64、Python 3.9.6 上，从公开包索引安装 requirements.txt。用两个原稿各连续跑两遍，输入第二遍的是第一遍结果。验证全部成图的原尺寸与完整透明通道。以下是按 RGBA 通道顺序逐行哈希的像素值（非 PNG 文件哈希），可排除编码器压缩差异。

| 输入 | 第一遍 SHA-256 | 第二遍 SHA-256 |
|---|---|---|
| momo | 0268bf2767e665cf90124c07acfbc671249965d6a124a26b2400a4667528ec1a | f119edc67fadd4e14562c83d6e9b2ce88fe7e76c97aeaa4d76c0446b1a84acfc |
| starfish | fe916748a4c186fb702182c26c2beabf8a2fd6da1a88ac25299c8f780656b1d0 | a4db852511393c637eb88b1300a2dcc9c0694a497c5df038949ced281900ee6b |

陌白两遍均与收版时保存的对应原尺寸成图逐像素一致。不同平台的底层数值库和依赖可能令边缘像素不同；出现差异时同时检查尺寸、蒙版、运行参数与局部结构。

```sh
python -c "from PIL import Image; import hashlib,sys; print(hashlib.sha256(Image.open(sys.argv[1]).convert('RGBA').tobytes()).hexdigest())" runs/momo/pass-1/result.png
python tests/verify_ornaments.py
```

装饰测试以程序生成的图形检验缩放、镜像、印记丢失、平滑表面、透明度、蒙版优先级及连续处理。真实图复核曾覆盖袜口穿绳孔、细链、眼睛反光、创可贴与交叉带。一个区域复测成功只支持该区域，不外推所有金属或所有画师。

可选降噪在独立工作解释器上通过了 L0 和 ESRGAN 两种 CPU 测试，包括几何锚点、透明度、低纹理噪声下降、零强度和全白保护绕过。具备相应依赖时执行：

```sh
python tests/verify_denoise.py --denoise l0_0.001 --denoise-python .denoise-venv/bin/python
python tests/verify_denoise.py --denoise esrgan60_l0 --denoise-python .denoise-venv/bin/python --denoise-models 权重目录
```

衣褶数值修正仅调整折面坐标投影的计算。定版前，陌白与一张卡面分别跑两遍，候选修正消除了数值警告；四次成图、衣褶阴影层和装饰蒙版均与原版逐像素相同。

## 审美验证与局限

历史全身盲测允许选择“持平”，之后才查看同坐标局部。部分改动仅放大可见；保留装饰和让画面更讨喜应分开评分。本次验证覆盖本地工具，不包含商业模型的在线能力排名。
