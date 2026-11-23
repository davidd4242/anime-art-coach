# Third-party software and artwork

[简体中文](THIRD_PARTY.md) · English

The daily renderer uses NumPy, OpenCV, scikit-image, and Pillow. Each dependency is distributed under its own terms. The Windows portable archive also contains CPython and its license.

Optional denoising incorporates CXP-2024’s gpt-anime-denoise v1.5.0. The distribution includes the upstream script, its MIT license, and the platform-specific requirements file. ESRGAN uses the separate weight file `RealESRGAN_x4plus_anime_6B.pth`. Its expected SHA-256 is `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da`. Only the ESRGAN mode needs this download. [Denoiser releases](https://github.com/CXP-2024/gpt-anime-denoise/releases) · [Official Real-ESRGAN weight](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)

The two public illustrations, Case 1 and Case 2, were selected by the user for demonstration. The project distributes metadata-stripped pixel copies of their originals and delivered images, comparison sheets, diagnostics, and a reviewed grayscale mask for Case 1. Artwork rights remain with the respective rights holders. The third-party software licenses cover their corresponding software. The private reference collection, its source archives, and model account configurations are excluded from distribution.
