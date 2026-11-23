"""Check optional denoising on a generated RGBA illustration using CPU."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw


LAYERS = ('flat', 'shadow', 'highlight', 'lines')
GRAIN_SAMPLE = np.s_[70:130, 90:145, 0]


def fixture(path):
    width, height = 426, 362
    image = Image.new('RGB', (width, height), '#efeee8')
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((16, 16, width - 17, height - 17), radius=28,
                           fill=(127, 154, 179), outline='#29394a', width=5)
    draw.ellipse((width - 116, 95, width - 39, 170), fill=(227, 191, 160),
                 outline=(92, 64, 71), width=4)
    draw.rectangle((36, 204, width - 40, height - 37), outline='#2c2031', width=4)
    rgb = np.array(image)
    ramp = np.linspace(0, 45, width - 83, dtype=np.uint8)
    rgb[208:height - 41, 40:width - 43] = np.array([78, 55, 70]) + ramp[None, :, None]
    rng = np.random.default_rng(941)
    rgb[57:163, 28:width - 120] += rng.integers(0, 11, (106, width - 148, 1), dtype=np.uint8)

    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[8:-8, 8:-8] = 79
    alpha[10:-10, 10:-10] = 167
    alpha[12:-12, 12:-12] = 255
    alpha[178:191, width - 51:width - 38] = 0
    hidden = alpha == 0
    rgb[hidden] = rng.integers(0, 256, (hidden.sum(), 3), dtype=np.uint8)
    source = np.dstack((rgb, alpha))
    Image.fromarray(source).save(path)
    return source


def read_png(path):
    with Image.open(path) as image:
        return np.array(image)


def check_anchored(directory, source, before, after):
    result_off, off, debug_off = before
    result_on, on, debug_on = after
    assert on['denoise']['applied'] and on['denoise']['analysis_mode'] == 'source_geometry', on['denoise']
    assert debug_off['gmap'].any() and debug_off['lines'].any() and off['light_confidence'] > 0, (
        'fixture must exercise regions, ink and directional light')
    for name in ('lines', 'silhouette', 'face', 'tiers', 'gmap', 'eyes'):
        np.testing.assert_array_equal(debug_off[name], debug_on[name], err_msg=name)
    for name in ('regions', 'light_to_source_xy', 'light_confidence'):
        assert off[name] == on[name], f'denoise changed {name}'
    for name in ('debug_protect.png', *(f'original_layers/layer_{layer}.png' for layer in LAYERS)):
        assert (directory / 'off' / name).read_bytes() == (directory / 'on' / name).read_bytes(), name

    alpha = source[..., 3]
    for name in ('result.png', 'layer_flat.png', 'original_layers/layer_flat.png'):
        pixels = read_png(directory / 'on' / name)
        assert pixels.shape == source.shape, name
        np.testing.assert_array_equal(pixels[..., 3], alpha, err_msg=name)
        np.testing.assert_array_equal(pixels[alpha == 0], source[alpha == 0], err_msg=name)

    noise_before = float(result_off[GRAIN_SAMPLE].std())
    noise_after = float(result_on[GRAIN_SAMPLE].std())
    assert 0 < noise_before and noise_after < 0.6 * noise_before, (noise_before, noise_after)
    print(f'Anchors, layer export and alpha preserved. Flat noise {noise_before * 255:.3f} -> '
          f'{noise_after * 255:.3f} RGB codes.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scripts', type=Path, default=Path(__file__).resolve().parents[1] / 'scripts',
                        help='directory containing redraw.py')
    parser.add_argument('--denoise-python', help='worker interpreter with optional dependencies')
    parser.add_argument('--denoise-models', type=Path, help='local model directory for ESRGAN')
    parser.add_argument('--denoise', default='l0_0.001',
                        choices=('l0_0.001', 'l0_0.0005', 'esrgan60_l0'))
    args = parser.parse_args()
    scripts = args.scripts.resolve()
    if not (scripts / 'redraw.py').is_file():
        parser.error(f'redraw.py was not found in {scripts}')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(scripts))
    import redraw

    cv2.setNumThreads(1)
    with TemporaryDirectory(prefix='verify-denoise-') as scratch:
        directory = Path(scratch)
        source_path = directory / 'source.png'
        source = fixture(source_path)
        unavailable = str(directory / 'unavailable-worker')

        def run(name, **options):
            return redraw.run(source_path, directory / name, denoise_device='cpu',
                              denoise_models=args.denoise_models, verbose=False,
                              return_debug=True, **options)

        off = run('off', denoise_python=unavailable)
        assert off[1]['denoise']['reason'] == 'disabled'
        on = run('on', denoise=args.denoise, denoise_amount=1, denoise_python=args.denoise_python)
        check_anchored(directory, source, off, on)

        mask = directory / 'white_mask.png'
        Image.new('L', (source.shape[1], source.shape[0]), color=255).save(mask)
        for expected, control, options in (
            ('fully_protected', 'white_off', {'mask': mask}),
            ('zero_strength', 'zero_off', {'strength': 0}),
            ('zero_amount', 'off', {'denoise_amount': 0}),
        ):
            if control != 'off':
                run(control, denoise_python=unavailable, **options)
            _, report, _ = run(expected, denoise=args.denoise, denoise_python=unavailable, **options)
            info = report['denoise']
            assert not info['applied'] and info['reason'] == expected, info
            assert (directory / expected / 'result.png').read_bytes() == (
                directory / control / 'result.png').read_bytes(), expected
        print('White mask, zero strength and zero amount bypass the worker.', flush=True)


if __name__ == '__main__':
    main()
