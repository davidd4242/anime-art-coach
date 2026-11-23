"""Portable regression checks for ornament selection and layer reconstruction."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image


def exercise_marks(module):
    source = np.full((256, 320, 3), (.72, .78, .86), np.float32)
    cv2.rectangle(source, (45, 48), (94, 75), (.25, .24, .26), -1)
    cv2.rectangle(source, (48, 51), (91, 72), (.55, .84, .91), -1)
    cv2.line(source, (58, 62), (82, 62), (.93, .96, 1), 3)
    cv2.circle(source, (160, 77), 13, (.14, .12, .14), 5)
    render = source.copy()
    render[57:68, 55:86] = (.55, .84, .91)
    render[61:85, 155:168] = (.72, .78, .86)
    gray = np.linspace(.65, .85, 128, dtype=np.float32)
    source[170:220, 24:152] = gray[None, :, None]
    render[170:220, 24:152] = .75
    groups = np.ones(source.shape[:2], np.int32)
    for factor, mirrored in ((1., False), (1.4, True)):
        def transform(arr, categorical=False):
            changed = cv2.resize(arr, None, fx=factor, fy=factor,
                                 interpolation=cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR)
            return np.ascontiguousarray(changed[:, ::-1] if mirrored else changed)
        original, edited = transform(source), transform(render)
        plane = transform(groups, True)
        mask = module.automatic_mask(original, edited, plane, {1: 'cloth_light'}, None)
        def evidence(box):
            select = np.zeros(groups.shape, np.uint8)
            l, t, r, b = box
            select[t:b, l:r] = 1
            return mask[transform(select, True).astype(bool)]
        assert evidence((58, 59, 82, 66)).mean() > .90, "faded sign"
        assert evidence((157, 62, 165, 68)).mean() > .65, "interrupted dark loop"
        assert evidence((30, 178, 146, 210)).max() == 0, "smooth tone"
        np.testing.assert_array_equal(mask, module.automatic_mask(original, edited, plane, {1:'cloth_light'}, None))
        assert not module.automatic_mask(original, original, plane, {1:'cloth_light'}, None).any()
        assert not module.automatic_mask(original, edited, plane, {1:'cloth_light'},
                                         np.zeros(plane.shape, np.uint8)).any()


def exercise_layers(module):
    from layers import Stack
    rng = np.random.default_rng(412)
    def random_stack():
        return Stack(*(rng.random((11, 19, n)).astype(np.float32) for n in (3, 3, 3, 4)))
    original, edited = random_stack(), random_stack()
    for stack in (original, edited):
        stack.lines[0, :, 3] = 0
        stack.lines[1, :, 3] = 1
    weight = rng.choice(np.float32([0, .27, .81, 1]), (11, 19))
    expected = original.composite() * weight[..., None] + edited.composite() * (1 - weight[..., None])
    kept = module.snapshot(edited, weight == 0)
    module.restore(edited, module.snapshot(original, weight > 0), weight)
    np.testing.assert_allclose(edited.composite(), expected, atol=3e-7)
    for name in ('flat', 'shadow', 'highlight', 'lines'):
        np.testing.assert_array_equal(getattr(edited, name)[weight == 0], getattr(kept, name))
        np.testing.assert_array_equal(getattr(edited, name)[weight == 1], getattr(original, name)[weight == 1])


def exercise_overrides(redraw):
    from verify_denoise import fixture
    with TemporaryDirectory(prefix='ornament-controls-') as temp:
        root = Path(temp)
        src = root / 'source.png'
        source = fixture(src)
        mask_path = root / 'detail.png'
        Image.fromarray(np.full(source.shape[:2], 255, np.uint8)).save(mask_path)
        out = root / 'reused-output'
        def run(**options):
            return redraw.run(src, out, verbose=False, **options)[1]['detail_preservation']
        assert run(detail_mask=mask_path)['mode'] == 'explicit'
        np.testing.assert_array_equal(np.array(Image.open(out/'result.png')), source)
        paint = np.zeros(source.shape[:2], np.uint8)
        paint[20:120, 35:96] = 189
        Image.fromarray(paint).save(mask_path)
        run(detail_mask=mask_path)
        previous = np.array(Image.open(out/'result.png'))
        again = root / 'chained'
        _, info = redraw.run(out/'result.png', again, verbose=False)
        assert info['detail_preservation']['carried_from_input']
        np.testing.assert_array_equal(np.array(Image.open(again/'result.png'))[paint != 0], previous[paint != 0])
        from detail_protection import previous_result_mask
        assert previous_result_mask(src, paint.shape) is None
        other = out / 'other.png'
        Image.fromarray(source).save(other)
        assert previous_result_mask(other, paint.shape) is None
        Image.fromarray(np.flipud(paint)).save(out/'debug_detail_protect.png')
        assert previous_result_mask(out/'result.png', paint.shape) is None
        assert run()['mode'] == 'automatic', "a previous explicit mask leaked into a new run"
        assert run(mask=mask_path)['mode'] == 'off', "legacy mask must control selection"
        assert run(strength=0)['mode'] == 'off'
        first = (out/'result.png').read_bytes()
        run(strength=0, detail_protect='off')
        assert (out/'result.png').read_bytes() == first
        Image.new('L', (9, 13)).save(mask_path)
        try:
            run(detail_mask=mask_path)
        except ValueError as exc:
            assert 'resolution' in str(exc)
        else:
            raise AssertionError('a mask for another canvas was accepted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scripts', type=Path, default=Path(__file__).resolve().parents[1]/'scripts')
    args = parser.parse_args()
    sys.path.insert(0, str(args.scripts.resolve()))
    import detail_protection
    import redraw
    exercise_marks(detail_protection)
    exercise_layers(detail_protection)
    exercise_overrides(redraw)
    print('PASS: lost marks, resize and mirror, smooth surfaces, alpha, layer math, mask precedence, repeat runs')
