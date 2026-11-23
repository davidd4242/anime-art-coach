# Validation and reproduction

[简体中文](VALIDATION.md) · English

## Identify the inputs

| Case | Original RGBA PNG | User-supplied final image | Inspect |
|---|---|---|---|
| Case 1 | `examples/momo.png`, 997 × 1578 | `examples/momo-delivered.png`, decoded from an 808 × 1280 JPEG | Bandage face, rabbit badge, collar lacing and optional collar shadow |
| Case 2 | `examples/starfish.png`, 1024 × 1536 | `examples/starfish-delivered.png`, decoded from an 853 × 1280 JPEG | Leg highlights, knee volume and marine ornaments |

The archived originals matched the SHA-256 of the uploaded files. Public copies preserve their decoded pixels and strip image metadata. Use the original PNGs for rendering and the supplied final images for the illustrated comparisons.

## Two-pass reference results

The development environment used macOS arm64, Python 3.9.6, and the dependencies in `requirements.txt`. Each original went through two consecutive passes, with dimensions and alpha checked for every result. These hashes cover row-major decoded RGBA bytes, so PNG compression differences do not affect the check.

| Input | Pass 1 SHA-256 | Pass 2 SHA-256 |
|---|---|---|
| Case 1 | `0268bf2767e665cf90124c07acfbc671249965d6a124a26b2400a4667528ec1a` | `f119edc67fadd4e14562c83d6e9b2ce88fe7e76c97aeaa4d76c0446b1a84acfc` |
| Case 2 | `fe916748a4c186fb702182c26c2beabf8a2fd6da1a88ac25299c8f780656b1d0` | `a4db852511393c637eb88b1300a2dcc9c0694a497c5df038949ced281900ee6b` |

Case 1 matched both approved full-size renders pixel for pixel. If another platform yields different edge values, compare dependency versions, settings, masks, and crop structure.

```sh
python -c "from PIL import Image; import hashlib,sys; print(hashlib.sha256(Image.open(sys.argv[1]).convert('RGBA').tobytes()).hexdigest())" runs/case1/pass-1/result.png
python tests/verify_ornaments.py
```

Synthetic fixtures exercise resizing, mirroring, lost marks, smooth areas, alpha, mask precedence, and repeat passes. Checks on real artwork cover eyelets, chain links, eye reflections, a bandage motif, and crossed lacing. The fold-projection correction was also tested on a portrait and a card, each run twice. All four renders, shadow layers, and protection masks matched the previous output, and the numerical warnings cleared.

## Optional denoiser tests

CPU tests for L0 and ESRGAN checked geometry anchors, alpha, suppression of flat-region noise, and bypass behavior for a fully protected mask or zero strength. With the optional dependencies installed, run:

```sh
python tests/verify_denoise.py --denoise l0_0.001 --denoise-python .denoise-venv/bin/python
python tests/verify_denoise.py --denoise esrgan60_l0 --denoise-python .denoise-venv/bin/python --denoise-models path/to/weights
```

## Interpreting the evidence

In blind reviews, viewers saw the whole figure first and could call a tie. Some differences were only visible in enlarged crops. Score structural preservation separately from overall preference. These checks evaluate the local tool. A comparison of commercial models’ visual judgment requires its own evaluation.
