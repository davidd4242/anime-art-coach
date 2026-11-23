"""Colour utilities and background-based white balance.

All images are float32 RGB in 0..1, shape H x W x 3.
"""
import cv2
import numpy as np


# ---------------------------------------------------------------- conversions
def to_lab(img):
    """float32 RGB 0..1 -> Lab (L 0..100, a/b roughly -128..127)."""
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def from_lab(lab):
    """Lab -> float32 RGB 0..1 (clipped)."""
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB), 0, 1)


def chroma(lab):
    return np.hypot(lab[..., 1], lab[..., 2])


def load_image(path):
    """Read any Pillow-supported image as float32 RGB 0..1."""
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def load_source(path):
    """Read straight RGB and, if present, the original opacity (including tRNS)."""
    from PIL import Image
    with Image.open(path) as im:
        alpha = None
        if "A" in im.getbands() or "transparency" in im.info:
            alpha = np.asarray(im.convert("RGBA").getchannel("A")).copy()
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return rgb, alpha


def save_image(path, img):
    from PIL import Image
    arr = np.clip(np.round(np.asarray(img, dtype=np.float32) * 255), 0, 255)
    Image.fromarray(arr.astype(np.uint8)).save(path)


# ---------------------------------------------------------------- background
def background_mask(img, lab=None, min_l=90.0, max_chroma=8.0, barrier=None):
    """Near-white, low-chroma pixels 4-connected to the image border.

    `barrier` (bool) pixels are never part of the background (used to stop the
    flood fill at line art so white clothing is not swallowed through gaps).
    """
    if lab is None:
        lab = to_lab(img)
    cand = (lab[..., 0] >= min_l) & (chroma(lab) <= max_chroma)
    if barrier is not None:
        cand &= ~barrier
    n, cc = cv2.connectedComponents(cand.astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
    border = border[border > 0]
    return np.isin(cc, border)


def white_balance(img, min_fraction=0.02):
    """Estimate the colour cast from the white background and remove it.

    Background = bright (L>=90), low-chroma (C<=8) pixels connected to the
    border.  The per-channel gain maps the background's robust mean to a
    neutral grey as bright as its brightest channel, so the
    background becomes neutral and the cast is removed from the subject too.
    If fewer than `min_fraction` of the pixels are background the image is
    returned unchanged.
    """
    lab = to_lab(img)
    bg = background_mask(img, lab)
    frac = float(bg.mean())
    info = {"background_fraction": round(frac, 4), "applied": False}
    if frac < min_fraction:
        info["reason"] = "not enough background"
        return img.copy(), info
    px = img[bg]
    # trimmed mean is robust to anti-aliased edge pixels that slipped in
    lo, hi = np.percentile(px, [5, 95], axis=0)
    keep = np.all((px >= lo) & (px <= hi), axis=1)
    mean = px[keep].mean(0) if keep.sum() > 100 else px.mean(0)
    # map the background to a neutral grey as bright as its brightest channel
    target = float(mean.max())
    gain = target / np.maximum(mean, 1e-4)
    cast = float(np.abs(mean - target).max())
    info.update({"bg_mean_rgb": [round(float(v), 4) for v in mean],
                 "gain": [round(float(v), 4) for v in gain],
                 "cast": round(cast, 4)})
    if cast < 1.0 / 255:  # already neutral: nothing to do
        info["reason"] = "background already neutral"
        return img.copy(), info
    info["applied"] = True
    return np.clip(img * gain[None, None, :], 0, 1).astype(np.float32), info


# ---------------------------------------------------------------- counting
def count_colors(lab_px, de=3.0, min_share=0.01, max_px=60000, seed=0):
    """Number of perceptually distinct colours in a set of Lab pixels.

    Greedy leader clustering: colours (binned at 1 ΔE) are visited from the
    most to the least frequent; each joins the first existing cluster centre
    closer than `de`, otherwise it founds a new cluster.  Clusters holding
    less than `min_share` of the pixels (anti-aliasing ramps between two
    levels, stray pixels) are not counted.
    """
    px = np.asarray(lab_px, np.float32).reshape(-1, 3)
    if len(px) == 0:
        return 0
    if len(px) > max_px:
        px = px[np.random.default_rng(seed).choice(len(px), max_px, replace=False)]
    q = np.round(px).astype(np.int32)
    uq, cnt = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-cnt)
    centres, weights = [], []
    for i in order:
        c = uq[i].astype(np.float32)
        if centres:
            d = np.linalg.norm(np.asarray(centres) - c, axis=1)
            j = int(d.argmin())
            if d[j] < de:
                weights[j] += cnt[i]
                continue
        centres.append(c)
        weights.append(int(cnt[i]))
    w = np.asarray(weights, np.float64) / len(px)
    return int((w >= min_share).sum())
