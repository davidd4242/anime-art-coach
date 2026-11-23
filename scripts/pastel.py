"""Retain original low-contrast pigment and detail in high-key paintings."""
import cv2
import numpy as np

from colorfix import to_lab
from masks import _ellipse
from pigment import paired_banks


def ramp(values, low, high):
    return np.clip((values - low) / (high - low), 0, 1)


def pastel_protection(img, silhouette, lines, alpha=None):
    """Protect source-supported paint and accents before material-wide edits.

    A pale palette is measured on the subject. Within it, coherent Lab
    pigment and faint dark ridges carry detail even when hard ink or the
    material classifier cannot resolve them.
    """
    subject = silhouette if alpha is None else silhouette & (alpha >= 32)
    if not subject.any():
        return None
    lab = to_lab(img)
    lightness = lab[..., 0]
    chroma = np.linalg.norm(lab[..., 1:], axis=2)
    visible_l = lightness[subject]
    visible_c = chroma[subject]
    lower, median = np.percentile(visible_l, [20, 50])
    high_key = float(ramp(lower, 63, 72) * ramp(median, 80, 86))
    subdued = 1 - float(ramp(np.mean(visible_c > 26), .12, .28))
    palette = high_key * subdued
    if palette < .01:
        return None

    w = subject.astype(np.float32)
    blur = lambda field, sigma: cv2.GaussianBlur(field, (0, 0), sigma)
    sigma = np.clip(min(w.shape) * .010, 2, 8)
    den = np.maximum(blur(w, sigma), 1e-5)
    surround_l = blur(w * lightness, sigma) / den
    mean_ab = blur(w[..., None] * lab[..., 1:], sigma) / den[..., None]
    stable_c = np.linalg.norm(mean_ab, axis=2)
    alignment = np.sum(mean_ab * lab[..., 1:], axis=2) / np.maximum(stable_c * chroma, 1.)
    tint = ramp(stable_c, 1.4, 5.1) * ramp(alignment, .45, .92)

    # A neutral bright cleft still needs nearby paint contrast to read.
    # Give the strongest retention to coherent pigment, and leave a larger
    # edit budget in neutral highlights and discontinuous colour.
    paint = .58 + .35 * tint
    # Sparse dark lines and small facial marks weigh much more than their
    # area in a bright painting. Give the existing pixels and their close
    # antialiasing banks priority before shadow or ink simplification.
    step = max(1, round(.34 * sigma))
    valley = np.maximum(paired_banks(lightness, step),
                        paired_banks(lightness, 2 * step))
    recess = ramp(surround_l - lightness, 3, 12)
    ridge = ramp(valley, 1.1, 3.1) * ramp(surround_l-lightness, .9, 2.7)
    accent = np.maximum(ridge, recess * (lines | (lightness < 76)))
    halo = cv2.dilate(np.float32(accent), _ellipse(2 * step + 3))
    detail = np.maximum(accent, blur(halo, max(.7, step * .42)))
    return np.float32(palette * np.maximum(paint, detail) * w)
