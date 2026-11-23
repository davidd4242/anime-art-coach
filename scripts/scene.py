"""Canvas coverage and conservative treatment of already-painted scenery."""
import cv2
import numpy as np

from colorfix import to_lab


def canvas_mode(rgb, alpha):
    if alpha is not None and np.any(alpha < 255):
        return "transparent"
    # A studio backdrop occupies much of the perimeter. A light wall or a
    # few white objects in a scene do not establish a neutral reference.
    h, w = rgb.shape[:2]
    band = max(1, round(min(h, w) * .012))
    parts = (rgb[:band], rgb[-band:], rgb[:, :band], rgb[:, -band:])
    boundary = np.concatenate([s.reshape(-1, 3) for s in parts])
    white = (boundary.min(1) > .89) & (np.ptp(boundary, axis=1) < .075)
    return "paper" if white.mean() > .70 else "scene"


def analysis_plate(rgb, alpha):
    """Extend straight edge colours a few pixels under zero-alpha samples.
    Arbitrary invisible RGB must not influence ink or material detection."""
    visible = alpha > 0
    plate = np.ones_like(rgb)
    if not visible.any():
        return plate
    distance, nearest = cv2.distanceTransformWithLabels(
        (~visible).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    halo = (distance > 0) & (distance <= 4)
    plate[visible] = rgb[visible]
    plate[halo] = rgb[visible][nearest[halo] - 1]
    return plate


def scene_protection(img, labels, lines):
    """Keep border-connected scenery and texture. Give enclosed, outlined
    surfaces the strongest edit where their colours differ from the perimeter."""
    h, w = labels.shape
    band = max(1, round(min(h, w) * .025))
    perimeter = np.zeros_like(labels, dtype=bool)
    perimeter[:band] = perimeter[-band:] = True
    perimeter[:, :band] = perimeter[:, -band:] = True
    border_ids = np.unique(labels[perimeter])
    anchored = np.isin(labels, border_ids)

    # A smoothed boundary colour histogram expresses the scene's own palette.
    # Only opacity of the edit is varied. No palette colour is pasted in.
    lab = to_lab(img)
    steps = np.array([8, 10, 10], dtype=np.float32)
    bins = np.floor((lab + np.array([0, 130, 130], dtype=np.float32)) / steps).astype(np.int32)
    shape = (14, 28, 28)
    index = np.ravel_multi_index(tuple(bins.transpose(2, 0, 1)), shape, mode="clip")
    counts = np.bincount(index[perimeter], minlength=np.prod(shape)).reshape(shape)
    # Adjacent quantisation bins must agree at boundaries of the same paint.
    kernel = np.array([.2, .6, .2], dtype=np.float32)
    freq = counts.astype(np.float32)
    for axis in range(3):
        freq = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), axis, freq)
    frequency = freq.ravel()[index] / perimeter.sum()
    distinct = np.exp(-frequency / .0025).astype(np.float32)
    # Fine repeated grain in skies and fabrics is subject to false ink
    # extraction. Reserve edits for surfaces with larger coherent interiors.
    sigma = max(2, min(h, w) * .013)
    near_ink = cv2.GaussianBlur(lines.astype(np.float32), (0, 0), sigma)
    structured = np.clip(near_ink / .12, 0, 1)
    confidence = (~anchored) * (.12 + .88 * distinct) * structured
    confidence = cv2.GaussianBlur(confidence.astype(np.float32), (0, 0), max(1, sigma * .32))
    # A full-frame scene has no explicit alpha or guaranteed semantic
    # silhouette. Apply a lighter edit to each plausible subject region.
    return 1 - .57 * confidence
