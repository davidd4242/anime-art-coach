"""Conservative candidate selection for paired narrow glints inside diffuse skin."""
from itertools import combinations
from typing import NamedTuple

import cv2
import numpy as np


class Ridge(NamedTuple):
    label: int
    center: np.ndarray
    axis: np.ndarray
    length: float
    contrast: float
    clearance: float


def detect(source, alpha):
    height, width = source.shape[:2]
    red, green, blue = source.transpose(2, 0, 1)
    skin = ((red > green + .018) & (red > blue + .022) & (red > .55)
            & (green > .34) & (np.abs(green-blue) < .18) & (red-green < .28))
    if alpha is not None:
        skin &= alpha > 120
    radius = max(3, round(min(width, height) * .006))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    skin = cv2.morphologyEx(skin.astype('uint8'), cv2.MORPH_CLOSE, disk)
    clearance = cv2.distanceTransform(skin, cv2.DIST_L2, 5)
    gray = cv2.GaussianBlur(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), (0, 0), .7)
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, disk)
    seeds = (clearance > radius * 1.6) & (top > .011)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(seeds.astype('uint8'))
    features = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 12 or max(w, h) < 3 * radius:
            continue
        yy, xx = np.nonzero(labels[y:y+h, x:x+w] == i)
        eigenvalues, axes = np.linalg.eigh(np.cov(np.stack((xx, yy))))
        if eigenvalues[1] < 12*max(1, eigenvalues[0]) or np.sqrt(eigenvalues[0]) > radius:
            continue
        coords = np.stack((xx+x, yy+y), axis=1)
        axis = axes[:, 1]
        offset = axes[:, 0] * max(radius, 3*np.sqrt(eigenvalues[0]))

        def sample(shift):
            ix, iy = np.rint(coords+shift).astype(int).T
            return gray[np.clip(iy, 0, height-1), np.clip(ix, 0, width-1)]

        contrast = np.median(sample(0)-np.maximum(sample(offset), sample(-offset)))
        if contrast >= .008:
            length = np.ptp(np.sum(coords*axis, axis=1))
            features.append(Ridge(i, centers[i], axis, length, contrast,
                                  np.median(clearance[yy+y, xx+x])))
    return features, labels, top, clearance, radius


def even_surroundings(luminance, ridge, radius):
    """Compare both outer banks to the ridge's own signal, in source pixels."""
    along = np.linspace(-.33, .33, 13) * ridge.length
    across = np.r_[np.linspace(-2.5, -1.5, 8), np.linspace(1.5, 2.5, 8)] * radius
    normal = np.array([-ridge.axis[1], ridge.axis[0]])
    points = ridge.center + along[:, None, None]*ridge.axis + across[None, :, None]*normal
    x, y = np.rint(points).astype(int).transpose(2, 0, 1)
    h, w = luminance.shape
    if x.min() < 0 or y.min() < 0 or x.max() >= w or y.max() >= h:
        return False
    banks = luminance[y, x]
    left = banks[:, :8].mean(axis=1)
    right = banks[:, 8:].mean(axis=1)
    tolerance = .4 * ridge.contrast
    return max(np.mean(np.abs(left-right)), left.std(), right.std()) <= tolerance


def paired_mask(source, alpha):
    features, labels, top, clearance, radius = detect(source, alpha)
    chosen = set()
    luminance = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    surroundings = {}
    for a, b in combinations(features, 2):
        distance = np.linalg.norm(a.center-b.center)
        direction = (a.center-b.center) / max(distance, 1.)
        # Only separate, collinear marks, with similar width of surrounding skin.
        shorter, longer = sorted((a.length, b.length))
        # A dimmer exposure can shorten the detected tip of one stroke.
        # Use both lengths to estimate spacing while rejecting tiny fragments.
        if shorter < .5*longer or not (1.7*longer < distance < 3.6*np.sqrt(shorter*longer)):
            continue
        if min(abs(np.dot(a.axis, direction)), abs(np.dot(b.axis, direction))) < .982:
            continue
        if min(a.clearance, b.clearance) < max(3.5*radius, .72*max(a.clearance, b.clearance)):
            continue
        if max(a.contrast, b.contrast) > min(.048, 1.65*min(a.contrast, b.contrast)):
            continue
        x, y = np.rint(np.linspace(a.center, b.center, 64)).astype(int).T
        if np.min(clearance[y, x]) < 1.6*radius:
            continue
        # A large colour or exposure swing along the limb suggests directional light.
        if np.max(np.ptp(source[y, x], axis=0)) > .19:
            continue
        for ridge in (a, b):
            if ridge.label not in surroundings:
                surroundings[ridge.label] = even_surroundings(luminance, ridge, radius)
        if surroundings[a.label] and surroundings[b.label]:
            chosen.update((a.label, b.label))
    mask = np.isin(labels, list(chosen)).astype('uint8')
    mask = cv2.dilate(mask, np.ones((3, 3), dtype='uint8'))
    return mask, top, len(features), len(chosen)


def apply(stack, source, alpha, protection, strength):
    mask, _, candidates, selected = paired_mask(source, alpha)
    active = (mask > 0) & (stack.lines[..., 3] < .08)
    if protection is not None:
        active &= protection == 0
    if active.any():
        paint = stack.flat * stack.shadow
        paint += (1-paint) * stack.highlight
        gray = cv2.cvtColor(paint, cv2.COLOR_RGB2GRAY)
        radius = max(3, round(min(gray.shape)*.006))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        top = cv2.morphologyEx(cv2.GaussianBlur(gray, (0, 0), .7), cv2.MORPH_TOPHAT, disk)
        fill = np.stack([cv2.inpaint(np.ascontiguousarray(channel), active.astype('uint8'),
                                     3, cv2.INPAINT_NS) for channel in paint.transpose(2, 0, 1)], axis=-1)
        target = paint[active] - strength*np.clip((paint-fill)[active], 0, 2*top[active, None])
        base = stack.flat[active]
        diffuse = np.minimum(base*stack.shadow[active], target)
        stack.shadow[active] = np.divide(diffuse, base, out=np.ones_like(base), where=base>0)
        stack.highlight[active] = np.divide(target-diffuse, 1-diffuse, out=np.zeros_like(base), where=diffuse<1)
    return {"candidates": candidates, "selected": selected, "eligible_pixels": int(active.sum())}
