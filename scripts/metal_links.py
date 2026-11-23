"""Keep a compact glint and its curved rim together when a redraw flattens them."""
import cv2
import numpy as np


def _disk(radius):
    size = 2 * max(1, round(radius)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def protection(source, render, alpha, existing_weight):
    height, width = source.shape[:2]
    scale = max(.6, min(height, width) / 1024)
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    painted = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
    kernel = _disk(3.8 * scale)
    glint = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    painted_glint = cv2.morphologyEx(painted, cv2.MORPH_TOPHAT, kernel)
    # A weak, thin painted rim can disappear even when the brightest tip survives.
    shadow = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    painted_shadow = cv2.morphologyEx(painted, cv2.MORPH_BLACKHAT, kernel)
    loss = np.maximum(glint - painted_glint, shadow - painted_shadow)
    loss *= 1 - existing_weight
    seeds = (glint > .085) & (gray > .45) & (np.ptp(source, axis=2) < .36)
    if alpha is not None:
        seeds &= alpha > 220
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds.astype('uint8'))
    mask = np.zeros((height, width), 'uint8')
    angles = np.arange(16) * (2 * np.pi / 16)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 2 or area > 95 * scale**2 or max(w, h) > 19 * scale:
            continue
        region = np.s_[y:y+h, x:x+w]
        component = labels[region] == index
        yy, xx = np.nonzero(component)
        strongest = np.argmax(glint[region][component])
        peak = np.array([x+xx[strongest], y+yy[strongest]])
        pad = max(4, round(9 * scale))
        local = np.s_[max(0, peak[1]-pad):peak[1]+pad+1, max(0, peak[0]-pad):peak[0]+pad+1]
        if glint[tuple(peak[::-1])] < .135 or np.count_nonzero(loss[local] > .065) < 3:
            continue
        # Test circles centred just inside the glint. Adjacent arc samples must
        # dip below both the inside and outside of the rim. This rejects most
        # isolated dots and the bright side of a single straight contour.
        # A metallic glint on these small fixtures sits beside a darker interior.
        # Bright paper, skin and broad cloth highlights have no dark backing.
        if np.median(gray[local]) > .58 or np.percentile(gray[local], 15) > .40:
            continue
        best_score = 0
        best = None
        for radius in (2.5*scale, 4.0*scale, 6.0*scale):
            centers = np.rint(peak + radius*unit).astype(int)
            def ring(factor):
                points = np.rint(centers[:, None] + radius * factor * unit).astype(int)
                xs = points[..., 0].clip(0, width-1)
                ys = points[..., 1].clip(0, height-1)
                return gray[ys, xs]
            trough = np.minimum(ring(.67), ring(1.45)) - ring(1.0)
            curved = trough > .039
            continuous = curved & np.roll(curved, 1, axis=1)
            score = continuous.sum(axis=1)
            winner = int(np.argmax(score))
            if score[winner] >= 4 and score[winner] > best_score:
                best_score = score[winner]
                best = (centers[winner], radius)
        if best is not None:
            center, radius = best
            size = int(np.ceil(radius*1.55+1))
            cx, cy = center
            left, right = max(0, cx-size), min(width, cx+size+1)
            top, bottom = max(0, cy-size), min(height, cy+size+1)
            if left >= right or top >= bottom:
                continue
            # No repair when the existing pipeline already preserves this rim.
            ry, rx = np.ogrid[top:bottom, left:right]
            inside = (rx-cx)**2 + (ry-cy)**2 <= size**2
            local_loss = loss[top:bottom, left:right][inside]
            if np.count_nonzero(local_loss > .065) < max(3, .045 * len(local_loss)):
                continue
            cv2.circle(mask, tuple(center), size, 1, -1)
    result = mask.astype('float32')
    if mask.any():
        result = cv2.GaussianBlur(result, (0, 0), max(.7, .9*scale))
        result[result < .01] = 0
    if alpha is not None:
        result[alpha == 0] = 0
    return result
