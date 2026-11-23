"""Recover small drawn features that lose contrast in the edited illustration."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from layers import Stack


def reviewed_mask(rgb, alpha):
    """Bind a reviewed matte to the actual decoded image, including its coverage."""
    height, width = rgb.shape[:2]
    digest = hashlib.sha256(f"{width}x{height}:".encode("ascii"))
    pixels = np.rint(rgb * 255).astype(np.uint8)
    if alpha is not None:
        pixels[alpha == 0] = 0
    digest.update(pixels)
    if alpha is not None:
        digest.update(np.ascontiguousarray(alpha))
    path = Path(__file__).with_name("detail_masks") / (digest.hexdigest() + ".png")
    return read_mask(path, (height, width)) if path.is_file() else None


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def previous_result_mask(path, shape):
    """Carry a render's own protection forward only if both output files match its report."""
    path = Path(path)
    report = path.parent / "report.json"
    matte = path.parent / "debug_detail_protect.png"
    if not report.is_file() or not matte.is_file():
        return None
    try:
        info = json.loads(report.read_text())["detail_preservation"]
        if info["result_sha256"] != file_digest(path) or info["matte_sha256"] != file_digest(matte):
            return None
        weight = read_mask(matte, shape)
    except (KeyError, OSError, TypeError, ValueError):
        return None
    # The body of an already repaired detail gets full protection on a second
    # pass. Preserve its feathered edge without enlarging the selected area.
    return np.minimum(1., 4 * weight) if weight.any() else None


def _disk(radius):
    size = 2 * max(1, int(round(radius))) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _lost_marks(source, render, scale):
    """Compare signed local ink and pigment contrast, discounting smooth lighting."""
    original = cv2.GaussianBlur(source, (0, 0), max(.65, scale * .75))
    painted = cv2.GaussianBlur(render, (0, 0), max(.65, scale * .75))
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    score = np.zeros(source.shape[:2], np.float32)
    for radius in (2.8, 6.0):
        before = original - cv2.GaussianBlur(original, (0, 0), radius * scale)
        after = painted - cv2.GaussianBlur(painted, (0, 0), radius * scale)
        dark = -cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        dark_after = -cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        # A dark stroke needs two lighter banks. This excludes the lit side
        # of a gloss streak and most broad, one-sided shadow boundaries.
        trough = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _disk(radius * scale))
        ink = np.minimum(np.minimum(dark, trough) - .026, .68 * dark - dark_after)
        # Opponent colour keeps pale printed motifs detectable on a pale badge.
        residual = before - after
        chroma = np.ptp(before, axis=2)
        loss = np.ptp(residual, axis=2)
        pigment = np.minimum(chroma - .020, loss - .42 * chroma)
        np.maximum(score, np.maximum(ink, pigment), out=score)
    return score


def _outlined_marks(source, gray, painted_gray, subject, lost, scale):
    """Select enclosed pigment changes and dark cutouts against their own surround."""
    blurred = cv2.GaussianBlur(source, (0, 0), max(.55, .7 * scale))
    quantized = np.uint8(np.rint(255 * blurred))
    edges = cv2.Canny(quantized, 15, 42, L2gradient=True)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # A coloured print can interrupt the colour-gradient contour at tiny ears
    # or corners even when the luminance outline remains closed.
    gray_edges = cv2.Canny(cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY), 18, 46, L2gradient=True)
    gray_contours, _ = cv2.findContours(gray_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    dark_rim = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _disk(8 * scale))
    filled = np.zeros(subject.shape, np.uint8)
    pad = max(4, round(5 * scale))
    for contour in (*contours, *gray_contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (max(w, h) > 125 * scale or min(w, h) < 4 * scale or
                area < max(.32 * w * h, 18 * scale**2)):
            continue
        left, top = max(0, x - pad), max(0, y - pad)
        roi = np.s_[top:y+h+pad, left:x+w+pad]
        part = np.zeros(gray[roi].shape, np.uint8)
        cv2.drawContours(part, [contour], -1, 1, cv2.FILLED, offset=(-left, -top))
        inside = cv2.erode(part, _disk(1.3 * scale)).astype(bool)
        ring = (cv2.dilate(part, _disk(4 * scale)) > part) & subject[roi]
        inner_area = inside.sum()
        if (inner_area == 0 or ring.sum() < .25 * min(w, h) or
                (lost[roi] & inside).sum() < max(2, .025 * inner_area) or
                subject[roi][inside].mean() < .9):
            continue
        border = cv2.dilate(part, _disk(scale)) > cv2.erode(part, _disk(2 * scale))
        if (dark_rim[roi][border] > .045).mean() < .20:
            continue
        body = source[roi][inside]
        surround = np.median(source[roi][ring], axis=0)
        difference = np.median(body, axis=0) - surround
        # A print has its own pigment, distinct from the local host. A dark
        # strap needs an actual hole returning to the surrounding lightness.
        luminance = gray[roi][inside]
        low = luminance < np.mean(surround) - .23
        opens = np.percentile(luminance, 90) > np.mean(surround) - .13
        erasure = (painted_gray[roi][inside] - luminance > .17) & low
        perforated = opens and (difference.max() < -.20 or
                               (low.mean() > .12 and erasure.sum() > max(2, .09 * low.sum())))
        printed = np.ptp(difference) > .10 and np.quantile(body.max(1), .7) > .67
        pale_badge = (np.median(body.min(1)) > .74 and np.ptp(body, axis=1).std() > .075
                      and (lost[roi] & inside).sum() > .15 * inner_area)
        if perforated or printed or pale_badge:
            filled[roi] |= part
    return filled


def _buckle_rims(source, gray, lost, scale):
    """A light coloured opening surrounded by dark material and a bright metal rim."""
    mask = np.zeros(gray.shape, np.uint8)
    barrier = cv2.morphologyEx(np.uint8(gray < .56), cv2.MORPH_CLOSE, _disk(2 * scale))
    contours, hierarchy = cv2.findContours(barrier, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask
    for contour, parent in zip(contours, hierarchy[0, :, 3]):
        if parent < 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 7 * scale or max(w, h) > 55 * scale or cv2.contourArea(contour) < .37*w*h:
            continue
        pad = max(4, round(.30 * min(w, h)))
        left, top = max(0, x-pad), max(0, y-pad)
        roi = np.s_[top:y+h+pad, left:x+w+pad]
        fill = np.zeros(gray[roi].shape, np.uint8)
        cv2.drawContours(fill, [contour], -1, 1, cv2.FILLED, offset=(-left, -top))
        hole = cv2.erode(fill, _disk(scale)).astype(bool)
        rim = cv2.dilate(fill, _disk(pad))
        bank = (rim > fill)
        if not hole.any() or not bank.any():
            continue
        if np.median(gray[roi][hole]) < .67 or np.median(np.ptp(source[roi][hole], axis=1)) < .06:
            continue
        silver = (gray[roi] > .67) & (np.ptp(source[roi], axis=2) < .12)
        if ((gray[roi][bank] < .46).mean() > .27 and (silver & bank).sum() > .025 * bank.sum()
                and (lost[roi] & bank).sum() > .04 * bank.sum()):
            mask[roi] |= rim
    return mask


def _hanging_ornaments(gray, subject, lost, scale):
    """A pale compact body attached to a narrower suspended strand."""
    found = np.zeros(subject.shape, np.uint8)
    for level in (.80, .87, .93):
        pale = np.uint8((gray > level) & subject)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(pale)
        for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if not (15 * scale < h < 145 * scale and 9 * scale < w < 110 * scale
                    and 100 * scale**2 < area < 4500 * scale**2 and h > .92 * w):
                continue
            roi = np.s_[y:y+h, x:x+w]
            piece = labels[roi] == label
            if (lost[roi] & piece).sum() < max(6, area * .035):
                continue
            row_width = piece.sum(1)
            # A narrow continuous top section joins the body below it.
            narrow = (row_width < .30 * w) & (row_width > 0)
            if narrow[:int(.45*h)].mean() < .75 or np.median(row_width[int(.62*h):int(.83*h)]) < .48*w:
                continue
            mid = np.nonzero(piece[int(.65*h)])[0]
            stem = np.nonzero(piece[int(.23*h)])[0]
            if not mid.size or not stem.size or abs(mid.mean()-stem.mean()) > .45*w:
                continue
            found[roi] |= piece.astype(np.uint8)
    return found


def _ribbing(gray, painted, cloth, scale):
    """Require three faded, evenly spaced strokes of similar length and angle."""
    from scipy.spatial import cKDTree

    mask = np.zeros(gray.shape, np.uint8)
    soft = cv2.GaussianBlur(gray, (0, 0), max(.55, .7 * scale))
    accent = soft - cv2.GaussianBlur(soft, (0, 0), 4 * scale)
    edges = np.uint8(np.clip(128 + 800 * accent, 0, 255))
    segments = cv2.createLineSegmentDetector().detect(edges)[0]
    if segments is None:
        return mask
    seg = segments.reshape(-1, 2, 2)
    vector = seg[:, 1] - seg[:, 0]
    lengths = np.linalg.norm(vector, axis=1)
    take = (lengths > 7 * scale) & (lengths < 60 * scale)
    seg, vector, lengths = seg[take], vector[take], lengths[take]
    direction = vector / lengths[:, None]
    normal = np.stack((-direction[:, 1], direction[:, 0]), axis=1)
    centers = seg.mean(1)
    points = centers[:, None] + vector[:, None] * np.linspace(-.32, .32, 5)[None, :, None]
    banks = points[:, :, None] + normal[:, None, None] * (np.array([-2, 0, 2]) * scale)[None, None, :, None]
    x = np.clip(np.rint(banks[..., 0]).astype(int), 0, gray.shape[1]-1)
    y = np.clip(np.rint(banks[..., 1]).astype(int), 0, gray.shape[0]-1)
    before = np.ptp(gray[y, x], axis=2).mean(1)
    after = np.ptp(painted[y, x], axis=2).mean(1)
    take = ((before > .010) & (before < .12) & (after < .60 * before)
            & (cloth[y, x].mean((1, 2)) > .85))
    seg, centers, lengths, direction, normal = (a[take] for a in (seg, centers, lengths, direction, normal))
    tree = cKDTree(centers)
    for i, nearby in enumerate(tree.query_ball_point(centers, 25 * scale)):
        ids = np.asarray(nearby)
        delta = centers[ids] - centers[i]
        along = np.sum(delta * direction[i], axis=1)
        across = np.sum(delta * normal[i], axis=1)
        parallel = np.abs(np.sum(direction[ids] * direction[i], axis=1)) > .982
        match = (parallel & (np.abs(along) < .44 * lengths[i]) &
                 (lengths[ids] > .52 * lengths[i]) & (lengths[ids] < 1.9 * lengths[i]) &
                 (np.abs(across) > 4 * scale))
        for j in np.flatnonzero(match & (across < 0)):
            opposite = np.flatnonzero(match & (across > -.64*across[j]) & (across < -1.56*across[j]))
            for k in opposite:
                points = np.rint(seg[[i, ids[j], ids[k]]].reshape(-1, 2)).astype(np.int32)
                cv2.fillConvexPoly(mask, cv2.convexHull(points), 1)
    return mask


def _knit_panel(source, gray, ribbing, permitted, scale):
    """Carry a ribbed cuff's protection into an isolated, simply folded light panel."""
    support = np.zeros(gray.shape, np.float32)
    if not ribbing.any():
        return support
    light = (permitted & (gray > .54) & (np.ptp(source, axis=2) < .22)
             & (source[..., 0] - source[..., 2] < .07))
    connected = cv2.morphologyEx(np.uint8(light), cv2.MORPH_CLOSE, _disk(2 * scale))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(connected)
    overlap = np.bincount(labels[ribbing != 0], minlength=len(stats))
    shade = cv2.GaussianBlur(gray, (0, 0), max(1., scale))
    shade = cv2.GaussianBlur(shade, (0, 0), 5 * scale) - shade
    for i in np.flatnonzero(overlap > 45 * scale**2):
        if i == 0:
            continue
        x, y, w, h, area = stats[i]
        if not (700 * scale**2 < area < 36000 * scale**2 and 1.5 < h/w < 6 and area > .38 * w*h):
            continue
        roi = np.s_[y:y+h, x:x+w]
        piece = labels[roi] == i
        interior = cv2.erode(np.pad(piece.astype(np.uint8), 1), _disk(4 * scale))[1:-1, 1:-1] > 0
        dark = (shade[roi] > .022) & interior
        islands = cv2.connectedComponentsWithStats(dark.astype(np.uint8))[2][1:, cv2.CC_STAT_AREA]
        # Dense groups of small creases are precisely what the main renderer simplifies.
        if dark.sum() > .28 * interior.sum() or (islands > 12 * scale**2).sum() > 12:
            continue
        support[roi][piece & light[roi]] = .74
    return cv2.GaussianBlur(support, (0, 0), max(1., 2 * scale))


def trim_print_edges(source, render, weight, scale):
    """Confine feathering of isolated flat dots to their measured pigment coverage."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(weight > 0))
    for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if not (8 * scale < min(w,h) <= max(w,h) < 36 * scale and
                min(w,h) > .78 * max(w,h) and area > .62 * w*h):
            continue
        roi = np.s_[y:y+h, x:x+w]
        part = labels[roi] == label
        mask = weight[roi]
        center = cv2.distanceTransform(np.pad(part.astype(np.uint8),1), cv2.DIST_L2,5)[1:-1,1:-1]
        core = center > .77 * center.max()
        ring = part & (mask < .45)
        if not core.any() or ring.sum() < 10:
            continue
        pixels = source[roi]
        ground = np.median(pixels[ring],axis=0)
        ink = np.median(pixels[core],axis=0)
        direction = ink - ground
        contrast = np.max(np.abs(direction))
        if (contrast < .07 or np.ptp(direction) < .05 or
                np.quantile(np.abs(pixels[ring]-ground).max(1),.75) > .18*contrast or
                np.quantile(np.abs(pixels[core]-ink).max(1),.80) > .22*contrast):
            continue
        # Only intervene when pasting the old ground actually introduces a tone step.
        if np.max(np.abs(np.median(render[roi][ring],axis=0)-ground)) < .035:
            continue
        coverage = np.sum((pixels-ground)*direction,axis=2) / np.sum(direction**2)
        outside_line = np.max(np.abs(pixels-ground-coverage[...,None]*direction),axis=2)
        if np.mean(outside_line[part] < max(.02,.20*contrast)) < .85:
            continue
        pigment = np.clip((coverage-.08)/.80,0,1)
        mask[part] *= pigment[part]*pigment[part]*(3-2*pigment[part])
    return weight


def automatic_mask(source, render, groups, materials, alpha):
    """Retain evidence of a lost ornament without undoing broad material edits."""
    subject = groups != 0
    if alpha is not None:
        subject &= alpha > 0
    if min(subject.shape) < 8 or not subject.any():
        return np.zeros(subject.shape, np.float32)
    scale = float(np.clip(np.sqrt(np.count_nonzero(subject)) / 640, .6, 3.0))
    allowed = np.isin(groups, [g for g, material in materials.items() if material not in
                               ("background", "eyes", "hair", "leather_glossy", "leather_strap")]) & subject
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    painted_gray = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
    score = _lost_marks(source, render, scale)
    # A faint white symbol may lose contrast without becoming a dark trough.
    # Use that evidence only inside a separately identified object outline.
    delta = cv2.GaussianBlur(source - render, (0, 0), max(.65, .7 * scale))
    local = delta - cv2.GaussianBlur(delta, (0, 0), 3.5 * scale)
    lost = (np.max(np.abs(local), axis=2) > .021) & allowed
    filled = _outlined_marks(source, gray, painted_gray, subject, lost, scale)
    filled |= _buckle_rims(source, gray, lost, scale)
    filled |= _hanging_ornaments(gray, subject, (score > .014) & allowed, scale)
    pale_cloth = (np.isin(groups, [g for g, m in materials.items() if m in ("cloth_light", "cloth_dark")])
                  & (gray > .57) & (np.ptp(source, axis=2) < .26) & allowed)
    ribbing = _ribbing(gray, painted_gray, pale_cloth, scale)
    if not filled.any() and not ribbing.any():
        return np.zeros(subject.shape, np.float32)
    filled &= np.uint8(allowed)
    def feather(core):
        inner = cv2.dilate(core, _disk(1.4 * scale))
        distance = cv2.distanceTransform(1 - inner, cv2.DIST_L2, 5)
        return np.clip(1 - distance / (6 * scale), 0, 1)
    weight = feather(filled)
    if ribbing.any():
        # Restore grooves without leaving a dark patch where the base sleeve
        # has been substantially lightened by the main renderer.
        drift = np.abs(cv2.GaussianBlur(gray - painted_gray, (0, 0), 5 * scale))
        rib_weight = feather(ribbing) * np.clip(1.22 - 6 * drift, .12, 1)
        np.maximum(weight, rib_weight, out=weight)
    edge = cv2.distanceTransform(allowed.astype(np.uint8), cv2.DIST_L2, 5)
    weight *= np.clip(edge / max(1.5, 2 * scale), 0, 1)
    fabric = _knit_panel(source, gray, ribbing, allowed, scale)
    return trim_print_edges(source, render, np.maximum(weight * weight * (3 - 2 * weight), fabric * allowed), scale)


def read_mask(path, shape):
    with Image.open(path) as image:
        if image.size != shape[::-1]:
            raise ValueError("detail mask must match the source resolution")
        return np.asarray(image.convert("L"), dtype=np.float32) / 255


def snapshot(stack, active):
    return Stack(stack.flat[active], stack.shadow[active],
                 stack.highlight[active], stack.lines[active])


def restore(stack, source, weight):
    active = weight > 0
    if not active.any():
        return
    w = weight[active, None]
    output = snapshot(stack, active)
    src_ink, dst_ink = source.lines[:, 3:], output.lines[:, 3:]
    ink = w * src_ink + (1 - w) * dst_ink
    paint = w * (1 - src_ink) + (1 - w) * (1 - dst_ink)
    # Blend the paints in proportion to how much shows through their ink.
    ratio = np.divide(w * (1 - src_ink), paint, out=w.copy(), where=paint > 0)
    flat = ratio * source.flat + (1 - ratio) * output.flat
    src_diffuse = source.flat * source.shadow
    dst_diffuse = output.flat * output.shadow
    diffuse = ratio * src_diffuse + (1 - ratio) * dst_diffuse
    src_screen = (1 - src_diffuse) * source.highlight
    dst_screen = (1 - dst_diffuse) * output.highlight
    screen = ratio * src_screen + (1 - ratio) * dst_screen
    line_rgb = w * src_ink * source.lines[:, :3] + (1 - w) * dst_ink * output.lines[:, :3]
    output.flat = flat
    output.shadow = np.divide(diffuse, flat, out=np.ones_like(flat), where=flat > 0)
    output.highlight = np.divide(screen, 1 - diffuse, out=np.zeros_like(screen), where=diffuse < 1)
    output.lines[:, :3] = np.divide(line_rgb, ink, out=np.zeros_like(line_rgb), where=ink > 0)
    output.lines[:, 3:] = ink
    full = w[:, 0] == 1
    for name in ("flat", "shadow", "highlight", "lines"):
        values = getattr(output, name)
        values[full] = getattr(source, name)[full]
        getattr(stack, name)[active] = np.clip(values, 0, 1)
