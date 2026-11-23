"""Finish local facial colour and distinguish hardware from surface glints."""
import cv2
import numpy as np

from colorfix import from_lab, to_lab
from highlights import hardware_coverage, primary_accent
from layers import norm_blur
from masks import _ellipse, region_slices

METAL_SURFACE = 1
GEM_SURFACE = 2
METAL_RIM = 3
PUNCHED_HOLE = 4


def ease(field, start, end):
    t = np.clip((field - start) / (end - start), 0, 1)
    return t * t * (3 - 2 * t)


def detect_additional_faces(source, labels, stats, lines, face, eyes, protect=None):
    """Find single exposed eyes with a second, skin-surrounded facial landmark.

    The prominent-eye classifier handles its usual frontal face. A compact
    warm mouth plus an iris with an inked upper edge and skin below can also
    establish a side view, even when its skin merges with surrounding hair.
    """
    extra_face = face.copy()
    extra_eyes = eyes.copy()
    profiles = []
    if not stats:
        return extra_face, extra_eyes, profiles
    lab = to_lab(source)
    luma, red, warm = lab.transpose(2, 0, 1)
    skin = (luma > 78) & (red > -2) & (red < 31) & (warm > .5) & (warm < 23) & ~lines
    if protect is not None:
        skin &= protect < .8
    height, width = labels.shape
    pixels = height * width
    mouths = [r for r, st in stats.items()
              if .000045 * pixels < st['area'] < .0015 * pixels and
              48 < st['median'][0] < 88 and st['median'][1] > 22 and
              5 < st['median'][2] < 36 and 15 < st['hue'] < 65]
    eye_colours = [r for r, st in stats.items() if 30 < st['median'][0] < 88 and st['chroma'] > 17]
    for mouth in mouths:
        st = stats[mouth]
        my, mx, mb, mr = st['bbox']
        mw, mh = mr - mx, mb - my
        if not 1 < mw / mh < 5:
            continue
        pad_x = round(.55 * mw)
        mouth_box = np.s_[max(0, my - mh):min(height, mb + mh), max(0, mx - pad_x):min(width, mr + pad_x)]
        surround = labels[mouth_box] != mouth
        if surround.sum() < 8 or skin[mouth_box][surround].mean() < .64:
            continue
        scores = []
        for r in eye_colours:
            eye = stats[r]
            y0, x0, y1, x1 = eye['bbox']
            ey, ex = eye['cy'], eye['cx']
            w, h = x1 - x0, y1 - y0
            rise = st['cy'] - ey
            drift = abs(st['cx'] - ex)
            if not (1.4 * mw < rise < 7.1 * mw and drift < 3.1 * mw and
                    .25 * st['area'] < eye['area'] < 4.5 * st['area'] and .48 < w / h < 4):
                continue
            left, right = max(0, x0 - round(.45 * w)), min(width, x1 + round(.45 * w))
            below = skin[y1 + round(.52 * h):min(height, y1 + round(1.5 * h) + 2), left:right]
            brow = lines[max(0, y0 - 3):y0 + max(1, round(.34 * h)), left:right]
            if not below.size or below.mean() < .39 or brow.mean() < .28:
                continue
            scores.append((below.mean() + .7 * brow.mean() - .04 * drift / mw, r))
        if not scores:
            continue
        _, selected = max(scores)
        eye = stats[selected]
        ey, ex = eye['cy'], eye['cx']
        if extra_eyes[int(ey), int(ex)]:
            continue
        y0, x0, y1, x1 = eye['bbox']
        w, h = x1 - x0, y1 - y0
        eye_box = np.s_[max(0, y0 - round(.44 * h)):min(height, y1 + round(.81 * h)),
                       max(0, x0 - round(.97 * w)):min(width, x1 + round(.72 * w))]
        extra_face[eye_box] = True
        extra_eyes[eye_box] = True
        extra_face[mouth_box] = True
        profiles.append((float(ey), float(ex), float(st['cy']), float(st['cx'])))
    return extra_face, extra_eyes, profiles


def profile_paint_protection(source, subject, profiles):
    """Keep coherent face colour before material-wide shadow simplification."""
    skin = np.zeros(subject.shape, np.float32)
    height, width = subject.shape
    for ey, ex, my, mx in profiles:
        span = my - ey
        xc = .55 * ex + .45 * mx
        y0, y1 = max(0, int(ey - .52 * span)), min(height, int(my + .94 * span))
        x0, x1 = max(0, int(xc - 1.14 * span)), min(width, int(xc + 1.14 * span))
        sl = np.s_[y0:y1, x0:x1]
        y, x = np.ogrid[y0:y1, x0:x1]
        luma, red, warm = to_lab(source[sl]).transpose(2, 0, 1)
        colour = ease(luma, 70, 84) * ease(red, .3, 2.1) * ease(warm, 1, 3.3)
        # The complexion carries through neighbouring soft shading, but the
        # dark hair/ink/background boundary falls outside the colour support.
        colour *= 1 - ease(warm, 20, 29)
        gate = (
            (1 - ease(np.abs(x - xc) / span, .64, 1.1))
            * ease((y - ey) / span, -.49, -.23)
            * (1 - ease((y - my) / span, .56, .92))
        )
        confidence = cv2.GaussianBlur((colour * gate).astype(np.float32), (0, 0), max(.7, .019 * span))
        np.maximum(skin[sl], confidence * ease(colour, 0, .18) * subject[sl], out=skin[sl])
    return skin


def mouth_stroke_protection(source, labels, kinds, stats, face, eyes):
    """Hold a short two-banked mouth stroke on a verified frontal skin plate.

    Weak mouth ink often belongs to the extracted skin paint. Use it as a
    local protection weight so the underlying colour and narrow tonal
    profile survive, including their original antialiasing.
    """
    guard = np.zeros(labels.shape, np.float32)
    _, face_parts = cv2.connectedComponents(face.astype(np.uint8))
    for face_id, sl in region_slices(face_parts):
        ids, counts = np.unique(labels[sl][face_parts[sl] == face_id], return_counts=True)
        plates = [(n, r) for r, n in zip(ids, counts) if kinds.get(r) == 'skin']
        if not plates:
            continue
        _, plate = max(plates)
        y0, x0, y1, x1 = stats[plate]['bbox']
        width, height = x1 - x0, y1 - y0
        if min(width, height) < 24:
            continue
        patch = np.s_[y0:y1, x0:x1]
        y, x = np.ogrid[:height, :width]
        eye_map = eyes[patch] & (y < .54 * height)
        left = np.nonzero(eye_map & (x < .5 * width))
        right = np.nonzero(eye_map & (x >= .5 * width))
        if min(left[0].size, right[0].size) < max(6, .004 * height * width):
            continue
        eye_x = [float(np.median(half[1])) for half in (left, right)]
        eye_y = [float(np.median(half[0])) for half in (left, right)]
        span = eye_x[1] - eye_x[0]
        centre = .5 * sum(eye_x)
        if span < .30 * width or abs(eye_y[1] - eye_y[0]) > .20 * span:
            continue
        floor = max(float(np.percentile(half[0], 95)) for half in (left, right))
        if floor > .53 * height:
            continue
        gate = (y > max(.56 * height, floor + .12 * span)) & (y < .85 * height)
        gate = gate & (np.abs(x - centre) < .20 * span) & (labels[patch] == plate)
        if not gate.any():
            continue
        luma = to_lab(source[patch])[..., 0]
        # A horizontal stroke has brighter banks above and below. Curved
        # ends may use a diagonal cross-section through the same paint.
        radius = max(1, min(7, round(.026 * span)))
        pad = cv2.copyMakeBorder(luma, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
        depth = np.zeros_like(luma)
        for dx in (-radius, 0, radius):
            upper = pad[:height, radius + dx:radius + dx + width]
            lower = pad[2 * radius:2 * radius + height, radius - dx:radius - dx + width]
            np.maximum(depth, np.minimum(upper, lower) - luma, out=depth)
        seed = np.uint8(gate & (depth > 3.4))
        bridge = np.ones((1, 1 + 2 * max(1, round(.014 * span))), np.uint8)
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, bridge)
        count, parts, sizes, centroids = cv2.connectedComponentsWithStats(seed)
        best = None
        score = 0.0
        for number in range(1, count):
            _, _, w, h, area = sizes[number]
            px, _ = centroids[number]
            if not (max(3, .038 * span) <= w <= .33 * span and w >= 1.45 * h
                    and area >= 3 and abs(px - centre) < .12 * span):
                continue
            candidate = parts == number
            energy = np.minimum(depth[candidate], 18).sum() * (1 - .4 * abs(px - centre) / span)
            if energy > score:
                best, score = candidate, energy
        if best is None:
            continue
        # Keep only the observed valley and a one-pixel antialias shoulder.
        core = cv2.dilate(best.astype(np.uint8), _ellipse(3)).astype(np.float32)
        feather = cv2.GaussianBlur(core, (0, 0), max(.65, .008 * span))
        keep = np.maximum(core, feather) * gate
        np.maximum(guard[patch], keep, out=guard[patch])
    return guard


def apply_cheek_pigment(stack, sl, original, clean, baseline, gate, width, protect, strength):
    sigma = max(1.1, .024 * width)
    body = clean & (original[..., 0] > baseline[0] - 19)
    if not body.any():
        return
    red, warm = [norm_blur(original[..., c], body, sigma) for c in (1, 2)]
    # Look for a local warm tint rather than attenuating every skin-coloured pixel.
    fit = (
        gate * ease(red - baseline[1], 2.2, 5.9)
        * ease(original[..., 0], baseline[0] - 17, baseline[0] - 10)
        * (1 - ease(np.abs(original[..., 1] - red), 2.3, 6.0))
    )
    edge = cv2.distanceTransform(np.pad(body.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    weight = .47 * strength * fit * ease(edge, .8, max(2.3, .026 * width)) * (1 - protect[sl])
    active = weight > .0005
    if not active.any():
        return
    flat = stack.flat[sl]
    shadow = stack.shadow[sl]
    screen = stack.highlight[sl]
    diffuse = flat * shadow
    under = diffuse + (1 - diffuse) * screen
    lab = to_lab(under)
    adjusted = lab.copy()
    red_excess = np.maximum(0, np.minimum(lab[..., 1], red) - baseline[1] - .85)
    warm_excess = warm - baseline[2]
    warm_remaining = (lab[..., 2] - baseline[2]) * np.sign(warm_excess)
    warm_correction = np.minimum(np.abs(warm_excess), np.maximum(0, warm_remaining))
    adjusted[..., 1] -= weight * red_excess
    adjusted[..., 2] -= weight * np.sign(warm_excess) * np.minimum(warm_correction, .65 * red_excess)
    colour = np.clip(under + (from_lab(adjusted) - from_lab(lab)), 0, 1)
    # Retain the previous diffuse where feasible and encode the changed
    # surface with the same Multiply and Screen operators.
    diffuse = np.minimum(diffuse, colour)
    shadow[active] = np.divide(diffuse, flat, out=np.ones_like(flat), where=flat > 0)[active]
    screen[active] = np.divide(colour - diffuse, 1 - diffuse, out=np.zeros_like(flat),
                               where=diffuse < 1)[active]


def attenuate_blush(stack, source, labels, kinds, stats, face, eyes, protect, strength, profiles=()):
    """Reduce supported cheek pigment at fixed L*, without altering the ink.

    Each connected face hint needs a skin plate and two eye anchors. The
    strength field comes from source pigment and fades at features and at
    the skin edge, so its shape follows the original paint.
    """
    _, face_ids = cv2.connectedComponents(face.astype(np.uint8))
    for index, sl in region_slices(face_ids):
        face_hint = face_ids[sl] == index
        ids, counts = np.unique(labels[sl][face_hint], return_counts=True)
        candidates = [(count, r) for r, count in zip(ids, counts) if kinds.get(r) == 'skin']
        if not candidates:
            continue
        _, skin_id = max(candidates)
        y0, x0, y1, x1 = stats[skin_id]['bbox']
        width, height = x1 - x0, y1 - y0
        if min(width, height) < 18:
            continue
        patch = np.s_[y0:y1, x0:x1]
        face_skin = labels[patch] == skin_id
        y, x = np.ogrid[:height, :width]
        anchors = eyes[patch] & (y < .55 * height)
        left_y = np.nonzero(anchors & (x < .5 * width))[0]
        right_y = np.nonzero(anchors & (x >= .5 * width))[0]
        if min(left_y.size, right_y.size) < max(4, .003 * width * height):
            continue
        eye_floor = np.array([np.percentile(left_y, 96), np.percentile(right_y, 96)])
        lower_eye = np.where(x < .5 * width, eye_floor[0], eye_floor[1])

        original = to_lab(source[patch])
        old_ink = stack.lines[patch][..., 3]
        clean = face_skin & (old_ink < .06) & ~eyes[patch]
        midface = clean & (np.abs(x / width - .5) < .13) & (y > lower_eye) & (y < .79 * height)
        if midface.sum() < max(8, .012 * width * height):
            continue
        light = np.percentile(original[..., 0][midface], 65)
        midface &= np.abs(original[..., 0] - light) < 4
        if midface.sum() < 8:
            continue
        baseline = np.median(original[midface], axis=0)
        gate = (
            ease(y - lower_eye, .032 * width, .073 * width)
            * (1 - ease(y / height, .75, .87))
            * ease(np.abs(x / width - .5), .10, .20)
        )
        apply_cheek_pigment(stack, patch, original, clean, baseline, gate, width, protect, strength)
    height, width = labels.shape
    for ey, ex, my, mx in profiles:
        span = my - ey
        cy = ey + .38 * span
        cx = ex + .17 * span * np.sign(ex - mx)
        y0, y1 = max(0, int(ey)), min(height, int(my + .30 * span))
        x0, x1 = max(0, int(cx - .65 * span)), min(width, int(cx + .65 * span))
        sl = np.s_[y0:y1, x0:x1]
        y, x = np.ogrid[y0:y1, x0:x1]
        original = to_lab(source[sl])
        light, red, warmth = original.transpose(2, 0, 1)
        clean = ((labels[sl] > 0) & ~eyes[sl] & (stack.lines[sl][..., 3] < .06) &
                 (light > 76) & (red > -2) & (red < 30) & (warmth > .5) & (warmth < 23))
        reference = clean & (np.abs(x - mx) < .22 * span) & (np.abs(y - my) < .22 * span)
        if reference.sum() < 15:
            continue
        baseline = np.median(original[reference], axis=0)
        gate = (1 - ease(np.abs(x - cx) / span, .12, .43)) * (1 - ease(np.abs(y - cy) / span, .10, .28))
        apply_cheek_pigment(stack, sl, original, clean, baseline, gate, 1.4 * span, protect, strength)


def mark_accessory(segmentation, kinds, source, labels, lines, core, optics, scale, group_pass):
    """Separate sharply bounded parts, including fragments split off a clasp."""
    limit = (.012 if group_pass else .0035) * scale ** 2
    radius = max(2, int(round(.004 * scale)))
    for key, sl in region_slices(segmentation, pad=2 * radius + 1):
        kind = kinds.get(key)
        if group_pass and kind not in ('metal', 'gem'):
            continue
        # A pale facet belonging to a traced horn is part of its shell.
        if not group_pass and kind in ('hair', 'eyes', 'horn', 'background'):
            continue
        n, cc, sizes, _ = cv2.connectedComponentsWithStats(np.uint8(segmentation[sl] == key))
        for i in range(1, n):
            _, _, width, height, area = sizes[i]
            if not (max(12, .000019 * scale ** 2) <= area <= limit and min(width, height) >= 4):
                continue
            piece = np.uint8(cc == i)
            inner = (piece > 0) & ~lines[sl]
            if inner.sum() < 6 or np.mean(core[sl][piece > 0]) > .96:
                continue
            annulus = (cv2.dilate(piece, _ellipse(4 * radius + 1)) > 0) & (piece == 0)
            around = annulus & (labels[sl] > 0) & ~lines[sl]
            if around.sum() < 12:
                continue
            boundary = (piece > 0) & (cv2.erode(piece, _ellipse(2 * radius + 1)) == 0)
            near_ink = cv2.dilate(lines[sl].astype(np.uint8), _ellipse(2 * radius + 1)) > 0
            ink_support = near_ink[boundary].mean()
            if ink_support < (.45 if group_pass else .63):
                continue
            lab = to_lab(source[sl])
            body, host = (np.median(lab[m], axis=0) for m in (inner, around))
            contrast = np.linalg.norm(body - host)
            neutral = np.hypot(body[1], body[2]) < 12
            expected = 10.0 if kind == 'metal' and not group_pass else 24.0
            bright_metal = neutral and 54 < body[0] < 98.5 and body[0] - host[0] > expected
            jewel = group_pass and kind == 'gem' and contrast > 20
            ribbon = (not group_pass and kind != 'skin' and np.hypot(body[1], body[2]) > 26
                      and np.linalg.norm(body[1:] - host[1:]) > 24 and contrast > 28)
            if not (bright_metal or jewel or ribbon):
                continue
            core[sl][piece > 0] = True
            if bright_metal or jewel:
                optics[sl][piece > 0] = GEM_SURFACE if jewel else METAL_SURFACE


def mark_fasteners(source, groups, materials, core, optics, scale):
    """Find punched holes, isolated round studs, and rims around lace apertures."""
    size = max(2, min(9, int(round(.005 * scale))))
    for g, sl in region_slices(groups, pad=2 * size + 1):
        material = materials.get(g)
        if material not in ('leather_glossy', 'leather_strap', 'cloth_dark'):
            continue
        region = groups[sl] == g
        if region.sum() < 80:
            continue
        tile = source[sl]
        value = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
        inside = dist >= max(2, .6 * size)
        if material in ('leather_glossy', 'leather_strap'):
            width = 2 * np.percentile(dist[region], 85)
            rings = hardware_coverage(value, region, dist, width)
            core[sl][region & (rings > .8)] = True
            # Leave these rims in the leather group for its lacing geometry.
            # Their own ownership still protects them from the leather response.
            rim = region & (rings > .8) & (value > np.median(value[region]) + .16)
            optics[sl][rim & (optics[sl] == 0)] = METAL_RIM
        kernel = _ellipse(4 * size + 1)
        for bright in (False, True):
            operation = cv2.MORPH_TOPHAT if bright else cv2.MORPH_BLACKHAT
            field = cv2.morphologyEx(value, operation, kernel)
            n, ids, boxes, _ = cv2.connectedComponentsWithStats(
                np.uint8(inside & (field > (.19 if bright else .12))))
            for i in range(1, n):
                x, y, w, h, area = boxes[i]
                if not (max(5, .32 * size ** 2) <= area <= 4.3 * size ** 2 and
                        min(w, h) >= .56 * max(w, h) and area >= .53 * w * h):
                    continue
                local = np.s_[max(0, y - size):y + h + size, max(0, x - size):x + w + size]
                part = ids[local] == i
                annulus = (cv2.dilate(part.astype(np.uint8), _ellipse(2 * size + 1)) > 0) & ~part
                if not annulus.any() or (annulus & region[local]).sum() < .85 * annulus.sum():
                    continue
                collar = value[local][annulus]
                centre = np.median(value[local][part])
                if bright:
                    rgb = np.median(tile[local][part], axis=0)
                    contrast = centre - np.percentile(collar, 85)
                    valid = contrast > .20 and np.ptp(rgb) < .19
                else:
                    contrast = np.percentile(collar, 20) - centre
                    valid = contrast > .065 and centre < .44
                if valid:
                    hold = cv2.dilate(part.astype(np.uint8), _ellipse(3)) > 0
                    core[sl][local][hold] = True
                    new_core = part & (optics[sl][local] == 0)
                    optics[sl][local][new_core] = METAL_SURFACE if bright else PUNCHED_HOLE


def accessory_protection(source, labels, kinds, groups, materials, lines, reflections):
    """Preserve designed geometry before the bulk material passes.

    Bright islands already reassigned to a glossy host are kept eligible for
    reflection handling. Hardware has either independent ink boundaries or
    a small, contrast-bounded aperture in the host.
    """
    scale = np.sqrt((labels > 0).sum())
    core = np.zeros(labels.shape, bool)
    optics = np.zeros(labels.shape, np.uint8)
    mark_accessory(groups, materials, source, labels, lines, core, optics, scale, True)
    mark_accessory(labels, kinds, source, labels, lines, core, optics, scale, False)
    mark_fasteners(source, groups, materials, core, optics, scale)
    core &= ~reflections
    optics[reflections] = 0
    edge = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 1.1)
    protection = np.maximum(core, edge) * (labels > 0)
    protection[reflections] = 0
    return protection, optics


def separate_accessories(groups, materials, optics):
    """Give a separated jewel or metal surface its own diffuse paint group."""
    new_groups = groups.copy()
    material_by_id = materials.copy()
    next_group = max(int(groups.max()), max(materials, default=0)) + 1
    for owner, material in ((METAL_SURFACE, 'metal'), (GEM_SURFACE, 'gem')):
        _, parts = cv2.connectedComponents(np.uint8(optics == owner))
        for part_id, sl in region_slices(parts):
            part = parts[sl] == part_id
            current = np.unique(groups[sl][part])
            if len(current) == 1 and materials.get(current[0]) == material:
                continue
            new_groups[sl][part] = next_group
            material_by_id[next_group] = material
            next_group += 1
    return new_groups, material_by_id


def finish_accessories(raw, stack, optics, strength, light, external_protect=None):
    """Trim secondary Screen excess inside a finished metal or jewel surface."""
    for material, trim in ((METAL_SURFACE, .35), (GEM_SURFACE, .24)):
        _, parts = cv2.connectedComponents(np.uint8(optics == material))
        for index, sl in region_slices(parts, pad=3):
            part = parts[sl] == index
            distance = cv2.distanceTransform(part.astype(np.uint8), cv2.DIST_L2, 5)
            clean = part & (distance > 1.4) & (raw.lines[sl][..., 3] < .10)
            if clean.sum() < 10:
                continue
            width = 2 * float(np.percentile(distance[part], 85))
            lift = (1 - raw.flat[sl] * raw.shadow[sl]) * raw.highlight[sl]
            intensity = cv2.cvtColor(lift, cv2.COLOR_RGB2GRAY)
            body = float(np.percentile(intensity[clean], 64))
            excess = np.maximum(0, intensity - body)
            if excess[clean].max() < .025:
                continue
            accent = primary_accent(excess, clean, width, light, False)
            weight = trim * strength * ease(distance, 1.3, 3.1) * (1 - accent) * clean
            if external_protect is not None:
                weight *= 1 - external_protect[sl]
            share = excess / np.maximum(intensity, .0001)
            stack.highlight[sl] *= (1 - weight * share)[..., None]


def whiten_eye_catchlights(source, stack, eyes, strength, external_protect=None):
    """Restore only the interior of an existing near-white spec on a dark iris."""
    if not eyes.any():
        return
    _, eye_parts = cv2.connectedComponents(eyes.astype(np.uint8))
    for index, sl in region_slices(eye_parts, pad=4):
        eye = eye_parts[sl] == index
        tile = source[sl]
        value = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        pale = (tile.min(2) > .933) & (np.ptp(tile, axis=2) < .045)
        n, cc, sizes, _ = cv2.connectedComponentsWithStats(np.uint8(pale & eye))
        limit = max(8, .045 * eye.sum())
        for i in range(1, n):
            _, _, w, h, area = sizes[i]
            if not (4 <= area <= limit and min(w, h) >= .45 * max(w, h)):
                continue
            spec = cc == i
            around = (cv2.dilate(spec.astype(np.uint8), _ellipse(7)) > 0) & ~spec
            if not around.any() or (value[around] < .64).mean() < .73:
                continue
            interior = cv2.distanceTransform(spec.astype(np.uint8), cv2.DIST_L2, 5)
            active = spec & (stack.lines[sl][..., 3] == 0)
            depth = float(interior.max())
            if depth < 1.2:
                # A resolved one-pixel centre can be enclosed by warm JPEG
                # fringes and consequently have no eroded neutral plateau.
                score = cv2.GaussianBlur(np.where(spec, tile.min(2), 0), (0, 0), .75)
                core = np.flatnonzero(active)
                whiten = np.zeros_like(interior)
                if core.size:
                    whiten.flat[core[np.argmax(score.flat[core])]] = 1
            else:
                whiten = ease(interior, 1, min(1.45, depth))
            opacity = strength * whiten[active]
            if external_protect is not None:
                opacity *= 1 - external_protect[sl][active]
            screen = stack.highlight[sl]
            screen[active] += (1 - screen[active]) * opacity[:, None]
