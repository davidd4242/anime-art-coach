"""Finish the paint, dark contact edges and metal of a traced leather belt."""
import cv2
import numpy as np

from highlights import apply_paint, primary_accent
from layers import norm_blur
from masks import _ellipse, region_slices
from pigment import paired_banks


def ease(field, begin, end):
    step = np.clip((field - begin) / (end - begin), 0, 1)
    return step * step * (3 - 2 * step)


def luma(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def with_luma(rgb, target):
    """Change brightness using the local RGB colour and its available range."""
    value = luma(rgb)
    delta = target - value
    dark = rgb * (np.minimum(delta, 0) / np.maximum(value, 1e-5))[..., None]
    light = (1 - rgb) * (np.maximum(delta, 0) / np.maximum(1 - value, 1e-5))[..., None]
    return np.clip(rgb + dark + light, 0, 1)


def strap_light_transition(observed, body, part, distance, size, fittings, dark_bank):
    """Round a narrow light step along a long strap's local direction of travel."""
    y, x = np.nonzero(part)
    covariance = np.cov(np.column_stack((x, y)), rowvar=False)
    variance, direction = np.linalg.eigh(covariance)
    if variance[1] < 4.3 * max(variance[0], 1):
        return None
    axis = direction[:, 1]
    tone = cv2.GaussianBlur(observed, (0, 0), .9)
    gx = cv2.Sobel(tone, cv2.CV_32F, 1, 0, scale=.125)
    gy = cv2.Sobel(tone, cv2.CV_32F, 0, 1, scale=.125)
    longitudinal = np.abs(gx * axis[0] + gy * axis[1])
    alignment = longitudinal / np.maximum(np.hypot(gx, gy), 1e-4)
    support = part & (distance > max(2.8, .13 * size)) & (fittings < .12)
    reflected = cv2.dilate(ease(observed - body, .035, .125), _ellipse(7))
    step = (ease(longitudinal, .023, .063) * ease(alignment, .47, .83) *
            reflected * (1 - ease(dark_bank, .018, .074)) * support)
    if step.max() < .06:
        return None
    length = np.clip(.135 * size, 2.8, 6.6)
    radius = int(np.ceil(3 * length))
    dy, dx = np.mgrid[-radius:radius+1, -radius:radius+1].astype(np.float32)
    u = dx * axis[0] + dy * axis[1]
    v = -dx * axis[1] + dy * axis[0]
    kernel = np.float32(np.exp(-.5 * ((u / length)**2 + (v / .85)**2)))
    kernel /= kernel.sum()
    confidence = cv2.dilate(step, _ellipse(2 * max(2,round(.09 * size)) + 1))
    confidence = cv2.filter2D(confidence.astype(np.float32), -1, kernel)
    margin = ease(distance, max(2, .075 * size), max(4.6, .19 * size))
    mix = .96 * ease(confidence, .035, .28) * margin * (1 - fittings)
    den = cv2.filter2D(support.astype(np.float32), -1, kernel)
    def smooth(value):
        local = cv2.filter2D(np.float32(value * support), -1, kernel) / np.maximum(den, 1e-5)
        return value + mix * ease(den, .18, .57) * (local - value)
    return smooth


def strap_response(source, paint, part, distance, size, fittings):
    """A curved band has broad paint, a coherent sheen and narrower edge ink."""
    observed = luma(source)
    sample = part & (distance > 1.8) & (fittings < .16)
    if sample.sum() < 25:
        return None
    rough = luma(paint)
    low, high = np.percentile(rough[sample], [11, 79])
    sigma = np.clip(.20 * size, 3.4, 18)
    body = norm_blur(np.clip(rough, low, high), sample, sigma)
    light = np.maximum(rough - body, 0)
    wide = norm_blur(np.minimum(light, max(.08, np.percentile(light[sample], 96))), sample,
                     np.clip(.075 * size, 1.3, 4.5))
    wide = np.minimum(wide, light)
    # A continuous dark valley within either edge bank is contact ink or an
    # impressed seam. The thin bright trim next to it belongs to the lighting.
    dark_bank = paired_banks(observed, max(2, round(.09 * size)))
    edge = 1 - ease(distance, 1.5, .26 * size + 2)
    contact = edge * ease(dark_bank, .020, .085) * ease(body-observed, .020, .075)
    soften = strap_light_transition(observed, body, part, distance, size, fittings, dark_bank)
    smooth_contact = norm_blur(np.minimum(np.maximum(body - rough, 0), .09), sample, 2.1)
    def response(value):
        d = value - body
        light_strength = np.minimum(wide,np.maximum(d,0))
        shade = np.minimum(d,0)
        value = np.clip(body + .23 * np.maximum(d,0) + .24 * light_strength +
                        (.57 + .43 * contact) * shade - .10 * smooth_contact, 0, 1)
        return value if soften is None else soften(value)
    # A light double edge is often a narrow reflected crest right next to an
    # inked bank. Let its own positive contrast drive limited extra trimming.
    crest = paired_banks(-observed, max(2, round(.055 * size)))
    rim_sheen = ease(crest, .024, .084) * ease(observed-body, .047, .17)
    rim_sheen = cv2.GaussianBlur(rim_sheen, (0,0), .65)
    return response, rim_sheen


def metal_opening(frame, strap):
    """A buckle is a metal frame around visible leather rather than a stud."""
    contours, hierarchy = cv2.findContours(np.uint8(frame), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    for i, shape in enumerate(contours):
        parent = hierarchy[0,i,3]
        if parent < 0 or cv2.contourArea(shape) < .14 * cv2.contourArea(contours[parent]):
            continue
        interior = np.zeros(frame.shape, np.uint8)
        cv2.drawContours(interior, [shape], 0, 1, thickness=cv2.FILLED)
        opening = (interior > 0) & ~frame
        if opening.sum() >= 24 and np.mean(strap[opening]) > .56:
            return True
    return False


def apply_response(stack, sl, active, weight, response):
    """Recompose the tone adjustment in the existing Multiply/Screen/ink layers."""
    flat = stack.flat[sl]
    diffuse = flat * stack.shadow[sl]
    under = diffuse + (1 - diffuse) * stack.highlight[sl]
    ink = stack.lines[sl]
    opacity = ink[...,3]
    visible = under * (1 - opacity[...,None]) + ink[...,:3] * opacity[...,None]
    fitted = []
    for rgb in (visible, under, diffuse):
        original_l = luma(rgb)
        fitted_l = original_l + weight * (response(original_l) - original_l)
        fitted.append(with_luma(rgb, fitted_l))
    pigment = ease(luma(under) - luma(visible), .013, .065)
    alpha = opacity * (1 - .65 * weight * (1 - pigment))
    shadow, highlight, ink = apply_paint(flat, fitted[1], fitted[0], alpha, fitted[2])
    stack.shadow[sl][active] = shadow[active]
    stack.highlight[sl][active] = highlight[active]
    stack.lines[sl][active] = ink[active]


def finish_belt_surfaces(raw, stack, img, groups, materials, hardware, external,
                         strength, light, optics=None):
    """Subdue leather glints and secondary metal reflections at existing geometry."""
    selected = [g for g,m in materials.items() if m == "leather_strap"]
    if not selected:
        return
    strap = np.isin(groups, selected)
    for g, sl in region_slices(groups, pad=4):
        material = materials.get(g)
        if material not in ("leather_strap", "metal") or not strap[sl].any():
            continue
        group = groups[sl] == g
        n, cc, stats, _ = cv2.connectedComponentsWithStats(np.uint8(group))
        for i in range(1, n):
            if stats[i,4] < 110:
                continue
            part = cc == i
            distance = cv2.distanceTransform(np.uint8(part), cv2.DIST_L2, 5)
            size = 2 * np.percentile(distance[part], 83)
            if size < 4:
                continue
            original = img[sl]
            source_keep = np.zeros(part.shape, np.float32)
            if external is not None:
                source_keep = external[sl]
            strength_map = strength * (1 - source_keep)
            if material == "leather_strap":
                # Global small-region protection also fires on detached
                # islands of glare. Punched holes and fittings have their
                # own geometry and dedicated source coverage.
                fittings = hardware[sl].copy()
                if optics is not None:
                    optical_core = cv2.GaussianBlur(np.float32(optics[sl] != 0), (0,0), .8)
                    fittings = np.maximum(fittings, optical_core)
                diffuse = raw.flat[sl] * raw.shadow[sl]
                original_paint = diffuse + (1 - diffuse) * raw.highlight[sl]
                treatment = strap_response(original, original_paint, part, distance, size, fittings)
                if treatment is None:
                    continue
                response, rim_sheen = treatment
                interior = ease(distance, .75, 3.8)
                interior = np.maximum(interior, .72 * rim_sheen * ease(distance, .65, 1.8))
                weight = strength_map * (1 - fittings) * interior * part
            elif metal_opening(part, strap[sl]):
                tone = luma(original)
                legal = part & (distance > 1.1) & (tone > .49)
                if legal.sum() < 24:
                    continue
                # The underlying bevels are larger than their small sparkle
                # peaks. Keep the main bright run and compress the rest.
                smooth = cv2.bilateralFilter(tone, 7, .19, 2.1)
                excess = np.maximum(tone - smooth, 0)
                accent = primary_accent(excess, legal, size, light, False)
                broad = norm_blur(np.clip(tone, .45, .90), legal, 2.5)
                trim = (.90 - .48 * accent) * np.maximum(excess - .009, 0)
                trim += (.32 - .20 * accent) * np.maximum(tone - np.maximum(.69, broad), 0)
                weight = strength_map * part * ease(distance, .7, 2.1) * ease(tone, .49, .69)
                def response(value):
                    return np.maximum(value - trim * ease(value, .49, .71), 0)
            else:
                continue
            active = weight > 1e-5
            if active.any():
                apply_response(stack, sl, active, weight, response)
