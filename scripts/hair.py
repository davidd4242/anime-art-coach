"""Simplify painted hair accents and pressure on secondary strand ink."""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from colorfix import to_lab
from layers import norm_blur
from masks import _adjacency, _ellipse, outline_band, region_slices, TIER_INTRA, TIER_INTER
from pigment import paired_banks
from strokes import paths_on_skeleton


def smoothstep(value, lo, hi):
    t = np.clip((value - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def accent_pigment(lab, part, ink):
    """Keep coherent dyed pigment whose hue differs from the hair base."""
    no_ink = part & (ink < .22)
    if no_ink.sum() < 16:
        return np.zeros(part.shape, np.float32)
    tone = np.median(lab[no_ink], axis=0)
    base = tone[1:]
    base_chroma = np.hypot(*base)
    if base_chroma < 4:
        return np.zeros(part.shape, np.float32)
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    aligned = (a * base[0] + b * base[1]) / np.maximum(chroma * base_chroma, 1e-4)
    distance = np.hypot(a - base[0], b - base[1])
    # The same pigment may have split into a second segment, leaving only
    # its antialiasing on the hair side. Admit adjacent source colour as
    # continuity evidence, while keeping any protection strictly in part.
    neighbour = cv2.dilate(part.astype(np.uint8), _ellipse(5)) > 0
    seed = neighbour & (chroma > 16) & (aligned < .72) & (distance > 16)
    seed &= lab[..., 0] > max(46, tone[0] - 25)
    count, islands, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8))
    if count < 2:
        return np.zeros(part.shape, np.float32)
    substantial = stats[:, cv2.CC_STAT_AREA] >= max(4, round(.000024 * part.sum()))
    substantial[0] = False
    core = substantial[islands]
    fringe = cv2.dilate(core.astype(np.uint8), _ellipse(3)) > 0
    fringe &= (aligned < .85) & (chroma > 10) & (distance > 12) & part
    return np.maximum(fringe, cv2.GaussianBlur(core.astype(np.float32), (0, 0), .8)) * part


def hair_parts(source, groups, materials, protect):
    """Attach a warm recessed segment only across its existing hair boundary.

    Gold hair shadows are sometimes classified as skin. Their dark ochre
    shift differs from skin's red-pink shadow. A candidate must touch one
    dominant head/tail patch and stay away from real skin, eye and horn.
    This support is used only by the hair pass.
    """
    hair_ids = {g for g, material in materials.items() if material == 'hair'}
    ownership = np.where(np.isin(groups, list(hair_ids)), groups, 0)
    if not hair_ids:
        return ownership
    skin_ids = {g for g, material in materials.items() if material == 'skin'}
    first, second, edge, _ = _adjacency(groups)
    neighbours = {}
    for a, b, weight in zip(first, second, edge):
        if a in skin_ids or b in skin_ids:
            neighbours.setdefault(a, []).append((b, weight))
            neighbours.setdefault(b, []).append((a, weight))
    relevant = hair_ids | {g for g, adj in neighbours.items() if
                           g in skin_ids and any(n in hair_ids for n, _ in adj)}
    medians = {}
    sizes = {}
    for g, sl in region_slices(groups):
        if g not in relevant:
            continue
        pixels = (groups[sl] == g) & (protect[sl] < .94)
        if pixels.sum() < 20:
            continue
        medians[g] = np.median(to_lab(source[sl])[pixels], axis=0)
        sizes[g] = int(pixels.sum())
    attached = {}
    for g in relevant - hair_ids:
        if g not in medians:
            continue
        adj = neighbours[g]
        candidates = [(weight, other) for other, weight in adj if other in medians and other in hair_ids]
        if not candidates:
            continue
        contact, host = max(candidates)
        l, a, b = medians[g]
        lh, ah, bh = medians[host]
        if not (.004 * sizes[host] <= sizes[g] <= .34 * sizes[host] and
                8 < lh - l < 34 and b > max(12, .92 * bh) and
                abs(b - bh) < 7 and -2 < a - ah < 12.4):
            continue
        all_edges = sum(weight for _, weight in adj)
        foreign = sum(weight for other, weight in adj if
                      materials.get(other) in ('skin', 'eyes', 'horn') and other not in hair_ids)
        chromatic = sum(weight for other, weight in adj if
                        materials.get(other) in ('cloth_dark', 'membrane', 'leather_glossy', 'gem'))
        if contact >= .22 * all_edges and foreign < .17 * all_edges and contact > chromatic:
            attached[g] = host
    for g, sl in region_slices(groups):
        if g in attached:
            ownership[sl][groups[sl] == g] = attached[g]
    return ownership


def simplify_ink(opacity, raw_opacity, source_luma, part, interior, tiers, width, seam):
    """Keep deep, long separators and taper the lesser free ends of strands."""
    allowed = (tiers == TIER_INTRA) | (tiers == TIER_INTER)
    corridor = cv2.dilate(allowed.astype(np.uint8), _ellipse(3)) > 0
    valley = paired_banks(source_luma, max(2, min(5, round(.085 * width))))
    core = part & corridor & (raw_opacity > .25) & (valley > 2.5)
    core = (cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(3)) > 0) & part
    if not core.any():
        return np.zeros_like(source_luma)
    spine = skeletonize(np.ascontiguousarray(core))
    thick = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    _, pieces, boxes, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8))
    reach = np.maximum(boxes[:, cv2.CC_STAT_WIDTH], boxes[:, cv2.CC_STAT_HEIGHT])
    pressure = np.zeros_like(source_luma)
    for xy, start, end in paths_on_skeleton(spine):
        if len(xy) < 2:
            continue
        xx, yy = xy.astype(np.int32).T
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        length = arc[-1]
        depth = float(np.percentile(valley[yy, xx], 67))
        breadth = float(np.median(thick[yy, xx]))
        component = int(pieces[yy[0], xx[0]])
        keep_length = smoothstep(max(length, reach[component]) / width, .44, 2.0)
        keep_shape = smoothstep(depth, 9, 24) * (.52 + .48 * smoothstep(breadth, .8, 1.9))
        structural = max(keep_length * (.48 + .52 * smoothstep(depth, 5, 16)), keep_shape)
        fade = np.full(len(arc), .62 * (1 - structural), np.float32)
        tip_length = max(2.5, min(.21 * length, .24 * width))
        if start == 1:
            fade = np.maximum(fade, (.49 - .29 * structural) * (1 - smoothstep(arc, .5, tip_length)))
        if end == 1:
            fade = np.maximum(fade, (.49 - .29 * structural) * (1 - smoothstep(length - arc, .5, tip_length)))
        # Preserve network junctions. An apparently short branch connected
        # on both ends can be a bend in a single continuous separator.
        if start > 2:
            fade *= smoothstep(arc, 0, min(3.5, tip_length))
        if end > 2:
            fade *= smoothstep(length - arc, 0, min(3.5, tip_length))
        np.maximum.at(pressure, (yy, xx), fade)
    nearest = cv2.distanceTransformWithLabels((~spine).astype(np.uint8), cv2.DIST_L2, 5,
                                               labelType=cv2.DIST_LABEL_PIXEL)
    distance, labels = nearest
    falloff = 1 - smoothstep(distance, 1.0, 3.0)
    observed = np.maximum(raw_opacity, opacity) * part * corridor
    field = norm_blur(pressure[spine][labels - 1].astype(np.float32), observed, 1.35) * falloff
    banks = cv2.GaussianBlur(cv2.dilate(valley, _ellipse(3)), (0, 0), 1.6)
    anchor = smoothstep(banks, 10, 21)
    return field * (1 - .87 * anchor) * (1 - .90 * seam) * part * interior * corridor


def highlight_limit(lift, source_lift, part, ink, width, light):
    """Estimate a broad highlight pool and retain one source-supported lobe.

    Narrow glints lose amplitude first. Pooling serves only as a cap on
    the existing Screen lift so the falloff never paints light on new pixels.
    """
    signal = cv2.cvtColor(source_lift, cv2.COLOR_RGB2GRAY)
    signal = np.where(ink < .23, signal, 0)
    eligible = part & (signal > .018)
    if eligible.sum() < max(12, .004 * part.sum()):
        return None
    coarse = float(np.clip(.16 * width, 3.0, 16.0))
    envelope = norm_blur(signal, part, coarse)
    width_support = cv2.morphologyEx(np.where(part, signal, 0), cv2.MORPH_OPEN,
                                    _ellipse(2 * max(1, round(.055 * width)) + 1))
    # Use only appreciable reflectance inside the source's light envelope.
    # A normalised score lets a narrow rim remain subordinate to a large lobe.
    score = (.64 * envelope + .36 * width_support) * part
    yy, xx = np.where(part)
    y, x = np.ogrid[:part.shape[0], :part.shape[1]]
    projected = ((x - xx.mean()) * light[0] + (y - yy.mean()) * light[1]) / max(width, 1)
    score *= 1 + .22 * np.tanh(projected)
    score[~eligible] = 0
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    centre_scale = np.clip(.52 * width, 8, 48)
    centre = np.exp(-((x - ix) ** 2 + (y - iy) ** 2) / (2 * centre_scale ** 2))
    prominence = smoothstep(width_support, .24 * np.max(signal), .72 * np.max(signal))
    broad = (.54 + .50 * centre) * envelope
    cap = broad + .29 * centre * prominence * signal
    previous = cv2.cvtColor(lift, cv2.COLOR_RGB2GRAY)
    mix = smoothstep(previous, .010, .048)
    ratio = np.minimum(1.0, cap / np.maximum(previous, 1e-6))
    return (1 - mix * (1 - ratio)).astype(np.float32)


def flowing_arc(source_lift, part, ink, source_luma, body_luma, width, light):
    """Fit the shared course of an elongated, source-supported reflection.

    Estimate its centre and width in the arc's principal direction. The
    profile can cross weak cuts in the light paint, but added light is
    rejected over actual dark ink and cast boundaries.
    """
    signal = cv2.cvtColor(source_lift, cv2.COLOR_RGB2GRAY)
    supported = part & (ink < .20) & (signal > .025)
    if supported.sum() < max(80, .012 * part.sum()):
        return None
    threshold = max(.024, .69 * float(np.percentile(signal[supported], 65)))
    seeds = np.uint8(part & (ink < .20) & (signal > threshold))
    trim = max(1, round(.036 * width))
    bridge = max(2, round(.195 * width))
    clean = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, _ellipse(2 * trim + 1))
    joined = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, _ellipse(2 * bridge + 1))
    n, ids, sizes, _ = cv2.connectedComponentsWithStats(joined)
    if n < 2:
        return None
    selected = ids == 1 + np.argmax(sizes[1:, cv2.CC_STAT_AREA])
    observed = selected & (clean > 0)
    if observed.sum() < max(80, .014 * part.sum()):
        return None
    rows, cols = np.nonzero(observed)
    points = np.column_stack((cols, rows)).astype(np.float32)
    centre = points.mean(0)
    horizontal, vertical = (points - centre).T
    diagonal = np.array([np.mean(horizontal ** 2), np.mean(vertical ** 2)])
    covariance = float(np.mean(horizontal * vertical))
    _, basis = np.linalg.eigh([[diagonal[0], covariance], [covariance, diagonal[1]]])
    normal, tangent = basis.T
    v = horizontal * normal[0] + vertical * normal[1]
    u = horizontal * tangent[0] + vertical * tangent[1]
    lo, hi = np.percentile(u, [2, 98])
    span = hi - lo
    if span < 1.85 * width:
        return None
    # Track the curve even when its centre moves appreciably across the
    # normal, while keeping narrow strand-level irregularities out of it.
    pitch = max(3.0, .075 * width)
    steps = np.arange(lo, hi + pitch, pitch)
    tracks = []
    observed_lift = signal[observed]
    for t in steps:
        near = np.abs(u - t) < 1.45 * pitch
        if near.sum() < 14:
            continue
        a, m, b = np.percentile(v[near], [16, 50, 84])
        tracks.append((t, m, max(2.0, .5 * (b - a)), np.median(observed_lift[near])))
    if len(tracks) < 6:
        return None
    ts, centres, widths, brightness = np.asarray(tracks).T
    centres = cv2.GaussianBlur(centres[None, :], (0, 0), 1.35)[0]
    widths = cv2.GaussianBlur(widths[None, :], (0, 0), 1.45)[0]
    brightness = cv2.GaussianBlur(brightness[None, :], (0, 0), 2.0)[0]
    if np.median(widths) > .24 * span:
        return None
    y, x = np.ogrid[:part.shape[0], :part.shape[1]]
    along = (x - centre[0]) * tangent[0] + (y - centre[1]) * tangent[1]
    cross = (x - centre[0]) * normal[0] + (y - centre[1]) * normal[1]
    traced_centre = np.interp(along, ts, centres)
    local_width = np.interp(along, ts, widths)
    across = (cross - traced_centre) / np.maximum(local_width + 1.1, 2.0)
    lit_side = float(tangent[0] * light[0] + tangent[1] * light[1])
    peak_fit = brightness * (1 + .13 * lit_side * (ts - ts.mean()) / span)
    peak_fit *= np.exp(-.5 * ((ts - ts.mean()) / (.44 * span)) ** 2)
    peak = ts[int(np.argmax(peak_fit))]
    # Centre one broad accent on the curve. Its ends vary with the observed
    # widths and are tapered independently of the source's rectilinear cuts.
    axial = np.exp(-((along - peak) / (.28 * span)) ** 4 / 2)
    radial = np.exp(-.5 * across ** 2)
    inside = smoothstep(along, lo - pitch, lo + .2 * span)
    inside *= 1 - smoothstep(along, hi - .21 * span, hi + pitch)
    shoulder = np.percentile(source_lift[observed], 64, axis=0)
    target = (.87 * axial * radial * inside)[..., None] * shoulder
    room = cv2.dilate(np.uint8(selected), _ellipse(2 * max(2, trim) + 1)) > 0
    local_paint = smoothstep(source_luma - body_luma, -10.5, -2.4)
    fill = room * part * (1 - smoothstep(ink, .06, .25)) * local_paint
    # A short gap must have genuine light on both ends along the curve.
    # This lets a pale painted cut rejoin the light without lifting broad
    # side shade or covering an inked separator.
    reach = max(3., .105 * width)
    indices = np.indices(part.shape, dtype=np.float32)
    dy, dx = np.float32(tangent[::-1] * reach)
    nearby = cv2.dilate(signal, _ellipse(5))
    bank_a = cv2.remap(nearby, indices[1] + dx, indices[0] + dy, cv2.INTER_LINEAR)
    bank_b = cv2.remap(nearby, indices[1] - dx, indices[0] - dy, cv2.INTER_LINEAR)
    paired = smoothstep(np.minimum(bank_a, bank_b), threshold * .65, threshold * 1.08)
    gap = (1 - smoothstep(signal, threshold * .55, threshold)) * paired
    bridge_confidence = (.83 * gap * fill * (axial * radial) ** .45).astype(np.float32)
    return target.astype(np.float32), fill.astype(np.float32), room & part, bridge_confidence


def finish_hair(raw, stack, source, groups, materials, tiers, silhouette, face, protect,
                strength, light, edit_highlights=True, scene_protect=None):
    if strength <= 0 or not any(material == 'hair' for material in materials.values()):
        return
    ownership = hair_parts(source, groups, materials, protect)
    ruler = cv2.distanceTransform(silhouette.astype(np.uint8), cv2.DIST_L2, 5)
    outside_ink = outline_band(silhouette)
    for g, sl in region_slices(ownership, pad=4):
        part = ownership[sl] == g
        area = part.sum()
        if area < 40:
            continue
        # The yellow-family material heuristic also marks some scenery and
        # cloth as hair. Withhold this extra finish on groups whose scene
        # protection already gives the source colour the dominant weight.
        if scene_protect is not None and np.median(scene_protect[sl][part]) > .62:
            continue
        distance = cv2.distanceTransform(part.astype(np.uint8), cv2.DIST_L2, 5)
        width = 2 * float(np.percentile(distance[part], 82))
        if width < 5:
            continue
        local_face = face[sl]
        near_face = cv2.GaussianBlur(local_face.astype(np.float32), (0, 0), 2.0)
        hold = (part * (~local_face) * (1 - near_face) * (~outside_ink[sl])
                * (1 - protect[sl]) * smoothstep(ruler[sl], 4.5, 9.0) * strength)
        lab = to_lab(source[sl])
        source_hair = part & (groups[sl] == g)
        accent = accent_pigment(lab, source_hair, raw.lines[sl][..., 3])
        hold *= 1 - accent
        if not hold.any():
            continue
        shadow = stack.shadow[sl]
        raw_diffuse = raw.flat[sl] * raw.shadow[sl]
        # Remove a narrow excess shadow only where it fails to persist in
        # the larger bundle shape. Thick shadow planes and rims stay put.
        density = 1 - cv2.cvtColor(shadow, cv2.COLOR_RGB2GRAY)
        plane = norm_blur(density, part, np.clip(width * .115, 2.15, 13.0))
        excess = np.maximum(density - plane, 0)
        persistent = smoothstep(plane, .053, .18)
        narrow = smoothstep(excess, .006, .042) * (1 - persistent)
        old_groups = groups[sl].astype(np.float32)
        different = cv2.dilate(old_groups, _ellipse(5)) != cv2.erode(old_groups, _ellipse(5))
        seam = cv2.GaussianBlur(different.astype(np.float32), (0, 0), 1)
        paint_weight = .56 * narrow * smoothstep(distance, 1.2, 4.5) * hold
        paint_weight *= 1 - .86 * seam
        scale = np.clip(excess / np.maximum(density, .002), 0, 1) * paint_weight
        shadow += (1 - shadow) * scale[..., None]
        luma = lab[..., 0]
        if edit_highlights:
            diffuse = stack.flat[sl] * shadow
            screen = stack.highlight[sl]
            lift = (1 - diffuse) * screen
            source_lift = (1 - raw_diffuse) * raw.highlight[sl]
            # An attached dark underside carries its own flat. Its Screen
            # coefficients may describe coloured diffuse paint on that flat.
            # Rank glints only on the source hair body's shared paint surface.
            lit_body = source_hair & (accent < .25)
            ratio = highlight_limit(lift, source_lift, lit_body, raw.lines[sl][..., 3], width, light)
            if ratio is not None:
                adjusted = lift * ratio[..., None]
                flow = flowing_arc(source_lift, lit_body, raw.lines[sl][..., 3], luma,
                                   to_lab(raw.flat[sl])[..., 0], width, light)
                if flow is not None:
                    arc, fill, room, bridge = flow
                    # Attenuation applies even at the original fragmented
                    # rectangles. Any increase needs both arc and paint evidence.
                    smoothed = np.minimum(adjusted, arc)
                    smoothed += fill[..., None] * np.maximum(arc - adjusted, 0)
                    smoothed += bridge[..., None] * np.maximum(raw.flat[sl] - diffuse, 0)
                    old_strength = cv2.cvtColor(lift, cv2.COLOR_RGB2GRAY)
                    competing = smoothstep(old_strength, .010, .043)
                    adjusted *= (1 - .54 * competing * (~room))[..., None]
                    adjusted[room] = smoothed[room]
                applied = lift + (hold * lit_body)[..., None] * (adjusted - lift)
                changed = lit_body & (hold > 0)
                screen[changed] = np.clip(applied[changed] / np.maximum(1 - diffuse[changed], 1e-6), 0, 1)
        line = stack.lines[sl]
        reduction = simplify_ink(line[..., 3], raw.lines[sl][..., 3], luma, part,
                                 ruler[sl] > 4.5, tiers[sl], width, seam)
        line[..., 3] *= 1 - hold * reduction
