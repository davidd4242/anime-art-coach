"""Recover rigid crown pieces and perforated strap surfaces from source geometry."""
from itertools import combinations

import cv2
import numpy as np

from masks import _adjacency, _ellipse, region_slices
from pigment import paired_banks


def crown_shells(labels, stats):
    """Trace tapering silhouette pieces whose cutout has a dark colored wall."""
    fg = labels > 0
    if not fg.any():
        return []
    y = np.flatnonzero(fg.any(axis=1))
    x = np.flatnonzero(fg.any(axis=0))
    height = y[-1] - y[0] + 1
    middle = .5 * (x[0] + x[-1])
    halo = cv2.dilate(np.uint8(~fg), _ellipse(3)) != 0
    exposed = np.bincount(labels[halo & fg], minlength=labels.max() + 1)
    a, b, count, _ = _adjacency(labels)
    neighbors = {}
    for p, q, n in zip(a, b, count):
        neighbors.setdefault(p, {})[q] = n
        neighbors.setdefault(q, {})[p] = n
    shells = []
    assigned = set()
    for r, st in stats.items():
        y0, x0, y1, x1 = st['bbox']
        span = y1 - y0
        if (r in assigned or y0 > y[0] + .06 * height or not .07 * height < span < .19 * height or
                not 58 < st['median'][0] < 88 or not 7 < st['chroma'] < 22 or
                st['hi_tail'] < 12 or exposed[r] < .19 * span):
            continue
        side = np.sign(st['cx'] - middle)
        members = {r}
        pending = [r]
        cutout = False
        while pending:
            p = pending.pop()
            for q, n in neighbors.get(p, {}).items():
                if q in members or q in assigned:
                    continue
                s = stats[q]
                # Narrow tip facets can lean inward; body facets must lie on
                # the exposed side of the shell, away from adjacent hair.
                tip = (s['bbox'][0] < y0 + .23 * span and
                       x0 - .02 * span < s['cx'] < x1 + .02 * span)
                lateral = side * (s['cx'] - st['cx']) > 0
                if (not (tip or lateral) or s['cy'] >= y0 + .68 * span or
                        not .006 * st['area'] < s['area'] < .48 * st['area'] or
                        exposed[q] < max(4, .05 * np.sqrt(s['area'])) or
                        n < max(2, .045 * np.sqrt(s['area']))):
                    continue
                L, C = s['median'][0], s['chroma']
                wall = L < 38 and C > 23
                if C < 20 or wall:
                    cutout |= wall
                    members.add(q)
                    pending.append(q)
        if cutout:
            shells.append(sorted(members))
            assigned.update(members)
    return shells


def aperture_centers(tone, scale, radius_fraction=.17):
    """Connected, nearly round, dark cores surrounded by lighter paint."""
    radius = max(2, round(radius_fraction * scale))
    response = cv2.morphologyEx(tone, cv2.MORPH_BLACKHAT, _ellipse(2 * radius + 1))
    n, ids, stats, centers = cv2.connectedComponentsWithStats(np.uint8(response > .18))
    hits = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w >= tone.shape[1] or y + h >= tone.shape[0]:
            continue
        if not (max(3, .038 * scale) <= min(w, h) <= max(w, h) <= .22 * scale):
            continue
        if not (.52 <= min(w, h) / max(w, h) and .40 <= area / (w * h) <= .93):
            continue
        cx, cy = np.rint(centers[i]).astype(int)
        cut = np.s_[y:y+h, x:x+w]
        core = ids[cut] == i
        if tone[cy, cx] < .18 and np.mean(tone[cut][core]) < .17:
            hits.append((cx, cy))
    return np.array(hits, dtype=int).reshape(-1, 2)


def belt_centerline(u, v, inner):
    """Fit a shallow curved row of apertures through a buckle opening."""
    if not inner.any() or len(u) < 4 or len(u) > 24:
        return None
    best_score = (0, -np.inf)
    best = None
    for ids in combinations(range(len(u)), 3):
        idx = np.asarray(ids)
        if np.ptp(u[idx]) < 1.30 or np.min(np.diff(np.sort(u[idx]))) < .19:
            continue
        A = np.column_stack((u[idx]**2, u[idx], np.ones(3)))
        coefficients = np.linalg.solve(A, v[idx])
        a, b, c = coefficients
        if abs(a) > .32 or abs(b) > .52 or abs(c) > .37:
            continue
        error = np.abs(np.polyval(coefficients, u) - v)
        close = error < .085
        if close.sum() < 4 or not (inner & close).any() or np.ptp(u[close]) < 2.10:
            continue
        score = (int(close.sum()), -float(np.sum(error[close]**2)))
        if score > best_score:
            best_score, best = score, close
    if best is None:
        return None
    return np.polyfit(u[best], v[best], 2), best


def belt_frame(tone, metal, surface):
    """A roughly rectangular ring with a large opening gives the band scale."""
    contours, hi = cv2.findContours(metal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hi is None:
        return None
    for i, c in enumerate(contours):
        parent = hi[0, i, 3]
        if parent < 0:
            continue
        outer = contours[parent]
        x0, y0, width, height = cv2.boundingRect(outer)
        if x0 == 0 or y0 == 0 or x0 + width >= metal.shape[1] or y0 + height >= metal.shape[0]:
            continue
        area = cv2.contourArea(outer)
        (cx, cy), (w, h), _ = rect = cv2.minAreaRect(outer)
        if (min(w, h) < 16 or not .56 <= min(w,h) / max(w,h) <= .92 or
                area < .74 * w * h or not .18 * area <= cv2.contourArea(c) <= .66 * area):
            continue
        corners = cv2.boxPoints(rect)
        sides = np.roll(corners, -1, axis=0) - corners
        axis = sides[np.argmax(np.linalg.norm(sides, axis=1))] / max(w,h)
        if abs(axis[0]) < .72:
            continue
        axis *= np.sign(axis[0])
        across = np.array([-axis[1], axis[0]])
        center = np.array([cx, cy])
        center_xy = aperture_centers(tone, min(w, h))
        center_xy = center_xy[surface[center_xy[:,1], center_xy[:,0]]]
        puv = (center_xy - center) @ np.stack((axis, across), axis=1) / min(w,h)
        u, v = puv.T
        close = (np.abs(u) < 3.1) & (np.abs(v) < 1.18)
        center_xy, u, v = center_xy[close], u[close], v[close]
        inner = np.array([cv2.pointPolygonTest(c, (float(x), float(y)), False) > 0
                          for x, y in center_xy], bool)
        result = belt_centerline(u, v, inner)
        if result is not None:
            fit, passed = result
            return center, axis, across, min(w,h), max(w,h), fit, center_xy[passed]
    return None


def strap_rivets(rgb, labels, u, off, scale):
    """Find compact bright neutral metal inside the dark band neighborhood."""
    for r, sl in region_slices(labels):
        part = labels[sl] == r
        area = part.sum()
        h, w = part.shape
        if not (.017 * scale**2 < area < .060 * scale**2 and
                min(w,h) > .53 * max(w,h) and area > .50 * w*h):
            continue
        position = np.mean(u[sl][part])
        if not .86 < abs(position) < 2.8 or abs(np.mean(off[sl][part])) > .65:
            continue
        values = rgb[sl][part]
        bright = values.mean(1) >= np.percentile(values.mean(1), 77)
        peak = np.median(values[bright], axis=0)
        if peak.mean() < .63 or np.ptp(peak) > .11:
            continue
        yy, xx = np.nonzero(part)
        center = np.array([xx[bright].mean() + sl[1].start, yy[bright].mean() + sl[0].start])
        yield r, center, np.sign(position)


def hanging_band(tone, visible, center, side, axis, across, scale):
    """Follow two dark seams outward from the rivet towards the silhouette."""
    if max(tone.shape) >= 32767:  # OpenCV remap uses signed 16-bit raster coordinates.
        return None
    distances = np.arange(.22, 2.76, .085, dtype=np.float32)
    widths = np.arange(.23, .34, .025, dtype=np.float32)
    bends = np.arange(-.035, .122, .04, dtype=np.float32)
    angles = np.deg2rad(np.arange(21, 56, 2, dtype=np.float32))
    d, width, bend, angle = np.meshgrid(distances, widths, bends, angles, indexing='ij')
    along = side * np.cos(angle)[...,None] * axis + np.sin(angle)[...,None] * across
    cross = side * np.sin(angle)[...,None] * axis - np.cos(angle)[...,None] * across
    points = center + scale * (d[...,None] * along + (bend * d*d)[...,None] * cross)
    ridge = paired_banks(tone, max(2, round(.055 * scale)))
    foreground = visible.astype(np.float32)

    def sample(field, pos):
        grid = pos.reshape(len(distances), -1, 2).astype(np.float32)
        value = cv2.remap(field, grid[...,0], grid[...,1], cv2.INTER_LINEAR)
        return value.reshape(d.shape)

    banks = []
    for sign in (-1, 1):
        values = [sample(ridge, points + sign * scale * (width + offset)[...,None] * cross)
                  for offset in (-.055, 0, .055)]
        banks.append(np.maximum.reduce(values))
    px, py = points[...,0], points[...,1]
    raster = (px > 2) & (py > 2) & (px < tone.shape[1] - 3) & (py < tone.shape[0] - 3)
    inside = sample(foreground, points) > .7
    dual = np.minimum(*banks) * inside
    score = np.clip(dual, 0, .3).mean(axis=0)
    run = np.cumprod(inside, axis=0).sum(axis=0)
    # Require a long run of paint that actually ends at a silhouette edge.
    end = np.minimum(run + 1, len(distances) - 1)[None]
    visible_end = np.take_along_axis(raster & ~inside, end, axis=0)[0]
    usable = (run >= 19) & visible_end & ((dual > .045).sum(axis=0) >= 14)
    best = np.unravel_index(np.argmax(score * usable), score.shape)
    if not usable[best] or score[best] < .051:
        return None
    k = (0,) + best
    yy, xx = np.indices(tone.shape, np.float32)
    x, y = (xx - center[0])/scale, (yy - center[1])/scale
    s = x * along[k][0] + y * along[k][1]
    t = x * cross[k][0] + y * cross[k][1] - bend[k] * s*s
    tail = distances[min(int(run[best]), len(distances)-1)]
    return (s > -.085) & (s < tail + .17) & (np.abs(t) < width[k] + .025)


def perforated_straps(img, labels, groups, materials):
    """Assign strap surfaces and protect observed apertures and buckle contact."""
    fixed = groups.copy()
    materials = materials.copy()
    protection = np.zeros(groups.shape, np.float32)
    regions = np.bincount(labels.ravel())
    next_group = int(groups.max()) + 1
    for g, sl in region_slices(groups):
        if materials.get(g) != 'metal':
            continue
        # Restrict analysis to several frame widths; the aperture row is local.
        h, w = (s.stop - s.start for s in sl)
        if min(w, h) < 16:
            continue
        pad = 3 * min(w, h)
        sl = tuple(slice(max(0, s.start - pad), min(n, s.stop + pad))
                   for s, n in zip(sl, groups.shape))
        old, label = groups[sl], labels[sl]
        metal = np.uint8(old == g)
        rgb = img[sl]
        tone = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        surface = (old > 0) & (old != g) & (label > 0)
        frame = belt_frame(tone, metal, surface)
        if frame is None:
            continue
        center, axis, across, scale, length, curve, apertures = frame
        yy, xx = np.indices(old.shape, np.float32)
        x, y = xx - center[0], yy - center[1]
        u = (x * axis[0] + y * axis[1]) / scale
        v = (x * across[0] + y * across[1]) / scale
        off = v - np.polyval(curve, u)
        holes_u = ((apertures - center) @ axis) / scale
        band = ((u > holes_u.min() - .90) & (u < holes_u.max() + .62) &
                (np.abs(off) < .34) & surface)
        owned = np.bincount(label[band], minlength=len(regions))
        nearby = np.bincount(label[old == g], minlength=len(regions))
        detected = np.unique(label[apertures[:, 1], apertures[:, 0]])
        candidate = (owned > .58 * regions) & (nearby < .2 * regions)
        # Index homogeneous slices, never reassign the background or buckle.
        detected = detected[detected > 0]
        candidate[detected] = True
        candidate[0] = False
        ids = np.flatnonzero(candidate)
        if len(detected) < 3:
            continue
        body = np.isin(label, ids) & surface
        fixed[sl][body] = next_group
        materials[next_group] = 'leather_strap'
        next_group += 1
        rivets = np.zeros(metal.shape, bool)
        for r, rivet, side in strap_rivets(rgb, label, u, off, scale):
            branch = hanging_band(tone, old > 0, rivet, side, axis, across, scale)
            if branch is None:
                continue
            coverage = np.bincount(label[branch], minlength=len(regions))
            members = (coverage > .64 * regions) & (nearby < .2 * regions)
            members[0] = members[r] = False
            # Detected belt holes define the main run; the overlap seam
            # belongs to the hanging piece only where it covers its own labels.
            members[detected] = False
            if np.count_nonzero(members) < 3:
                continue
            part = np.isin(label, np.flatnonzero(members)) & surface
            fixed[sl][part] = next_group
            materials[next_group] = 'leather_strap'
            rivet_pixels = (label == r) & surface
            fixed[sl][rivet_pixels] = next_group + 1
            materials[next_group + 1] = 'metal'
            rivets |= rivet_pixels
            next_group += 2
        hold = np.zeros(metal.shape, np.uint8)
        # Once the curve and panel are established, look for smaller-scale apertures
        # attached to its existing leather. They do not vote on segmentation.
        weaker = aperture_centers(tone, scale, radius_fraction=.115)
        hx, hy = weaker.T
        on_row = body[hy,hx] & (np.abs(off[hy,hx]) < .12)
        for cx, cy in np.vstack((apertures, weaker[on_row])):
            cv2.circle(hold, (int(cx), int(cy)), max(2, round(.089 * scale)), 1, -1)
        source = cv2.GaussianBlur(hold.astype(np.float32), (0,0), max(.7,.014 * scale)) * surface
        # Keep the hardware plate and its existing lower occlusion. The
        # distance feather uses only observed, dark foreground under the rim.
        distance = cv2.distanceTransform(1 - metal, cv2.DIST_L2, 5)
        contact = np.clip((.22 * scale - distance) / (.12 * scale), 0, 1)
        contact *= np.clip((.33 - tone) / .095, 0, 1)
        contact *= np.clip((v - .26) / .16, 0, 1)
        contact *= np.clip((.65 * length / scale - np.abs(u)) / .22, 0, 1) * (old > 0)
        protection[sl] = np.maximum(protection[sl], np.maximum(source, np.maximum(metal | rivets, contact)))
    return fixed, materials, protection
